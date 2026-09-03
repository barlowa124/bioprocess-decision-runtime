/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <exception>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#define USE_ASYNC_STREAM
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)

enum class RecvThreadState {
    INIT,
    WORKING,
    STOP,
    FINISHED,
};

struct CTXstate {
    ChannelDev* channel_dev = nullptr;
    ChannelHost channel_host;
    CUmodule tool_module;
    CUfunction flush_channel_func;
    volatile RecvThreadState recv_thread_done = RecvThreadState::INIT;
    bool need_sync = false;
};

#include "tool_func/flush_channel.c"

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;

std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int trace_enabled = 1;
int list_functions_only = 0;
int print_launches = 1;
int start_target_launch = 1;
int max_target_launches = 1;
uint32_t target_launch_count = 0;
uint32_t intervention_offset = UINT32_MAX;
uint32_t pair_low_offset = UINT32_MAX;
uint32_t pair_middle_offset = UINT32_MAX;
uint32_t pair_high_offset = UINT32_MAX;
int intervention_class = -1;
int baseline_target_launch = 1;
int intervention_target_launch = 2;
int intervention_repetitions = 1;
int ureg_probe_reg = -1;
std::string kernel_filter;
std::unordered_set<uint32_t> target_offsets;

/* opcode to id map and reverse map  */
std::map<std::string, int> sass_to_id_map;
std::map<int, std::string> id_to_sass_map;

std::string to_hex_reverse(const std::vector<uint8_t>& bytes) {
    std::stringstream stream;
    stream << std::hex << std::setfill('0');
    for (auto iterator = bytes.rbegin(); iterator != bytes.rend(); ++iterator) {
        stream << std::setw(2) << static_cast<int>(*iterator);
    }
    return stream.str();
}

bool parse_offset(const std::string& raw, uint32_t* parsed) {
    size_t begin = raw.find_first_not_of(" \t\r\n");
    size_t end = raw.find_last_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return false;
    }
    std::string value = raw.substr(begin, end - begin + 1);
    try {
        size_t consumed = 0;
        unsigned long result = std::stoul(value, &consumed, 0);
        if (consumed != value.size() || result > UINT32_MAX) {
            return false;
        }
        *parsed = static_cast<uint32_t>(result);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

uint32_t require_offset(const char* name, const std::string& value) {
    uint32_t parsed = 0;
    if (!parse_offset(value, &parsed)) {
        fprintf(stderr, "Invalid %s value: %s\n", name, value.c_str());
        exit(EXIT_FAILURE);
    }
    return parsed;
}

void* recv_thread_fun(void* args);

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(trace_enabled, "TRACE_ENABLED", 1,
                "Inject carry trace calls or only discover instructions");
    GET_VAR_INT(list_functions_only, "LIST_FUNCTIONS_ONLY", 0,
                "Print matching function names without decoding instructions");
    GET_VAR_INT(print_launches, "PRINT_LAUNCHES", 1,
                "Print matching kernel launch descriptions");
    GET_VAR_INT(start_target_launch, "START_TARGET_LAUNCH", 1,
                "First matching kernel launch to instrument");
    GET_VAR_INT(max_target_launches, "MAX_TARGET_LAUNCHES", 1,
                "Maximum matching kernel launches to instrument");
    const char* filter = getenv("KERNEL_FILTER");
    kernel_filter = filter ? filter : "";
    printf("%20s = %s - Kernel name substring filter\n", "KERNEL_FILTER",
           kernel_filter.c_str());
    const char* offsets = getenv("TARGET_OFFSETS");
    if (offsets) {
        std::string value(offsets);
        size_t start = 0;
        while (start < value.size()) {
            size_t end = value.find(',', start);
            std::string token = value.substr(start, end - start);
            target_offsets.insert(require_offset("TARGET_OFFSETS", token));
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
    }
    printf("%20s = %s - Comma-separated instruction offsets\n",
           "TARGET_OFFSETS", offsets ? offsets : "");
    const char* intervention = getenv("INTERVENTION_OFFSET");
    if (intervention) {
        intervention_offset = require_offset("INTERVENTION_OFFSET", intervention);
    }
    const char* pair_low = getenv("PAIR_LOW_OFFSET");
    const char* pair_middle = getenv("PAIR_MIDDLE_OFFSET");
    const char* pair_high = getenv("PAIR_HIGH_OFFSET");
    if ((pair_low == nullptr) != (pair_high == nullptr)) {
        fprintf(stderr, "PAIR_LOW_OFFSET and PAIR_HIGH_OFFSET must be set together\n");
        exit(EXIT_FAILURE);
    }
    if (pair_low && pair_high) {
        pair_low_offset = require_offset("PAIR_LOW_OFFSET", pair_low);
        pair_high_offset = require_offset("PAIR_HIGH_OFFSET", pair_high);
        if (pair_middle) {
            pair_middle_offset = require_offset("PAIR_MIDDLE_OFFSET", pair_middle);
        }
    }
    GET_VAR_INT(intervention_class, "INTERVENTION_CLASS", -1,
                "Controlled synthetic carry class");
    GET_VAR_INT(baseline_target_launch, "BASELINE_TARGET_LAUNCH", 1,
                "Matching launch used for baseline post-state");
    GET_VAR_INT(intervention_target_launch, "INTERVENTION_TARGET_LAUNCH", 2,
                "First matching launch used for controlled intervention");
    GET_VAR_INT(intervention_repetitions, "INTERVENTION_REPETITIONS", 1,
                "Number of consecutive controlled intervention launches");
    GET_VAR_INT(ureg_probe_reg, "UREG_PROBE_REG", -1,
                "Uniform register used for synthetic read/write/restore probe");
    printf("%20s = %s - Synthetic intervention instruction offset\n",
           "INTERVENTION_OFFSET", intervention ? intervention : "");
    printf("%20s = %s,%s,%s - Paired low/middle/high instruction offsets\n",
           "PAIR_OFFSETS", pair_low ? pair_low : "",
           pair_middle ? pair_middle : "", pair_high ? pair_high : "");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    /* set mutex as recursive */
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&cuda_event_mutex, &attr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }
        std::string function_name = nvbit_get_func_name(ctx, f);
        if (!kernel_filter.empty() &&
            function_name.find(kernel_filter) == std::string::npos) {
            continue;
        }
        if (list_functions_only) {
            printf("FUNCTION %s\n", function_name.c_str());
            continue;
        }
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        if (verbose) {
            printf("Inspecting function %s at address 0x%lx\n",
                   function_name.c_str(), nvbit_get_func_addr(ctx, f));
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
                cnt++;
                continue;
            }
            if (!target_offsets.empty() &&
                target_offsets.find(instr->getOffset()) == target_offsets.end()) {
                cnt++;
                continue;
            }
            if (verbose) {
                std::vector<uint8_t> binary;
                instr->getSassBinary(
                    [](uint8_t byte, void* data) {
                        reinterpret_cast<std::vector<uint8_t>*>(data)->push_back(
                            byte);
                    },
                    &binary);
                printf("/*%04x*/ %s /* 0x%s */\n", instr->getOffset(),
                       instr->getSass(), to_hex_reverse(binary).c_str());
            }

            std::string opcode = instr->getOpcode();
            if (opcode.compare(0, 5, "IADD3") != 0 &&
                opcode.compare(0, 3, "LEA") != 0 &&
                opcode.compare(0, 6, "UIADD3") != 0 &&
                opcode.compare(0, 4, "ULEA") != 0 &&
                opcode.compare(0, 3, "P2R") != 0) {
                cnt++;
                continue;
            }
            if (!trace_enabled) {
                cnt++;
                continue;
            }

            if (sass_to_id_map.find(instr->getSass()) == sass_to_id_map.end()) {
                int opcode_id = sass_to_id_map.size();
                sass_to_id_map[instr->getSass()] = opcode_id;
                id_to_sass_map[opcode_id] = std::string(instr->getSass());
            }

            int opcode_id = sass_to_id_map[instr->getSass()];
            std::vector<int> reg_num_list;
            std::vector<bool> reg_is_uniform;
            std::vector<int> pred_num_list;
            for (int i = 0; i < instr->getNumOperands(); i++) {
                const InstrType::operand_t* op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::REG) {
                    reg_num_list.push_back(op->u.reg.num);
                    reg_is_uniform.push_back(false);
                } else if (op->type == InstrType::OperandType::UREG) {
                    reg_num_list.push_back(op->u.reg.num);
                    reg_is_uniform.push_back(true);
                } else if (op->type == InstrType::OperandType::PRED &&
                           op->u.pred.num < InstrType::PT) {
                    pred_num_list.push_back(op->u.pred.num);
                }
            }
            if (reg_num_list.size() > 8 || pred_num_list.size() > 8) {
                cnt++;
                continue;
            }

            if (ureg_probe_reg >= 0) {
                nvbit_insert_call(instr, "probe_ureg", IPOINT_BEFORE);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, ureg_probe_reg);
            }

            for (int capture_point = 0; capture_point < 2; capture_point++) {
                nvbit_insert_call(instr, "record_reg_val",
                                  capture_point == 0 ? IPOINT_BEFORE
                                                     : IPOINT_AFTER);
                nvbit_add_call_arg_guard_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, opcode_id);
                nvbit_add_call_arg_const_val64(
                    instr, (uint64_t)ctx_state->channel_dev);
                nvbit_add_call_arg_const_val32(instr, capture_point);
                nvbit_add_call_arg_pred_reg(instr);
                nvbit_add_call_arg_upred_reg(instr);
                nvbit_add_call_arg_launch_val32(instr, 0);
                nvbit_add_call_arg_const_val32(instr, instr->getOffset());
                nvbit_add_call_arg_const_val32(instr, intervention_offset);
                nvbit_add_call_arg_const_val32(instr, pair_low_offset);
                nvbit_add_call_arg_const_val32(instr, pair_middle_offset);
                nvbit_add_call_arg_const_val32(instr, pair_high_offset);
                nvbit_add_call_arg_const_val32(instr, intervention_class);
                nvbit_add_call_arg_const_val32(instr, baseline_target_launch);
                nvbit_add_call_arg_const_val32(instr, intervention_target_launch);
                nvbit_add_call_arg_const_val32(instr, intervention_repetitions);
                nvbit_add_call_arg_const_val32(
                    instr, opcode.find(".X") != std::string::npos);
                nvbit_add_call_arg_const_val32(
                    instr, opcode.compare(0, 5, "IADD3") == 0);
                nvbit_add_call_arg_const_val32(
                    instr, pred_num_list.empty() ? -1 : pred_num_list[0]);
                for (size_t position = 0; position < 4; position++) {
                    int reg_num = -1;
                    if (position < reg_num_list.size()) {
                        reg_num = reg_is_uniform[position]
                                      ? -(reg_num_list[position] + 2)
                                      : reg_num_list[position];
                    }
                    nvbit_add_call_arg_const_val32(instr, reg_num);
                }
                nvbit_add_call_arg_const_val32(instr, pred_num_list.size());
                nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
                for (int num : pred_num_list) {
                    nvbit_add_call_arg_pred_val_at(instr, num, true);
                }
                for (size_t index = 0; index < reg_num_list.size(); index++) {
                    if (reg_is_uniform[index]) {
                        nvbit_add_call_arg_ureg_val(
                            instr, reg_num_list[index], true);
                    } else {
                        nvbit_add_call_arg_reg_val(
                            instr, reg_num_list[index], true);
                    }
                }
            }
            cnt++;
        }
    }
}

void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    CUDA_SAFECALL(
        cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev)));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

static void leave_kernel_launch(CUcontext ctx, CTXstate* ctx_state) {
#ifdef USE_ASYNC_STREAM
    /* make sure current kernel is completed */
    CUDA_SAFECALL(cudaDeviceSynchronize());
    /* issue flush of channel so we are sure all the memory accesses
     * have been pushed */
    void* args[] = {&ctx_state->channel_dev};
    nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0,
                        nullptr, args, nullptr);
    CUDA_SAFECALL(cudaDeviceSynchronize());
#endif
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    /* Skip callbacks for contexts not yet initialized in ctx_state_map
     * (e.g. green-context-derived CUcontexts where nvbit_at_ctx_init has
     * not yet run). Using operator[] on an unknown key would insert a null
     * CTXstate* and cause a segfault when dereferenced below. */
    if (ctx_state_map.find(ctx) == ctx_state_map.end()) {
        skip_callback_flag = false;
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            /* Make sure GPU is idle */
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            int nregs = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

            instrument_function_if_needed(ctx, func);

            std::string launch_name = nvbit_get_func_name(ctx, func);
            bool matches_filter =
                kernel_filter.empty() ||
                launch_name.find(kernel_filter) != std::string::npos;
            if (matches_filter) {
                target_launch_count++;
                nvbit_set_at_launch(ctx, func, target_launch_count);
            }
            bool enable =
                matches_filter &&
                target_launch_count >= (uint32_t)start_target_launch &&
                (max_target_launches <= 0 ||
                 target_launch_count <
                     (uint32_t)(start_target_launch + max_target_launches));
            nvbit_enable_instrumented(ctx, func, enable);
            ctx_state->need_sync = enable;

            if (print_launches && enable &&
                (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                 cbid == API_CUDA_cuLaunchKernelEx)) {
                cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
                printf(
                    "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - "
                    "nregs "
                    "%d - shmem %d - cuda stream id %ld\n",
                    nvbit_get_func_name(ctx, func), p->config->gridDimX,
                    p->config->gridDimY, p->config->gridDimZ,
                    p->config->blockDimX, p->config->blockDimY,
                    p->config->blockDimZ, nregs,
                    shmem_static_nbytes + p->config->sharedMemBytes,
                    (uint64_t)p->config->hStream);
            } else if (print_launches && enable) {
                cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                printf(
                    "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - "
                    "nregs "
                    "%d - shmem %d - cuda stream id %ld\n",
                    nvbit_get_func_name(ctx, func), p->gridDimX, p->gridDimY,
                    p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ,
                    nregs, shmem_static_nbytes + p->sharedMemBytes,
                    (uint64_t)p->hStream);
            }
        } else if (ctx_state->need_sync) {
            leave_kernel_launch(ctx, ctx_state);
            ctx_state->need_sync = false;
        }
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);

        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes + sizeof(reg_info_t) <= num_recv_bytes) {
                reg_info_t* ri = (reg_info_t*)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (ri->cta_id_x == -1) {
                    break;
                }

                printf("CTA %d,%d,%d - warp %d - %s - %s:\n", ri->cta_id_x,
                       ri->cta_id_y, ri->cta_id_z, ri->warp_id,
                       id_to_sass_map[ri->opcode_id].c_str(),
                       ri->capture_point == 0 ? "BEFORE" : "AFTER");
                printf("* PAIR_STATE launch=%d role=%d ready=0x%x intervention=%d\n",
                       ri->launch_index, ri->pair_role, ri->pair_ready_mask,
                       ri->pair_intervention);

                printf("* ACTIVE_MASK 0x%08x\n* THREAD_COORDS ",
                       ri->active_mask);
                for (int i = 0; i < 32; i++) {
                    printf("T%d:%u,%u,%u ", i, ri->thread_x[i],
                           ri->thread_y[i], ri->thread_z[i]);
                }
                printf("\n");

                printf("* PRED_REG ");
                for (int i = 0; i < 32; i++) {
                    printf("T%d:0x%08x ", i, ri->predicate_regs[i]);
                }
                printf("\n* UPRED_REG ");
                for (int i = 0; i < 32; i++) {
                    printf("T%d:0x%08x ", i,
                           ri->uniform_predicate_regs[i]);
                }
                printf("\n");

                for (int pred_idx = 0; pred_idx < ri->num_preds; pred_idx++) {
                    printf("* Pred%d ", pred_idx);
                    for (int i = 0; i < 32; i++) {
                        printf("T%d:%u ", i,
                               ri->predicate_values[i][pred_idx]);
                    }
                    printf("\n");
                }

                for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
                    printf("* ");
                    for (int i = 0; i < 32; i++) {
                        printf("Reg%d_T%d: 0x%08x ", reg_idx, i,
                               ri->reg_vals[i][reg_idx]);
                    }
                    printf("\n");
                }

                printf("\n");
                num_processed_bytes += sizeof(reg_info_t);
            }
        }
    }
    free(recv_buffer);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    CTXstate* ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;

    nvbit_load_tool_module(ctx, (const void*)flush_channel_bin,
                           &ctx_state->tool_module);
    nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel",
                                &ctx_state->flush_channel_func);

    pthread_mutex_unlock(&mutex);
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    init_context_state(ctx);
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    if (ctx_state->need_sync) {
        void* args[] = {&ctx_state->channel_dev};
        nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1,
                            1, 0, nullptr, args, nullptr);
        CUDA_SAFECALL(cudaDeviceSynchronize());
    }

    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    if (ctx_state->recv_thread_done != RecvThreadState::INIT) {
        ctx_state->recv_thread_done = RecvThreadState::STOP;
        while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);
    }

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    ctx_state_map.erase(ctx);
    pthread_mutex_unlock(&mutex);
}
