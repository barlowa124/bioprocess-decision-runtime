/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>
#include <cstdarg>

#include "utils/utils.h"
#include "instr_types.h"
#include "nvbit_reg_rw.h"

/* for channel */
#define USE_ASYNC_STREAM
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

__device__ uint32_t baseline_destination[16384];
__device__ uint32_t baseline_predicate_register[16384];
__device__ uint32_t saved_sources[16384][3];
__device__ uint32_t baseline_ready[16384];
__device__ uint32_t pair_baseline_destination[2][16384];
__device__ uint32_t pair_baseline_predicate_register[2][16384];
__device__ uint32_t pair_baseline_sources[2][16384][3];
__device__ int32_t pair_operand_codes[2][16384][4];
__device__ uint32_t pair_baseline_ready[2][16384];
__device__ uint32_t middle_baseline_destination[16384];
__device__ int32_t middle_destination_code[16384];
__device__ uint32_t middle_baseline_ready[16384];

__device__ __forceinline__ void write_operand(int32_t operand, uint32_t value) {
    if (operand >= 0 && operand < InstrType::RZ) {
        nvbit_write_reg(operand, value);
    } else if (operand <= -2 && -operand - 2 < InstrType::URZ) {
        nvbit_write_ureg(-operand - 2, value);
    }
}

extern "C" __device__ __noinline__ void probe_ureg(int pred, int reg_num) {
    if (!pred || get_laneid() != 0 || reg_num < 0 ||
        reg_num >= InstrType::URZ) {
        return;
    }
    uint32_t original = nvbit_read_ureg(reg_num);
    uint32_t test_value = 0x13579bdf;
    nvbit_write_ureg(reg_num, test_value);
    uint32_t observed = nvbit_read_ureg(reg_num);
    nvbit_write_ureg(reg_num, original);
    uint32_t restored = nvbit_read_ureg(reg_num);
    printf("UREG_PROBE reg=%d original=0x%08x observed=0x%08x restored=0x%08x valid=%d\n",
           reg_num, original, observed, restored,
           observed == test_value && restored == original);
}

extern "C" __device__ __noinline__ void record_reg_val(
    int pred, int opcode_id, uint64_t pchannel_dev, int32_t capture_point,
    int32_t predicate_reg, int32_t uniform_predicate_reg,
    uint32_t launch_index, uint32_t instruction_offset,
    uint32_t intervention_offset, uint32_t pair_low_offset,
    uint32_t pair_middle_offset, uint32_t pair_high_offset,
    int32_t intervention_class,
    uint32_t baseline_target_launch, uint32_t intervention_target_launch,
    uint32_t intervention_repetitions, int32_t instruction_is_high,
    int32_t instruction_is_iadd, int32_t intervention_predicate_num,
    int32_t destination_reg,
    int32_t source_reg_0, int32_t source_reg_1,
    int32_t source_reg_2, int32_t num_preds, int32_t num_regs...) {
    if (!pred || num_preds < 0 || num_preds > 8 || num_regs < 0 ||
        num_regs > 8) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    reg_info_t ri;

    int4 cta = get_ctaid();
    ri.cta_id_x = cta.x;
    ri.cta_id_y = cta.y;
    ri.cta_id_z = cta.z;
    ri.warp_id = get_warpid();
    ri.opcode_id = opcode_id;
    ri.capture_point = capture_point;
    ri.launch_index = launch_index;
    ri.pair_role = -1;
    ri.pair_ready_mask = 0;
    ri.pair_intervention = 0;
    ri.num_regs = num_regs;
    ri.num_preds = num_preds;
    ri.active_mask = active_mask;

    for (int tid = 0; tid < 32; tid++) {
        ri.thread_x[tid] = __shfl_sync(active_mask, threadIdx.x, tid);
        ri.thread_y[tid] = __shfl_sync(active_mask, threadIdx.y, tid);
        ri.thread_z[tid] = __shfl_sync(active_mask, threadIdx.z, tid);
        ri.predicate_regs[tid] = __shfl_sync(active_mask, predicate_reg, tid);
        ri.uniform_predicate_regs[tid] =
            __shfl_sync(active_mask, uniform_predicate_reg, tid);
    }

    uint32_t local_reg_values[8] = {};
    if (num_preds || num_regs) {
        va_list vl;
        va_start(vl, num_regs);

        for (int i = 0; i < num_preds; i++) {
            uint32_t val = va_arg(vl, uint32_t);
            for (int tid = 0; tid < 32; tid++) {
                ri.predicate_values[tid][i] =
                    __shfl_sync(active_mask, val, tid);
            }
        }
        for (int i = 0; i < num_regs; i++) {
            uint32_t val = va_arg(vl, uint32_t);
            local_reg_values[i] = val;
            for (int tid = 0; tid < 32; tid++) {
                ri.reg_vals[tid][i] = __shfl_sync(active_mask, val, tid);
            }
        }
        va_end(vl);
    }

    uint32_t block_index =
        blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    uint32_t thread_index =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    uint32_t state_index =
        block_index * (blockDim.x * blockDim.y * blockDim.z) + thread_index;
    bool pair_offsets_authorized =
        (pair_low_offset == 0xd0 && pair_high_offset == 0xe0) ||
        (pair_low_offset == 0xf0 && pair_high_offset == 0x100) ||
        (pair_low_offset == 0x1770 && pair_high_offset == 0x17b0) ||
        (pair_low_offset == 0x1fa0 && pair_high_offset == 0x1fc0) ||
        (pair_low_offset == 0x3350 && pair_middle_offset == 0x3370 &&
         pair_high_offset == 0x3390);
    bool pair_mode =
        pair_offsets_authorized && pair_low_offset != 0xffffffff &&
        pair_high_offset != 0xffffffff &&
        (instruction_offset == pair_low_offset ||
         instruction_offset == pair_middle_offset ||
         instruction_offset == pair_high_offset) &&
        intervention_class >= 0 &&
        intervention_class <=
            (instruction_offset == pair_middle_offset ? 1 : num_preds) &&
        destination_reg >= 0 && source_reg_0 != -1 &&
        (instruction_offset == pair_middle_offset || source_reg_1 != -1) &&
        state_index < 16384;
    bool pair_is_low = instruction_offset == pair_low_offset;
    bool pair_is_middle = instruction_offset == pair_middle_offset;
    bool pair_is_high = instruction_offset == pair_high_offset;
    int pair_slot = pair_is_high ? 1 : 0;
    ri.pair_role = pair_is_low ? 0 : pair_is_middle ? 1 : pair_is_high ? 2 : -1;
    if (pair_mode && !pair_is_middle &&
        launch_index == baseline_target_launch && capture_point == 1) {
        if (pair_baseline_ready[pair_slot][state_index] == 0) {
            pair_baseline_destination[pair_slot][state_index] = local_reg_values[0];
            pair_baseline_predicate_register[pair_slot][state_index] = predicate_reg;
            pair_baseline_sources[pair_slot][state_index][0] = local_reg_values[1];
            pair_baseline_sources[pair_slot][state_index][1] = local_reg_values[2];
            pair_baseline_sources[pair_slot][state_index][2] =
                source_reg_2 == -1 ? 0 : local_reg_values[3];
            pair_operand_codes[pair_slot][state_index][0] = destination_reg;
            pair_operand_codes[pair_slot][state_index][1] = source_reg_0;
            pair_operand_codes[pair_slot][state_index][2] = source_reg_1;
            pair_operand_codes[pair_slot][state_index][3] = source_reg_2;
            pair_baseline_ready[pair_slot][state_index] = 1;
        } else {
            pair_baseline_ready[pair_slot][state_index] = 2;
        }
        __threadfence_system();
    }
    if (pair_mode && pair_is_middle &&
        launch_index == baseline_target_launch && capture_point == 1) {
        if (middle_baseline_ready[state_index] == 0) {
            middle_baseline_destination[state_index] = local_reg_values[0];
            middle_destination_code[state_index] = destination_reg;
            middle_baseline_ready[state_index] = 1;
        } else {
            middle_baseline_ready[state_index] = 2;
        }
        __threadfence_system();
    }
    bool pair_intervention =
        pair_mode && launch_index >= intervention_target_launch &&
        launch_index < intervention_target_launch + intervention_repetitions &&
        pair_baseline_ready[0][state_index] == 1 &&
        pair_baseline_ready[1][state_index] == 1 &&
        (pair_middle_offset == 0xffffffff ||
         middle_baseline_ready[state_index] == 1);
    ri.pair_ready_mask =
        pair_baseline_ready[0][state_index] |
        (middle_baseline_ready[state_index] << 1) |
        (pair_baseline_ready[1][state_index] << 2);
    ri.pair_intervention = pair_intervention;
    if (pair_intervention && capture_point == 0) {
        if (pair_is_low) {
            if (instruction_is_iadd) {
                uint32_t first = intervention_class == 0 ? 1 : 0xffffffff;
                uint32_t second = intervention_class == 0 ? 2 :
                                  intervention_class == 1 ? 1 : 0xffffffff;
                uint32_t third = intervention_class == 0 ? 3 :
                                 intervention_class == 1 ? 0 : 0xffffffff;
                write_operand(source_reg_0, first);
                write_operand(source_reg_1, second);
                write_operand(source_reg_2, third);
            } else {
                write_operand(source_reg_0, 1);
                write_operand(source_reg_1,
                              intervention_class == 0 ? 2 : 0xffffffff);
            }
        } else if (pair_is_high) {
            write_operand(source_reg_0, instruction_is_iadd ? 0 : 1);
            write_operand(source_reg_1, 0);
            if (source_reg_2 != -1) {
                write_operand(source_reg_2, 0);
            }
        }
    }
    if (pair_intervention && pair_is_middle && capture_point == 1) {
        write_operand(middle_destination_code[state_index],
                      middle_baseline_destination[state_index]);
    }
    if (pair_intervention && pair_is_high && capture_point == 1) {
        for (int slot = 0; slot < 2; slot++) {
            write_operand(pair_operand_codes[slot][state_index][0],
                          pair_baseline_destination[slot][state_index]);
            for (int source = 0; source < 3; source++) {
                int operand = pair_operand_codes[slot][state_index][source + 1];
                if (operand != -1) {
                    write_operand(operand,
                                  pair_baseline_sources[slot][state_index][source]);
                }
            }
        }
        nvbit_write_pred_reg(pair_baseline_predicate_register[0][state_index]);
        nvbit_write_pred_reg(pair_baseline_predicate_register[1][state_index]);
    }

    bool intervention_offset_allowed =
        intervention_offset == 0xd0 || intervention_offset == 0xe0 ||
        intervention_offset == 0xf0 || intervention_offset == 0x100 ||
        intervention_offset == 0x1770 || intervention_offset == 0x17b0 ||
        intervention_offset == 0x1fa0 || intervention_offset == 0x1fc0 ||
        intervention_offset == 0x3350 || intervention_offset == 0x3390;
    bool intervention_valid =
        !pair_mode && intervention_offset_allowed &&
        instruction_offset == intervention_offset &&
        intervention_class >= 0 &&
        intervention_class <=
            (instruction_is_high || source_reg_2 == -1 ? 1 : 2) &&
        (!instruction_is_high || intervention_predicate_num >= 0) &&
        destination_reg >= 0 && source_reg_0 != -1 && source_reg_1 != -1 &&
        state_index < 16384;
    if (intervention_valid && launch_index == baseline_target_launch &&
        capture_point == 1) {
        if (baseline_ready[state_index] == 0) {
            baseline_destination[state_index] = local_reg_values[0];
            baseline_predicate_register[state_index] = predicate_reg;
            baseline_ready[state_index] = 1;
        } else {
            baseline_ready[state_index] = 2;
        }
        __threadfence_system();
    }
    if (intervention_valid && launch_index >= intervention_target_launch &&
        launch_index < intervention_target_launch + intervention_repetitions &&
        baseline_ready[state_index] == 1) {
        if (capture_point == 0) {
            saved_sources[state_index][0] = local_reg_values[1];
            saved_sources[state_index][1] = local_reg_values[2];
            saved_sources[state_index][2] =
                source_reg_2 == -1 ? 0 : local_reg_values[3];
            bool two_source = source_reg_2 == -1;
            uint32_t first = instruction_is_high ? 0 :
                             intervention_class == 0 ? 1 :
                             two_source ? 1 : 0xffffffff;
            uint32_t second = instruction_is_high ? 0 :
                              intervention_class == 0 ? 2 :
                              two_source ? 0xffffffff :
                              intervention_class == 1 ? 1 : 0xffffffff;
            uint32_t third = instruction_is_high ? 0 :
                             intervention_class == 0 ? 3 :
                             intervention_class == 1 ? 0 : 0xffffffff;
            write_operand(source_reg_0, first);
            write_operand(source_reg_1, second);
            if (source_reg_2 != -1) {
                write_operand(source_reg_2, third);
            }
            if (instruction_is_high) {
                uint32_t mask = 1u << intervention_predicate_num;
                uint32_t controlled_predicate =
                    intervention_class ? predicate_reg | mask
                                       : predicate_reg & ~mask;
                nvbit_write_pred_reg(controlled_predicate);
            }
        } else {
            nvbit_write_reg(destination_reg, baseline_destination[state_index]);
            write_operand(source_reg_0, saved_sources[state_index][0]);
            write_operand(source_reg_1, saved_sources[state_index][1]);
            if (source_reg_2 != -1) {
                write_operand(source_reg_2, saved_sources[state_index][2]);
            }
            nvbit_write_pred_reg(baseline_predicate_register[state_index]);
        }
    }

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ri, sizeof(reg_info_t));
    }
}
