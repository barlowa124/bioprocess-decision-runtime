// SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: BSD-3-Clause

#include "utils/channel.hpp"

#include "common.h"

extern "C" __global__ void flush_channel(ChannelDev* ch_dev) {
    /* push memory access with negative cta id to communicate the kernel is
     * completed */
    reg_info_t ri;
    ri.cta_id_x = -1;
    ch_dev->push(&ri, sizeof(reg_info_t));

    /* flush channel */
    ch_dev->flush();
}
