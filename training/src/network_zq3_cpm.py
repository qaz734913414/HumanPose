# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

from network_base_zq import max_pool, upsample, inverted_bottleneck, separable_conv, convb, dwconvb, is_trainable

N_KPOINTS = 14
STAGE_NUM = 3

def build_network(input, trainable):
    is_trainable(trainable)

    net = convb(input, 3, 3, 8, 1, name="Conv2d_0")
    net = dwconvb(net, 7, 7, 2, name="Conv2d_1_dw")
    net = convb(net, 1, 1, 8, 1, name="Conv2d_1_sep")

    with tf.variable_scope('MV2'):

        # 128, 112
        mv2_branch_0 = slim.stack(net, inverted_bottleneck,
                                  [
                                      (1, 16, 0, 5),
                                      (1, 16, 0, 5),
                                  ], scope="part0")

        # 64, 56
        mv2_branch_1 = slim.stack(mv2_branch_0, inverted_bottleneck,
                                  [
                                      (1, 16, 1, 5),
                                      (1, 16, 0, 5),
                                  ], scope="part1")

        # 32, 28
        mv2_branch_2 = slim.stack(mv2_branch_1, inverted_bottleneck,
                                  [
                                      (1, 32, 1, 5),
                                      (1, 32, 0, 5),
                                  ], scope="part2")

        # 16, 14
        mv2_branch_3 = slim.stack(mv2_branch_2, inverted_bottleneck,
                                  [
                                      (1, 48, 1, 5),
                                      (1, 48, 0, 5),
                                  ], scope="part3")

        # 8, 7
        mv2_branch_4 = slim.stack(mv2_branch_3, inverted_bottleneck,
                                  [
                                      (1, 64, 1, 5),
                                      (1, 64, 0, 5),
                                  ], scope="part4")

        concat_mv2_3_4 = tf.concat(
            [
                mv2_branch_3,
                upsample(mv2_branch_4, 2, name="mv2_4_upsample")
            ]
            , axis=3)
			
        feat_3_4 = slim.stack(concat_mv2_3_4, inverted_bottleneck,
                                  [
                                      (1, 64, 0, 5),
                                      (1, 64, 0, 5),
                                  ], scope="part5")
								  

        concat_mv2_2_3 = tf.concat(
            [
                mv2_branch_2,
                upsample(feat_3_4, 2, name="feat_3_4_upsample")
            ]
            , axis=3)
			
        feat_2_3 = slim.stack(concat_mv2_2_3, inverted_bottleneck,
                                  [
                                      (1, 48, 0, 5),
                                      (1, 48, 0, 5),
                                  ], scope="part6")

        concat_mv2 = tf.concat(
            [
                max_pool(mv2_branch_0, 2, 2, 2, 2, name="mv2_0_max_pool"),
                mv2_branch_1,
                upsample(feat_2_3, 2, name="feat_2_3_upsample")
            ]
            , axis=3)

    with tf.variable_scope("CPM"):
        l2s = []
        prev = None
        for stage_number in range(STAGE_NUM):
            if prev is not None:
                inputs = tf.concat([concat_mv2, prev], axis=3)
            else:
                inputs = concat_mv2

            kernel_size = 7
            lastest_channel_size = 256
            if stage_number == 0:
                kernel_size = 7
                lastest_channel_size = 256

            _ = slim.stack(inputs, inverted_bottleneck,
                           [
                               (1, 64, 0, kernel_size),
                               (1, 64, 0, kernel_size),
                               (1, 64, 0, kernel_size),
                           ], scope="stage_%d_mv2" % stage_number)
            _ = convb(_, 1, 1, lastest_channel_size, 1, "stage_%d_mv1_0"%stage_number, relu=True)
            _ = convb(_, 1, 1, N_KPOINTS, 1, "stage_%d_mv1_1"%stage_number, relu=True)

            prev = _
            cpm_out = upsample(_, 2, "stage_%d_out" % stage_number)
            l2s.append(cpm_out)

    return cpm_out, l2s
