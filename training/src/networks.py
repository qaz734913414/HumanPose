# -*- coding: utf-8 -*-
# @Time    : 18-3-6 3:20 PM
# @Author  : edvard_hua@live.com
# @FileName: data_filter.py
# @Software: PyCharm

import network_mv2_cpm
import network_zq_cpm
import network_zq1_cpm
import network_zq2_cpm
import network_zq3_cpm
import network_zq4_cpm
import network_zq5_cpm
import network_zq6_cpm
import network_zq7_cpm
import network_zq8_cpm
import network_zq9_cpm
import network_zq10_cpm
import network_zq11_cpm
import network_zq12_cpm
import network_zq13_cpm
import network_zq14_cpm
import network_zq15_cpm
import network_zq16_cpm
import network_zq17_cpm
import network_zq18_cpm
import network_zq19_cpm
import network_zq20_cpm
import network_zq21_cpm
import network_zq22_cpm
import network_mv2_hourglass

def get_network(type, input, trainable=True):
    if type == 'mv2_cpm':
        net, loss = network_mv2_cpm.build_network(input, trainable)
    elif type == "zq_cpm":
        net, loss = network_zq_cpm.build_network(input, trainable)
    elif type == "zq1_cpm":
        net, loss = network_zq1_cpm.build_network(input, trainable)
    elif type == "zq2_cpm":
        net, loss = network_zq2_cpm.build_network(input, trainable)
    elif type == "zq3_cpm":
        net, loss = network_zq3_cpm.build_network(input, trainable)
    elif type == "zq4_cpm":
        net, loss = network_zq4_cpm.build_network(input, trainable)
    elif type == "zq5_cpm":
        net, loss = network_zq5_cpm.build_network(input, trainable)
    elif type == "zq6_cpm":
        net, loss = network_zq6_cpm.build_network(input, trainable)
    elif type == "zq7_cpm":
        net, loss = network_zq7_cpm.build_network(input, trainable)
    elif type == "zq8_cpm":
        net, loss = network_zq8_cpm.build_network(input, trainable)
    elif type == "zq9_cpm":
        net, loss = network_zq9_cpm.build_network(input, trainable)
    elif type == "zq10_cpm":
        net, loss = network_zq10_cpm.build_network(input, trainable)
    elif type == "zq11_cpm":
        net, loss = network_zq11_cpm.build_network(input, trainable)
    elif type == "zq12_cpm":
        net, loss = network_zq12_cpm.build_network(input, trainable)
    elif type == "zq13_cpm":
        net, loss = network_zq13_cpm.build_network(input, trainable)
    elif type == "zq14_cpm":
        net, loss = network_zq14_cpm.build_network(input, trainable)
    elif type == "zq15_cpm":
        net, loss = network_zq15_cpm.build_network(input, trainable)
    elif type == "zq16_cpm":
        net, loss = network_zq16_cpm.build_network(input, trainable)
    elif type == "zq17_cpm":
        net, loss = network_zq17_cpm.build_network(input, trainable)
    elif type == "zq18_cpm":
        net, loss = network_zq18_cpm.build_network(input, trainable)
    elif type == "zq19_cpm":
        net, loss = network_zq19_cpm.build_network(input, trainable)
    elif type == "zq20_cpm":
        net, loss = network_zq20_cpm.build_network(input, trainable)
    elif type == "zq21_cpm":
        net, loss = network_zq21_cpm.build_network(input, trainable)
    elif type == "zq22_cpm":
        net, loss = network_zq22_cpm.build_network(input, trainable)
    elif type == "mv2_hourglass":
        net, loss = network_mv2_hourglass.build_network(input, trainable)        
    return net, loss
