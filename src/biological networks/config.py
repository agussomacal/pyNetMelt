# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:21:54 2018

@author: crux
"""

import os


root_dir = os.getcwd().split("/src")[0]  # root directory
GTEx_dir = "/media/crux/Data/datasets/GTEx"

Wnetworks_dir = GTEx_dir + "/W_networks_py"
if not os.path.exists(Wnetworks_dir):
    os.mkdir(Wnetworks_dir)

PLOTS_dir = root_dir + "/Graficos"
if not os.path.exists(PLOTS_dir):
    os.mkdir(PLOTS_dir)

byTissue_data_dir = GTEx_dir + "/byTissue_data"
if not os.path.exists(byTissue_data_dir):
    os.mkdir(byTissue_data_dir)


