# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     paras
   Description :
   Author :       walnut
   date:          2021/1/5
-------------------------------------------------
   Change Activity:
                  2021/1/5:
-------------------------------------------------
"""
__author__ = 'walnut'


# Parameters of train and validate process
EPOCH = 500
TRAIN_BATCH = 64
TEST_BATCH = 32
LR = 0.5
LR_DECAY = 0.9
DECAY_PERIOD = 25
GPU_ID = "0"
GPU_IDS = [2, 3]

SAVE_FLAG = 10000000000000

# Parameters for sequence data and  RNN layers
INPUT_SIZE = 1
SEQ_LEN = 6
CALIBRATE_INTERVAL = 1


# temporalNet paras
LOCAL_HIDDEN_SIZE = 64
NUM_PART = 5
REGRESS_SIZE = 2048


# Data file
DATA_FILE = "/home/hetao/projects/FlowPrediction/Data/dataUse.json"
DATA_FILE_PATH = " "

RES_PATH = "/home/hetao/Data/FlowPrediction/"

