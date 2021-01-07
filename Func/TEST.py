# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     TEST
   Description :
   Author :       walnut
   date:          2021/1/5
-------------------------------------------------
   Change Activity:
                  2021/1/5:
-------------------------------------------------
"""
__author__ = 'walnut'

from classes import *
from paras import *
from Utils.Reader import read_from_json


if __name__ == "__main__":
    data = read_from_json("D:/Workplace/PyCharm/FlowPrediction/Data/dataUse.json")
    arr = np.array(data)
    shape = arr.shape
    len = len(shape)
    size = arr.size
    print("EOF")
    pass

