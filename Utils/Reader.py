# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     reader
   Description :
   Author :       walnut
   date:          2021/1/5
-------------------------------------------------
   Change Activity:
                  2021/1/5:
-------------------------------------------------
"""
__author__ = 'walnut'


import json


def read_from_json(file):
    # default is empty
    data = ""
    with open(file, 'r') as f:
        data = json.load(f)
    return data

