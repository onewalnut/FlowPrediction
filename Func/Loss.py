# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     Loss
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


# custom loss function
def pos_loss(input, target):
    x = torch.norm(input-target, dim=1)
    x = torch.mean(x)

    return x


# mean localization error
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
