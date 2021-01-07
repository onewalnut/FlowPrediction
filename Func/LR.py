# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     LR
   Description :
   Author :       walnut
   date:          2021/1/5
-------------------------------------------------
   Change Activity:
                  2021/1/5:
-------------------------------------------------
"""
__author__ = 'walnut'


# learning rate setting
def adjust_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    # lr = 1e-4 * (0.1 ** (epoch // 80))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr



