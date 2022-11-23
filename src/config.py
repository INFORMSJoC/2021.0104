#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 10:23
# @Author  : Jack Zhao
# @Site    : 
# @File    : config.py
# @Software: PyCharm

# #Desc: 这里的config针对focal loss，单元测试记得修改model中对应的config
import warnings


class Config:
    GPU_USED = True
    LR = 0.004
    WEIGHT_DECAY = 1e-3 # 正则化
    BATCH_SIZE = 128
    EPOCHES = 30
    LOG_FREQ = 30
    # WEIGHTS = r"E:/Code/Pycharm/JOC/result/focal/"
    WEIGHTS = r'/data/JOC/result/focal/'
    DATASET = 'let'


def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribute %s" % k)
        setattr(self, k, v)
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith("__"):
            print(k, getattr(self, k))


Config.parse = parse
opt = Config()


if __name__ == '__main__':
    pass