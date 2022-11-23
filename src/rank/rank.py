#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 16:26
# @Author  : Jack Zhao
# @Site    :
# @File    : note.py
# @Software: PyCharm

# #Desc:给结果排序


import pandas as pd
import re
import numpy as np
import os

def imp_argsort(data, sort_value):
    """优化argsort，相同的一致"""
    flag_i = np.array([True] * len(data)) # 是否使用过
    for index_i, value_i in enumerate(data):
        if not flag_i[index_i]:
            continue
        rep_tmp = []
        for index_j, value_j in enumerate(data):
            if index_j <= index_i:
                continue
            if value_j == value_i:
                rep_tmp.append(index_j)
        if not rep_tmp:
            continue
        else:
            indexes = [index_i] + rep_tmp
            true_index = min(sort_value[indexes]) # 最小index
            sort_value[indexes] = true_index
            mask = sort_value > true_index
            sort_value[mask] -= 1
            flag_i[true_index] = False
    return sort_value


def process(data,columns,pattern):
    # 提取avg
    for column in columns:
        str_value = float(re.search(pattern, data[column]).group())
        data[column] = str_value
        data[column] = float(data[column])
    # 排序
    sort_value = np.argsort(np.argsort(-data.values))+1
    sort_value = imp_argsort(data.values,sort_value) # 相同
    for index,column in enumerate(columns):
        data[column] = sort_value[index]
    return data


def concat_df(data, split_row=10):
    """切分"""
    data_p1,data_p2 = data.iloc[:split_row, :], data.iloc[split_row+1:, :].reset_index(drop=True)
    data_p2.columns = data.iloc[split_row,:] # 第一个是dataset
    # print(data_p1.columns,data_p2.columns)
    data_p2 = data_p2.drop(columns=['Data sets'])
    data = pd.concat((data_p1,data_p2),axis=1) # 在行上操作
    data = data.set_index('Data sets', drop=True)
    return data



if __name__ == '__main__':
    files = os.listdir('.')
    file = 'f1.txt'
    data = pd.read_table(file, sep=',',header=0)
    data.columns = [i.strip() for i in data.columns]
    data = concat_df(data,split_row=10) # 3,10
    pattern = re.compile(r'\d*\.?\d*(\d*)')
    df_value_sort = data.apply(lambda x: process(x, data.columns, pattern), axis=1) # row
    df_value_sort.loc['Average', :] = df_value_sort.mean().values
    df_value_sort.to_csv('./{}_sort.csv'.format(file))

    # for file in files:
    #     if file.endswith('txt'):
    #         if file.endswith('1.txt'):
    #             split_row = 3
    #         else:
    #             split_row = 10
    #         data = pd.read_table(file, sep=',',header=0)
    #         data.columns = [i.strip() for i in data.columns]
    #         data = concat_df(data,split_row=split_row)
    #         pattern = re.compile(r'\d*\.?\d*(\d*)')
    #         df_value_sort = data.apply(lambda x: process(x, data.columns, pattern), axis=1) # row
    #         data.loc['Average', :] = df_value_sort.mean().values
    #         df_value_sort.to_csv('./{}_sort.csv'.format(file))
    #     else:
    #         continue

    # 测试img_sort
    # data = np.array([1,2,3,4,4,5])
    # sort_value = np.argsort(np.argsort(-data))+1
    # sort_value = imp_argsort(data,sort_value)
    # print(sort_value)




