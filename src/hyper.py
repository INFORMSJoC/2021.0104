#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 10:32
# @Author  : Jack Zhao
# @Site    : 
# @File    : hyper.py
# @Software: PyCharm

# #Desc:这里是超参数实验环节。

from model.bcwf_s import BcwfS
from main import run
import pandas as pd

def run_T(dataset_name,T,grad_constraints):
    """执行模型"""
    model_class = BcwfS(dataset_name, T, grad_constraints) # 这里的self.T,grad_constraints可以试着修改
    model_class.apply_all()
    metrics = model_class.display()

    return metrics




if __name__ == '__main__':
    # 这是T的检验
    # dataset_names = ['aba', 'let']
    # metrics_baseline = ['roc', 'avg_roc', 'std_roc', 'pos_f1', 'avg_pos_f1', 'std_pos_f1', 'weight_f1', 'avg_weight_f1',
    #                     'std_weight_f1', 'gmean', 'avg_gmean', 'std_gmean']  # 需要保证和返回的顺序一致
    # metrics = metrics_baseline + list(map(lambda x: x + '_t', metrics_baseline))
    # model = "BCWF_s"
    # Ts = list(range(7,25,2))
    # # 开始训练
    # print("==============开始超参数检验===========")
    # for dataset_name in dataset_names:
    #     # 用于存储数据
    #     df_metrics = pd.DataFrame(columns=metrics, index=Ts)
    #     # 遍历T
    #     for T in Ts:
    #         print("=====当前正在执行对{}数据集T={}的预测=========".format(dataset_name,T))
    #         metric = run_T(dataset_name,T,0)
    #         df_metrics.loc[T] = metric
    #     df_metrics.to_csv('./result/hyper/T/{}.csv'.format(dataset_name))
    #     print('{}数据存储成功!'.format(dataset_name))
    # print("==============All Done!!!===========")


    # """下面代码的执行需要手动更换基分类器为Bagging"""
    # dataset_names = ['aba', 'bal', 'hab', 'hou', 'let', 'wdbc', 'wpbc', 'yea', 'pim', 'p1', 'p2', 'p3', 'cre']  # p3报错移除
    # metrics_baseline = ['roc', 'avg_roc', 'std_roc', 'pos_f1', 'avg_pos_f1', 'std_pos_f1', 'weight_f1', 'avg_weight_f1',
    #                     'std_weight_f1', 'gmean', 'avg_gmean', 'std_gmean']  # 需要保证和返回的顺序一致
    # metrics = metrics_baseline + list(map(lambda x: x + '_t', metrics_baseline))
    # # FocalLoss需要单独调参
    # model = "BCWF_s"
    # # 开始训练
    # df_metrics = pd.DataFrame(columns=metrics, index=dataset_names)
    # print("==============开始批量训练===========")
    # for dataset_name in dataset_names:
    #     # 用于存储数据
    #     # 遍历model
    #     print("=====当前正在执行Bagging基分类器对{}数据集的预测=========".format(dataset_name))
    #     metric = run(model, dataset_name)
    #     df_metrics.loc[dataset_name] = metric
    # df_metrics.to_csv('./result/hyper/base/{}.csv'.format("Bagging"))
    # print('{}数据存储成功!'.format("Bagging"))
    # print("==============All Done!!!===========")

    # 这是grad的检验
    dataset_names = ['aba', 'let']
    metrics_baseline = ['roc', 'avg_roc', 'std_roc', 'pos_f1', 'avg_pos_f1', 'std_pos_f1', 'weight_f1', 'avg_weight_f1',
                        'std_weight_f1', 'gmean', 'avg_gmean', 'std_gmean']  # 需要保证和返回的顺序一致
    metrics = metrics_baseline + list(map(lambda x: x + '_t', metrics_baseline))
    model = "BCWF_s"
    grad_aba = [0,2,4,5, 6,7, 8,9, 10]
    grad_let = [0,3,6,9,12,15,18,21,24]
    # 开始训练
    print("==============开始超参数检验===========")
    for dataset_name in dataset_names:
        # 用于存储数据
        if dataset_name == 'aba':
            grads = grad_aba
        elif dataset_name == 'let':
            grads = grad_let

        df_metrics = pd.DataFrame(columns=metrics, index=grads)
        # 遍历grad
        for grad in grads:
            print("=====当前正在执行对{}数据集grad={}的预测=========".format(dataset_name,grad))
            metric = run_T(dataset_name,15,grad)
            df_metrics.loc[grad] = metric
        df_metrics.to_csv('./result/hyper/grad/{}.csv'.format(dataset_name))
        print('{}数据存储成功!'.format(dataset_name))
    print("==============All Done!!!===========")