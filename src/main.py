#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 10:20
# @Author  : Jack Zhao
# @Site    : 
# @File    : main.py
# @Software: PyCharm

# #Desc:
from model.bcwf_h import BcwfH
from model.bcwf_s import BcwfS
from model.focal_loss import train
from model.hemc import HemClass
from model.mod import Mod
from model.svm_smote import SvmS
from model.kmeans_smote import KS
from model.other_bs import OtherBaseline
from config import opt
import pandas as pd

def run(model,dataset_name):
    """执行模型"""
    metrics = []
    if model in ["Adaboost","BaggingClassifier","EasyEnsemble","RUSBoost","SelfPacedEnsemble","DES"]:
        model_class = OtherBaseline(dataset_name)
        model_class.apply_all(model)
        metrics = model_class.display()

    elif model == "MOD":
        model_class = Mod(dataset_name)
        model_class.apply_all()
        metrics = model_class.display()

    elif model == "SmoteSvm":
        model_class = SvmS(dataset_name)
        model_class.apply_all()
        metrics = model_class.display()

    elif model == "FocalLoss":
        opt.DATASET = dataset_name
        metrics = train(dataset_name)

    elif model == "KS":
        model_class = KS(dataset_name)
        model_class.apply_all()
        metrics = model_class.display()


    elif model == "HEMAdaboost":
        model_class = HemClass(dataset_name)
        model_class.apply_all()
        metrics = model_class.display()

    elif model == "BCWF_h":
        model_class = BcwfH(dataset_name, 15) # 这里的self.T可以试着修改
        model_class.apply_all()
        metrics = model_class.display()

    elif model =="BCWF_s":
        model_class = BcwfS(dataset_name, 15,0) # 这里的self.T,grad_constraints可以试着修改
        model_class.apply_all()
        metrics = model_class.display()


    return metrics


if __name__ == '__main__':
    print("Hello JOC!")
    # dataset_names = ['aba','bal','hab','hou','let','wdbc','wpbc','yea','pim','p1','p2','p3','cre'] # p3报错移除
    dataset_names = ['p1','p2','p3']
    metrics_baseline = ['roc','avg_roc','std_roc','pos_f1','avg_pos_f1','std_pos_f1','weight_f1','avg_weight_f1',
               'std_weight_f1','gmean','avg_gmean','std_gmean'] # 需要保证和返回的顺序一致
    metrics = metrics_baseline + list(map(lambda x: x+'_t', metrics_baseline))
    # FocalLoss需要单独调参
    # models = ["Adaboost","BaggingClassifier","EasyEnsemble","RUSBoost","SelfPacedEnsemble","MOD","SmoteSvm","HEMAdaboost","BCWF_h","BCWF_s"]
    # 三轮修改后加
    models = ["KS","DES"]
    # 开始训练
    print("==============开始批量训练===========")
    for dataset_name in dataset_names:
        # 用于存储数据
        df_metrics_baseline = pd.DataFrame(columns=metrics_baseline, index=models[:-2])
        df_metrics = pd.DataFrame(columns=metrics, index=models[-2:])
        # 遍历model
        for model in models:
            print("=====当前正在执行{}对{}数据集的预测=========".format(model,dataset_name))
            metric = run(model, dataset_name)
            if model in ["BCWF_h","BCWF_s"]:
                df_metrics.loc[model] = metric
            else:
                df_metrics_baseline.loc[model] = metric  # 解包
        df_metrics_baseline.to_csv('./result/{}_bs.csv'.format(dataset_name))
        df_metrics.to_csv('./result/{}.csv'.format(dataset_name))
        print('{}数据存储成功!'.format(dataset_name))
    print("==============All Done!!!===========")



