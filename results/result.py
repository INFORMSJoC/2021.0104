#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 13:40
# @Author  : Jack Zhao
# @Site    : 
# @File    : result.py
# @Software: PyCharm

# #Desc: 这里用于提取所需的指标合集
import os
import pandas as pd

def get_file_lis(path):
    file_lis = os.listdir(path)
    file_lis.remove('focal')
    return file_lis

def merge_bs(dataset_name,feature_column):
    """合并bs和我们的方法文件"""
    df_bs = pd.read_csv('./{}_bs.csv'.format(dataset_name))
    all_df = df_bs
    # 三轮JOC修改
    # df = pd.read_csv('./{}.csv'.format(dataset_name))
    # df = df[feature_column]
    # all_df = pd.concat((df_bs,df),axis=0)
    return all_df



def get_metric_comp(metric,dataset_names,dataset_df_dic,model_names):
    """
    某个metric上，所有算法在所有数据集上的表现
    :param metric: 比较的metric
    :param dataset_names: all datasets names
    :param dataset_df_dic: all datasets
    :return: df.columns = model df.index=dataset
    """
    metrics = ['avg_'+metric,'std_'+metric]
    df_metric = pd.DataFrame(columns=model_names,index=dataset_names)
    for dataset_name in dataset_names:
        avg,std = metrics[0],metrics[1]
        data_name_metric = dataset_df_dic[dataset_name].apply(lambda x: str(x[avg])+"+/-"+str(x[std]),axis=1).values
        df_metric.loc[dataset_name] = data_name_metric
    return df_metric


def base_comp(dataset_names,model,metrics):
    """生成和Bagging对应的比较文件"""
    df_ada = pd.DataFrame(index=dataset_names,columns=metrics)
    # dataset_names = map(lambda x:x+"_bs.csv",dataset_names)
    if model=='BCWF_s':
        for dataset_name in dataset_names:
            df = pd.read_csv(dataset_name+".csv",index_col=0)
            df_ada.loc[dataset_name] = df.loc[model]
    return df_ada


def concat_focal(metric,df):
    """该函数用于拼接focal loss结果"""
    metrics = ['avg_' + metric, 'std_' + metric]
    avg, std = metrics[0], metrics[1]
    focal_dataset = pd.read_csv('./comp/focal_loss.csv',index_col=0)
    df['FocalLoss'] = focal_dataset.apply(lambda x:str(x[avg])+"+/-"+str(x[std]),axis=1)
    return df


if __name__ == '__main__':
    # file_lis = get_file_lis('./')
    # dataset_names = [ 'aba', 'bal', 'hab', 'hou', 'let', 'wdbc', 'wpbc', 'yea', 'pim', 'p1', 'p2', 'p3','cre']
    # dataset_names = ['wdbc','pim','hab','wpbc','cre','hou','yea','aba','bal','let'] # 按论文顺序
    dataset_names = ['p1','p2','p3']
    # model_names = ["Adaboost", "BaggingClassifier", "EasyEnsemble", "RUSBoost", "SelfPacedEnsemble", "MOD", "SmoteSvm",
    #           "HEMAdaboost", "BCWF_h", "BCWF_s"] # Focal Loss被排除，如果main加上，这里也得加上
    # 三轮修改后加
    model_names = ["KS", "DES"]
    feature_column = ['roc','avg_roc','std_roc','pos_f1','avg_pos_f1','std_pos_f1','weight_f1','avg_weight_f1',
               'std_weight_f1','gmean','avg_gmean','std_gmean']
    dataset_df_dic = {} # 存储合并的df
    for dataset_name in dataset_names:
        dataset_df_dic[dataset_name] = merge_bs(dataset_name,feature_column)
    metrics_general = ['roc','pos_f1','weight_f1','gmean']
    for metric in metrics_general:
        df = get_metric_comp(metric,dataset_names,dataset_df_dic,model_names) # 两个数据对应
        # df = concat_focal(metric,df)
        df.to_csv('./comp/{}_1.csv'.format(metric)) # 转置和paper一样维度
        print("{}比较矩阵以生成完毕!".format(metric))


    # 下面是生成Adaboost的比较矩阵
    # metrics_baseline = ['roc', 'avg_roc', 'std_roc', 'pos_f1', 'avg_pos_f1', 'std_pos_f1', 'weight_f1', 'avg_weight_f1',
    #                     'std_weight_f1', 'gmean', 'avg_gmean', 'std_gmean']  # 需要保证和返回的顺序一致
    # metrics = metrics_baseline + list(map(lambda x: x + '_t', metrics_baseline))
    # model = 'BCWF_s'
    # df_base = base_comp(dataset_names,model,metrics)
    # df_base.to_csv('./hyper/base/{}.csv'.format('Adaboost'))
    # print("Adaboost生成完毕，见hyper文件夹！")


