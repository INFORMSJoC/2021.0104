#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 15:33
# @Author  : Jack Zhao
# @Site    : 
# @File    : test_file.py
# @Software: PyCharm

# #Desc: 做一些杂事
import pandas as pd

def get_sample(dataset_name,ratio):
    for i in range(1,6):
        train_df = pd.read_csv(r'E:/Code/Pycharm/JOC/data/train_{}{}.csv'.format(dataset_name,i))
        test_df = pd.read_csv(r'E:/Code/Pycharm/JOC/data/test_{}{}.csv'.format(dataset_name,i))
        # 改名
        train_df = train_df.rename(columns={'BoolDefaultLabel':"TARGET"})
        test_df = test_df.rename(columns={'BoolDefaultLabel':"TARGET"})
        # 计算ratio
        ratio_train =train_df[train_df['TARGET']==-1].shape[0]//ratio/train_df[train_df['TARGET']==1].shape[0]
        ratio_test = test_df[test_df['TARGET']==-1].shape[0]//ratio/test_df[test_df['TARGET']==1].shape[0]
        # 采样
        train_df_pos = train_df[train_df['TARGET']==1].sample(frac=ratio_train,axis=0,replace=True)
        test_df_pos = test_df[test_df['TARGET']==1].sample(frac=ratio_test,axis=0,replace=True)
        # 和负例拼接
        train_df = pd.concat((train_df[train_df['TARGET']==-1],train_df_pos),axis=0)
        test_df = pd.concat((test_df[test_df['TARGET']==-1],test_df_pos),axis=0)
        print(train_df[train_df['TARGET']==-1].shape[0],train_df[train_df['TARGET']==1].shape[0])
        print(test_df[test_df['TARGET']==-1].shape[0],test_df[test_df['TARGET']==1].shape[0])
        # 数据存储
        train_df.to_csv(r'E:/Code/Pycharm/JOC/data/train_{}{}.csv'.format(dataset_name,i))
        test_df.to_csv(r'E:/Code/Pycharm/JOC/data/test_{}{}.csv'.format(dataset_name,i))



if __name__ == '__main__':
    dataset_names = [('p1',10),('p2',5),('p3',2)]
    for i in range(len(dataset_names)):
        dataset_name, ratio = dataset_names[i]
        get_sample(dataset_name, ratio)

    # for i in range(1,6): # 改p1-p3数据集
    #     dataset_train = r'E:\Code\Pycharm\JOC\data\train_{}'.format('p3')+str(i)+'.csv'
    #     dataset_test = r'E:\Code\Pycharm\JOC\data\test_{}'.format('p3')+str(i)+'.csv'
    #     train_df = pd.read_csv(dataset_train)
    #     test_df = pd.read_csv(dataset_test)
    #     train_df.loc[train_df.TARGET == 1, 'TARGET'] = 0
    #     train_df.loc[train_df.TARGET == -1, 'TARGET'] = 1
    #     train_df.loc[train_df.TARGET == 0, 'TARGET'] = -1
    #     test_df.loc[test_df.TARGET == 1, 'TARGET'] = 0
    #     test_df.loc[test_df.TARGET == -1, 'TARGET'] = 1
    #     test_df.loc[test_df.TARGET == 0, 'TARGET'] = -1
    #     train_df.to_csv(dataset_train,index=None)
    #     test_df.to_csv(dataset_test,index=None)


