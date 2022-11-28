#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 13:19
# @Author  : Jack Zhao
# @Site    : 
# @File    : bcwf_h.py
# @Software: PyCharm

# #Desc: 这里是bcwf_hard的实现
from model.hemc import HemClass
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.metrics import geometric_mean_score

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class BcwfH(HemClass):
    def __init__(self,dataset,T):
        super(BcwfH,self).__init__(dataset)
        self.roc_t = []
        self.pos_f1_t = []
        self.weight_f1_t = []
        self.gmean_t = []
        self.T = T # 迭代次数


    def apply_all(self):
        """主函数"""
        for i in range(1,6):
            train_df, test_df, feature_column, train_copy, test_copy, pos_num, neg_num,pos_train_df,ratio = self.data_read(i)
            # 删除负样本后的trainset
            pos_hard_nums, neg_hard_nums, pos_hard_index, neg_hard_index = self.whether_divide(ratio,train_df,feature_column,pos_train_df,train_copy,pos_num)
            # 集成boosting
            self.predict(10,train_df,test_df,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index)

    def predict(self,nums,train_df,test_df,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index):
        for i in range(nums): # 计算10次啦
            iter_est_lis,iter_pred_lis,iter_label_lis = [],[],[]
            # 迭代终止条件
            self.iterations(train_df,test_df,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis)
            # 不同的vote方式计算指标
            # 投票法1
            test_df['a_vote'] = sum(iter_pred_lis) / len(iter_est_lis) # 有维度的
            test_preds =  [1 if pred > 0.5 else -1 for pred in test_df['a_vote']]
            self.roc.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1.append(round(pos_f1score, 4))
            self.weight_f1.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))
            # new
            self.gmean.append(round(geometric_mean_score(test_df['TARGET'].values, test_preds), 4))


            # 投票法2
            test_df['m_sum'] = sum(iter_label_lis)
            test_preds = [1 if pred > 0 else -1 for pred in test_df['m_sum']]
            self.roc_t.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1_t.append(round(pos_f1score, 4))
            self.weight_f1_t.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))
            # new
            self.gmean_t.append(round(geometric_mean_score(test_df['TARGET'].values, test_preds), 4))



    def iterations(self,iter_train_df, test_df ,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis):
        """检测是否达到迭代条件"""
        iter_train_df_copy = iter_train_df.copy()
        # 数据集已经平衡或者最大迭代次数已经达到
        cur_iter = 0
        while iter_train_df_copy.loc[iter_train_df_copy.TARGET == 1].shape[0] <= iter_train_df_copy.loc[iter_train_df_copy.TARGET == -1].shape[0]:
            iter_train_df_copy = self.train_change(iter_train_df,test_df, feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis)
            cur_iter +=1
            if cur_iter >= self.T:
                break
        print('迭代结束')
        # 看看最后训练集的标签分布
        return iter_train_df_copy.TARGET.value_counts()


    def train_change(self,iter_train_df,test_df, feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis):
        """
        用于删减样本,以训练boosting
        :param iter_train_df: 每次训练的数据集
        :param test_df: 每次测试的数据集
        :param feature_column:
        :param pos_hard_nums: hardest exam num
        :param neg_hard_nums:
        :param pos_hard_index: hardest exam index
        :param neg_hard_index:
        :param iter_est_lis: 每次的分类器，for boost
        :param iter_pred_lis: 每次预测的概率
        :param iter_label_lis: 保存每次预测的label
        :return: 删除了困难样本的df
        """
        neg_train_df = iter_train_df.loc[iter_train_df.TARGET == -1]
        pos_train_df = iter_train_df.loc[iter_train_df.TARGET == 1]

        # 随机采样,训练一个分类器
        pos_num = pos_train_df.shape[0]
        neg_sample = neg_train_df.sample(pos_num)
        sub_sample = pd.concat((neg_sample, pos_train_df), axis=0)
        print("训练样本大小为{}".format(sub_sample.shape[0]))
        base_est_fit = self.fit_base_est('Adaboost', 10, sub_sample[feature_column].values,
                                         sub_sample["TARGET"].values)
        iter_est_lis.append(base_est_fit)

        # 预测S中的grad值
        preds = base_est_fit.predict_proba(iter_train_df[feature_column].values)[:,-1]
        result = [1 if pred > 0.5 else -1 for pred in preds]
        iter_train_df['iter_pro'] = preds
        iter_train_df['iter_result'] = result

        iter_train_df_copy = iter_train_df.copy()

        # 预测错误负样本的index[sort]
        mis_neg_index = list(
            iter_train_df.loc[(iter_train_df['iter_result'] != iter_train_df['TARGET']) & (iter_train_df['TARGET'] == -1)]['iter_pro'].sort_values(ascending=False).index)
        # 取交集MNE,NHE/PHE,不用set以保证有序
        total_mis_dict = dict(Counter(mis_neg_index + neg_hard_index))
        drop_index = [key for key, value in total_mis_dict.items() if value > 1]
        drop_num = len(drop_index)
        print("Neg交集有{}个数据".format(drop_num))
        if neg_hard_nums / self.T > 1:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:int(neg_hard_nums/self.T)+1])
        else:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:1]) # 只删1个

        mis_pos_index = list(
            iter_train_df.loc[(iter_train_df['iter_result'] != iter_train_df['TARGET']) & (iter_train_df['TARGET'] == 1)]['iter_pro'].sort_values(ascending=True).index)
        total_mis_dict = dict(Counter(mis_pos_index + pos_hard_index))
        drop_index = [key for key, value in total_mis_dict.items() if value > 1]
        drop_num = len(drop_index)
        print("Pos交集有{}个数据".format(drop_num))
        if pos_hard_nums/self.T >1:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:int(pos_hard_nums/self.T)+1])
        else:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:1]) # 只删1个

        iter_pred_lis.append(iter_est_lis[-1].predict_proba(test_df[feature_column].values)[:,-1])
        iter_label_lis.append(iter_est_lis[-1].predict(test_df[feature_column].values))

        return iter_train_df_copy


    def display(self):
        print("roc : ", self.roc, ", pos_f1: ", self.pos_f1, ", weight_f1", self.weight_f1, ", gmean", self.gmean)
        print("avg_roc : ", np.mean(self.roc), ", avg_pos_f1: ", np.mean(self.pos_f1),
              ", avg_weight_f1", np.mean(self.weight_f1),
              ", avg_gmean", np.mean(self.gmean))
        print("std_roc : ", np.std(self.roc), ", std_pos_f1: ", np.std(self.pos_f1),
              ", std_weight_f1", np.std(self.weight_f1),
              ", std_gmean", np.std(self.gmean))
        print("========================================================")
        print("roc_t : ", self.roc_t, ", pos_f1_t: ", self.pos_f1_t, ", weight_f1_t", self.weight_f1_t, ", gmean_t", self.gmean_t)
        print("avg_roc_t : ", np.mean(self.roc_t), ", avg_pos_f1: ", np.mean(self.pos_f1_t),
              ", avg_weight_f1", np.mean(self.weight_f1_t),
              ", avg_gmean_t", np.mean(self.gmean_t))
        print("std_roc_t : ", np.std(self.roc_t), ", std_pos_f1_t: ", np.std(self.pos_f1_t),
              ", std_weight_f1", np.std(self.weight_f1_t),
              ", std_gmean_t", np.std(self.gmean_t))
        return self.roc, np.mean(self.roc), np.std(self.roc), self.pos_f1, np.mean(self.pos_f1), np.std(self.pos_f1), \
               self.weight_f1, np.mean(self.weight_f1), np.std(self.weight_f1), self.gmean, np.mean(self.gmean), np.std(self.gmean), \
               self.roc_t, np.mean(self.roc_t), np.std(self.roc_t), self.pos_f1_t, np.mean(self.pos_f1_t), \
               np.std(self.pos_f1_t), self.weight_f1_t, np.mean(self.weight_f1_t), np.std(self.weight_f1_t), self.gmean_t, \
               np.mean(self.gmean_t), np.std(self.gmean_t)


if __name__ == '__main__':
    model = BcwfH('wdbc',15)
    model.apply_all()
    model.display()



