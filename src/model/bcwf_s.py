#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 18:51
# @Author  : Jack Zhao
# @Site    : 
# @File    : bcwf_s.py
# @Software: PyCharm

# #Desc:这里是BCWF_S的实现
from model.bcwf_h import BcwfH
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.metrics import geometric_mean_score

import pandas as pd


class BcwfS(BcwfH):
    def __init__(self,dataset,T, grad_constraints):
        super(BcwfS,self).__init__(dataset,T)
        self.grad_constraints = grad_constraints

    def whether_divide(self,ratio,train_df,feature_column,pos_train_df,train_copy,pos_num):
        """
        两种情况讨论
        :param ratio:neg/pos
        :param train_df:
        :param feature_column:
        :param pos_train_df:
        :param train_copy:==train_df，避免在train上修改
        :param pos_num:==pos_train_df.shape
        :return:
        """
        if round(ratio) > ratio:
            # 多取样一次
            removed_neg_df = train_df.copy()
            for i in range(1, round(ratio)):
                removed_neg_df = self.remove_hard_exam(pos_train_df, feature_column, train_copy, removed_neg_df)
            add_sample = train_df.drop(removed_neg_df.index).sample(2 * pos_num - removed_neg_df.shape[0])  # 额外抽取 TODO: sample的随机种子
            last_subsample = pd.concat((add_sample, removed_neg_df), axis=0)
            base_est_fit = self.fit_base_est('Adaboost', 10, last_subsample[feature_column].values,
                                             last_subsample["TARGET"].values)
            preds = base_est_fit.predict(train_copy[feature_column].values)
            train_copy['iter_result'] = preds
            train_copy['sum'] += train_copy['iter_result']
        else:
            removed_neg_df = train_df.copy()
            for i in range(1, round(ratio) + 1):
                removed_neg_df = self.remove_hard_exam(pos_train_df, feature_column, train_copy, removed_neg_df)
        pos_hard_index = list(
            train_copy.loc[(train_copy['sum'] == (-round(ratio))) & (train_copy['TARGET'] == 1)].index)
        neg_hard_index = list(
            train_copy.loc[(train_copy['sum'] > self.grad_constraints) & (train_copy['TARGET'] == -1)].index) # new here

        pos_hard_nums = len(pos_hard_index)
        neg_hard_nums = len(neg_hard_index)
        return pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index

    def apply_all(self):
        for i in range(1,6):
            train_df, test_df, feature_column, train_copy, test_copy, pos_num, neg_num,pos_train_df,ratio = self.data_read(i)
            pos_hard_nums, neg_hard_nums, pos_hard_index, neg_hard_index = self.whether_divide(ratio, train_df,
                                                                                               feature_column,
                                                                                               pos_train_df, train_copy,
                                                                                               pos_num)
            # 额外需要减去的neg数
            neg_delete_num = int((neg_num - (pos_num - pos_hard_nums)) / self.T)
            if neg_num - (pos_num - pos_hard_nums) != self.T * neg_delete_num:
                neg_delete_num = neg_delete_num + 1

            self.predict(10,train_df,test_df,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,neg_delete_num)


    def predict(self,nums,train_df,test_df,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,neg_delete_num):
        for i in range(nums): # 计算10次啦
            iter_est_lis,iter_pred_lis,iter_label_lis = [],[],[]
            # 迭代终止条件
            self.iterations(train_df,test_df,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis,neg_delete_num)
            # 不同的vote方式计算指标
            # 投票法1
            test_df['a_vote'] = sum(iter_pred_lis) / len(iter_est_lis) # 有维度的
            test_preds =  [1 if pred > 0.5 else -1 for pred in test_df['a_vote']]
            self.roc.append(round(roc_auc_score(test_df['TARGET'].values, test_preds), 4))
            pos_f1score = float(classification_report(test_df['TARGET'].values, test_preds).split()[12])
            self.pos_f1.append(round(pos_f1score, 4))
            self.weight_f1.append(round(f1_score(test_df['TARGET'].values, test_preds, average='weighted'), 4))

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



    def iterations(self,iter_train_df, test_df ,feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis,neg_delete_num):
        """检测是否达到迭代条件"""
        iter_train_df_copy = iter_train_df.copy()
        # 数据集已经平衡或者最大迭代次数已经达到
        cur_iter = 0
        while iter_train_df_copy.loc[iter_train_df_copy.TARGET == 1].shape[0] <= iter_train_df_copy.loc[iter_train_df_copy.TARGET == -1].shape[0]:
            iter_train_df_copy = self.train_change(iter_train_df,test_df, feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis,neg_delete_num)
            cur_iter +=1
            if cur_iter >= self.T:
                break
        print('迭代结束')
        # 看看最后训练集的标签分布
        return iter_train_df_copy.TARGET.value_counts()

    def train_change(self,iter_train_df,test_df, feature_column,pos_hard_nums,neg_hard_nums,pos_hard_index,neg_hard_index,iter_est_lis,iter_pred_lis,iter_label_lis,neg_delete_num):
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
        preds = base_est_fit.predict_proba(iter_train_df[feature_column].values)[:, -1]
        result = [1 if pred > 0.5 else -1 for pred in preds]
        iter_train_df['iter_pro'] = preds
        iter_train_df['iter_result'] = result

        iter_train_df_copy = iter_train_df.copy()

        # 预测错误负样本的index[sort]
        mis_neg_index = list(
            iter_train_df.loc[
                (iter_train_df['iter_result'] != iter_train_df['TARGET']) & (iter_train_df['TARGET'] == -1)][
                'iter_pro'].sort_values(ascending=False).index)
        # 取交集MNE,NHE/PHE,不用set以保证有序
        total_mis_dict = dict(Counter(mis_neg_index + neg_hard_index))
        drop_index = [key for key, value in total_mis_dict.items() if value > 1]
        drop_num = len(drop_index)
        print("Neg交集有{}个数据".format(drop_num))

        if neg_hard_nums/self.T >1:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:int(neg_hard_nums/self.T)+1])
            add_num = neg_delete_num - (int(neg_hard_nums / self.T) + 1)
        else:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:1]) # 只删1个
            add_num = neg_delete_num - 1 # n2 ??

        mis_pos_index = list(
            iter_train_df.loc[
                (iter_train_df['iter_result'] != iter_train_df['TARGET']) & (iter_train_df['TARGET'] == 1)][
                'iter_pro'].sort_values(ascending=True).index)
        total_mis_dict = dict(Counter(mis_pos_index + pos_hard_index))
        drop_index = [key for key, value in total_mis_dict.items() if value > 1]
        drop_num = len(drop_index)
        print("Pos交集有{}个数据".format(drop_num))
        if pos_hard_nums / self.T > 1:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:int(pos_hard_nums / self.T) + 1])
        else:
            iter_train_df_copy = iter_train_df_copy.drop(drop_index[:1])  # 只删1个

        # news-> 需要额外减去的neg_num
        neg_delete_index = list(
            iter_train_df.loc[(iter_train_df['iter_result'] == iter_train_df['TARGET']) & (iter_train_df['TARGET'] == -1)]['iter_pro'].sort_values(
                ascending=True).index)

        iter_train_df_copy = iter_train_df_copy.drop(neg_delete_index[:add_num])

        iter_pred_lis.append(iter_est_lis[-1].predict_proba(test_df[feature_column].values)[:, -1])
        iter_label_lis.append(iter_est_lis[-1].predict(test_df[feature_column].values))

        return iter_train_df_copy


if __name__ == '__main__':
    model = BcwfS('wdbc', 15, 0)
    model.apply_all()
    model.display()









