import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report as class_report
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt

class ModelThreshold(object):
    def __init__(self, y_probs_nmf, y_probs_lda, topic_dfs_nmf, topic_dfs_lda):
        self.nm_abc_y_pb, self.nm_gbc_y_pb, self.nm_rfc_y_pb = y_probs_nmf
        self.ld_abc_y_pb, self.ld_gbc_y_pb, self.ld_rfc_y_pb = y_probs_lda
        t_d_n = self.load_split(topic_dfs_nmf)
        t_d_l = self.load_split(topic_dfs_lda)
        self.nm_X_train,self.nm_y_train,self.nm_X_test,self.nm_y_test = t_d_n
        self.ld_X_train,self.ld_y_train,self.ld_X_test,self.ld_y_test = t_d_l

        self.thresholds = np.arange(0.30, 0.75, 0.05)

    def load_split(self, topic_dfs_model):
        '''
        Loads and splits data by nmf/lda, y/X, train/test in that order.

        Returns: X_train, y_train, X_test, y_test
        '''
        topic_df_train, topic_df_test = topic_dfs_model
        df_train, df_test = topic_df_train.copy(), topic_df_test.copy()
        y_train = df_train.pop('is_nut').values
        y_test = df_test.pop('is_nut').values
        X_train, X_test = df_train.values, df_test.values

        return X_train, y_train, X_test, y_test

    def show_class_report(self):
        '''
        Prints outs the precision, recall, and f1 score by nmf/lda, threshold,
        and abc/gbc/rfc.
        '''
        for i in self.thresholds:
            print('\n' + '=' * 56 + '\n%s nmf\n' % i + '=' * 56)
            abc_y_pred  = [1 if prob[1] > i else 0 for prob in self.nm_abc_y_pb]
            gbc_y_pred  = [1 if prob[1] > i else 0 for prob in self.nm_gbc_y_pb]
            rfc_y_pred  = [1 if prob[1] > i else 0 for prob in self.nm_rfc_y_pb]
            print('abc', class_report(self.nm_y_test, abc_y_pred, digits=3))
            print('-' * 56)
            print('gbc', class_report(self.nm_y_test, gbc_y_pred, digits=3))
            print('-' * 56)
            print('rfc', class_report(self.nm_y_test, rfc_y_pred, digits=3))
        for i in self.thresholds:
            print('\n' + '=' * 56 + '\n%s lda\n' % i + '=' * 56)
            abc_y_pred  = [1 if prob[1] > i else 0 for prob in self.ld_abc_y_pb]
            gbc_y_pred  = [1 if prob[1] > i else 0 for prob in self.ld_gbc_y_pb]
            rfc_y_pred  = [1 if prob[1] > i else 0 for prob in self.ld_rfc_y_pb]
            print('abc', class_report(self.ld_y_test, abc_y_pred, digits=3))
            print('-' * 56)
            print('gbc', class_report(self.ld_y_test, gbc_y_pred, digits=3))
            print('-' * 56)
            print('rfc', class_report(self.ld_y_test, rfc_y_pred, digits=3))

    def confusion_terms(self, terms=None):
        '''
        Makes dataframes for each predictive model. Columns are 'tn','fp','fn','tp' and whatever is specified in terms. 
        '''

        columns = ['threshold','tn','fp','fn','tp']
        self.nm_abc_df = pd.DataFrame(columns=columns)
        self.nm_gbc_df = pd.DataFrame(columns=columns)
        self.nm_rfc_df = pd.DataFrame(columns=columns)
        for i in self.thresholds:
            abc_y_pred  = [1 if prob[1] > i else 0 for prob in self.nm_abc_y_pb]
            gbc_y_pred  = [1 if prob[1] > i else 0 for prob in self.nm_gbc_y_pb]
            rfc_y_pred  = [1 if prob[1] > i else 0 for prob in self.nm_rfc_y_pb]
            abc_mat = confusion_matrix(self.nm_y_test, abc_y_pred).ravel()
            gbc_mat = confusion_matrix(self.nm_y_test, gbc_y_pred).ravel()
            rfc_mat = confusion_matrix(self.nm_y_test, rfc_y_pred).ravel()

            tn, fp, fn, tp = abc_mat
            new_df = pd.DataFrame(data=[[i, tn, fp, fn, tp]], columns=columns)
            self.nm_abc_df = self.nm_abc_df.append(new_df, ignore_index=True)
            tn, fp, fn, tp = gbc_mat
            new_df = pd.DataFrame(data=[[i, tn, fp, fn, tp]], columns=columns)
            self.nm_gbc_df = self.nm_gbc_df.append(new_df, ignore_index=True)
            tn, fp, fn, tp = rfc_mat
            new_df = pd.DataFrame(data=[[i, tn, fp, fn, tp]], columns=columns)
            self.nm_rfc_df = self.nm_rfc_df.append(new_df, ignore_index=True)

        self.nm_abc_df.set_index('threshold', inplace=True)
        self.nm_gbc_df.set_index('threshold', inplace=True)
        self.nm_rfc_df.set_index('threshold', inplace=True)

        term_dict = {'tpr': lambda x: x.loc['tp']/(x.loc['tp'] + x.loc['fn']),
                     'tnr': lambda x: x.loc['tn']/(x.loc['tn'] + x.loc['fp']),
                     'ppv': lambda x: x.loc['tp']/(x.loc['tp'] + x.loc['fp']),
                     'fnr': lambda x: x.loc['fn']/(x.loc['fn'] + x.loc['tp']),
                     'fpr': lambda x: x.loc['fp']/(x.loc['fp'] + x.loc['tn']),
                     'f1': lambda x: (2 * x.loc['tpr'] * x.loc['ppv']) / (x.loc['tpr'] + x.loc['ppv'])
                     }

        if terms != None:
            for term in terms:
                func = term_dict[term]
                self.nm_abc_df[term] = self.nm_abc_df.apply(func, axis=1)
                self.nm_gbc_df[term] = self.nm_gbc_df.apply(func, axis=1)
                self.nm_rfc_df[term] = self.nm_rfc_df.apply(func, axis=1)

        self.nm_abc_df.fillna(0.0, inplace=True)
        self.nm_gbc_df.fillna(0.0, inplace=True)
        self.nm_rfc_df.fillna(0.0, inplace=True)
