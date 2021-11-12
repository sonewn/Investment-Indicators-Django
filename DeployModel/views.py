import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import multiprocessing
import copy
import pickle
import warnings
from datetime import datetime, timedelta
from time import time, sleep, mktime
from matplotlib import font_manager as fm, rc, rcParams
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re


import numpy as np
from numpy import array, nan, random as rnd, where
import pandas as pd
from pandas import DataFrame as dataframe, Series as series, isna, read_csv
from pandas.tseries.offsets import DateOffset, BDay
import statsmodels.api as sm
from scipy.stats import f_oneway

from sklearn import preprocessing as prep
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split as tts, GridSearchCV as GridTuner, StratifiedKFold, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from sklearn.pipeline import make_pipeline

from sklearn import linear_model as lm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn import neighbors as knn
from sklearn import ensemble

# ===== tensorflow =====
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import metrics as tf_metrics
from tensorflow.keras import callbacks as tf_callbacks
from tqdm.keras import TqdmCallback
# import tensorflow_addons as tfa
# import keras_tuner as kt
# from keras_tuner import HyperModel

# # ===== NLP =====
# from selenium import webdriver
# from konlpy.tag import Okt
# from KnuSentiLex.knusl import KnuSL

# ===== timeseries =====
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.preprocessing import timeseries_dataset_from_array as make_ts_tensor

warnings.filterwarnings(action='ignore')
rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# font setting
# font_path = 'C:/Users/user/PycharmProjects/pythonProject/myfonts/NanumSquareB.ttf'
font_path = './DeployModel/myfonts/NanumSquareB.ttf'
font_obj = fm.FontProperties(fname=font_path, size=12).get_name()
rc('font', family=font_obj)

# %reset -f

# ===== utility functions =====
# label encoding for categorical column with excepting na value
def which(bool_list):
    idx_array = where(bool_list)[0]
    return idx_array[0] if len(idx_array) == 1 else idx_array
def easyIO(x=None, path=None, op="r"):
    tmp = None
    if op == "r":
        with open(path, "rb") as f:
            tmp = pickle.load(f)
        return tmp
    elif op == "w":
        tmp = {}
        print(x)
        if type(x) is dict:
            for k in x.keys():
                if "MLP" in k:
                    tmp[k] = {}
                    for model_comps in x[k].keys():
                        if model_comps != "model":
                            tmp[k][model_comps] = copy.deepcopy(x[k][model_comps])
                    print(F"INFO : {k} model is removed (keras)")
                else:
                    tmp[k] = x[k]
        if input("Write [y / n]: ") == "y":
            with open(path, "wb") as f:
                pickle.dump(tmp, f)
            print("operation success")
        else:
            print("operation fail")
    else:
        print("Unknown operation type")
def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]
def findIdx(data_x, col_names):
    return [int(i) for i, j in enumerate(data_x) if j in col_names]
def orderElems(for_order, using_ref):
    return [i for i in using_ref if i in for_order]
# concatenate by row
def ccb(df1, df2):
    if type(df1) == series:
        tmp_concat = series(pd.concat([dataframe(df1), dataframe(df2)], axis=0, ignore_index=True).iloc[:,0])
        tmp_concat.reset_index(drop=True, inplace=True)
    elif type(df1) == dataframe:
        tmp_concat = pd.concat([df1, df2], axis=0, ignore_index=True)
        tmp_concat.reset_index(drop=True, inplace=True)
    elif type(df1) == np.ndarray:
        tmp_concat = np.concatenate([df1, df2], axis=0)
    else:
        print("Unknown Type: return 1st argument")
        tmp_concat = df1
    return tmp_concat
def change_width(ax, new_value):
    for patch in ax.patches :
        current_width = patch.get_width()
        adj_value = current_width - new_value
        # we change the bar width
        patch.set_width(new_value)
        # we recenter the bar
        patch.set_x(patch.get_x() + adj_value * .5)
def week_of_month(date):
    month = date.month
    week = 0
    while date.month == month:
        week += 1
        date -= timedelta(days=7)
    return week
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
def dispPerformance(result_dic, result_metrics):
    perf_table = dataframe(columns=result_metrics)
    for k, v in result_dic.items():
        perf_table = pd.concat([perf_table, v["performance"]], ignore_index=True, axis=0)
    print(perf_table)
    return perf_table

class MyLabelEncoder:
    def __init__(self, preset={}):
        # dic_cat format -> {"col_name": {"value": replace}}
        self.dic_cat = preset
    def fit_transform(self, data_x, col_names):
        tmp_x = copy.deepcopy(data_x)
        for i in col_names:
            # type check
            if not ((tmp_x[i].dtype.name == "object") or (tmp_x[i].dtype.name == "category")):
                print(F"WARNING : {i} is not object or category")
            # if key is not in dic, update dic
            if i not in self.dic_cat.keys():
                tmp_dic = dict.fromkeys(sorted(set(tmp_x[i]).difference([nan])))
                label_cnt = 0
                for j in tmp_dic.keys():
                    tmp_dic[j] = label_cnt
                    label_cnt += 1
                self.dic_cat[i] = tmp_dic
            # transform value which is not in dic to nan
            tmp_x[i] = tmp_x[i].astype("object")
            conv = tmp_x[i].replace(self.dic_cat[i])
            for conv_idx, j in enumerate(conv):
                if j not in self.dic_cat[i].values():
                    conv[conv_idx] = nan
            # final return
            tmp_x[i] = conv.astype("float")
        return tmp_x
    def transform(self, data_x):
        tmp_x = copy.deepcopy(data_x)
        for i in list(self.dic_cat.keys()):
            if not ((tmp_x[i].dtype.name == "object") or (tmp_x[i].dtype.name == "category")):
                print(F"WARNING : {i} is not object or category")
            # transform value which is not in dic to nan
            tmp_x[i] = tmp_x[i].astype("object")
            conv = tmp_x[i].replace(self.dic_cat[i])
            for conv_idx, j in enumerate(conv):
                if j not in self.dic_cat[i].values():
                    conv[conv_idx] = nan
            # final return
            tmp_x[i] = conv.astype("float")
        return tmp_x
    def clear(self):
        self.dic_cat = {}
class MyOneHotEncoder:
    def __init__(self, label_preset={}):
        self.dic_cat = {}
        self.label_preset = label_preset
    def fit_transform(self, data_x, col_names):
        tmp_x = dataframe()
        for i in data_x:
            if i not in col_names:
                tmp_x = pd.concat([tmp_x, dataframe(data_x[i])], axis=1)
            else:
                if not ((data_x[i].dtype.name == "object") or (data_x[i].dtype.name == "category")):
                    print(F"WARNING : {i} is not object or category")
                self.dic_cat[i] = OneHotEncoder(sparse=False, handle_unknown="ignore")
                conv = self.dic_cat[i].fit_transform(dataframe(data_x[i])).astype("int")
                col_list = []
                for j in self.dic_cat[i].categories_[0]:
                    if i in self.label_preset.keys():
                        for k, v in self.label_preset[i].items():
                            if v == j:
                                col_list.append(str(i) + "_" + str(k))
                    else:
                        col_list.append(str(i) + "_" + str(j))
                conv = dataframe(conv, columns=col_list)
                tmp_x = pd.concat([tmp_x, conv], axis=1)
        return tmp_x
    def transform(self, data_x):
        tmp_x = dataframe()
        for i in data_x:
            if not i in list(self.dic_cat.keys()):
                tmp_x = pd.concat([tmp_x, dataframe(data_x[i])], axis=1)
            else:
                if not ((data_x[i].dtype.name == "object") or (data_x[i].dtype.name == "category")):
                    print(F"WARNING : {i} is not object or category")
                conv = self.dic_cat[i].transform(dataframe(data_x[i])).astype("int")
                col_list = []
                for j in self.dic_cat[i].categories_[0]:
                    if i in self.label_preset.keys():
                        for k, v in self.label_preset[i].items():
                            if v == j: col_list.append(str(i) + "_" + str(k))
                    else:
                        col_list.append(str(i) + "_" + str(j))
                conv = dataframe(conv, columns=col_list)
                tmp_x = pd.concat([tmp_x, conv], axis=1)
        return tmp_x
    def clear(self):
        self.dic_cat = {}
        self.label_preset = {}
class MyKNNImputer:
    def __init__(self, k=5):
        self.imputer = KNNImputer(n_neighbors=k)
        self.cat_dic = {}
    def fit_transform(self, x, y, cat_vars=None):
        naIdx = dict.fromkeys(cat_vars)
        for i in cat_vars:
            self.cat_dic[i] = diff(list(sorted(set(x[i]))), [nan])
            naIdx[i] = list(which(array(x[i].isna()))[0])
        x_imp = dataframe(self.imputer.fit_transform(x, y), columns=x.columns)

        # if imputed categorical value are not in the range, adjust the value
        for i in cat_vars:
            x_imp[i] = x_imp[i].apply(lambda x: int(round(x, 0)))
            for j in naIdx[i]:
                if x_imp[i][j] not in self.cat_dic[i]:
                    if x_imp[i][j] < self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][0]
                    elif x_imp[i][j] > self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][len(self.cat_dic[i]) - 1]
        return x_imp
    def transform(self, x):
        naIdx = dict.fromkeys(self.cat_vars)
        for i in self.cat_dic.keys():
            naIdx[i] = list(which(array(x[i].isna())))
        x_imp = dataframe(self.imputer.transform(x), columns=x.columns)

        # if imputed categorical value are not in the range, adjust the value
        for i in self.cat_dic.keys():
            x_imp[i] = x_imp[i].apply(lambda x: int(round(x, 0)))
            for j in naIdx[i]:
                if x_imp[i][j] not in self.cat_dic[i]:
                    if x_imp[i][j] < self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][0]
                    elif x_imp[i][j] > self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][len(self.cat_dic[i]) - 1]
        return x_imp
    def clear(self):
        self.imputer = None
        self.cat_dic = {}
folder_path = "./projects/dacon_stockprediction/"

# ===== task specific =====
from pykrx import stock
def getBreakthroughPoint(df, col1, col2, patient_days, fill_method="fb"):
    '''
    :param df: dataframe (including col1, col2)
    :param col1: obj
    :param col2: obj moving average
    :param patient_days: patient days detected as breakthrough point
    :return: signal series
    '''
    sigPrice = []
    flag = -1  # A flag for the trend upward/downward

    for i in range(0, len(df)):
        if df[col1][i] > df[col2][i] and flag != 1:
            tmp = df['Close'][i:(i + patient_days + 1)]
            if len(tmp) == 1:
                sigPrice.append("buy")
                flag = 1
            else:
                if (tmp.iloc[1:] > tmp.iloc[0]).all():
                    sigPrice.append("buy")
                    flag = 1
                else:
                    sigPrice.append(nan)
        elif df[col1][i] < df[col2][i] and flag != 0:
            tmp = df['Close'][i:(i + patient_days + 1)]
            if len(tmp) == 1:
                sigPrice.append("sell")
                flag = 0
            else:
                if (tmp.iloc[1:] < tmp.iloc[0]).all():
                    sigPrice.append("sell")
                    flag = 0
                else:
                    sigPrice.append(nan)
        else:
            sigPrice.append(nan)

    sigPrice = series(sigPrice)
    for idx, value in enumerate(sigPrice):
        if not isna(value):
            if value == "buy":
                sigPrice.iloc[1:idx] = "sell"
            else:
                sigPrice.iloc[1:idx] = "buy"
            break
    # if fill_method == "bf":
    #
    # elif fill_method == ""
    sigPrice.ffill(inplace=True)
    return sigPrice
def stochastic(df, n=14, m=5, t=5):
    #데이터 프레임으로 받아오기 때문에 불필요

    #n 일중 최저가
    ndays_high = df['High'].rolling(window=n, min_periods=n).max()
    ndays_low = df['Low'].rolling(window=n, min_periods=n).min()
    fast_k = ((df['Close'] - ndays_low) / (ndays_high - ndays_low) * 100)
    slow_k = fast_k.ewm(span=m, min_periods=m).mean()
    slow_d = slow_k.ewm(span=t, min_periods=t).mean()
    df = df.assign(fast_k=fast_k, fast_d=slow_k, slow_k=slow_k, slow_d=slow_d)
    return df

# modeling function
kfolds_spliter = TimeSeriesSplit(n_splits=5, test_size=1, gap=1)
targetType = "numeric"
targetTask = None
class_levels = [0,1]
cut_off = 0

def doLinear(train_x, train_y, test_x=None, test_y=None, model_export=False, preTrained=None):
    result_dic = {}
    scaler_standard = prep.StandardScaler()
    train_x = scaler_standard.fit_transform(train_x)
    test_x = scaler_standard.transform(test_x)

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            result_dic["model"] = lm.LinearRegression(n_jobs=multiprocessing.cpu_count())

        result_dic["model"].fit(train_x, train_y)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:

                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

def doElasticNet(train_x, train_y, test_x=None, test_y=None, tuningMode=True,
                 c=None, alpha=None, l1_ratio=None,
                 kfolds=KFold(10, shuffle=True, random_state=2323),
                 model_export=False, preTrained=None, seed=1000):
    result_dic = {}
    scaler_standard = prep.StandardScaler()
    train_x = scaler_standard.fit_transform(train_x)

    runStart = time()
    if targetType == "numeric":
        tuner_params = {"alpha": np.linspace(1e-3, 1e+3, 100).tolist(),
                        "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:
                model_tuner = GridTuner(lm.ElasticNet(max_iter=1000, random_state=seed),
                                        refit=False,
                                        param_grid=tuner_params,
                                        scoring="neg_root_mean_squared_error",
                                        cv=kfolds.split(train_x, train_y),
                                        n_jobs=multiprocessing.cpu_count())
                model_tuner.fit(train_x, train_y)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = lm.ElasticNet(max_iter=1000, alpha=model_tuner.best_params_["alpha"],
                                                    l1_ratio=model_tuner.best_params_["l1_ratio"],
                                                    random_state=seed)
            else:
                result_dic["model"] = lm.ElasticNet(max_iter=1000,
                                                    alpha=alpha,
                                                    l1_ratio=l1_ratio,
                                                    random_state=seed)

        result_dic["model"].fit(train_x, train_y)
        if test_x is not None:
            test_x = scaler_standard.transform(test_x)
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

def doKNN(train_x, train_y, test_x=None, test_y=None, kSeq=[3, 5, 7], k=5,
          kfolds=KFold(10, shuffle=True, random_state=2323), tuningMode=True,
          model_export=False, preTrained=None, seed=7777):
    result_dic = {}
    np.random.seed(seed)
    scaler_minmax = MinMaxScaler()
    train_x = scaler_minmax.fit_transform(train_x)
    tuner_params = {"n_neighbors": kSeq}

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:
                knn_model = knn.KNeighborsRegressor(n_jobs=None)
                model_tuner = GridTuner(knn_model, param_grid=tuner_params,
                                        cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        pre_dispatch=multiprocessing.cpu_count(),
                                        scoring="neg_root_mean_squared_error")
                model_tuner.fit(train_x, train_y)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = knn.KNeighborsRegressor(n_neighbors=model_tuner.best_params_["n_neighbors"],
                                                              n_jobs=multiprocessing.cpu_count())
            else:
                result_dic["model"] = knn.KNeighborsRegressor(n_neighbors=k,
                                                              n_jobs=multiprocessing.cpu_count())
        result_dic["model"].fit(train_x, train_y)
        if test_x is not None:
            test_x = scaler_minmax.transform(test_x)
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

def doXGB(train_x, train_y, test_x=None, test_y=None, ntrees=5000, eta=5e-3,
          depthSeq=[4, 6, 8], subsampleSeq=[0.6, 0.8], colsampleSeq=[0.6, 0.8],
          l2Seq=[0.1, 1.0, 5.0], mcwSeq=[1, 3, 5], gammaSeq=[0.0, 0.2],
          kfolds=KFold(10, shuffle=True, random_state=2323),
          model_export=False, preTrained=None, seed=11, tuningMode=True):
    result_dic = {}
    np.random.seed(seed)
    tuner_params = {"max_depth": depthSeq, "subsample": subsampleSeq, "colsample_bytree": colsampleSeq,
                    "reg_lambda": l2Seq, "min_child_weight": mcwSeq, "gamma": gammaSeq}
    patientRate = 0.2

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:
                xgb_model = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                             n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                             n_jobs=None, random_state=seed,
                                             verbosity=0, use_label_encoder=False)
                model_tuner = GridTuner(xgb_model, param_grid=tuner_params, cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        scoring="neg_root_mean_squared_error")
                model_tuner.fit(train_x, train_y, verbose=False)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                                       n_estimators=ntrees, learning_rate=eta,
                                                       max_depth=model_tuner.best_params_["max_depth"],
                                                       subsample=model_tuner.best_params_["subsample"],
                                                       colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                       reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                       min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                       gamma=model_tuner.best_params_["gamma"],
                                                       n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                       verbosity=0, use_label_encoder=False)

                result_dic["best_params"]["best_trees"] = 0
                for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                    result_dic["model"].fit(train_x.iloc[nonkIdx,:], train_y[nonkIdx],
                                            eval_set=[(train_x.iloc[kIdx,:], train_y[kIdx])],
                                            eval_metric="rmse", verbose=False,
                                            early_stopping_rounds=int(ntrees * patientRate))
                    result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration / kfolds.get_n_splits()
                result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                result_dic["model"] = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                                       n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                       max_depth=model_tuner.best_params_["max_depth"],
                                                       subsample=model_tuner.best_params_["subsample"],
                                                       colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                       reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                       min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                       gamma=model_tuner.best_params_["gamma"],
                                                       n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                       verbosity=0, use_label_encoder=False)
            else:
                result_dic["model"] = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                                       n_estimators=ntrees, learning_rate=eta,
                                                       max_depth=depthSeq,
                                                       subsample=subsampleSeq,
                                                       colsample_bytree=colsampleSeq,
                                                       reg_lambda=l2Seq,
                                                       min_child_weight=mcwSeq,
                                                       gamma=gammaSeq,
                                                       n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                       verbosity=0, use_label_encoder=False)

        result_dic["model"].fit(train_x, train_y, verbose=False)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

def doLGB(train_x, train_y, test_x=None, test_y=None, categoIdx=None, boostingType="goss", ntrees=5000, eta=5e-3,
          leavesSeq=[pow(2, i) - 1 for i in [4, 6, 8]], subsampleSeq=[0.6, 0.8], gammaSeq=[0.0, 0.2],
          colsampleSeq=[0.6, 0.8], l2Seq=[0.1, 1.0, 5.0], mcsSeq=[5, 10, 20], mcwSeq=[1e-4, 1e-3, 1e-2],
          kfolds=KFold(10, shuffle=True, random_state=2323), model_export=False, preTrained=None, seed=22, tuningMode=True):
    result_dic = {}
    np.random.seed(seed)
    tuner_params = {"num_leaves": leavesSeq, "subsample": subsampleSeq, "colsample_bytree": colsampleSeq,
                    "reg_lambda": l2Seq, "min_child_samples": mcsSeq, "min_child_weight": mcwSeq, "min_split_gain": gammaSeq}
    patientRate = 0.2

    runStart = time()

    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:
                if boostingType == "rf":
                    lgb_model = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                  n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                  silent=True, n_jobs=None,
                                                  subsample_freq=2, random_state=seed)
                    model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                            cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_root_mean_squared_error")
                    model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            subsample_freq=2, silent=True)

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="rmse",
                                                verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            subsample_freq=2, silent=True)
                elif boostingType == "goss":
                    lgb_model = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                  n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                  silent=True, n_jobs=None, random_state=seed)
                    model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                            cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_root_mean_squared_error")
                    model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)
                    result_dic["model"] = lgb_model.fit(train_x, train_y, categorical_feature=categoIdx,
                                                        eval_set=[(test_x, test_y)], eval_metric="rmse",
                                                        verbose=False, early_stopping_rounds=int(ntrees * patientRate))

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="rmse",
                                                verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)
            else: # not tuning mode
                if boostingType == "rf":
                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=leavesSeq,
                                                            subsample=subsampleSeq,
                                                            colsample_bytree=colsampleSeq,
                                                            reg_lambda=l2Seq,
                                                            min_child_weight=mcwSeq,
                                                            min_child_samples=mcsSeq,
                                                            min_split_gain=gammaSeq,
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            subsample_freq=2, silent=True)
                elif boostingType == "goss":
                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=leavesSeq,
                                                            subsample=subsampleSeq,
                                                            colsample_bytree=colsampleSeq,
                                                            reg_lambda=l2Seq,
                                                            min_child_weight=mcwSeq,
                                                            min_child_samples=mcsSeq,
                                                            min_split_gain=gammaSeq,
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)


        result_dic["model"].fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

def createNetwork(nCols, mlpName, hiddenLayers=128, dropoutRate=1/2**2, seqLength=5, eta=2e-3):

    if mlpName == "MLP_LSTM_V1":
        # === input layers ===
        B0_input = layers.Input(shape=(seqLength, nCols), dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_lstm = layers.LSTM(units=hiddenLayers, dropout=dropoutRate, return_sequences=True)(B0_embedding)
        B2_lstm = layers.LSTM(units=hiddenLayers, dropout=dropoutRate)(B1_lstm)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 2), activation="relu", name="layer_final")(B2_lstm)
    else:
        print("Available Model list --->", ["MLP_Desc_V1", "MLP_ResNet_V1", "MLP_DenseNet_V1", "MLP_LP_V1"])
        return None

    if targetType == "numeric":
        layer_regressor = layers.Dense(units=1, name="Regressor")(layer_final)
        model_mlp = Model(B0_input, layer_regressor)
        model_mlp.compile(optimizer=optimizers.Adam(learning_rate=eta), loss="mean_squared_error",
                          metrics=tf_metrics.RootMeanSquaredError(name="rmse"))
    else:
        if targetTask == "binary":
            layer_classifier = layers.Dense(units=1, activation="sigmoid", name="Classifier")(layer_final)
            model_mlp = Model(B0_input, layer_classifier)
            model_mlp.compile(optimizer=optimizers.Adam(learning_rate=eta), loss="binary_crossentropy",
                              metrics=tf_metrics.AUC(name="roc_auc"))
        else:
            layer_classifier = layers.Dense(units=len(class_levels), activation="softmax", name="Classifier")(
                layer_final)
            model_mlp = Model(B0_input, layer_classifier)
            model_mlp.compile(optimizer=optimizers.Adam(learning_rate=eta), loss="categorical_crossentropy",
                              metrics=tf_metrics.CategoricalAccuracy(name="accuracy"))

    return model_mlp
# # "Available Model list --->", ["MLP_Desc_V1", "MLP_ResNet_V1", "MLP_DenseNet_V1", "MLP_LP_V1", "MLP_MultiActs_V1"]
# targetType = "numeric"
# createNetwork(nCols=10, mlpName="MLP_MultiActs_V2").summary()
def doMLP(train_x, train_y, test_x, test_y, mlpName="MLP_Desc_V1", seqLength=None,
          hiddenLayers={"min": 32, "max": 128, "step": 32}, dropoutRate=1/2**2,
          epochs=10, batch_size=32, model_export=False, preTrained=None, seed=515):
    result_dic = {}
    scaler_minmax = prep.MinMaxScaler()
    train_x = scaler_minmax.fit_transform(train_x)
    test_x = scaler_minmax.transform(test_x)
    patientRate = 0.2
    tf.random.set_seed(seed)
    cb_earlystopping = tf_callbacks.EarlyStopping(patience=int(epochs * patientRate),
                                                  restore_best_weights=True)
    cb_reduceLR = tf_callbacks.ReduceLROnPlateau(patience=int((epochs * patientRate)/10), factor=0.8, min_lr=1e-4)
    cb_lists = [cb_earlystopping, cb_reduceLR, TqdmCallback(verbose=0)]

    runStart = time()
    if targetType == "numeric":
        # LSTM model
        if seqLength is not None:
            train_ts_dataset = make_ts_tensor(train_x, train_y, sequence_length=seqLength, batch_size=batch_size)
            test_ts_dataset = make_ts_tensor(test_x, test_y, sequence_length=seqLength, batch_size=batch_size)
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if type(hiddenLayers) is dict:
                    pass
                    # model_tuner = kt.BayesianOptimization(
                    #     MyHyperModel(train_x.shape[1], mlpName=mlpName, eta=1e-4, seqLength=seqLength,
                    #                  hiddenLayers=hiddenLayers, dropoutRate=dropoutRate),
                    #     objective="val_loss",
                    #     max_trials=int(epochs * 0.2),
                    #     seed=seed + 1,
                    #     overwrite=True)
                    # model_tuner.search(train_ts_dataset, validation_data=test_ts_dataset, shuffle=False)
                    #
                    # result_dic["best_params"] = {"hiddenLayers": model_tuner.get_best_hyperparameters()[0].get("hiddenLayers")}
                    # print("\nTuning Result ---> Hidden layers :", result_dic["best_params"]["hiddenLayers"])
                else:
                    result_dic["best_params"] = {"hiddenLayers": hiddenLayers}

                result_dic['model'] = createNetwork(nCols=train_x.shape[1], mlpName=mlpName,
                                                    hiddenLayers=result_dic["best_params"]["hiddenLayers"],
                                                    dropoutRate=dropoutRate)

                result_dic["best_params"]["epochs"] = result_dic['model'].fit(train_ts_dataset,
                                                                              epochs=epochs,
                                                                              validation_data=test_ts_dataset,
                                                                              verbose=0, shuffle=False,
                                                                              callbacks=cb_lists)
                result_dic["best_params"]["epochs"] = np.argmin(result_dic["best_params"]["epochs"].history["val_loss"])

            if test_x is not None:
                result_dic["pred"] = result_dic["model"].predict(test_ts_dataset)
                if test_y is not None:
                    mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                    rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                    result_dic["performance"] = {"MAE": mae,
                                                 "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                                 "NMAE": mae / test_y.abs().mean(),
                                                 "RMSE": rmse,
                                                 "NRMSE": rmse / test_y.abs().mean(),
                                                 "R2": metrics.r2_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic





# ===== django ===============================================
from django.http import HttpResponse
from django.shortcuts import render
import joblib
from django.http import Http404

def home(request):

    # stock_list = read_csv("./Stock_List.csv")
    # stock_list = read_csv("C:/Users/flash/PycharmProjects/pythonProject/k3_django2/Stock_List.csv")
    tickers = []
    tickers += stock.get_market_ticker_list(market="KOSPI")
    tickers += stock.get_market_ticker_list(market="KOSDAQ")

    ticker_names = []
    for i in tickers:
        ticker_names.append(stock.get_market_ticker_name(i))
    stock_list = dataframe(columns=["종목명", "종목코드"])
    stock_list["종목명"] = ticker_names
    stock_list["종목코드"] = tickers
    if stock_list.isna().sum().sum() != 0:
        raise Http404("stock list has NA values")

    response = {
        "stock_list": stock_list["종목명"].sort_values()
    }

    return render(request, "home.html", response)



def result(request):

    process_runningtime = time()

    inputDic = {"target_corp": request.GET['selectedCom'],
                "input_model": request.GET['selectedModel'],
                "pred_start": request.GET['예측날짜'].replace("-", ""),
                "target_ntime": int(request.GET['예측기간']),
                "train_period": int(request.GET['훈련기간'])}

    target_timegap = inputDic["target_ntime"]

    # Get Data & Modeling
    # 분석할 date 변수 지정
    start_date = (datetime.strptime(inputDic["pred_start"], "%Y%m%d") - BDay(inputDic["train_period"])).strftime("%Y%m%d")
    end_date = inputDic["pred_start"]
    business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

    tickers = []
    tickers += stock.get_market_ticker_list(market="KOSPI")
    tickers += stock.get_market_ticker_list(market="KOSDAQ")

    ticker_names = []
    for i in tickers:
        ticker_names.append(stock.get_market_ticker_name(i))
    stock_list = dataframe(columns=["종목명", "종목코드"])
    stock_list["종목명"] = ticker_names
    stock_list["종목코드"] = tickers
    if stock_list.isna().sum().sum() != 0:
        raise Http404("stock list has NA values")

    stock_list.set_index("종목명", inplace=True)
    selected_codes = stock_list.index[stock_list.index == inputDic["target_corp"]].to_list()
    stock_list = stock_list.loc[selected_codes]["종목코드"]

    stock_dic = dict.fromkeys(selected_codes)
    error_list = []
    corr_list = []
    metric_days = 14
    cat_vars = []
    bin_vars = []
    cat_vars.append("weekday")
    cat_vars.append("weeknum")
    bin_vars.append("mfi_signal")

    # ==== selected feature =====
    selected_features = ["date", "close", "kospi", "obv", "trading_amount", "mfi_signal"]
    logtrans_vec = ["close", "kospi", "trading_amount"]
    pvalue_check = series(0, index=selected_features)

    dataloading_runningtime = time()
    for stock_name, stock_code in stock_list.items():
        print("=====", stock_name, "=====")
        # 종목 주가 데이터 로드
        try:
            stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
            stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
            investor_df = stock.get_market_trading_volume_by_date(start_date, end_date, stock_code)[["기관합계", "외국인합계"]].reset_index()
            kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()
            # sleep(0.5)

            stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            investor_df.columns = ["Date", "inst", "fore"]
            kospi_df.columns = ["Date", "kospi"]
            # 영업일과 주가 정보를 outer 조인
            train_x = pd.merge(business_days, stock_df, how='left', on="Date")
            train_x = pd.merge(train_x, investor_df, how='left', on="Date")
            train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
            # 종가데이터에 생긴 na 값을 선형보간 및 정수로 반올림
            train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)
        except:
            stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
            stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
            sleep(0.5)
            # investor_df = stock.get_market_trading_volume_by_date(start_date, end_date, stock_code)[["기관합계", "외국인합계"]].reset_index()
            # sleep(0.5)
            # kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()
            # sleep(0.5)

            stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            # investor_df.columns = ["Date", "inst", "fore"]
            # kospi_df.columns = ["Date", "kospi"]
            # 영업일과 주가 정보를 outer 조인
            train_x = pd.merge(business_days, stock_df, how='left', on="Date")
            # train_x = pd.merge(train_x, investor_df, how='left', on="Date")
            # train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
            # 종가데이터에 생긴 na 값을 선형보간 및 정수로 반올림
            train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)

        # ===== feature engineering =====
        # 거래대금 파생변수 추가
        train_x['trading_amount'] = train_x["Close"] * train_x["Volume"]

        # OBV 파생변수 추가
        # 매수 신호: obv > obv_ema
        # 매도 신호: obv < obv_ema
        obv = [0]
        for i in range(1, len(train_x.Close)):
            if train_x.Close[i] >= train_x.Close[i - 1]:
                obv.append(obv[-1] + train_x.Volume[i])
            elif train_x.Close[i] < train_x.Close[i - 1]:
                obv.append(obv[-1] - train_x.Volume[i])
            # else:
            #     obv.append(obv[-1])
        train_x['obv'] = obv
        train_x['obv'][0] = nan

        # MFI 파생변수 추가
        # MFI = 100 - (100 / 1 + MFR)
        # MFR = 14일간의 양의 MF / 14일간의 음의 MF
        # MF = 거래량 * (당일고가 + 당일저가 + 당일종가) / 3
        # MF 컬럼 만들기
        train_x["mf"] = train_x["Volume"] * ((train_x["High"] + train_x["Low"] + train_x["Close"]) / 3)
        # 양의 MF와 음의 MF 표기 컬럼 만들기
        p_n = []
        for i in range(len(train_x['mf'])):
            if i == 0:
                p_n.append(nan)
            else:
                if train_x['mf'][i] >= train_x['mf'][i - 1]:
                    p_n.append('p')
                else:
                    p_n.append('n')
        train_x['p_n'] = p_n
        # 14일간 양의 MF/ 14일간 음의 MF 계산하여 컬럼 만들기
        mfr = []
        for i in range(len(train_x['mf'])):
            if i < metric_days - 1:
                mfr.append(nan)
            else:
                train_x_ = train_x.iloc[(i - metric_days + 1):i]
                a = (sum(train_x_['mf'][train_x['p_n'] == 'p']) + 1) / (sum(train_x_['mf'][train_x['p_n'] == 'n']) + 10)
                mfr.append(a)
        train_x['mfr'] = mfr
        # 최종 MFI 컬럼 만들기
        train_x['mfi'] = 100 - (100 / (1 + train_x['mfr']))
        train_x["mfi_signal"] = train_x['mfi'].apply(lambda x: "buy" if x > 50 else "sell")

        # 지표계산을 위해 쓰인 컬럼 drop
        train_x.drop(["mf", "p_n", "mfr", "Open", "High", "Low"], inplace=True, axis=1)

        train_x = train_x.dropna()
        train_x.reset_index(drop=True, inplace=True)
        print("NA values --->", train_x.isna().sum().sum())

        # create target list
        target_list = []
        target_list.append(train_x["Close"])
        for i in range(1,target_timegap+1,1):
            target_list.append(train_x["Close"].shift(-i))
        for idx, value in enumerate(target_list):
            value.name = "target_shift" + str(idx)

        # 컬럼이름 소문자 변환 및 정렬
        train_x.columns = train_x.columns.str.lower()
        train_x = pd.concat([train_x[["date"]], train_x.iloc[:, 1:].sort_index(axis=1)], axis=1)

        # # <visualization>
        # # 시각화용 데이터프레임 생성
        # train_bi = pd.concat([target_list[timeunit_gap_forviz], train_x], axis=1)[:-timeunit_gap_forviz]
        #
        # # 기업 평균 상관관계를 측정하기 위한 연산
        # corr_obj = train_bi.corr().round(3)
        # corr_rows = corr_obj.index.tolist()
        # corr_cols = corr_obj.columns.tolist()
        # corr_list.append(corr_obj.to_numpy().round(3)[..., np.newaxis])

        # <feature selection>
        if len(selected_features) != 0:
            train_x = train_x[np.intersect1d(train_x.columns, selected_features)]

        # <feature scaling>
        # log transform
        for i in logtrans_vec:
            if i in train_x.columns:
                train_x[i] = train_x[i].apply(np.log1p)

        # onehot encoding
        onehot_encoder = MyOneHotEncoder()
        train_x = onehot_encoder.fit_transform(train_x, cat_vars + bin_vars)

        stock_dic[stock_name]["df"] = train_x.copy()
        stock_dic[stock_name]["target_list"] = target_list.copy()
    dataloading_runningtime = time() - dataloading_runningtime

    # ===== Automation Predict =====
    # validation data evaluation
    model_names = ["Linear", "ElasticNet", "KNN", "XGB_GBT",
                   "LGB_RandomForest", "LGB_GOSS", "ARIMA", "LSTM"]


    seqLength = 5
    output_str = ""
    output_list = []

    # 데이터를 저장할 변수 설정
    total_perf = None
    for stock_name, stock_data in stock_dic.items():
        stock_data["perf_list"] = dict.fromkeys(model_names)
        stock_data["pred_list"] = dict.fromkeys(model_names)
        total_perf = dict.fromkeys(model_names)
        for i in model_names:
            stock_data["perf_list"][i] = dict.fromkeys(range(1,target_timegap+1,1), 0)
            stock_data["pred_list"][i] = dict.fromkeys(range(1,target_timegap+1,1), 0)
            total_perf[i] = dict.fromkeys(range(1,target_timegap+1,1), 0)
            for j in total_perf[i].keys():
                total_perf[i][j] = series(0, index=["MAE", "MAPE", "NMAE", "RMSE", "NRMSE", "R2", "Running_Time"])

    fit_runningtime = time()
    for time_ngap in range(1, target_timegap + 1):
        print(F"=== Target on N+{time_ngap} ===")
        # time_ngap = 3
        for stock_name, stock_data in stock_dic.items():
            # remove date
            # break

            test_x = stock_data["df"].iloc[-1:]
            test_x_lstm = stock_data["df"].iloc[-seqLength:]

            full_x = stock_data["df"][:-time_ngap]
            full_y = stock_data["target_list"][time_ngap][:-time_ngap]
            full_x_lstm = stock_data["df"][:-time_ngap]
            full_y_lstm = full_y[seqLength - 1:]
            arima_full = stock_data["target_list"][0]

            # create dataset for fitting
            data_forfit = stock_data["df"][:-time_ngap]
            target_forfit = stock_data["target_list"][time_ngap][:-time_ngap]

            val_x = data_forfit.iloc[-1:]
            val_y = target_forfit.iloc[-1:]
            val_x_lstm = data_forfit.iloc[-seqLength:]
            val_y_lstm = target_forfit.iloc[-1:]

            train_x = data_forfit[:-time_ngap]
            train_y = target_forfit[:-time_ngap]
            train_x_lstm = data_forfit[:-time_ngap]
            train_y_lstm = train_y[seqLength - 1:]
            arima_train = stock_data["target_list"][0][:-time_ngap][:-time_ngap]



            full_x.drop("date", axis=1, inplace=True)
            full_x_lstm.drop("date", axis=1, inplace=True)
            train_x.drop("date", axis=1, inplace=True)
            train_x_lstm.drop("date", axis=1, inplace=True)
            val_x.drop("date", axis=1, inplace=True)
            val_x_lstm.drop("date", axis=1, inplace=True)
            test_x.drop("date", axis=1, inplace=True)
            test_x_lstm.drop("date", axis=1, inplace=True)

            if inputDic["input_model"] == "Linear":
                # <선형회귀>
                tmp_runtime = time()
                print("Linear Regression on", stock_name)
                # evaludation on validation set
                model = doLinear(train_x, train_y, val_x, val_y)
                print(model["performance"])
                stock_data["perf_list"]["Linear"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])

                # prediction on test set
                model = doLinear(full_x, full_y, test_x, None)
                stock_data["pred_list"]["Linear"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["Linear"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["Linear"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["Linear"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "ElasticNet":
                # <엘라스틱넷>
                tmp_runtime = time()
                print("ElasticNet on", stock_name)
                # evaludation on validation set
                model = doElasticNet(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter)
                print(model["performance"])
                stock_data["perf_list"]["ElasticNet"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])
                # prediction on test set
                model = doElasticNet(full_x, full_y, test_x, None, kfolds=kfolds_spliter, tuningMode=False,
                                     alpha=model["best_params"]["alpha"], l1_ratio=model["best_params"]["l1_ratio"])
                stock_data["pred_list"]["ElasticNet"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["ElasticNet"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["ElasticNet"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["ElasticNet"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "KNN":
                # <KNN>
                tmp_runtime = time()
                print("KNN on", stock_name)
                # evaludation on validation set
                model = doKNN(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter)
                print(model["performance"])
                stock_data["perf_list"]["KNN"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])
                # prediction on test set
                model = doKNN(full_x, full_y, test_x, None, k=model["best_params"]["n_neighbors"],
                              tuningMode=False, kfolds=kfolds_spliter)
                stock_data["pred_list"]["KNN"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["KNN"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["KNN"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["KNN"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "XGB_GBT":
                # <XGBoost>
                tmp_runtime = time()
                print("XGB_GBT on", stock_name)
                # evaludation on validation set
                model = doXGB(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter,
                              ntrees=1000, eta=1e-2,
                              depthSeq=[4], subsampleSeq=[0.8], colsampleSeq=[1.0], gammaSeq=[0.0])
                print(model["performance"])
                stock_data["perf_list"]["XGB_GBT"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])
                print(model["best_params"])
                # prediction on test set
                model = doXGB(full_x, full_y, test_x, None, tuningMode=False,
                              ntrees=model["best_params"]["best_trees"],
                              depthSeq=model["best_params"]["max_depth"],
                              mcwSeq=model["best_params"]["min_child_weight"],
                              l2Seq=model["best_params"]["reg_lambda"],
                              gammaSeq=model["best_params"]["gamma"],
                              subsampleSeq=model["best_params"]["subsample"],
                              colsampleSeq=model["best_params"]["colsample_bytree"])
                stock_data["pred_list"]["XGB_GBT"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["XGB_GBT"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["XGB_GBT"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["XGB_GBT"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "LGB_RandomForest":
                # <LightGBM 랜덤포레스트>
                tmp_runtime = time()
                # GridSearchCV의 param_grid 설정
                print("LGB_RandomForest on", stock_name)
                params = {
                    'learning_rate': [0.01],
                    'num_leaves': [2 ** i - 1 for i in [4, 6]],
                    'n_estimators': [100]
                }

                model = lgb.LGBMRegressor(boosting_type='rf', objective="regression",
                                          subsample=0.8, subsample_freq=2,
                                          n_jobs=None, random_state=321)
                grid = GridTuner(estimator=model, param_grid=params,
                                 n_jobs=multiprocessing.cpu_count(),
                                 refit=False, cv=kfolds_spliter)
                grid.fit(train_x, train_y)

                model = lgb.LGBMRegressor(boosting_type='rf', objective="regression",
                                          subsample=0.8, subsample_freq=2,
                                          n_estimators=grid.best_params_["n_estimators"],
                                          num_leaves=grid.best_params_["num_leaves"],
                                          learning_rate=grid.best_params_["learning_rate"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=20, eval_metric='rmse', verbose=0)
                pred = model.predict(val_x)
                print(grid.best_params_)
                print("best iteration --->", model.best_iteration_)

                # recode performance
                tmp_mae = metrics.mean_absolute_error(val_y, pred)
                tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
                model_perf = {"MAE": tmp_mae,
                              "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                              "NMAE": tmp_mae / val_y.abs().mean(),
                              "RMSE": tmp_rmse,
                              "NRMSE": tmp_rmse / val_y.abs().mean(),
                              "R2": metrics.r2_score(val_y, pred)}
                print(model_perf)
                stock_data["perf_list"]["LGB_RandomForest"][time_ngap] = model_perf

                # prediction on test data
                model = lgb.LGBMRegressor(boosting_type='rf', objective="regression",
                                          subsample=0.8, subsample_freq=2,
                                          n_estimators=model.best_iteration_,
                                          num_leaves=grid.best_params_["num_leaves"],
                                          learning_rate=grid.best_params_["learning_rate"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(full_x, full_y, verbose=0)
                pred = model.predict(test_x)
                stock_data["pred_list"]["LGB_RandomForest"][time_ngap] = pred
                # recode running time
                tmp_runtime = time() - tmp_runtime
                print(tmp_runtime)
                total_perf["LGB_RandomForest"][time_ngap] += series(model_perf).append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["LGB_RandomForest"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["LGB_RandomForest"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "LGB_GOSS":
                # <LightGBM Gradient-based One-Side Sampling>
                tmp_runtime = time()
                # GridSearchCV의 param_grid 설정
                print("LGB_GOSS on", stock_name)
                params = {
                    'learning_rate': [5e-4],
                    'n_estimators': [2000],
                    'num_leaves': [2 ** i - 1 for i in [4, 6]],
                    'reg_lambda': [0.1, 1.0, 5.0],
                    'min_child_samples': [5, 10, 20]
                }

                model = lgb.LGBMRegressor(boosting_type='goss', objective="regression",
                                          subsample=0.8, n_jobs=None, random_state=321)
                grid = GridTuner(estimator=model, param_grid=params,
                                 n_jobs=multiprocessing.cpu_count(),
                                 refit=False, cv=kfolds_spliter)
                grid.fit(train_x, train_y)

                model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8,
                                          n_estimators=5000,
                                          num_leaves=grid.best_params_["num_leaves"],
                                          reg_lambda=grid.best_params_["reg_lambda"],
                                          min_child_samples=grid.best_params_["min_child_samples"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=500, eval_metric='rmse', verbose=0)
                pred = model.predict(val_x)
                print(grid.best_params_)
                print("best iteration --->", model.best_iteration_)

                # recode performance
                tmp_mae = metrics.mean_absolute_error(val_y, pred)
                tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
                model_perf = {"MAE": tmp_mae,
                              "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                              "NMAE": tmp_mae / val_y.abs().mean(),
                              "RMSE": tmp_rmse,
                              "NRMSE": tmp_rmse / val_y.abs().mean(),
                              "R2": metrics.r2_score(val_y, pred)}
                print(model_perf)
                stock_data["perf_list"]["LGB_GOSS"][time_ngap] = model_perf

                # prediction on test data
                model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8,
                                          n_estimators=model.best_iteration_,
                                          num_leaves=grid.best_params_["num_leaves"],
                                          reg_lambda=grid.best_params_["reg_lambda"],
                                          min_child_samples=grid.best_params_["min_child_samples"],
                                          n_jobs=multiprocessing.cpu_count(), random_state=321)
                model.fit(full_x, full_y, verbose=0)
                pred = model.predict(test_x)
                stock_data["pred_list"]["LGB_GOSS"][time_ngap] = pred
                # recode running time
                tmp_runtime = time() - tmp_runtime
                print(tmp_runtime)
                total_perf["LGB_GOSS"][time_ngap] += series(model_perf).append(series({"Running_Time": tmp_runtime}))
                output_list.append(stock_data["pred_list"]["LGB_GOSS"][time_ngap][0])
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["LGB_GOSS"][time_ngap][0]) + "\n"
            elif inputDic["input_model"] == "ARIMA":
                # <ARIMA>
                tmp_runtime = time()
                print("ARIMA on", stock_name)
                # order=(p: Auto regressive, q: Difference, d: Moving average)
                # 일반적 하이퍼파라미터 공식
                # 1. p + q < 2
                # 2. p * q = 0
                # 근거 : 실제로 대부분의 시계열 자료에서는 하나의 경향만을 강하게 띄기 때문 (p 또는 q 둘중 하나는 0)
                model = ARIMA(arima_train, order=(1, 2, 0))
                model_fit = model.fit()
                pred = array([model_fit.forecast(target_timegap).iloc[-1]])

                tmp_mae = metrics.mean_absolute_error(val_y, pred)
                tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
                tmp_perf = {"MAE": tmp_mae,
                            "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                            "NMAE": tmp_mae / val_y.abs().mean(),
                            "RMSE": tmp_rmse,
                            "NRMSE": tmp_rmse / val_y.abs().mean(),
                            "R2": metrics.r2_score(val_y, pred)}
                print(tmp_perf)
                stock_data["perf_list"]["ARIMA"][target_timegap] = tmp_perf

                # prediction on test data
                model = ARIMA(arima_full, order=(1, 2, 0))
                model_fit = model.fit()

                for idx, value in enumerate(model_fit.forecast(target_timegap)):
                    stock_data["pred_list"]["ARIMA"][idx+1] = value

                tmp_runtime = time() - tmp_runtime
                total_perf["ARIMA"][target_timegap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))
                for idx, value in stock_data["pred_list"]["ARIMA"].items():
                    output_list.append(value)
                    output_str += str(idx) + " ---> " + str(value) + "\n"
            elif inputDic["input_model"] == "LSTM":
                pass
                # <LSTM>
                tmp_runtime = time()
                print("LSTM on", stock_name)
                model = doMLP(train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, mlpName="MLP_LSTM_V1",
                              hiddenLayers=64, epochs=100, batch_size=4, seqLength=seqLength, model_export=True)
                print(model["performance"])
                stock_data["perf_list"]["LSTM"][time_ngap] = model["performance"]
                tmp_perf = series(model["performance"])

                model = doMLP(full_x_lstm, full_y_lstm, test_x_lstm, None,
                              seqLength=seqLength, preTrained=model["model"])
                stock_data["pred_list"]["LSTM"][time_ngap] = model["pred"]
                tmp_runtime = time() - tmp_runtime
                total_perf["LSTM"][time_ngap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))
                output_str += str(time_ngap) + " ---> " + str(stock_data["pred_list"]["LSTM"][time_ngap][0]) + "\n"
            else:
                print("WARNNING : Unknown model")
        print(output_str)
        if inputDic["input_model"] == "ARIMA":
            break
    fit_runningtime = time() - fit_runningtime
    print(fit_runningtime)

    # 성능평가 테이블 생성
    perf_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1, target_timegap+1,1)])
    runningtime_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1, target_timegap+1,1)])
    for i in list(total_perf.keys()):
        if array(list(total_perf[i].values())).sum() == 0:
            pass
        else:
            perf_table.loc[i] = dataframe(total_perf[i]).loc["NMAE"].values
            runningtime_table.loc[i] = dataframe(total_perf[i]).loc["Running_Time"].values

    perf_table = perf_table.iloc[:, :target_timegap]
    perf_table = perf_table * 100
    perf_table.loc["best_model"] = perf_table.min(axis=0)
    perf_table["avg"] = perf_table.iloc[:, :target_timegap].mean(axis=1)
    perf_table["std"] = perf_table.iloc[:, :target_timegap].std(axis=1)
    perf_table["running_time"] = runningtime_table.mean(axis=1).append(series({"best_model": -1}))
    print(perf_table)
    process_runningtime = time() - process_runningtime
    print(process_runningtime)
    output_list = np.round(output_list).tolist()

    date_list = []
    for i in range(1,len(output_list)+1):
        date_list.append(datetime.strptime(inputDic["pred_start"], "%Y%m%d") + BDay(i))

    concat_1 = stock_dic[inputDic["target_corp"]]["df"].iloc[:,:2].set_index("date")
    concat_1["close"] = stock_dic[inputDic["target_corp"]]["target_list"][0].values
    concat_2 = series(output_list, index=date_list, name="close").to_frame()



    # <전체기간 plot>
    plot_df = pd.concat([concat_1, concat_2], axis=0)["close"]

    # empty figure 출력
    fig, ax = plt.subplots(figsize=(12, 6))
    # 예측 이전 가격 plot
    graph_act = sns.lineplot(x=plot_df.index[:-inputDic["target_ntime"]+1], y=plot_df.values[:-inputDic["target_ntime"]+1],
                             ci=None, color=sns.color_palette("Set2", 8)[0])

    # 예측 후 가격 plot
    sns.lineplot(x=plot_df.index[-inputDic["target_ntime"]:], y=plot_df.values[-inputDic["target_ntime"]:],
                              ci=None, color=sns.color_palette("Set2", 8)[1])

    # 20일 이평선 plot
    sns.lineplot(x=plot_df.rolling(window=20, min_periods=20).mean().index,
                              y=plot_df.rolling(window=20, min_periods=20).mean().values, linewidth=1.3,
                              ci=None, linestyle="--", color=sns.color_palette("Set2", 8)[3])

    graph_act.set_xlabel("일자", fontsize=14, fontweight="bold", labelpad=20)
    graph_act.set_ylabel("가격", fontsize=14, fontweight="bold", labelpad=20)

    ax.set_xticks(plot_df.index[::-1][::20][::-1])
    plt.xticks(rotation=45)
    plt.legend(loc="upper left", labels=['과거주가', '예측치', '20일 이평선'], fontsize=10)
    plt.subplots_adjust(bottom=0.2)
    graph_act.set_title(inputDic["target_corp"] + " 가격 예측 차트", fontsize=20, fontweight="bold", pad=15)

    # 예측 데이터 강조 vertical line plot
    plt.axvline(x=plot_df.index[:-inputDic["target_ntime"]+1][-1], color='gray', linestyle='--', linewidth=1.5)

    plt.savefig("DeployModel/static/" + inputDic["target_corp"] + "_alltime", dpi=300)
    plt.close()


    # <예측과 같은기간 plot>
    plot_df = pd.concat([concat_1[-inputDic["target_ntime"]:], concat_2], axis=0)["close"]

    # empty figure 출력
    fig, ax = plt.subplots(figsize=(12, 6))
    # 예측 이전 가격 plot (20일)
    graph_act = sns.lineplot(x=plot_df.index[(-inputDic["target_ntime"] + 1 - 20):-inputDic["target_ntime"] + 1],
                             y=plot_df.values[(-inputDic["target_ntime"] + 1 - 20):-inputDic["target_ntime"] + 1],
                             ci=None, color=sns.color_palette("Set2", 8)[0])

    # 예측 후 가격 plot
    sns.lineplot(x=plot_df.index[-inputDic["target_ntime"]:], y=plot_df.values[-inputDic["target_ntime"]:],
                 ci=None, color=sns.color_palette("Set2", 8)[1])

    # 5일 이평선 plot
    plot_df_5mv = plot_df.rolling(window=5, min_periods=5).mean()
    sns.lineplot(x=plot_df_5mv.index, y=plot_df_5mv.values, linewidth=1.3,
                              ci=None, linestyle="--", color=sns.color_palette("Set2", 8)[2])

    graph_act.set_xlabel("일자", fontsize=14, fontweight="bold", labelpad=20)
    graph_act.set_ylabel("가격", fontsize=14, fontweight="bold", labelpad=20)
    ax.set_xticks(plot_df.index[::-1][::5][::-1])
    plt.xticks(rotation=45)
    plt.legend(loc="upper left", labels=['과거주가', '예측치', '5일 이평선'], fontsize=10)
    plt.subplots_adjust(bottom=0.2)
    graph_act.set_title(inputDic["target_corp"] + " 가격 예측 차트_all", fontsize=20, fontweight="bold", pad=15)

    # 예측 데이터 강조 vertical line plot
    plt.axvline(x=plot_df.index[:-inputDic["target_ntime"] + 1][-1], color='gray', linestyle='--', linewidth=1.5)
    plt.savefig("DeployModel/static/" + inputDic["target_corp"], dpi=300)
    plt.close()

    corp_name = inputDic["target_corp"]
    with open('DeployModel/sector_html.pickle', "rb") as f:
        sector = pickle.load(f)


    st_key = list(sector.keys())

    for i in st_key:
        chk_lst = sector[i].keys()
        print(i)
        if corp_name in chk_lst:
            cis = sector[i][corp_name]["cis"]
            bs = sector[i][corp_name]["bs"]
            cf = sector[i][corp_name]["cf"]
            met = pd.DataFrame(sector["it"]["NAVER"]["metrics"],index=[0])
            cis.to_pickle("cis.pkl")
            bs.to_pickle("bs.pkl")
            cf.to_pickle("cf.pkl")
            met.to_pickle("met.pkl")
            cis.to_html("DeployModel/cis.html")
            bs.to_html("DeployModel/bs.html")
            cf.to_html("DeployModel/cf.html")
            met.to_html("DeployModel/metrics.html")

            break
        elif i == st_key[-1]:
            cis_ht = None
            bs_ht = None
            cf_ht = None

    #met.to_html('met.html')


    response = {
        "corp_name": inputDic["target_corp"],
        "prediction": output_list,
        "validation_score": perf_table.loc[inputDic["input_model"]].values[:target_timegap],
        "validation_score_avg": perf_table.loc[inputDic["input_model"]].values[-3],
        "validation_score_std": perf_table.loc[inputDic["input_model"]].values[-2],
        "process_runningtime": round(process_runningtime,5),
        "dataloading_runningtime": round(dataloading_runningtime,5),
        "fit_runningtime": round(fit_runningtime,5),
        "img_name1" : inputDic["target_corp"]+".png",
        "img_name2" : inputDic["target_corp"] + "_alltime"+ ".png",
    }

    return render(request, "result.html", response)



'''
for i in sector.keys():
    print(i)
    try:
        cis = sector[i]["삼성전자"]["cis"]
        bs = sector[i]["삼성전자"]["bs"]
        cf = sector[i]["삼성전자"]["cf"]
        met = sector[i]["삼성전자"]["metrics"]
        break
    except:
        event = "재무데이터가 없습니다."
        print(event)
'''


import pickle
from django.views.decorators.csrf import csrf_exempt
import webbrowser
@csrf_exempt
def popup(request):

    try:
        cis = pd.read_pickle('cis.pkl').to_html()
        bs = pd.read_pickle('bs.pkl').to_html()
        cf = pd.read_pickle('cf.pkl').to_html()
        met = pd.read_pickle('met.pkl').T.to_html()


    except:
        cis = None
        bs = None
        cf = None

    response = {"cis" : cis,
                "bs" : bs,
                "cf" : cf,
                "metrics" : met,
    }
    return render(request, 'popup.html', response)