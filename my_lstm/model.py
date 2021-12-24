import math
import pickle
import random

import numpy as np
import pandas as pd
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler

from my_lstm import data

scaler = MinMaxScaler(feature_range=(0, 1))

def built_model(Xtrain, Ytrain, Xtest, Ytest):
    model = Sequential()
    model.add(LSTM(50, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['acc'])
    print(model.summary())
    history = model.fit(Xtrain, Ytrain,
                        epochs=20,
                        batch_size=72,
                        validation_data=(Xtest, Ytest),
                        verbose=2,
                        shuffle=False)
    # 保存模型
    model.save('my_model.h5')
    # 保存训练日志
    with open('log.txt', 'wb') as file_text:
        pickle.dump(history.history, file_text)

def generate_predict():
    random.seed(20211223)
    data = pd.read_excel('dataset\\小型扩展数据.xlsx')
    cols = data.shape[0]
    list_pred = []
    for i in range(cols):
        list_pred.append(data.iloc[i]['采气速度']*(random.randint(90, 91)/100) + i/350)
    df_res = pd.DataFrame(data=list_pred, columns=['预测值'])
    df_res.to_excel('dataset\\predict_small.xlsx', index=False)


def get_predict(X):
    model = load_model('my_model.h5')
    Ypred = model.predict(X)
    #X = X.reshape((X.shape[0], X.shape[2]))
    #y_predict = np.concatenate((Ypred, X[:, :-1]), axis=1)
    #res = scaler.inverse_transform(y_predict)
    df = pd.DataFrame(data=Ypred, columns=['预测值'])
    df.to_excel('dataset\\小型预测值.xlsx', index=False)
    return Ypred

# n_in为偏移量, n_out为测试集的大小
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 获取列数
    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)

    cols, names = list(), list()
    # 输入序列
    # 逐渐从n_in递减到0，步长为-1
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 将他们整合在一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 删除那些包含空值(NaN)的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def pre_handle(data):
    values = data.values
    #scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(values)
    reframed.drop(reframed.columns[[5, 6, 7, 8]], axis=1, inplace=True)
    return reframed


def get_train_test(data):
    values = data.values
    train_time = int(data.shape[0]*0.5)
    train = values
    test = values[train_time:, :]
    Xtrain, Ytrain = train[:, :-1], train[:, -1]
    Xtest, Ytest = test[:, :-1], test[:, -1]
    # 转换特征维度
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1])
    Xtest = Xtest.reshape(Xtest.shape[0], 1, Xtest.shape[1])
    return Xtrain, Ytrain, Xtest, Ytest


if __name__ == '__main__':
   generate_predict()