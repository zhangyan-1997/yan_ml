import pickle

import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler

from my_lstm import data, picture


def built_model(Xtrain, Ytrain, Xtest, Ytest, log_path, model_path):
    model = Sequential()
    model.add(LSTM(50, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), return_sequences=True))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    print(model.summary())
    history = model.fit(Xtrain, Ytrain,
                        epochs=50,
                        batch_size=72,
                        validation_data=(Xtest, Ytest),
                        verbose=2,
                        shuffle=False)
    # 保存模型
    model.save(model_path)
    # 保存训练日志
    with open(log_path, 'wb') as file_text:
        pickle.dump(history.history, file_text)



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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled)
    reframed.drop(reframed.columns[[8, 9, 10, 11, 12, 13]], axis=1, inplace=True)
    return reframed


def get_train_test(data, train_nums):
    values = data.values
    n_hours = int(365*24*train_nums)
    train = values[:n_hours, :]
    test = values[n_hours:, :]
    Xtrain, Ytrain = train[:, :-1], train[:, -1]
    Xtest, Ytest = test[:, :-1], test[:, -1]
    # 转换特征维度
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1])
    Xtest = Xtest.reshape(Xtest.shape[0], 1, Xtest.shape[1])
    return Xtrain, Ytrain, Xtest, Ytest



def train_process():
    '''
    df = data.load_data('pollution.csv')
    df = data.data_clean_pollution(df)
    picture.show_columns(df)
    df = pre_handle(df)
    '''
    log_lists = ['small', 'medium', 'large']
    train_datas_nums = [2, 1.2, 2.4]
    for log, num in zip(log_lists, train_datas_nums):
        log_path = 'log_' + log + '.txt'
        model_path = 'model_' + log + '.h5'
        title = 'Model Train loss tendency -MSE on ' + log + ' Type'
        #Xtrain, Ytrain, Xtest, Ytest = get_train_test(df, num)
        #built_model(Xtrain, Ytrain, Xtest, Ytest, log_path=log_path, model_path=model_path)
        picture.show_loss(path=log_path, title=title)

if __name__ == '__main__':
    train_process()