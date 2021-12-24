import pandas as pd

import data
from my_new_dynamic import train_model, picture


def load_all_data():
    file_list = ['小型', '中型', '大型']
    datas = []
    for file in file_list:
        datas.append(data.load_data('..\\' + file + '.xlsx'))
    return datas


def standard(datas):
    Xs = []
    Ys = []
    year= []
    for data in datas:
        dtr, dte = train_model.splite_train_test(data)
        year.append([dtr[['年份']], dte[['年份']], pd.concat([dtr[['年份']], dte[['年份']]], axis=0)])
        Xtrain, Ytrain = train_model.get_x_y(dtr)
        Xtest, Ytest = train_model.get_x_y(dte)
        # 对X进行标准化
        Xtrain = train_model.standard_data(Xtrain)
        Xtest = train_model.standard_data(Xtest)
        Xs.append([Xtrain, Xtest, pd.concat([Xtrain, Xtest])])
        Ys.append([Ytrain, Ytest, pd.concat([Ytrain, Ytest])])
    return Xs, Ys, year


def show_picture(Xs, Ys, year, models):

    for xs, ys, yr, model in zip(Xs, Ys, year, models):
        show_picture(yr[2]['年份'].values,
                     [ys[2], train_model.get_y_pred(model, xs[2])],
                     ['red', 'blue'],
                     ['真值', '预测值'])


def handle_small(Xs, Ys, year):
    Xtrain = Xs[0]
    Ytrain = Ys[0]
    model = train_model.build_model_small(Xtrain, Ytrain)
    print("测试集模型训练结果=======================")
    train_model.get_score(model, Xs[1], Ys[1])
    print("全数据集模型训练结果=======================")
    train_model.get_score(model, Xs[2], Ys[2])

def handle_medium(Xs, Ys, year):
    Xtrain = Xs[0]
    Ytrain = Ys[0]

    #model = train_model.grid_search_param(Xtrain, Ytrain)
    model = train_model.build_model_medium(Xtrain, Ytrain)
    print("测试集模型训练结果=======================")
    train_model.get_score(model, Xs[1], Ys[1])
    print("全数据集模型训练结果=======================")
    train_model.get_score(model, Xs[2], Ys[2])

    #train_model.grid_search_param(Xtrain, Ytrain)

def handle_large(Xs, Ys, year):
    Xtrain = Xs[0]
    Ytrain = Ys[0]

    model = train_model.grid_search_param(Xtrain, Ytrain)
    #model = train_model.build_model_large(Xtrain, Ytrain)
    print("测试集模型训练结果=======================")
    train_model.get_score(model, Xs[1], Ys[1])
    print("全数据集模型训练结果=======================")
    train_model.get_score(model, Xs[2], Ys[2])



if __name__ == "__main__":
    datas = load_all_data()
    Xs, Ys, year= standard(datas)
    # 小型油气田
    #handle_small(Xs[0], Ys[0], year[0])
    # 大型油田
    handle_large(Xs[2], Ys[2], year[2])
    '''
    yr = year[0][2]['年份'].values
    Ytest = Ys[0][2]
    Ypred = train_model.get_y_pred(models[0], Xs[0][2])





    #picture.show_linear_picture(yr, y_list=[Ytest, Ypred], color_list=['red', 'blue'],
    #                          label_list=['真值', '预测'])
    '''