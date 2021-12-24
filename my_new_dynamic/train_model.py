import numpy as np
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def standard_data(data):
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    return data


def splite_train_test(data):
    data_train = data.loc[((data['年份'] > 1995) & (data['年份'] < 2016))]
    data_test = data.loc[((data['年份'] > 2015) & (data['年份'] < 2021))]
    return data_train, data_test


def get_x_y(data):
    x = data[['产量', '累计产量', '可采储量', '采出程度']]
    y = data[['采气速度']]
    return x, y


def build_model_small(Xtrain, Ytrain):
    svr = SVR(kernel='poly', gamma='auto', cache_size=5000)
    svr = svr.fit(Xtrain, Ytrain)
    return svr


def build_model_medium(Xtrain, Ytrain):
    #svr = SVR(kernel='sigmoid', gamma= 0.01, C= 100,cache_size=5000)
    #svr = ensemble.AdaBoostRegressor(n_estimators=50)
    svr = MLPRegressor()
    svr = svr.fit(Xtrain, Ytrain)
    return svr


def build_model_large(Xtrain, Ytrain):
    #svr = SVR(kernel='sigmoid', gamma= 0.01, C= 100,cache_size=5000)
    svr = SVR()
    svr = svr.fit(Xtrain, Ytrain)
    return svr


def get_score(model, X, Y):
    print("模型得分--------")
    print(model.score(X, Y))
    Ypred = get_y_pred(model, X)
    score1 = mean_absolute_error(Y, Ypred)
    score2 = r2_score(Y, Ypred)
    score3 = explained_variance_score(Y, Ypred)
    print("绝对值误差: " + str(score1))
    print("r2得分： " + str(score2))
    print("解释偏差得分 " + str(score3))
    return score1, score2, score3


def get_y_pred(model, X):
    Ypred = model.predict(X)
    return Ypred


def grid_search_param(Xtrain, Ytrain):
    parameters = [{'kernel': ['sigmoid', 'rbf', 'poly', 'linear'],
                   'gamma': np.logspace(-2, 2, 5),
                   'C': [1e0, 1e1, 1e2, 1e3]
                   }]
    model = GridSearchCV(SVR(), param_grid=parameters, scoring='r2', cv=5)
    model.fit(Xtrain, Ytrain)
    print("最佳训练参数: " + str(model.best_params_))
    print("最优得分： " + str(model.best_score_))
    print("最优模型 " + str(model.best_estimator_))
    return model.best_estimator_
