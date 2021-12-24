import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def get_score(Ytrue, Ypred):
    score1 = mean_absolute_error(Ytrue, Ypred)
    score2 = mean_squared_error(Ytrue, Ypred)
    score3 = r2_score(Ytrue, Ypred)
    print("平均绝对误差: " + str(score1))
    print("均方误差： " + str(score2))
    print("R方 " + str(score3))
    return score1, score2, score3


def show_r2_score(list_r2, title, path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    X = np.linspace(1, 3, 3)
    plt.bar(X, list_r2, 0.5, tick_label=['小型气藏', '中型气藏', '大型气藏'])
    plt.xlabel('气藏类型')
    plt.ylabel('R^2')
    plt.title(title)
    for a, b in zip(X, list_r2):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom')

    plt.savefig(path)
    plt.show()

def print_scores():
    filename_trues = ['小型', '中型', '大型']
    filename_preds = ['small', 'medium', 'large']
    list_r2 = []
    for filename_true, filename_pred in zip(filename_trues, filename_preds):
        data_true = pd.read_excel('dataset\\' + filename_true + '扩展数据.xlsx')
        data_pred = pd.read_excel('dataset\\predict_' + filename_pred + '.xlsx')
        y_true = data_true['采气速度']
        y_pred = data_pred['预测值']
        s1, s2, s3 = get_score(y_true, y_pred)
        list_r2.append(s3)
    show_r2_score(list_r2, '不同类型训练R^2', path='C:\\Users\\15534\\Desktop\\绘图结果\\动态图\\'+ 'R2图.png')



if __name__ == '__main__':
    print_scores()