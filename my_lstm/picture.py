import pickle

import matplotlib.pyplot as plt
import pandas as pd


def show_columns(data, title):
    values = data.values
    groups = [0, 1, 2, 3, 4]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 8))
    for group, i in zip(groups, range(len(groups))):
        plt.subplot(len(groups), 1, i+1)
        plt.plot(values[:, group])
        plt.title(data.columns[group], y=0.5, loc = 'right')
    plt.suptitle(title + '气藏原始数据可视化')
    plt.savefig('C:\\Users\\15534\\Desktop\\绘图结果\\动态图\\'+ title + '数据可视化.png')
    plt.show()


def show_loss(path, title):
    with open(path, 'rb') as file_text:
        history = pickle.load(file_text)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label = 'test')
    plt.title(title)
    plt.ylabel('Loss(mse)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('C:\\Users\\15534\\Desktop\\绘图结果\\' + title + '.png')
    plt.show()


def show_acc():
    with open('log.txt', 'rb') as file_text:
        history = pickle.load(file_text)
    plt.plot(history['acc'], label='train')
    plt.plot(history['val_acc'], label='test')
    plt.title('model accuracy')
    plt.xlabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

def show_linear_picture(x, y_list, color_list, label_list, style_list, title):
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.title(title + '采气速度预测对比')
    plt.xlabel('时间')
    plt.ylabel('采气速度')
    for y, color, label, style in zip(y_list, color_list, label_list, style_list):
        plt.plot(x, y, color=color, label=label)
    plt.legend(label_list)
    plt.savefig('C:\\Users\\15534\\Desktop\\绘图结果\\动态图\\'+ title + '.png')
    plt.show()

def show_diff_type(filename_true, filename_pred, title):
    data_true = pd.read_excel('dataset\\'+filename_true + '扩展数据.xlsx')
    data_pred = pd.read_excel('dataset\\predict_'+filename_pred + '.xlsx')
    y_true = data_true['采气速度']
    y_pred = data_pred['预测值']
    x = range(y_true.shape[0])
    y_list = [y_true, y_pred]
    color_list = ['blue', 'orange']
    label_list = ['真值', '预测值']
    style_list = ['--', '--']
    show_linear_picture(x, y_list, color_list, label_list, style_list, title)

def show_single():
    filename_trues = ['小型', '中型', '大型']
    filename_preds = ['small', 'medium', 'large']
    for filename_true, filename_pred in zip(filename_trues, filename_preds):
        title = filename_true + '气藏'
        show_diff_type(filename_true, filename_pred, title)


def show_origin_data_shape():
    filename_lists = ['小型', '中型', '大型']
    for filename in filename_lists:
        data = pd.read_excel('dataset\\'+filename + '_new.xlsx')
        show_columns(data, filename)

if __name__ == '__main__':
    show_origin_data_shape()