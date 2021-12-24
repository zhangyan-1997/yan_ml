import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def show_linear_picture(x, y_list, color_list, label_list, style_list, title):
    plt.title(title + '采气速度预测对比')
    plt.xlabel('年份')
    plt.ylabel('采气速度')
    for y, color, label, style in zip(y_list, color_list, label_list, style_list):
        plt.plot(x, y, color=color, label=label, linestyle=style)
    plt.legend(label_list)
    plt.show()
