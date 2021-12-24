import datetime

import pandas as pd


def extend_data():
    df = pd.read_excel('dataset\\大型扩展数据.xlsx')
    print(df)
    columns = ['产量', '累计产量', '可采储量', '采出程度', '采气速度']
    list_mean = []
    cols = df.shape[0]
    for index in range(0, cols-1):
        list = []
        for item in columns:
            list.append((df.iloc[index][item] + df.iloc[index+1][item])/2)
        list_mean.append(list)
    df_new = pd.DataFrame(data=list_mean, columns=columns)

    list_res = []
    for i in range(0, cols):
        list = []
        for item in columns:
            list.append(df.iloc[i][item])
        list_res.append(list)
        if i != cols - 1:
            list = []
            for item in columns:
                list.append(df_new.iloc[i][item])
            list_res.append(list)

    df_res = pd.DataFrame(data=list_res, columns=columns)
    print(df_res)
    df_res.to_excel('dataset\\大型扩展数据.xlsx', index=False)

def add_index():
    df = pd.read_excel('dataset\\小型_new.xlsx')
    df['年份'] = range(df.shape[0])
    df.to_excel('dataset\\小型_new.xlsx', index=False)

def load_data(filemname):
    df = pd.read_csv(filemname)
    return df


def load_new_data():
    data = pd.read_excel('小型临时.xlsx', header=None)
    df = pd.DataFrame()
    df['年份'] = range(1965, 2021, 1)
    df['产量'] = data.iloc[0, :]
    df['累计产量'] = data.iloc[1, :]
    df['可采储量'] = data.iloc[2, :]
    df['采出程度'] = data.iloc[4, :]
    df['采气速度'] = data.iloc[3, :]
    df.to_csv('小型.csv', index=False, encoding='utf_8_sig')


def data_clean_pollution(data):
    data['date'] = data.apply(lambda x: datetime.datetime(x["year"], x["month"], x["day"], x["hour"]), axis=1)
    data = data.set_index(['date'])
    data.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)
    data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    data.drop(['wnd_dir'], axis=1, inplace=True)
    data['pollution'].fillna(0, inplace=True)
    data = data[24:]
    return data


def data_clean_oil(data):
    data = data.set_index(['年份'])
    return data




if __name__ == '__main__':
    # load_new_data()
    #df = load_data()
    #data = data_clean(df)
    #print(data)


    for i in range(3):
        extend_data()

