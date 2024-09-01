import os
import torch
import pandas as pd


# 写入数据
def write_to_file(path, file_name):
    print("===write_to_file===")
    try:
        os.makedirs(os.path.join('..', path), exist_ok=True)
        data_file = os.path.join('..', path, file_name)
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write('NumRooms,Alley,Price\n')  # 列名
            f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
            f.write('2,NA,106000\n')
            f.write('4,NA,178100\n')
            f.write('NA,NA,140000\n')
    except Exception as e:
        print(f"Error occurred: {e}")
    return


# 读取数据
def load_data(path, file_name):
    print("===load_data===")
    datas = pd.read_csv(os.path.join('..', path, file_name))
    print(datas)
    return datas


def fill_nan_with_mean(dataframe):
    """
    对dataframe中的NaN值填充平均值，并返回处理后的dataframe。

    :param dataframe: 输入的DataFrame
    :return: 处理后的DataFrame
    """
    print("===fill_nan_with_mean===")
    # 填充NaN值
    inputs, outputs = dataframe.iloc[:, 0:2], dataframe.iloc[:, 2]
    # 忽略数值型数据列
    inputs = inputs.select_dtypes(include=['number'])
    inputs = inputs.fillna(inputs.mean())
    print(inputs)
    print(outputs)


def nan_convert_column(dataframe):
    """
    将dataframe中的NaN值转换为列名，并返回处理后的dataframe。

    :param dataframe: 输入的DataFrame
    :return: 处理后的DataFrame
    """
    print("===nan_convert_column===")
    inputs = dataframe.iloc[:, 0:2]
    # 列 忽略空值 处理数值
    inputs = inputs.fillna(inputs.mean(axis=0, numeric_only=True))
    #  由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
    inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
    print(inputs)


def to_ndarrys(dataframe):
    """
    转为张量

    :param dataframe: 输入的DataFrame
    :return: 处理后的numpy数组
    """
    print("===to_ndarrys===")
    inputs, outputs = dataframe.iloc[:, 0:2], dataframe.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean(axis=0, numeric_only=True))
    inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
    print(inputs, outputs)
    print(inputs.to_numpy(dtype=float))
    x_ndarray = torch.tensor(inputs.to_numpy(dtype=float))
    y_ndarray = torch.tensor(outputs.to_numpy(dtype=float))
    print(x_ndarray, y_ndarray)


if __name__ == '__main__':
    read_data = load_data('data', 'house_tiny.csv')
    fill_nan_with_mean(read_data)
    nan_convert_column(read_data)
    to_ndarrys(read_data)
