import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

import constants

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def draw_pic(eval_table: DataFrame, pic_path: str = 'evaluation.png'):
    """
    对测评结果进行绘图
    """
    data = pd.DataFrame(np.zeros((len(constants.class_list), 3)), index=constants.class_list,
                        columns=eval_table.index.values)
    for class_name in constants.class_list:
        for metric in eval_table.index.values:
            data.at[class_name, metric] = eval_table.at[metric, class_name]

    data.plot(kind='bar', legend=True)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2)

    plt.xlabel('类别')
    plt.ylabel('各项评估得分(%)')
    plt.title('评估结果')

    plt.tight_layout()
    plt.savefig(pic_path)


def evaluate(csv_path: str, output_path='evaluation.csv') -> float:
    """
    根据混淆矩阵计算正确率, 召回率和F1值
    返回整体的F1值
    """
    df = pd.read_csv(csv_path, index_col=0).apply(
        pd.to_numeric, errors='coerce').astype(int)
    pd.set_option('display.max_columns', None)
    print("混淆矩阵: \n", df, "\n")

    eval_table = pd.DataFrame(np.zeros((3, len(constants.class_list) + 1)), columns=constants.class_list + ['整体'],
                              index=['正确率', '召回率', 'F1'])
    precision_all = 0
    recall_all = 0
    for class_name in constants.class_list:
        precision = df.at[class_name, class_name] / df[class_name].sum()
        precision_all += precision
        eval_table.at['正确率', class_name] = round(precision * 100, 2)

        recall = df.at[class_name, class_name] / df.loc[class_name, :].sum()
        recall_all += recall
        eval_table.at['召回率', class_name] = round(recall * 100, 2)

        f1 = 2 * (precision * recall) / (precision + recall)
        eval_table.at['F1', class_name] = round(f1 * 100, 2)

    precision_all = precision_all / len(constants.class_list)
    recall_all = recall_all / len(constants.class_list)
    f1_all = 2 * (precision_all * recall_all) / (precision_all + recall_all)

    eval_table.at['正确率', '整体'] = round(precision_all * 100, 2)
    eval_table.at['召回率', '整体'] = round(recall_all * 100, 2)
    eval_table.at['F1', '整体'] = round(f1_all * 100, 2)

    pd.set_option('display.float_format', '{:.2f}'.format)
    print("评估结果: \n", eval_table, "\n")
    eval_table.to_csv(output_path)

    draw_pic(eval_table)

    return f1_all


if __name__ == '__main__':
    evaluate('./confusion_matrix_svm.csv')
