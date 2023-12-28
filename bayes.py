import pickle
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import evaluate
import constants


def train_bayes(_keywords: list, datasets, output: str):
    """
    朴素贝叶斯训练
    """
    max_idx = len(_keywords)
    word_df = pd.DataFrame(np.zeros((max_idx, len(constants.class_list))), columns=constants.class_list,
                           index=_keywords)
    bayes_df = pd.DataFrame(np.zeros((max_idx, len(constants.class_list))), columns=constants.class_list,
                            index=_keywords)

    # 构建关键词词频矩阵
    for _, class_name_1 in enumerate(constants.class_list):
        with open(datasets + class_name_1 + '/tf.pkl', 'rb') as file:
            tf_dict = pickle.load(file)
            for word in word_df.index:
                if tf_dict.get(word) is None:
                    continue
                else:
                    word_df.at[word, class_name_1] = tf_dict.get(word)

    df_sum = word_df.values.sum()

    # 构建条件概率矩阵
    for word in bayes_df.index:
        for class_name_1 in constants.class_list:
            bayes_df.at[word, class_name_1] = (word_df.at[word, class_name_1] + 1) / (
                    word_df[class_name_1].sum() + df_sum)

    with open(output, 'wb') as file:
        pickle.dump(bayes_df, file)


def _predict(text_pos: int, column, row, _keywords, bayes_df):
    pred = {}
    for class_name in constants.class_list:
        pred[class_name] = 1
        for v in column[row[text_pos]:row[text_pos + 1]]:
            w = _keywords[v]
            pred[class_name] *= bayes_df.at[w, class_name]

    res = sorted(pred.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    return res[0][0]


def predict_bayes(npz_path, confusion_matrix_path: str, test_size: int, _keywords: list, bayes_train_path: str):
    """
    朴素贝叶斯预测
    """
    test = load_npz(npz_path).tocsr()
    column = test.indices
    row = test.indptr

    with open(bayes_train_path, 'rb') as file:
        bayes_df = pickle.load(file)

    confusion_matrix = pd.DataFrame(np.zeros((len(constants.class_list), len(constants.class_list))),
                                    columns=constants.class_list, index=constants.class_list)
    class_index = 0
    for class_name in constants.class_list:
        for i in range(test_size):
            s = _predict(class_index * test_size + i, column, row, _keywords, bayes_df)
            confusion_matrix.at[class_name, s] += 1
        class_index += 1
    confusion_matrix.to_csv(confusion_matrix_path)


if __name__ == '__main__':
    with open('pkls/keywords.pkl', 'rb') as f:
        keywords_dict = pickle.load(f)
    keywords = list(keywords_dict.keys())

    start = time.time()
    train_bayes(keywords, 'pkls/', 'pkls/bayes_train.pkl')
    end = time.time()
    print('Train time: %s Seconds\n' % (end - start))

    start = time.time()
    predict_bayes('coo_test.npz', 'confusion_matrix_bayes.csv', 5000, keywords, 'pkls/bayes_train.pkl')
    end = time.time()
    print('Test time: %s Seconds\n' % (end - start))

    evaluate.evaluate('confusion_matrix_bayes.csv')
