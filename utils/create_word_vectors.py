import pickle
import numpy as np
from scipy.sparse import coo_matrix, save_npz

import constants


def create_word_vectors(dataset, output: str, dataset_size: int, _keywords: list):
    """
    生成词向量
    """
    max_idx = len(_keywords)
    print(max_idx)

    arr = np.zeros(shape=(dataset_size * len(constants.class_list), max_idx))
    index = 0

    for class_name in constants.class_list:
        print(class_name)
        arr, index = process_data(dataset + class_name + '/all.txt', _keywords, arr, index)

    res = coo_matrix(arr)
    save_npz(output, res)


def process_data(path, _keywords, arr, index):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        for word in line.split():
            if word not in _keywords:
                continue
            else:
                word_index = _keywords.index(word)
                arr[index][word_index] += 1
        index += 1
    return arr, index


if __name__ == '__main__':
    with open('../pkls/keywords.pkl', 'rb') as f:
        keywords_dic = pickle.load(f)
    keywords = list(keywords_dic.keys())

    create_word_vectors('../data_train/', '../coo_train.npz', 8000, keywords)
    create_word_vectors('../data_test/', '../coo_test.npz', 5000, keywords)
