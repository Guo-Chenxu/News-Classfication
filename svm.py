import pickle
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC

import evaluate
import constants


def train_svc(npz_path, output: str, train_size: int, c: float = 0.1, gamma: float = 0.1, max_iter: int = 1000000,
              kernel: str = 'linear'):
    """
    svm训练, 自定义核函数
    """

    coo_train = load_npz(npz_path)
    train_arr = np.array([int(i / train_size) for i in range(train_size * len(constants.class_list))])  # 样本分类

    if kernel == 'linear':
        print(f"kernel: linear, C: {c}, max_iter: {max_iter}")
        model = LinearSVC(C=c, max_iter=max_iter)
    elif kernel == 'rbf':
        print(f"kernel: rbf, C: {c}, gamma: {gamma}")
        model = SVC(kernel='rbf', C=c, gamma=gamma)
    else:
        raise Exception('kernel type error, only support linear and rbf')

    start = time.time()
    model.fit(coo_train.tocsr(), train_arr)
    end = time.time()
    print('Train time: %s Seconds\n' % (end - start))

    with open(output, 'wb') as f:
        pickle.dump(model, f)


def predict_svc(test_npz, model_path, confusion_matrix_path: str, test_size: int):
    """
    svm预测
    """
    coo_test = load_npz(test_npz)
    test_arr = np.array([int(i / test_size) for i in range(test_size * len(constants.class_list))])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    start = time.time()
    pre = model.predict(coo_test.tocsr())
    end = time.time()
    print('Test time: %s Seconds\n' % (end - start))

    c = metrics.confusion_matrix(test_arr, pre)
    confusion_matrix = pd.DataFrame(c, columns=constants.class_list,
                                    index=constants.class_list)
    confusion_matrix.to_csv(confusion_matrix_path)


if __name__ == '__main__':
    train_svc(c=0.1, max_iter=100000, npz_path='coo_train.npz', output='pkls/svm_train.pkl', train_size=8000,
              kernel='linear')
    predict_svc('coo_test.npz', 'pkls/svm_train.pkl', 'confusion_matrix_svm.csv', 5000)
    evaluate.evaluate('confusion_matrix_svm.csv')
