import itertools

import pandas as pd
import matplotlib.pyplot as plt

import evaluate
import svm

# 定义要尝试的参数范围
C_values = [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]

best_score = 0
best_params = {}

scores = []

for C in C_values:
    # 训练模型, 迭代次数也不断增加保证每次都能收敛
    max_iter = 100000
    if C >= 1:
        max_iter *= C
    svm.train_svc(c=C, max_iter=max_iter, npz_path='coo_train.npz', output='pkls/svm_train.pkl', train_size=8000,
                  kernel='linear')

    # 预测并评估模型
    svm.predict_svc('coo_test.npz', 'pkls/svm_train.pkl', 'confusion_matrix_svm.csv', 5000)
    score = evaluate.evaluate('confusion_matrix_svm.csv')

    scores.append({'C': C, 'score': score})

    # 更新最佳得分和参数组合
    if score > best_score:
        best_score = score
        best_params = {'C': C}

print("Best Parameters:", best_params)
print("Best Score:", best_score)

scores = pd.DataFrame(scores)
scores.to_csv('linear-c-score.csv', index=False)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.plot(scores['C'], scores['score'], marker='x')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.title('线性核C和F1-Score关系折线图')

plt.savefig('linear-c-score.png')
