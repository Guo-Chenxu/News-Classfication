import os
import pickle
from collections import Counter

import jieba.analyse

import constants


def init_word_counters(path, output: str) -> (dict, dict):
    """
    初始化每个分类的词典和词频统计器
    """
    words = {}
    words_cnt = {}

    for class_name in constants.class_list:
        with open(path + class_name + '/all.txt', 'r', encoding='utf-8') as file:
            words[class_name] = ''.join(file.readlines())
            words_cnt[class_name] = Counter(words[class_name].split())

        if not os.path.exists(output + class_name):
            os.makedirs(output + class_name)

        with open(output + class_name + '/tf.pkl', 'wb') as file:
            pickle.dump(dict(words_cnt[class_name]), file)

    return words, words_cnt


def calculate_chi_square(data_path, word, class_str: str):
    """
    计算卡方统计量
    """
    a = 0
    b = 0
    c = 0
    d = 0

    for class_name in constants.class_list:
        with open(data_path + class_name + '/all.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()

            if class_name == class_str:
                for line in lines:
                    if word in line.split():
                        a += 1
                    else:
                        c += 1
            else:
                for line in lines:
                    if word in line.split():
                        b += 1
                    else:
                        d += 1

    if b == 0:
        return 1

    return (a + b + c + d) * ((a * d - b * c) ** 2) / ((a + c) * (b + d) * (a + b) * (c + d))


def extract_keywords(data_path, output_path: str):
    """
    提取关键词并计算权重
    """
    words, words_cnt = init_word_counters(data_path, output_path)
    all_keywords = []

    for class_name in constants.class_list:
        # 统计词频大于30的词, 并计算tf-idf值
        top_words = words_cnt[class_name].most_common()
        top_k = next((i for i, (word, freq) in enumerate(top_words) if freq < 30), 2000)
        tags = jieba.analyse.extract_tags(words[class_name], topK=top_k, withWeight=True)
        tags_dict = dict(tags)
        index = 0

        # 计算卡方, 并将每个词的tf-idf值乘以卡方值得到新的权重
        for tag_word, value in tags_dict.items():
            index += 1
            tags_dict[tag_word] = value * calculate_chi_square(data_path, tag_word, class_name)

        # 选取前1000的词作为关键词
        all_keywords += [k for k, v in Counter(tags_dict).most_common(1000)]

    all_dict = Counter(all_keywords)

    with open(output_path + 'keywords.pkl', 'wb') as f:
        pickle.dump(all_dict, f)


if __name__ == '__main__':
    extract_keywords('../data_train/', '../pkls/')
