import os

import jieba
import jieba.posseg as ps
import paddle

import constants

with open('../stop_words_ch.txt', 'r') as file:
    stop_words = [i.strip() for i in file.readlines()]
with open('../stop_sign.txt', 'r', encoding='utf-8') as file:
    stop_sign = [i.strip() for i in file.readlines()]


def is_stop_word(word: str) -> bool:
    """
    判断是否为停用词
    """
    if word in stop_words:
        return True
    for i in word:
        if i in stop_sign:
            return True
    return False


def del_stop_words(src, out: str):
    """
    删除停用词
    并将同一类的数据整合到一个文件中, 方便后续处理

    src: 待处理文件路径
    out: 处理后文件路径
    """
    # 整个数据集分词后整合的结果
    _all = ''
    for class_name in constants.class_list:
        print(f'正在处理 {class_name} 类')

        if not os.path.exists(src + class_name):
            raise Exception(f'{src + class_name}  文件夹不存在')

        if not os.path.exists(out + class_name):
            os.makedirs(out + class_name)

        # 单个种类分词并合成整合
        all_single = ''
        for _file in os.listdir(src + class_name):
            single = ''
            with open(src + class_name + '/' + _file, 'r', encoding='utf-8') as f:
                lines = f.read()
                words = ps.cut(lines, use_paddle=True)
                for word, flag in words:
                    # 若为名词且不在停用词表中，则加入写入串
                    if flag == 'n' and not is_stop_word(word):
                        single += (word + ' ')
                all_single += single + '\n'

        with open(out + class_name + '/all.txt', 'w', encoding='utf-8') as f:
            f.write(all_single)

        _all += all_single

    with open(out + '/all.txt', 'w', encoding='utf-8') as f:
        f.write(_all)


if __name__ == '__main__':
    paddle.enable_static()
    jieba.enable_paddle()
    del_stop_words('../my_data_train/', '../data_train/')
    del_stop_words('../my_data_test/', '../data_test/')
