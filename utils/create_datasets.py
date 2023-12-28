import os
import shutil

import constants


def create_datasets(src, train_dir, test_dir: str, train_size, test_size: int):
    for class_name in constants.class_list:
        dir_path = src + class_name
        file_list = os.listdir(dir_path)
        print(class_name + ':' + str(len(file_list)))

        if not os.path.exists(train_dir + class_name):
            os.makedirs(train_dir + class_name)
        for i in range(train_size):
            shutil.copy(dir_path + '/' + file_list[i], train_dir + class_name + '/' + str(i) + '.txt')

        if not os.path.exists(test_dir + class_name):
            os.makedirs(test_dir + class_name)
        for i in range(train_size, train_size + test_size):
            shutil.copy(dir_path + '/' + file_list[i], test_dir + class_name + '/' + str(i - train_size) + '.txt')


if __name__ == '__main__':
    create_datasets('D:/datasets/THUCNews/THUCNews/', '../my_data_train/', '../my_data_test/', 8000, 5000)
