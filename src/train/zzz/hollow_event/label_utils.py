import os
from typing import List

import pandas as pd

from train.devtools import os_utils


def read_label_csv() -> pd.DataFrame:
    return pd.read_csv(os.path.join(
        os_utils.get_path_under_work_dir('labels', 'zzz'),
        'hollow_events.csv'
    ))


def get_raw_dir() -> str:
    return os_utils.get_path_under_work_dir('label_studio', 'zzz', 'hollow_event', 'raw')


def create_label_studio_dirs() -> None:
    """
    按分类创建打标用的文件夹名称
    """
    raw_dir = get_raw_dir()

    label_df = read_label_csv()
    for index, row in label_df.iterrows():
        sub_dir_name = '%04d-%s' % (row['label'], row['entry_name'])
        sub_dir_path = os.path.join(raw_dir, sub_dir_name)
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)


def rename_file_in_raw() -> None:
    """
    对原图的文件夹进行重命名
    """
    raw_dir = get_raw_dir()
    for sub_dir_name in os.listdir(raw_dir):
        sub_dir = os.path.join(raw_dir, sub_dir_name)
        if not os.path.isdir(sub_dir):
            continue

        max_idx: int = 0
        for file_name in os.listdir(sub_dir):
            if not file_name.endswith('.png'):
                continue
            if file_name.startswith(sub_dir_name):
                idx = int(file_name[-8:-4])
                max_idx = max(idx, max_idx)
        max_idx += 1

        for file_name in os.listdir(sub_dir):
            if not file_name.endswith('.png'):
                continue

            if file_name.startswith(sub_dir_name):
                continue

            old_file_path = os.path.join(sub_dir, file_name)
            new_file_path = os.path.join(sub_dir, '%s-%04d.png' % (sub_dir_name, max_idx))
            os.rename(old_file_path, new_file_path)
            max_idx += 1


def get_labels() -> List[str]:
    label_df = read_label_csv()
    return ['%04d' % i for i in label_df['label'].tolist()]


def get_labels_with_name() -> List[str]:
    label_df = read_label_csv()
    result = []
    for index, row in label_df.iterrows():
        result.append('%04d-%s' % (row['label'], row['entry_name']))
    return result


def print_ls_labels():
    label_df = read_label_csv()
    for index, row in label_df.iterrows():
        print('<Label value="%04d-%s" />' % (row['label'], row['entry_name']))


if __name__ == '__main__':
    create_label_studio_dirs()
    rename_file_in_raw()