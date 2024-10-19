import os
import random
import shutil
from typing import Optional, List

import cv2
import numpy as np
import pandas as pd
from ultralytics.data.utils import autosplit

from train.devtools import ultralytics_utils, label_studio_utils, os_utils


def get_raw_labels(raw_dataset_name: str) -> List[str]:
    raw_dir = ultralytics_utils.get_dataset_dir(raw_dataset_name)
    txt_path = os.path.join(raw_dir, 'classes.txt')
    df = pd.read_csv(txt_path, sep=' ', header=None, encoding='utf-8', names=['label'])
    return df['label'].tolist()


def init_labels_bk(
        dataset_name: str,
        raw_dataset_name: str,
        usage_labels: Optional[List[str]] = None
) -> List[str]:
    """
    初始化一个数据集的原始标签 到labels_bk文件夹
    并按要求 仅保留需要用来训练的
    @param dataset_name: 数据集id
    @param raw_dataset_name: 原始数据集id
    @param usage_labels: 需要使用的标签
    @return 返回过滤后的标签
    """
    source_labels_dir = ultralytics_utils.get_dataset_labels_dir(raw_dataset_name)
    target_labels_dir = ultralytics_utils.get_dataset_labels_bk_dir(dataset_name)

    # 重新创建标签
    if os.path.exists(target_labels_dir):
        shutil.rmtree(target_labels_dir)
    os.mkdir(target_labels_dir)

    # 过滤要用来训练的标签
    raw_labels = get_raw_labels(raw_dataset_name)
    labels_to_use_real = raw_labels if usage_labels is None else usage_labels

    # 映射到新标签的id
    id_old_2_new = {}
    label_2_idx_new = {}

    # 原标签的id
    label_2_idx_old = {}
    for idx in range(len(raw_labels)):
        label_2_idx_old[raw_labels[idx]] = idx

    id_new = 0
    for label in labels_to_use_real:
        id_old = label_2_idx_old[label]
        id_old_2_new[id_old] = id_new
        label_2_idx_new[label] = id_new
        id_new += 1

    for label_txt in os.listdir(source_labels_dir):
        if not label_txt.endswith('.txt'):
            continue
        label_txt_path = os.path.join(source_labels_dir, label_txt)
        df = read_label_txt(label_txt_path)

        # 转化成新的下标
        df['idx'] = df['idx'].map(id_old_2_new)
        df.dropna(inplace=True)
        df['idx'] = df['idx'].astype(int)

        if len(df) > 0:  # 过滤之后 还有标签的才保存
            new_label_txt_path = os.path.join(target_labels_dir, label_txt)
            df.to_csv(new_label_txt_path, sep=' ', index=False, header=False)

    return labels_to_use_real


def init_dataset_images_and_labels(
        dataset_name: str,
        raw_images_dir_path: str,
        target_img_size: int = 2176
) -> bool:
    """
    初始化一个数据集的图片和标签
    原图是 1920*1080 会使用两张图片合并成
    - 2176*2176 (2176=32*68)
    - 2208*2208 (2208=32*69)
    同时将对应标签合并
    需要先使用 init_labels_bk 初始化原标签文件夹 labels_bk
    :param dataset_name: 数据集名称
    :param target_img_size: 两张图片合并后的图片大小 需要>=1080*2=2160. 需要是32的倍数
    :return:
    """
    if (target_img_size < 1080 * 2) or (target_img_size % 32 != 0):
        print('传入的图片大小不合法')
        return False

    name_2_img_path = label_studio_utils.get_img_name_2_path(raw_images_dir_path)

    target_label_bk_dir = ultralytics_utils.get_dataset_labels_bk_dir(dataset_name)
    target_label_bk_list = os.listdir(target_label_bk_dir)

    target_dataset_dir = ultralytics_utils.get_dataset_dir(dataset_name)
    if not os.path.exists(target_dataset_dir):
        os.mkdir(target_dataset_dir)

    target_img_dir = ultralytics_utils.get_dataset_images_dir(dataset_name)
    if os.path.exists(target_img_dir):
        shutil.rmtree(target_img_dir)
    os.mkdir(target_img_dir)

    target_label_dir = ultralytics_utils.get_dataset_labels_dir(dataset_name)
    if os.path.exists(target_label_dir):
        shutil.rmtree(target_label_dir)
    os.mkdir(target_label_dir)

    total_cnt = len(target_label_bk_list)

    for case1_idx in range(total_cnt):
        case2_idx = random.randint(0, total_cnt-1)
        
        case1 = target_label_bk_list[case1_idx][:-4]
        case2 = target_label_bk_list[case2_idx][:-4]

        img1_path = name_2_img_path[case1]
        img1 = cv2.imread(img1_path)
        label1_path = os.path.join(target_label_bk_dir, target_label_bk_list[case1_idx])
        label1_df = read_label_txt(label1_path)

        img2_path = name_2_img_path[case2]
        img2 = cv2.imread(img2_path)
        label2_path = os.path.join(target_label_bk_dir, target_label_bk_list[case2_idx])
        label2_df = read_label_txt(label2_path)

        height = img1.shape[0]
        width = img1.shape[1]
        radius = target_img_size

        save_img = np.full((radius, radius, 3), 114, dtype=np.uint8)
        save_img[0:height, 0:width, :] = img1
        save_img[height:height+height, 0:width, :] = img2

        label1_df['x'] *= width
        label1_df['y'] *= height
        label1_df['w'] *= width
        label1_df['h'] *= height

        label2_df['x'] *= width
        label2_df['y'] *= height
        label2_df['y'] += height
        label2_df['w'] *= width
        label2_df['h'] *= height

        save_label_df = pd.concat([label1_df, label2_df])
        save_label_df['x'] /= radius
        save_label_df['y'] /= radius
        save_label_df['w'] /= radius
        save_label_df['h'] /= radius

        save_img_path = os.path.join(target_img_dir, '%s-%s.png' % (case1, case2))
        cv2.imwrite(save_img_path, save_img)

        save_label_path = os.path.join(target_label_dir, '%s-%s.txt' % (case1, case2))
        save_label_df.to_csv(save_label_path, sep=' ', index=False, header=False)

    return True


def read_label_txt(txt_path) -> pd.DataFrame:
    """
    读取一个标签文件
    """
    return pd.read_csv(txt_path, sep=' ', header=None, encoding='utf-8', names=['idx', 'x', 'y', 'w', 'h'])


def init_dataset(
        dataset_name: str,
        raw_dataset_name: str,
        raw_images_dir_path: str,
        target_img_size: int = 2176,
        split_weights=(0.9, 0.1, 0),
        usage_labels: List[str] = None
):
    target_dataset_dir = ultralytics_utils.get_dataset_dir(dataset_name)
    if os.path.exists(target_dataset_dir):
        shutil.rmtree(target_dataset_dir)
    os.mkdir(target_dataset_dir)

    labels_to_use_real = init_labels_bk(
        dataset_name=dataset_name,
        raw_dataset_name=raw_dataset_name,
        usage_labels=usage_labels
    )

    init_dataset_images_and_labels(
        dataset_name=dataset_name,
        raw_images_dir_path=raw_images_dir_path,
        target_img_size=target_img_size
    )

    # 划分数据集
    autosplit(path=ultralytics_utils.get_dataset_images_dir(dataset_name), weights=split_weights, annotated_only=True)
    if split_weights[1] == 0:
        train_txt_path = os.path.join(target_dataset_dir, 'autosplit_train.txt')
        val_txt_path = os.path.join(target_dataset_dir, 'autosplit_val.txt')
        shutil.copy(train_txt_path, val_txt_path)

    # 保存dataset.yaml
    with open(os.path.join(target_dataset_dir, 'dataset.yaml'), 'w', encoding='utf-8') as file:
        file.write('path: %s\n' % dataset_name)
        file.write('train: autosplit_train.txt\n')
        file.write('val: autosplit_val.txt\n')
        file.write('test: autosplit_test.txt\n')
        file.write('names:\n')
        label_idx = 0
        for label in labels_to_use_real:
            file.write('  %d: %s\n' % (label_idx, label))
            label_idx += 1


def count_labels(raw_dataset_name: str) -> dict[str, int]:
    """
    统计数据集中各标签出现的次数
    """
    labels = get_raw_labels(raw_dataset_name)
    idx_2_label = {}
    for idx, label in enumerate(labels):
        idx_2_label[idx] = label

    labels_dir = ultralytics_utils.get_dataset_labels_dir(raw_dataset_name)
    labels_count: dict[str, int] = {}
    for label_txt in os.listdir(labels_dir):
        if not label_txt.endswith('.txt'):
            continue
        txt_df = read_label_txt(os.path.join(labels_dir, label_txt))
        for idx, row in txt_df.iterrows():
            label = idx_2_label[int(row['idx'])]
            if label not in labels_count:
                labels_count[label] = 1
            else:
                labels_count[label] = labels_count[label] + 1

    return labels_count


if __name__ == '__main__':
    # init_dataset(
    #     dataset_name='zzz_hollow_event_2208',
    #     raw_dataset_name='zzz_hollow_event_raw',
    #     raw_images_dir_path=os_utils.get_path_under_work_dir('label_studio', 'zzz', 'hollow_event', 'raw'),
    #     target_img_size=2208,
    #     split_weights=(1, 0, 0),
    #     usage_labels=['临时拍档-0014']
    # )
    print(count_labels('zzz_hollow_event_raw'))
