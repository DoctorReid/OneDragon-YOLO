import os
import re
import shutil
from typing import Optional, List, Set

import cv2
import numpy as np
import pandas as pd
from ultralytics.data.utils import autosplit

from train.sr.detector import label_utils
from train.devtools import ultralytics_utils, label_studio_utils

_BASE_DETECT = 'base-detect'


def get_labels_dir(dataset_name: str, bk: bool = False) -> str:
    """
    获取数据集中 标签的文件夹路径
    :param dataset_name: 数据集名称
    :param bk: 是否原标签备份
    """
    dir_path = os.path.join(ultralytics_utils.get_dataset_dir(dataset_name), 'labels_bk' if bk else 'labels')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def get_dataset_images_dir(dataset_name: str) -> str:
    dir_path = os.path.join(ultralytics_utils.get_dataset_dir(dataset_name), 'images')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return dir_path


def read_label_txt(txt_path) -> pd.DataFrame:
    """
    读取一个标签文件
    """
    return pd.read_csv(txt_path, sep=' ', header=None, encoding='utf-8', names=['idx', 'x', 'y', 'w', 'h'])


def clear_dataset_labels(dataset_name: str):
    """
    清除一个dataset的全部标签文件
    """
    labels_dir = get_labels_dir(dataset_name)
    shutil.rmtree(labels_dir)


def init_labels_bk(dataset_name: str,
                   objects_to_use: Optional[List[str]] = None,
                   labels_version: str = 'v1'
                   ) -> List[str]:
    """
    初始化一个数据集的原始标签
    从base-detect数据集中复制过来
    并按要求 仅保留需要用来训练的
    @param dataset_name: 数据集id
    @param objects_to_use: 限定使用的内容
    @param labels_version: 标签版本
    @return 返回过滤后的标签
    """
    source_labels_dir = get_labels_dir(_BASE_DETECT)
    target_labels_dir = get_labels_dir(dataset_name, bk=True)

    # 重新创建标签
    if os.path.exists(target_labels_dir):
        shutil.rmtree(target_labels_dir)
    os.mkdir(target_labels_dir)

    labels_df = label_utils.read_label_csv()

    # 过滤要用来训练的标签
    labels_to_use_real = []

    for index, row in labels_df.iterrows():
        if objects_to_use is not None and row['name'] not in objects_to_use:
            continue
        if row[labels_version] in labels_to_use_real:
            continue
        labels_to_use_real.append(row[labels_version])

    # 映射到新标签的id
    id_old_2_new = {}
    label_2_idx_new = {}

    # 原标签的id
    label_2_idx_old = {}
    for label in labels_df[labels_version]:
        label_2_idx_old[label] = int(label[:4]) - 1

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


def init_dataset_images(dataset_name: str):
    """
    按照使用的标签 复制原图到数据集中
    未做数据增强
    """
    labels_dir = get_labels_dir(dataset_name)
    label_case_ids = set()  # 有标签的样例
    if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
        for label_txt in os.listdir(labels_dir):
            if not label_txt.endswith('.txt'):
                continue
            label_case_ids.add(label_txt[:-4])

    images_dir = get_dataset_images_dir(dataset_name)
    image_case_ids = set()  # 已经复制了图片的样例
    for img in os.listdir(images_dir):
        if not img.endswith('.png'):
            continue
        image_case_ids.add(img[:-4])

    # 完全删除后重新复制过去
    shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    raw_img_dir = label_studio_utils.get_raw_images_dir()
    for prefix in os.listdir(raw_img_dir):
        sub_img_dir = os.path.join(raw_img_dir, prefix)
        if not os.path.isdir(sub_img_dir):
            continue

        for img_name in os.listdir(sub_img_dir):
            if not img_name.endswith('.png'):
                continue
            case_id = img_name[:-4]

            if case_id not in label_case_ids:  # 没有包含标签
                pass

            if case_id in image_case_ids:  # 已经复制过图片了
                pass

            old_path = os.path.join(sub_img_dir, img_name)
            new_path = os.path.join(images_dir, img_name)
            shutil.copyfile(old_path, new_path)


def init_dataset_images_and_labels(dataset_name: str, img_size: int = 2176) -> bool:
    """
    初始化一个数据集的图片和标签
    原图是 1920*1080 会使用两张图片合并成 
    - 2176*2176 (2176=32*68)
    - 2208*2208 (2208=32*69)
    同时将对应标签合并
    需要先使用 init_labels_bk 初始化原标签文件夹 labels_bk
    :param dataset_name: 数据集名称
    :param img_size: 两张图片合并后的图片大小 需要>=1080*2=2160. 需要是32的倍数
    :return:
    """
    if (img_size < 1080 * 2) or (img_size % 32 != 0):
        return False
    base_img_dir = get_dataset_images_dir(_BASE_DETECT)

    labels_bk_dir = get_labels_dir(dataset_name, bk=True)
    labels_dir = get_labels_dir(dataset_name, bk=False)
    images_dir = get_dataset_images_dir(dataset_name)

    label_txt_names = os.listdir(labels_bk_dir)
    total_cnt = len(label_txt_names)
    idx = 0

    while idx < total_cnt:
        case1 = label_txt_names[idx][:-4]

        img1_path = os.path.join(base_img_dir, '%s.png' % case1)
        img1 = cv2.imread(img1_path)
        label1_path = os.path.join(labels_bk_dir, label_txt_names[idx])
        label1_df = read_label_txt(label1_path)

        if idx + 1 < total_cnt:  # 跟下一张合并
            case2 = label_txt_names[idx + 1][:-4]
            img2_path = os.path.join(base_img_dir, '%s.png' % case2)
            img2 = cv2.imread(img2_path)
            label2_path = os.path.join(labels_bk_dir, label_txt_names[idx + 1])
            label2_df = read_label_txt(label2_path)
        elif idx > 0:  # 跟上一张合并 只有总数=奇数的最后一张会触发
            case2 = label_txt_names[idx - 1][:-4]
            img2_path = os.path.join(base_img_dir, '%s.png' % case2)
            img2 = cv2.imread(img2_path)
            label2_path = os.path.join(labels_bk_dir, label_txt_names[idx - 1])
            label2_df = read_label_txt(label2_path)
        else:  # 跟自己合并 只有总数=1会触发
            case2 = case1
            img2 = img1.copy()
            label2_df = label1_df.copy()

        idx += 2

        height = img1.shape[0]
        width = img1.shape[1]
        radius = img_size

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

        save_img_path = os.path.join(images_dir, '%s-%s.png' % (case1, case2))
        cv2.imwrite(save_img_path, save_img)

        save_label_path = os.path.join(labels_dir, '%s-%s.txt' % (case1, case2))
        save_label_df.to_csv(save_label_path, sep=' ', index=False, header=False)

    return True


def prepare_dateset(dataset_name: str,
                    split_weights=(0.9, 0.1, 0),
                    objects_to_use: Optional[List[str]] = None,
                    labels_version: str = 'v1'):
    """
    从基础数据集中 生成一个子数据集

    1. 保留目标标签
    2. 过滤没有标签的图片
    2. 数据增强 未加入
    3. 划分数据集
    4. 写入 dataset.yaml 同时剔除标签的中文

    :param dataset_name: 子数据集名称
    :param split_weights: 自动划分数据集的比例
    :param objects_to_use: 限定使用的内容
    :param labels_version: 标签版本
    """
    target_dataset_dir = ultralytics_utils.get_dataset_dir(dataset_name)
    if os.path.exists(target_dataset_dir):
        shutil.rmtree(target_dataset_dir)
    os.mkdir(target_dataset_dir)

    # 初始化标签
    labels_to_use_real = init_labels_bk(dataset_name, objects_to_use=objects_to_use, labels_version=labels_version)

    # 初始化图片
    init_dataset_images_and_labels(dataset_name)

    # 划分数据集
    autosplit(path=os.path.join(target_dataset_dir, 'images'), weights=split_weights, annotated_only=True)

    # 保存dataset.yaml
    with open(os.path.join(target_dataset_dir, 'dataset.yaml'), 'w', encoding='utf-8') as file:
        file.write('path: %s\n' % dataset_name)
        file.write('train: autosplit_train.txt\n')
        file.write('val: autosplit_val.txt\n')
        file.write('test: autosplit_test.txt\n')
        file.write('names:\n')
        label_idx = 0
        for label in labels_to_use_real:
            file.write('  %d: %s\n' % (label_idx, re.sub(r'[^a-zA-Z0-9-]', '', label).replace('--', '-')))
            label_idx += 1


def count_labels(dataset_name: str, label_version: str = 'v1') -> dict[str, int]:
    """
    统计数据集中各标签出现的次数
    """
    labels_df = label_utils.read_label_csv()
    idx_2_label = {}
    for idx, row in labels_df.iterrows():
        idx_2_label[int(row[label_version][:4]) - 1] = row[label_version]

    labels_dir = get_labels_dir(dataset_name)
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


def clear_raw_images_without_label(dataset_name: str, confirm_delete: bool = False) -> List[str]:
    """
    清理没有标签的原图 包含 label-studio/raw_images目录 和 ultralytics/datasets/{dataset_name}/images目录
    @param dataset_name: 使用的数据集标签
    @param confirm_delete: 确认删除
    @return 没有标签的原图路径
    """
    case_id_with_labels: Set[str] = set()  # 标签

    # 保留特定标签
    labels_dir = get_labels_dir(dataset_name)

    for label_txt in os.listdir(labels_dir):
        if not label_txt.endswith('.txt'):
            continue
        case_id_with_labels.add(label_txt[:-4])

    to_delete_paths: List[str] = []
    raw_img_dir = label_studio_utils.get_raw_images_dir()
    for cate_img_dir in os.listdir(raw_img_dir):
        cate_img_path = os.path.join(raw_img_dir, cate_img_dir)
        if not os.path.isdir(cate_img_path):
            continue

        for img in os.listdir(cate_img_path):
            if not img.endswith('.png'):
                continue
            case_id = img[:-4]
            if case_id in case_id_with_labels:
                continue

            to_delete_paths.append(os.path.join(cate_img_path, img))

    dataset_img_dir = get_dataset_images_dir(dataset_name)
    for img in os.listdir(dataset_img_dir):
        if not img.endswith('.png'):
            continue
        case_id = img[:-4]
        if case_id in case_id_with_labels:
            continue

        to_delete_paths.append(os.path.join(dataset_img_dir, img))

    if confirm_delete:
        for to_delete in to_delete_paths:
            os.remove(to_delete)

    return to_delete_paths


def check_no_self_label_cases(dataset_name: str = 'base-detect') -> List[str]:
    """
    检查并返回图片中没有自身标签的样例 大概率是标注错了
    """
    labels_dir = get_labels_dir(dataset_name)

    without_self_label_list = []
    for label_txt_name in os.listdir(labels_dir):
        if not label_txt_name.endswith('.txt'):
            continue
        idx = int(label_txt_name[:4])
        txt_df = read_label_txt(os.path.join(labels_dir, label_txt_name))
        if (idx - 1) not in txt_df['idx'].unique():
            without_self_label_list.append(label_txt_name)

    return without_self_label_list
