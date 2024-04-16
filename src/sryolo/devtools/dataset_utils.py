import os
import re
import shutil
from typing import Optional, List, Set

import pandas as pd
from ultralytics.data.utils import autosplit

from sryolo.devtools import label_utils
from sryolo.utils import os_utils


def get_datasets_dir() -> str:
    """
    数据集的根目录
    """
    return os.path.join(os_utils.get_work_dir(), 'datasets')


def get_dataset_dir(dataset_name: str) -> str:
    """
    获取特定数据集的目录
    """
    datasets_dir = get_datasets_dir()
    return os.path.join(datasets_dir, dataset_name)


def get_raw_images_dir() -> str:
    """
    原图的根目录
    """
    return os.path.join(os_utils.get_work_dir(), '.env', 'data', 'raw_images')


def get_labels_dir(dataset_name: str) -> Optional[str]:
    """
    获取数据集中 标签的文件夹路径
    """
    dataset_dir = get_dataset_dir(dataset_name)
    labels_bk_dir = os.path.join(dataset_dir, 'labels_bk')
    if os.path.exists(labels_bk_dir) and os.path.isdir(labels_bk_dir):
        return labels_bk_dir
    labels_dir = os.path.join(dataset_dir, 'labels')
    if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
        return labels_dir
    return None


def get_dataset_images_dir(dataset_name: str) -> Optional[str]:
    return os.path.join(get_dataset_dir(dataset_name), 'images')


def read_label_txt(txt_path) -> pd.DataFrame:
    """
    读取一个标签文件
    """
    return pd.read_csv(txt_path, sep=' ', header=None, encoding='utf-8', names=['idx', 'x', 'y', 'w', 'h'])


def reorganize_labels(dataset_name: str,
                      labels_to_use: Optional[List[str]] = None,
                      cate_to_use: Optional[List[str]] = None,
                      labels_version: str = 'v1'
                      ) -> List[str]:
    """
    重新整理标签 仅保留需要用来训练的
    @param dataset_name: 数据集id
    @param labels_to_use: 限定使用的标签
    @param cate_to_use: 限定使用的类别
    @param labels_version: 标签版本
    @return 返回过滤后的标签
    """
    target_dataset_dir = get_dataset_dir(dataset_name)

    # 保留特定标签
    labels_dir = os.path.join(target_dataset_dir, 'labels')
    labels_bk_dir = os.path.join(target_dataset_dir, 'labels_bk')

    # 备份原有标签
    if not os.path.exists(labels_bk_dir):
        os.rename(labels_dir, labels_bk_dir)

    # 重新创建标签
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    os.mkdir(labels_dir)

    labels_df = label_utils.read_label_csv()

    # 过滤要用来训练的标签
    labels_to_use_real = []

    for index, row in labels_df.iterrows():
        if labels_to_use is not None and row[labels_version] not in labels_to_use:
            continue
        if cate_to_use is not None and row['cate'] not in cate_to_use:
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

    for label_txt in os.listdir(labels_bk_dir):
        if not label_txt.endswith('.txt'):
            continue
        label_txt_path = os.path.join(labels_bk_dir, label_txt)
        df = pd.read_csv(label_txt_path, sep='\s+', header=None, names=['id', 'x', 'y', 'w', 'h'])

        df['id'] = df['id'].map(id_old_2_new)
        df.dropna(inplace=True)
        df['id'] = df['id'].astype(int)

        if len(df) > 0:  # 过滤之后 还有标签的才保存
            new_label_txt_path = os.path.join(labels_dir, label_txt)
            df.to_csv(new_label_txt_path, sep=' ', index=False, header=False)

    return labels_to_use_real


def reorganize_images(dataset_name: str):
    """
    将没有标签的图片移到备份的图片文件夹中
    """
    target_dataset_dir = get_dataset_dir(dataset_name)

    # 有标签的样例
    label_case_ids: Set[str] = set()
    labels_dir = os.path.join(target_dataset_dir, 'labels')
    for label_txt in os.listdir(labels_dir):
        if not label_txt.endswith('.txt'):
            continue
        label_case_ids.add(labels_dir[:-4])

    image_usage_dir = os.path.join(target_dataset_dir, 'images')  # 用来训练的图片文件夹
    image_bk_dir = os.path.join(target_dataset_dir, 'images_bk')  # 备份的图片文件夹

    for img in os.listdir(image_usage_dir):
        if not img.endswith('.png'):
            continue
        case_id = img[:-4]
        if case_id in label_case_ids:
            continue

        # 没有标签的图片 移到备份文件夹中
        old_img_path = os.path.join(image_usage_dir, '%s.png' % case_id)
        new_img_path = os.path.join(image_bk_dir, '%s.png' % case_id)
        shutil.move(old_img_path, new_img_path)

    for img in os.listdir(image_bk_dir):
        if not img.endswith('.png'):
            continue
        case_id = img[:-4]
        if case_id not in label_case_ids:
            continue

        # 有标签的图片 移到使用文件夹中
        old_img_path = os.path.join(image_bk_dir, '%s.png' % case_id)
        new_img_path = os.path.join(image_usage_dir, '%s.png' % case_id)
        shutil.move(old_img_path, new_img_path)


def sync_raw_images_to_dataset(dataset_name: str):
    """
    按照使用的标签 复制原图到数据集中
    """
    labels_dir = get_labels_dir(dataset_name)
    label_case_ids = set()  # 有标签的样例
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

    raw_img_dir = get_raw_images_dir()
    for prefix in os.listdir(raw_img_dir):
        sub_img_dir = os.path.join(raw_img_dir, prefix)
        if not os.path.isdir(sub_img_dir):
            continue

        for img_name in os.listdir(sub_img_dir):
            if not img_name.endswith('.png'):
                continue
            case_id = img_name[:-4]

            if case_id in image_case_ids:  # 已经复制过图片了
                pass

            old_path = os.path.join(sub_img_dir, img_name)
            new_path = os.path.join(images_dir, img_name)
            shutil.copyfile(old_path, new_path)


def prepare_dateset(dataset_name: str,
                    split_weights=(0.7, 0.2, 0.1),
                    labels_to_use: Optional[List[str]] = None,
                    cate_to_use: Optional[List[str]] = None,
                    labels_version: str = 'v1'):
    """
    从label-studio导出后 在训练前进一步处理数据集

    1. 保留目标标签
    2. 过滤没有标签的图片
    2. 数据增强
    3. 划分数据集
    4. 写入 dataset.yaml 同时剔除标签的中文

    :param dataset_name: 数据集id
    :param split_weights: 自动划分数据集的比例
    :param labels_to_use: 限定使用的标签
    :param cate_to_use: 限定使用的类别
    :param labels_version: 标签版本
    """
    target_dataset_dir = get_dataset_dir(dataset_name)

    labels_to_use_real = reorganize_labels(dataset_name, labels_to_use=labels_to_use, cate_to_use=cate_to_use, labels_version=labels_version)
    # reorganize_images(dataset_name)  # 划分数据集部分可以只选用有标签的文件

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
    清理没有标签的原图
    @param dataset_name: 使用的数据集标签
    @param confirm_delete: 确认删除
    @return 没有标签的原图路径
    """
    labels: Set[str] = set()  # 标签

    target_dataset_dir = get_dataset_dir(dataset_name)

    # 保留特定标签
    labels_dir = os.path.join(target_dataset_dir, 'labels')
    labels_bk_dir = os.path.join(target_dataset_dir, 'labels_bk')
    labels_dir_to_use = labels_bk_dir if os.path.exists(labels_bk_dir) else labels_dir

    for label_txt in os.listdir(labels_dir_to_use):
        if not label_txt.endswith('.txt'):
            continue
        labels.add(label_txt[:-4])

    to_delete_paths: List[str] = []
    raw_img_dir = get_raw_images_dir()
    for cate_img_dir in os.listdir(raw_img_dir):
        cate_img_path = os.path.join(raw_img_dir, cate_img_dir)
        if not os.path.isdir(cate_img_path):
            continue

        for img in os.listdir(cate_img_path):
            if not img.endswith('.png'):
                continue
            case_id = img[:-4]
            if case_id in labels:
                continue

            to_delete_paths.append(os.path.join(cate_img_path, img))

    if confirm_delete:
        for to_delete in to_delete_paths:
            os.remove(to_delete)

    return to_delete_paths


def check_no_self_label_cases(dataset_name: str) -> List[str]:
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
