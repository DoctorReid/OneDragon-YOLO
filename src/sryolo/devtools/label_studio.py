from typing import List, Set

from sryolo.devtools import label_utils
import os
import json

from sryolo.utils import os_utils
from urllib.parse import quote, unquote
import shutil


def get_raw_images_dir() -> str:
    """
    原图的根目录
    """
    return os.path.join(os_utils.get_work_dir(), '.env', 'data', 'raw_images')


def get_annotations_dir() -> str:
    """
    标注的根目录 label-studio 自动同步Target的目录
    """
    return os.path.join(os_utils.get_work_dir(), '.env', 'data', 'label-studio-annotations')


def get_tasks_dir() -> str:
    """
    标注任务的根目录 label-studio 自动同步Source的目录
    """
    path = os.path.join(os_utils.get_work_dir(), '.env', 'data', 'label-studio-tasks')
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def list_label_template(col: str):
    """
    显示label-studio需要用的分类模板
    :param col: 分类列
    """
    df = label_utils.read_label_csv()
    for v1 in df[col].drop_duplicates():
        print('<Label value="%s"/>' % v1)


def get_img_name_2_annotations() -> dict:
    """
    获取当前的标注
    key=图片文件名
    value=Label-Studio自动同步保存的标注
    """
    annotations_dir = get_annotations_dir()
    img_2_annotations = {}
    for annotation_file_name in os.listdir(annotations_dir):
        if annotation_file_name.find('.') > -1:
            continue
        annotation_file_path = os.path.join(annotations_dir, annotation_file_name)
        with open(annotation_file_path, 'r') as file:
            label_old = json.load(file)
            label_new = {
                'data': {},
                'annotations': []
            }
            label_new['data']['image'] = label_old['task']['data']['image']
            label_new['annotations'].append({
                'result': label_old['result']
            })
            file_path = unquote(label_new['data']['image'][21:])  # 21位开始是raw_images文件夹
            img_2_annotations[file_path[file_path.rfind('\\') + 1:]] = label_new

    return img_2_annotations


def generate_tasks_from_annotations(renew: bool = False):
    """
    从已有的标注文件中生成task 适合用于导入其他Label-Studio项目标注的数据
    并对原图文件夹中的图片进行重命名 按子文件夹名称做前缀
    @param renew: 是否全新 否的话只重命名未符合规范的 是的话全部重命名 使用前应该做好备份
    """
    raw_img_dir = get_raw_images_dir()

    old_img_2_annotations = get_img_name_2_annotations()
    new_img_2_annotations = {}

    img_path_old_to_new = {}
    for prefix in os.listdir(raw_img_dir):
        sub_img_dir = os.path.join(raw_img_dir, prefix)
        if not os.path.isdir(sub_img_dir):
            continue

        to_name_list = []  # 需要重命名的图片
        existed_ids = set()  # 已经存在的id
        for img_name in os.listdir(sub_img_dir):
            if not img_name.endswith('.png'):
                continue
            if not renew and img_name.startswith(prefix):
                curr_id = img_name[-9:-4]
                existed_ids.add(int(curr_id))
            else:
                to_name_list.append(img_name)

        idx = 0
        for old_name in to_name_list:
            idx += 1
            while not renew and idx in existed_ids:
                idx += 1
            existed_ids.add(idx)

            new_name = '%s-%05d.png' % (prefix, idx)
            new_path = os.path.join(sub_img_dir, new_name)
            img_path_old_to_new[os.path.join(sub_img_dir, old_name)] = new_path
            if old_name in old_img_2_annotations:
                new_annotations = old_img_2_annotations[old_name].copy()
                new_annotations['data']['image'] = '/data/local-files/?d=' + quote(new_path[new_path.find('raw_images'):])
                new_img_2_annotations[new_name] = new_annotations

    old_path_list = list(img_path_old_to_new.keys())
    old_path_list.sort()
    for old_path in old_path_list:
        new_path = img_path_old_to_new[old_path]
        # print(old_path, new_path)
        shutil.move(old_path, new_path)

    tasks_dir = get_tasks_dir()
    for img_name, annotations in new_img_2_annotations.items():
        new_task_path = os.path.join(tasks_dir, '%s.json' % img_name[:-4])
        with open(new_task_path, 'w') as file:
            json.dump(annotations, file, indent=4)
