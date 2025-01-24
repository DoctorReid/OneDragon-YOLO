from train.devtools import os_utils, label_studio_utils


# 重命名分类子文件夹中的图片
if __name__ == '__main__':
    project_dir = os_utils.get_path_under_work_dir('label_studio', 'zzz', 'lost_void_det')

    label_studio_utils.rename_file_in_raw_sub_dir(
        project_dir=project_dir
    )