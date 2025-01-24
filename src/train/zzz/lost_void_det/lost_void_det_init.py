from train.devtools import os_utils, label_studio_utils
from train.zzz.lost_void_det import lost_void_label

# 初始化文件夹 以及输出label_studio的标签
if __name__ == '__main__':
    label_df = lost_void_label.get_label_df()
    project_dir = os_utils.get_path_under_work_dir('label_studio', 'zzz', 'lost_void_det')

    label_studio_utils.create_sub_dir_in_raw(
        project_dir=project_dir,
        label_df=label_df,
        label_col='label',
        class_col='entry_name'
    )

    label_studio_utils.print_labeling_interface(
        label_df=label_df,
        label_col='label',
        class_col='entry_name'
    )