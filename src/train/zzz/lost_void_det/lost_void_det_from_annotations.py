from train.devtools import label_studio_utils, os_utils

if __name__ == '__main__':
    project_dir = os_utils.get_path_under_work_dir('label_studio', 'zzz', 'lost_void_det')
    label_studio_utils.generate_tasks_from_annotations(
        project_dir,
        # old_img_path_prefix='raw_images',
        # new_img_path_prefix='zzz\\lost_void_det\\raw',
    )