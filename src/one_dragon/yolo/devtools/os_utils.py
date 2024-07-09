import os


def get_work_dir() -> str:
    """
    返回项目根目录的路径 OneDragon-YOLO/
    :return: 项目根目录
    """
    dir_path: str = os.path.abspath(__file__)
    up_times = 5
    for _ in range(up_times):
        dir_path = os.path.dirname(dir_path)
    return dir_path
