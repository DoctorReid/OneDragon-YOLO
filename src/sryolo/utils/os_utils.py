import os


def get_work_dir() -> str:
    """
    返回项目根目录的路径 StarRail-YOLO/
    :return: 项目根目录
    """
    dir_path: str = os.path.abspath(__file__)
    # 打包后运行
    up_times = 4
    for _ in range(up_times):
        dir_path = os.path.dirname(dir_path)
    return dir_path
