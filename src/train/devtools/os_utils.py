import os


def get_work_dir() -> str:
    """
    返回项目根目录的路径 OneDragon-YOLO/
    :return: 项目根目录
    """
    dir_path: str = os.path.abspath(__file__)
    up_times = 4
    for _ in range(up_times):
        dir_path = os.path.dirname(dir_path)
    return dir_path


def join_dir_path_with_mk(path: str, *subs) -> str:
    """
    拼接目录路径和子目录
    如果拼接后的目录不存在 则创建
    :param path: 目录路径
    :param subs: 子目录路径 可以传入多个表示多级
    :return: 拼接后的子目录路径
    """
    target_path = path
    for sub in subs:
        if sub is None:
            continue
        target_path = os.path.join(target_path, sub)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    return target_path


def get_path_under_work_dir(*sub_paths: str) -> str:
    """
    获取当前工作目录下的子目录路径
    :param sub_paths: 子目录路径 可以传入多个表示多级
    :return: 拼接后的子目录路径
    """
    return join_dir_path_with_mk(get_work_dir(), *sub_paths)