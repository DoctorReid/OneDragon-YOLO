import re

def without_cn_id_str(with_cn):
    return re.sub(r'[^a-zA-Z0-9\-]+', '', with_cn).replace('--', '-')