import hashlib
import os
import tarfile
import zipfile
import requests

"""
文件处理
"""

# @save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'


def download(name, cache_dir=os.path.join('..', 'data')):  # @save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        # 创建一个新的 SHA-1 哈希对象，用于后续计算文件的 SHA-1 校验码。
        sha1 = hashlib.sha1()
        # 打开文件 fname 以二进制读取模式 ('rb')
        # 上下文管理器 (with 语句) 确保文件在操作完成后正确关闭。
        with open(fname, 'rb') as f:
            while True:
                # 每次从文件中读取 1MB（1048576 字节）的数据块。f.read() 方法返回的是字节数据。
                data = f.read(1048576)
                # 如果读取到的数据为空（即到达文件末尾），则退出循环。
                if not data:
                    break
                # 将读取到的数据块更新到 SHA-1 哈希对象中，逐步累积哈希值。
                sha1.update(data)
        # sha1.hexdigest():计算整个文件的 SHA-1 校验码，并将其转换为十六进制字符串表示。
        if sha1.hexdigest() == sha1_hash:
            # 如果校验码匹配，则返回文件的本地路径 fname，表示文件已成功下载且完整，命中缓存。
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  # @save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    # 如果提供了 folder 参数，则返回 base_dir 和 folder 拼接后的路径。否则，返回 data_dir，即不带扩展名的文件路径。
    return os.path.join(base_dir, folder) if folder else data_dir


def download_all():  # @save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


