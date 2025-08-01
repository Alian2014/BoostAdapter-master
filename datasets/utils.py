import os
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image

import numpy as np
import torchvision.transforms as transforms
from datasets.augmix_ops import augmentations
from datasets.resize_crop import RandomResizedCrop


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print('Cannot read image from {}, '
                  'probably due to heavy IO. Will re-try'.format(path))


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [
        f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f
    ]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = ''  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data

        self._num_classes = self.get_num_classes(test)
        self._lab2cname, self._classnames = self.get_lab2cname(test)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError('Input domain must belong to {}, '
                                 'but got [{}]'.format(self.domains, domain))

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


'''
DatasetWrapper(TorchDataset)：将原始数据集 data_source 包装成标准的 PyTorch Dataset 接口，
并支持图像读取 + 多视图增强 + label 提取 + CLIP 标准归一化处理。
'''


class DatasetWrapper(TorchDataset):
    '''
    data_source: 原始的数据集对象（必须是一个支持 __getitem__ 和 __len__ 的对象，且每个样本包含 .impath, .label, .domain 字段）
    input_size: 目标图像尺寸（如 224）
    transform: 图像增强方法（可以是单个，也可以是 list/tuple）
    is_train: 是否是训练集（影响是否允许 k_tfm > 1 多次增强）
    return_img0: 是否返回未经 transform 的原始图像
    k_tfm: 每张图是否增强 k 次（用于多视图，如 TTA、AugMix）
    '''

    def __init__(self,
                 data_source,
                 input_size,
                 transform=None,
                 is_train=False,
                 return_img0=False,
                 k_tfm=1):
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        # 防止没有传 transform 却要增强多次（k_tfm > 1）
        if self.k_tfm > 1 and transform is None:
            raise ValueError('Cannot augment the image {} times '
                             'because transform is None'.format(self.k_tfm))

        # Build transform that doesn't apply any data augmentation
        # 构造一个 “无增强的默认 transform”（用于原图 img0 的转换）
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    # 数据集大小
    def __len__(self):
        return len(self.data_source)

    # 核心数据获取逻辑，每当从该类获取数据时，该函数触发
    def __getitem__(self, idx):
        # 每个样本应该是一个数据集类实例
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        # 从数据集类实例中获取图像数据
        img0 = read_image(item.impath)

        # 如果图像增强输入不为空
        if self.transform is not None:
            # self.transform 是一个 list/tuple
            if isinstance(self.transform, (list, tuple)):
                # 依次将每个 transform 应用于原始图像 img0
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            # 单一 transform 的情况
            else:
                # 将单个 transform 应用于原始图像 img0
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        # 是否返回未经增强的原图
        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        # 只返回第一个图列表（因为只有一种增强方法）和对应的标签
        return output['img'], output['label']

    # 对同一张原始图像 img0 执行 k_tfm 次 指定的变换 tfm，并根据 k_tfm 的大小返回单个结果或列表结果
    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


'''
build_data_loader(...) 是一个 统一构建 PyTorch 数据加载器（DataLoader） 的封装器
其核心是：用 DatasetWrapper 包裹数据源，然后交给 PyTorch 的 DataLoader 来处理。

data_source=None,          # 通常是一个已有的 Dataset 实例，如 dataset.test
batch_size=64,             # 每个 batch 的样本数量
input_size=224,            # 图像输入大小（一般用于 wrapper）
tfm=None,                  # 图像预处理 transform，如 ToTensor()+Normalize
is_train=True,             # 训练集 or 测试集
shuffle=False,             # 是否打乱样本顺序
dataset_wrapper=None       # Dataset 包裹器，默认用 DatasetWrapper 类
'''


def build_data_loader(data_source=None,
                      batch_size=64,
                      input_size=224,
                      tfm=None,
                      is_train=True,
                      shuffle=False,
                      dataset_wrapper=None):

    # 设置默认包裹器
    if dataset_wrapper is None:
        # DatasetWrapper 是你自定义的类，用于封装数据源、处理 transform、input_size、是否是训练数据等逻辑
        dataset_wrapper = DatasetWrapper

    # Build data loader
    # 再传入 DataLoader，PyTorch 会根据这个 DatasetWrapper 构造一个可迭代的 DataLoader
    data_loader = torch.utils.data.DataLoader(
        # 创建了一个 Dataset 对象
        # 调用你自己的 DatasetWrapper 类的 __init__ 方法，并封装原始 dataset，DatasetWrapper的类方法将为 PyTorch 提供数据信息和处理方法等
        # PyTorch 就可以用这个 wrapper 来自动迭代样本
        dataset_wrapper(data_source,
                        input_size=input_size,
                        transform=tfm,
                        is_train=is_train),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()))
    # 用于确保你没有传入空的 dataset.test，避免训练时报错
    assert len(data_loader) > 0
    '''
    PyTorch 会根据这个 DatasetWrapper 构造一个可迭代的 DataLoader，它具有以下功能：
    
    batch_size=batch_size	每次迭代返回一批（batch）数据
    num_workers=8	用 8 个子进程并行加载数据，提高 IO 吞吐
    shuffle=shuffle	是否每个 epoch 打乱数据顺序
    drop_last=False	如果最后一批不足 batch_size，是否丢弃（这里保留）
    pin_memory=True	如果用 GPU，数据会更快复制到 CUDA
    
    接着作为函数返回值返回
    '''
    return data_loader


'''
get_preaugment()，返回一个组合增强操作，常用于深度学习训练阶段的图像预处理 pipeline
'''


def get_preaugment():
    # transforms.Compose 用于将多个图像变换按顺序组合成一个整体 transform
    # 返回的是一个可以作用于 PIL 图像或 Tensor 的可调用对象
    return transforms.Compose([
        # 随机裁剪图像的一部分，并缩放到 224×224 尺寸
        RandomResizedCrop(224),
        # 以 0.5 概率将图像左右翻转
        transforms.RandomHorizontalFlip(),
    ])


'''
augmix() 是AugMix 数据增强方法的核心流程，具体执行图像增强操作

image               # 原始图像
preprocess          # 标准预处理操作，例如 Resize、ToTensor、Normalize 等
aug_list            # 包含多个数据增强操作的函数列表（如旋转、模糊等）
severity            # 图像增强操作的强度等级（severity level），其作用是控制每个数据增强函数（如模糊、噪声、颜色扰动等）的变换幅度
'''


def augmix(image, preprocess, aug_list, severity=1):
    # get_preaugment() 是一个返回图像预处理函数的接口（比如统一大小、颜色空间等）
    preaugment = get_preaugment()
    # 预处理过的“原图”
    x_orig = preaugment(image)
    # 对原图 x_orig 应用标准预处理（如 Resize+ToTensor+Normalize），得到张量 x_processed
    x_processed = preprocess(x_orig)
    # 如果 aug_list 是空的，则直接返回预处理后的图像
    if len(aug_list) == 0:
        return x_processed
    # w 是从 [1,1,1] 的 狄利克雷分布采样的 3 个随机权重，和为 1，用于后续融合增强图像
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    # m 是从 Beta(1,1) 分布中采样的一个标量（范围在 0 到 1），用于决定“原图”和“增强图”的融合程度
    m = np.float32(np.random.beta(1.0, 1.0))

    # 构造增强图并加权求和
    mix = torch.zeros_like(x_processed)
    # 外循环做 3 个增强版本（对应 Dirichlet 的 3 个分支）
    for i in range(3):
        # 是从原图 x_orig 开始复制
        x_aug = x_orig.copy()
        # 随机施加 1~3 次增强操作（从 aug_list 中随机选）
        for _ in range(np.random.randint(1, 4)):
            # 再通过标准 preprocess 转换为张量
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        # 按照权重 w[i] 加到最终混合图像 mix 中
        mix += w[i] * preprocess(x_aug)
    # 最后将 原图 和 增强图 混合：m 越接近 1，保留越多原图信息，1 - m 越大，保留越多增强图的扰动信息
    mix = m * x_processed + (1 - m) * mix
    return mix


'''
AugMixAugmenter(object) 是一个 可调用的图像增强器类
'''


class AugMixAugmenter(object):
    '''
    base_transform：初步几何处理，如 Resize 和 CenterCrop
    preprocess：ToTensor + Normalize，转换为模型可接受的格式
    n_views：生成多少个增强图
    augmix：是否启用 AugMix（否则只用 base_transform）
    severity：控制 AugMix 扰动强度（越大越难）
    '''

    def __init__(self,
                 base_transform,
                 preprocess,
                 n_views=2,
                 augmix=False,
                 severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        # 若启用则应用增强操作
        if augmix:
            self.aug_list = augmentations
        else:
            self.aug_list = []
        self.severity = severity

    # 这是一个可调用类，使得你可以像函数一样用它：outputs = aug(x)
    def __call__(self, x):
        # 对输入图像 x 首先做 base_transform（如 resize/crop）
        # 然后 preprocess（to tensor + normalize）
        # 得到干净的标准视图
        image = self.preprocess(self.base_transform(x))
        # 调用外部函数 augmix(...) 来对输入图像进行多次增强
        # 每次增强返回的是 Tensor 格式（已预处理）
        views = [
            augmix(x, self.preprocess, self.aug_list, self.severity)
            for _ in range(self.n_views)
        ]

        # 返回一个 list，共包含：原图（干净视图），n_views 个增强图像
        return [image] + views
