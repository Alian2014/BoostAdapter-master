from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenetv2 import ImageNetV2
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_sketch import ImageNetSketch

# 工厂映射表,字典的一种，key 依然为字符，但 value 为类，用于动态构造对象
dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet-a": ImageNetA,
                "imagenet-v": ImageNetV2,
                "imagenet-r": ImageNetR,
                "imagenet-s": ImageNetSketch,
                }

'''
build_dataset(...) 根据数据集名称（字符串）返回一个封装好的数据集类实例
'''
def build_dataset(dataset, root_path):
    # 利用工厂映射，动态初始化对应的类实例，返回对应的数据集类
    return dataset_list[dataset](root_path)