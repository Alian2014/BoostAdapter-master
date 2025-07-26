import os
import yaml
import torch
import math
import numpy as np
import clip
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader, AugMixAugmenter
import torchvision.transforms as transforms
from PIL import Image
import random
from enum import Enum

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

'''
get_entropy(...)将一个预测熵 loss 归一化为 [0, 1] 区间的比例值，表示当前预测的不确定性占理论最大值的多少

loss：通常是 softmax entropy 计算出的值，衡量预测分布的“不确定性”；
clip_weights：形状为 [num_classes, dim]，其中 num_classes 是类别数，用于确定“最大可能熵”
'''
def get_entropy(loss, clip_weights):
    # 计算最大熵
    max_entropy = math.log2(clip_weights.size(1))
    # 归一化当前熵
    # 把实际的熵 loss 除以理论最大熵，得到一个比例值 ∈ [0, 1]
    # 比例越高，说明模型预测越不确定;比例越低，说明模型预测越 confident
    return float(loss / max_entropy)

'''
get_output_entropy(outputs) 用来计算每个样本自身预测分布的熵
'''
def get_output_entropy(outputs):
    # 这行将每个样本的 logits 转换为 log-softmax
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    # 返回对每个样本的预测分布计算熵
    return -(logits * torch.exp(logits)).sum(dim=-1)

'''
softmax_entropy(x) 是在计算每个样本的预测分布熵
与 get_output_entropy(outputs) 功能一样
'''
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

'''
select_confident_samples(...)是典型的高置信样本筛选逻辑

feat：张量，形状为 [B, D]，表示每个样本的特征
logits：张量，形状为 [B, C]，表示每个样本对 C 个类的输出 logits
topTPT：浮点数 ∈ (0, 1)，保留最 confident 样本的比例（例如 0.1 代表保留前 10%）
'''
def select_confident_samples(feat, logits, topTPT):
    # 计算每个样本的 softmax 熵
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    # 筛选最 confident 的样本（低熵）
    idxTPT = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * topTPT)]
    # return feat[idxTPT], logits[idxTPT]
    '''
    feat[idxTPT]: 高置信特征；
    logits[idxTPT]: 高置信 logits；
    idxTPT: 被选中的样本下标
    '''
    return feat[idxTPT], logits[idxTPT], idxTPT


'''
avg_entropy(outputs) 计算的是多个样本预测分布的平均熵（entropy of the mean distribution），常用于评估一个模型输出的一致性
'''
def avg_entropy(outputs):
    # 这一行的作用是将 logits 转换为 log-softmax 形式
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    # 用 log-sum-exp 技巧计算平均分布的对数
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    # 这是为了获取当前数据类型支持的最小浮点数值
    min_real = torch.finfo(avg_logits.dtype).min
    # 防止 log(0)、exp(-inf)、nan 等数值不稳定问题
    avg_logits = torch.clamp(avg_logits, min=min_real)
    # 返回的是 outputs 的平均 softmax 概率分布的 熵（一个标量）
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

'''
cls_acc(...)是用于计算 Top-k 分类准确率

output：模型输出 logits，形状为 [B, C]，即每个样本对 C 类的预测分数；
target：真实标签，形状为 [B]，每个元素为整数类别 ID；
topk：默认是 1，表示计算 top-1 accuracy，也可以改为 top-5 等
'''
def cls_acc(output, target, topk=1):
    # 返回每个样本预测的概率最大的前 topk 个类别的索引，并转置
    pred = output.topk(topk, 1, True, True)[1].t()
    # 将 target reshape 成 [1, B]，再扩展成 [topk, B]
    # pred.eq(...) 会返回一个 [topk, B] 的布尔张量，表示第 k 个预测是否与真值相等
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # 将布尔 correct 张量拉平为 1D，再转成 float，再求和，得到预测正确的总数
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    # 最后除以样本数（即 batch size），再乘以 100 转成百分比形式
    acc = 100 * acc / target.shape[0]
    return acc

'''
clip_classifier(...) 是用获取每个类别的 Text Embedding，构造一个Text Embedding（类中心原型）矩阵，
可直接用于与图像 embedding 做 logits = image_feat @ clip_weights

classnames: 一个类名列表，如 ["dog", "cat", "airplane"]
template: 一个 prompt 模板列表，如 ["a photo of a {}.", "a picture of a {}."]
clip_model: OpenAI 的 CLIP 模型对象（必须支持 encode_text）
'''
def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        # 用于存储每个类别的最终权重（text embedding）
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            # 构造文本提示（prompt）
            texts = [t.format(classname) for t in template]
            # 将 prompt 们转换为 CLIP 输入的 token，再编码成 text embedding，形状为 [num_prompts, dim]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            # 对每个 prompt embedding 做 L2 归一化，以便后续使用点积作为相似度
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # 多个 prompt 的 embedding 做 prompt ensemble 平均
            class_embedding = class_embeddings.mean(dim=0)
            # 再 L2 归一化，得到该类别的最终 embedding，作为类中心（text prototype）
            class_embedding /= class_embedding.norm()
            # 将当前类别的文本向量 class_embedding 加入 clip_weights 列表中 
            clip_weights.append(class_embedding)

        # 把 [num_classes] 个 [D] 向量堆叠成一个二维张量,把堆叠后的权重张量移动到 GPU 上
        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    # 返回形状为 [embedding_dim, num_classes] 的 Text Embedding（类中心原型）矩阵
    return clip_weights

'''
get_clip_logits(...) 用于对输入图像计算 CLIP 模型的特征向量、logits，以及相关的预测信息（如 loss、prob_map、pred），并根据是否启用 infer_ori_image 控制返回哪些特征或做哪些聚合

images: 输入图像，可为单个张量或图像列表（通常用于 data augmentation）
clip_model: 已加载的 CLIP 模型，用于提取图像特征
clip_weights: 文本特征向量（即 prompt embedding），通常是 shape 为 [num_classes, feature_dim]
infer_ori_image: 控制是否以“原图方式”进行推理（通常在增强图像上聚合再推理，而原图只用一张）
'''
def get_clip_logits(images, clip_model, clip_weights, infer_ori_image=False):
    with torch.no_grad():
        # 如果 images 是列表（说明有多个增强版本），就拼接成一个 batch，然后无论如何都 .cuda() 将其移动到 GPU
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()
            
        # 用 CLIP 编码器提取图像特征，形状为 [B, D]
        image_features = clip_model.encode_image(images)
        # 每个向量做 L2 归一化，以便与文本特征做 cosine 相似度计算
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # logits = 图像特征 @ 文本特征，相当于 cosine similarity × 100
        clip_logits = 100. * image_features @ clip_weights
        # detach().clone() 是为了后面 BoostAdapter 筛选用（防止梯度传播 / 被修改）
        ori_feat = image_features.detach().clone()
        ori_output = clip_logits.detach().clone()

        #多张图 + 启用 infer_ori_image（仅用第一张原图）
        if image_features.size(0) > 1:
            if infer_ori_image:
                # loss：预测分布的熵
                loss = softmax_entropy(clip_logits[:1])
                # prob_map：类别概率
                prob_map = clip_logits[:1].softmax(1)
                # pred：预测类别
                pred = int(clip_logits[:1].topk(1, 1, True, True)[1].t()[0])
                return image_features[:1], clip_logits[:1], loss, prob_map, pred, ori_feat, ori_output
            # 多张图 + 非原图推理（对增强图集聚合）
            else:
                # 对所有增强图像计算熵（不确定性）
                batch_entropy = softmax_entropy(clip_logits)
                # 选出最有信心（熵最小）的前 10% 样本，用它们做特征聚合
                selected_idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * 0.1)]
                # 将这些最 confident 的图像特征、logits 做平均，输出的是 [1, D] 和 [1, C]
                output = clip_logits[selected_idx]
                image_features = image_features[selected_idx].mean(0).unsqueeze(0)
                clip_logits = output.mean(0).unsqueeze(0)

                # 对这 k 张图的 logits 计算平均熵（衡量不确定性）
                loss = avg_entropy(output)
                # 计算平均概率分布
                prob_map = output.softmax(1).mean(0).unsqueeze(0)
                # 最终预测类别（取均值后 top-1）
                # Top-1 指的是模型输出中，得分（logit 或概率）最高的那个类别
                pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        # 单张图像
        else:
            # 单张图像直接做预测
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred, ori_feat, ori_output

'''
get_ood_preprocess(args) 根据 CLIP 的要求构造一套增强逻辑，输入一张 PIL 图像，
输出多视图（views）的增强图像，用于 OOD 测试时的鲁棒性评估或不确定性分析
'''
def get_ood_preprocess(args):
    # 这组均值/标准差是 OpenAI CLIP 模型预训练所用的图像归一化参数
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    # 将图像缩放并居中裁剪为 224x224（CLIP 的默认输入尺寸）
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    # 将 PIL 图像转为 [C, H, W] 的 Tensor，值 ∈ [0, 1]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    # aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)
    # 增强模块
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=args.views-1, augmix=False)

    # 返回的是一个可调用的图像增强器对象
    return aug_preprocess

'''
get_cross_dataset_preprocess() 是用来构造一个用于 cross-dataset evaluation（跨数据集评估） 的图像预处理管道

preprocess: 是传入的 CLIP 或自定义的标准预处理（如 ToTensor + Normalize）
args.views: 控制生成的视图数量（如 total_views=4 → 原图 + 3 个增强）
'''
def get_cross_dataset_preprocess(preprocess, args):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    # 定义了一个空的 base_transform，对原始图像不进行几何变化
    base_transform = transforms.Compose([])
    # aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)
    # 表示除了原始图像之外，还要生成 args.views - 1 个额外视图
    # 返回的是一个多视图图像处理器，通常每次调用返回多个版本的图像
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=args.views-1, augmix=False)

    return aug_preprocess

'''
get_config_file(...)用于根据数据集名从 config_path 中读取对应的配置文件（YAML 格式）

dataset_name：数据集名称
config_path：配置文件地址
'''
def get_config_file(config_path, dataset_name):
    # 根据数据集名选择配置文件名
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    # 拼接路径并读取配置文件
    config_file = os.path.join(config_path, config_name)
    
    # 读取配置文件
    with open(config_file, 'r') as file:
        # 用 PyYAML 安全加载 YAML 文件，解析为字典
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    # 文件是否存在检查（疑似错误，应在打开之前）
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg

'''
build_test_data_loader(...) 根据你传入的数据集名称 dataset_name，加载对应的数据，
并返回用于测试的 DataLoader（图像加载器）以及该数据集的类名和模板，用于 Zero-Shot CLIP 推理任务

dataset_name：数据集简称，如 "I", "A", "V", "caltech101" 等
root_path：数据集存储的根目录
preprocess：输入图像预处理函数（可被 override）
args：其他控制参数（如 args.views）
'''
def build_test_data_loader(dataset_name, root_path, preprocess, args):
    # 用自定义 ImageNet 类加载数据
    if dataset_name == 'I':
        # 返回一个图像-类别映射字典
        dataset = ImageNet(root_path, preprocess)
        # 创建一个 PyTorch 的 DataLoader 类实例，用于批量加载测试图像数据 dataset.test
        '''
        dataset.test：PyTorch 的 Dataset 实例，测试集图像
        batch_size=1：每个 batch 的图像数量，1张一批
        num_workers=4：加载图像的子进程数，并行加载加快 IO 速度
        shuffle=True：是否打乱样本顺序，测试时一般为 False（这里设为 True 可能用于数据增强随机性）
        '''
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=4, shuffle=True)
    
    # 使用专门的 OOD 图像增强器：get_ood_preprocess(args)
    # OOD（Out-of-Distribution）数据集：和模型训练时所见数据分布不同的数据，用来测试模型在陌生场景下的表现（泛化能力、鲁棒性、不确定性等）
    elif dataset_name in ['A','V','R','S']:
        # 用专门的 OOD 图像增强器生成图像增强类实例
        preprocess = get_ood_preprocess(args)
        # 返回对应的数据集类实例
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        # 根据数据集实例和图像增强方法，返回一个可迭代的 DataLoader
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    # 使用 get_cross_dataset_preprocess() 构造跨数据集视图增强器
    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        # 构造跨数据集视图增强器
        preprocess = get_cross_dataset_preprocess(preprocess, args)
        # 返回对应的数据集类实例
        dataset = build_dataset(dataset_name, root_path)
        # 根据数据集实例和图像增强方法，返回一个可迭代的 DataLoader
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    '''
    test_loader: 加载测试数据的迭代器
    classnames: 类别名（用于构造文本 prompt）
    template: prompt 模板列表（如 "a photo of a {}"）
    '''
    return test_loader, dataset.classnames, dataset.template

'''
set_random_seed(seed) 是一个常见的随机数种子初始化函数，用于确保你在运行模型训练、推理、评估等过程中的结果具有可重复性
'''
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

'''
一个枚举类（Enum），它定义了四种统计方式的枚举常量，通常用于控制指标（如准确率、损失）的聚合方式
'''
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

'''
AverageMeter 类是训练或评估过程中常用的指标统计器（metric tracker），
结合前面定义的 Summary 枚举，可以灵活地记录当前值、平均值、总和、计数等指标，并支持多种输出格式
'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    '''
    name：该指标的名字（如 "acc@1", "loss"）
    fmt：控制打印数字的格式，默认 :f 即 float（可用 :.3f 设置小数位数）
    summary_type：决定 .summary() 方法返回什么（平均、总和、计数等）
    '''
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    # 清除所有状态
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    # 传入一个新值 val 和它的权重 n（通常是 batch size）
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    # 控制 print(meter) 输出什么
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    # 根据 summary_type 控制最终汇报时的输出
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

'''
ProgressMeter 类是训练或评估过程中用来格式化输出多个指标（如 loss、accuracy 等）随 batch 或 epoch 变化的进度信息的工具类
'''
class ProgressMeter(object):
    '''
    num_batches：总共的 batch 数（用于格式化进度 [ 23/100]）
    meters：一个 AverageMeter 实例列表
    prefix：前缀字符串（比如 "Test:"，"Train:"）
    '''
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    # 显示当前 batch 的进度 + 各指标的当前值和平均值
    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
    # 显示最终汇总 summary（如 epoch 结束后）
    def display_summary(self, logger):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))
    
    # 构造 [xx/total] 这样的格式化字符串
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'