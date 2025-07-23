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

def get_output_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    return -(logits * torch.exp(logits)).sum(dim=-1)

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

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

'''
cls_acc(...)是用于计算 Top-k 分类准确率

output：模型输出 logits，形状为 [B, C]，即每个样本对 C 类的预测分数；
target：真实标签，形状为 [B]，每个元素为整数类别 ID；
topk：默认是 1，表示计算 top-1 accuracy，也可以改为 top-5 等
'''
def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
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

                # 计算平均 entropy，预测类别，输出概率图
                loss = avg_entropy(output)
                prob_map = output.softmax(1).mean(0).unsqueeze(0)
                pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        # 单张图像
        else:
            # 单张图像直接做预测
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred, ori_feat, ori_output


def get_ood_preprocess(args):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    # aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=args.views-1, augmix=False)

    return aug_preprocess

def get_cross_dataset_preprocess(preprocess, args):
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
    base_transform = transforms.Compose([])
    # aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=63, augmix=True)
    aug_preprocess = AugMixAugmenter(base_transform, preprocess, n_views=args.views-1, augmix=False)

    return aug_preprocess


def get_config_file(config_path, dataset_name):
    if dataset_name == "I":
        config_name = "imagenet.yaml"
    elif dataset_name in ["A", "V", "R", "S"]:
        config_name = f"imagenet_{dataset_name.lower()}.yaml"
    else:
        config_name = f"{dataset_name}.yaml"
    
    config_file = os.path.join(config_path, config_name)
    
    with open(config_file, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")

    return cfg


def build_test_data_loader(dataset_name, root_path, preprocess, args):
    if dataset_name == 'I':
        dataset = ImageNet(root_path, preprocess)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=1, num_workers=4, shuffle=True)
    
    elif dataset_name in ['A','V','R','S']:
        preprocess = get_ood_preprocess(args)
        dataset = build_dataset(f"imagenet-{dataset_name.lower()}", root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)

    elif dataset_name in ['caltech101','dtd','eurosat','fgvc','food101','oxford_flowers','oxford_pets','stanford_cars','sun397','ucf101']:
        preprocess = get_cross_dataset_preprocess(preprocess, args)
        dataset = build_dataset(dataset_name, root_path)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=1, is_train=False, tfm=preprocess, shuffle=True)
    
    else:
        raise "Dataset is not from the chosen list"
    
    return test_loader, dataset.classnames, dataset.template



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
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

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
        
    def display_summary(self, logger):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'