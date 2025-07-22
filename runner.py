import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import clip
from utils import *
from loguru import logger
import sys
from IPython import embed
from utils import AverageMeter
import time
import copy


# 函数 get_plpd_image(x) 主要的作用是对图像进行局部 patch 打乱重排
def get_plpd_image(x):
    import torchvision
    from einops import rearrange
    # 设置 patch 的大小为 4×4
    patch_len = 4
    # 实例化对象，该对象可将图像 resize 成可以整除 patch_len 的大小，确保后续 patch 分割不出错
    resize_t = torchvision.transforms.Resize(
        ((x.shape[-1] // patch_len) * patch_len,
         (x.shape[-1] // patch_len) * patch_len))
    # 实例化对象，该对象可将图像恢复到原始尺寸，防止分块重排后尺寸变化
    resize_o = torchvision.transforms.Resize((x.shape[-1], x.shape[-1]))
    # 调用对象 _call_ 方法，调整尺寸到可整除 patch 长宽
    x = resize_t(x)
    # 将图像分成若干个大小为 patch_len x patch_len 的 patch
    x = rearrange(x,
                  'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w',
                  ps1=patch_len,
                  ps2=patch_len)
    # 对每个样本的 patch 顺序进行随机打乱
    perm_idx = torch.argsort(torch.rand(x.shape[0], x.shape[1]), dim=-1)
    x = x[torch.arange(x.shape[0]).unsqueeze(-1), perm_idx]
    # 把打乱的 patch 重新拼接成图像
    x = rearrange(x,
                  'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)',
                  ps1=patch_len,
                  ps2=patch_len)
    # 调用对象 _call_ 方法，恢复为原始图像尺寸
    x = resize_o(x)
    return x


# 命令行参数解析函数，从命令行读取和解析参数，返回一个包含所有参数的 args 对象
def get_arguments():
    """Get arguments of the test-time adaptation."""
    # 实例化对象，parser.add_argument 用于向命令行解析器添加参数规则
    parser = argparse.ArgumentParser()
    # 指定配置文件路径（YAML 格式），包含 TDA 的各种设置
    parser.add_argument(
        '--config',
        dest='config',
        required=True,
        help='settings of TDA on specific dataset in yaml format.')
    # 是否开启 Weights & Biases 的日志记录（实验可视化工具）
    parser.add_argument(
        '--wandb-log',
        dest='wandb',
        action='store_true',
        help=
        'Whether you want to log to wandb. Include this flag to enable logging.'
    )
    # 指定要处理的数据集（可能有多个，用 / 分隔）
    parser.add_argument(
        '--datasets',
        dest='datasets',
        type=str,
        required=True,
        help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S"
    )
    # 指定数据集根目录的路径
    parser.add_argument(
        '--data-root',
        dest='data_root',
        type=str,
        default='../data/',
        help='Path to the datasets directory. Default is ./dataset/')
    # 指定 CLIP 模型的骨干网络
    parser.add_argument('--backbone',
                        dest='backbone',
                        type=str,
                        choices=['RN50', 'ViT-B/16'],
                        required=True,
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')
    # 实验名（用于命名保存目录、日志等）
    parser.add_argument('--exp_name', type=str)
    # 运行模式，例如 tta, eval, adapt，取决于程序逻辑
    parser.add_argument('--mode', type=str)
    # 调节参数
    parser.add_argument('--delta', type=int, default=0)
    # 指定一个样本生成的视角/增强版本数量
    parser.add_argument('--views', type=int, default=64)
    # 根据命令行返回 args，一个包含所有参数的命名空间对象
    args = parser.parse_args()

    return args


# 缓存更新函数，将某个类别 pred 对应的新样本（包含其特征与损失信息）加入缓存中，并按照容量上限进行替换策略管理
def update_cache(cache,
                 pred,
                 features_loss,
                 shot_capacity,
                 include_prob_map=False,
                 fifo=True):
    # 表示这个过程是非训练性的
    with torch.no_grad():
        # 如果 include_prob_map=True，说明 features_loss 中包含 [feature, loss, prob_map]，否则就直接拿全部内容作为缓存元素
        # prob_map.shape == [1, C], feature.shape == [1, D]
        item = features_loss if not include_prob_map else features_loss[:2] + [
            features_loss[2]
        ]
        # 类别 pred 已存在缓存中，尝试加入新样本
        if pred in cache:
            # 将当前样本（item）添加进缓存字典 cache 中，属于某个类别 pred 的样本列表
            cache[pred].append(item)
            # 使用先进先出策略（默认）
            if fifo:
                # 如果超过长度限制，删除最旧的那个（即最前面的一个）
                if len(cache[pred]) > shot_capacity:
                    cache[pred] = cache[pred][1:]
            # 否则，使用“损失值排序”策略
            else:
                # 按 item[1] 即 loss 进行排序（从小到大）
                cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
                # 只保留 loss 最小的前 shot_capacity 个样本
                cache[pred] = cache[pred][:shot_capacity]
        # 第一次遇到该类别，初始化其缓存列表
        else:
            cache[pred] = [item]


# 这段函数 compute_cache_logits(...) 是典型的 基于缓存（cache）的 logit 计算模块
def compute_cache_logits(image_features,
                         cache,
                         alpha,
                         beta,
                         clip_weights,
                         neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        # 所有缓存特征图
        cache_keys = []
        # 每个缓存对应的类别或概率图
        cache_values = []
        # 将 feature 添加到特征图列表
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                # 如果启用了 neg_mask_thresholds，说明当前你希望使用 item[2] 的概率图（prob_map）进行筛选或加权
                if neg_mask_thresholds:
                    # 缓存值中存入概率图
                    cache_values.append(item[2])
                else:
                    # 否则，就是普通的分类任务，将类别存入缓存
                    cache_values.append(class_index)
        if len(cache_keys) == 0:
            # 这个写法用于在没有缓存内容时，返回一个合法的 torch.Tensor 类型的 0 值，而不是 Python 原生的 0 或 None
            return torch.zeros(1)[0]

        # 拼接所有缓存图像特征为 [N, D]，然后转置为 [D, N]
        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        # 如果启用阈值过滤
        if neg_mask_thresholds:
            # torch.cat(..., dim=0) 把它们拼成一个 [N, D] 的 tensor
            cache_values = torch.cat(cache_values, dim=0)
            # 阈值过滤 (val > th0) & (val < th1) 得到 [N, D] 的 mask，过滤缓存中不可靠的特征维度
            # 随后将 bool 类型转成 int8（0 或 1），再转成 float16 并搬到 GPU
            cache_values = (((cache_values > neg_mask_thresholds[0]) &
                             (cache_values < neg_mask_thresholds[1])).type(
                                 torch.int8)).cuda().half()
        else:
            # F.one_hot(...) → [N, C]：将每个类别索引转换成 one-hot 标签
            # .cuda().half()：转为 float16 并搬到 CUDA
            cache_values = (F.one_hot(
                torch.Tensor(cache_values).to(torch.int64),
                num_classes=clip_weights.size(1))).cuda().half()
        # 当前图像特征与 N 个缓存样本的特征向量相乘后得到每个缓存样本与该图像的相似度
        # affinity.shape == [1, N]
        affinity = image_features @ cache_keys
        # beta - beta * affinity = beta * (1 - affinity) 越相似的 affinity 趋近于 1，这项趋近于 0；越不相似的 affinity 趋近于 0，这项趋近于 beta
        # (-1 * ...).exp() = exp(beta * (affinity - 1)) 使得越相似的 affinity 得到的权重越大（接近 1），越不相似的权重接近 0
        # cache_logits = weight @ cache_values 是一个典型的加权平均操作
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # return alpha * cache_logits, affinity
        # 最后输出一个加权后的向量（特征或伪 logit），乘以调节系数 alpha
        # 返回值 shape 保持为 [1, D]
        return alpha * cache_logits


def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, logger,
                 args, cfg):
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    batch_time = AverageMeter('Time', ':6.3f', Summary.AVERAGE)
    progress = ProgressMeter(len(loader), [top1], prefix='Test: ')

    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []

        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']

        if args.mode in ["clip"]:
            pos_enabled = False
            neg_enabled = False

        if pos_enabled:
            pos_params = {
                k: pos_cfg[k]
                for k in ['shot_capacity', 'alpha', 'beta']
            }
        if neg_enabled:
            neg_params = {
                k: neg_cfg[k]
                for k in [
                    'shot_capacity', 'alpha', 'beta', 'entropy_threshold',
                    'mask_threshold'
                ]
            }

        end = time.time()
        #Test-time adaptation
        for idx, (images, target) in enumerate(loader):
            infer_ori_image = args.datasets in "caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101/ \
            contrast/jpeg/pixelate/elastic/ \
            defocus/glass/zoom/motion/ \
            saturate/ \
            gaussian/shot/impulse/speckle/ \
            fog/frost/snow/brightness"

            # infer_ori_image = True
            image_features, clip_logits, loss, prob_map, pred, ori_feat, ori_output = get_clip_logits(
                images,
                clip_model,
                clip_weights,
                infer_ori_image=infer_ori_image)
            target, prop_entropy = target.cuda(), get_entropy(
                loss, clip_weights)

            if pos_enabled:
                if args.mode in ["tda"]:
                    update_cache(pos_cache,
                                 pred, [image_features, loss],
                                 pos_params['shot_capacity'],
                                 fifo=False)
                else:
                    update_cache(pos_cache,
                                 pred, [image_features, loss],
                                 pos_params['shot_capacity'],
                                 fifo=False)

                if args.mode in ["boostadapter"]:
                    select_feat, select_output, select_idx = select_confident_samples(
                        ori_feat, ori_output, 0.1)
                    select_entropy = get_output_entropy(select_output)

                    cur_pos_cache = copy.deepcopy(pos_cache)
                    for i in range(select_entropy.shape[0]):
                        cur_pred = int(select_output[i].argmax(dim=-1).item())
                        cur_feat = select_feat[i]
                        update_cache(
                            cur_pos_cache,
                            cur_pred,
                            [cur_feat.unsqueeze(0), select_entropy[i].item()],
                            pos_params['shot_capacity'] + cfg['delta'],
                            fifo=False)

            if neg_enabled and neg_params['entropy_threshold'][
                    'lower'] < prop_entropy < neg_params['entropy_threshold'][
                        'upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map],
                             neg_params['shot_capacity'], True)

            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                if args.mode in ["tda"]:
                    final_logits += compute_cache_logits(
                        image_features, pos_cache, pos_params['alpha'],
                        pos_params['beta'], clip_weights)
                elif args.mode in ["boostadapter"]:
                    final_logits += compute_cache_logits(
                        image_features, cur_pos_cache, pos_params['alpha'],
                        pos_params['beta'], clip_weights)

            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(
                    image_features, neg_cache, neg_params['alpha'],
                    neg_params['beta'], clip_weights,
                    (neg_params['mask_threshold']['lower'],
                     neg_params['mask_threshold']['upper']))

            acc = cls_acc(final_logits, target)
            accuracies.append(acc)
            end = time.time()

            top1.update(acc, 1)
            batch_time.update(time.time() - end, 1)
            end = time.time()

            if idx % 1 == 0:
                progress.display(idx, logger)
        progress.display_summary(logger)

        return sum(accuracies) / len(accuracies)


def main():
    torch.set_num_threads(8)
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    model_path = args.backbone
    clip_model, preprocess = clip.load(model_path)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    fmt = "[{time: MM-DD hh:mm:ss}] {message}"
    config = {
        "handlers": [
            {
                "sink": sys.stderr,
                "format": fmt
            },
        ],
    }
    logger.configure(**config)

    if "search" in args.exp_name:
        file_root = f"log/search/{args.datasets}/"
    else:
        file_root = f"log/eval/{args.datasets}/"
    file_path = f"{file_root}/{args.exp_name}.txt"
    if not os.path.exists(file_root):
        os.makedirs(file_root)
    if os.path.exists(file_path):
        print("Experiment exists. Skipping...")
        exit()
    open(file_path, 'w').close()
    logger.add(file_path, format=fmt)

    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        logger.info(args)
        logger.info("\nRunning dataset configurations:")
        logger.info(cfg)

        test_loader, classnames, template = build_test_data_loader(
            dataset_name, args.data_root, preprocess, args)
        clip_weights = clip_classifier(classnames, template, clip_model)

        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader,
                           clip_model, clip_weights, logger, args, cfg)


if __name__ == "__main__":
    main()
