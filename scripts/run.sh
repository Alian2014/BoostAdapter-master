#!/bin/bash
# 删除旧的 debug.txt 文件，防止干扰当前实验
rm -rf log/eval/*/debug.txt
args=(
    # --mode tda
    --mode boostadapter
    --config configs
    --datasets A 
    # --datasets R 
    # --datasets V
    # --datasets S
    # -- datasets food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101
    # --datasets eurosat
    # --datasets food101
    # --datasets ucf101
    # --datasets dtd
    # --datasets fgvc
    # --datasets sun397
    # --datasets caltech101
    # --datasets oxford_flowers
    # --datasets oxford_pets
    # --datasets stanford_cars
    --backbone ViT-B/16 
    --delta 3
    --views 64
    --exp_name $2
)

#CUDA_VISIBLE_DEVICES=$1	使用第 $1 个命令行参数作为 GPU ID
#python runner.py	        启动主程序（BoostAdapter 实验入口）
#"${args[@]}"	            展开 args 列表作为命令行参数传入 runner.py
CUDA_VISIBLE_DEVICES=$1 python runner.py "${args[@]}"   