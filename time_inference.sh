#!/usr/bin/env bash

for BATCHSIZE in 1 5 10 20 50
do
    echo $BATCHSIZE
    # cpu
    python3 time_inference.py --model beyer_mod_relu_depth_126_48 --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model mobilenetv2_small_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model mobilenetv2_large_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model efficientnet_b0_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model efficientnet_b1_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model resnet18_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model resnet34_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model resnet50_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE
    python3 time_inference.py --model resnext50_additional_dense_pretrained --device cpu --batch_size $BATCHSIZE

    # gpu
    python3 time_inference.py --model beyer_mod_relu_depth_126_48 --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model mobilenetv2_small_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model mobilenetv2_large_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model efficientnet_b0_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model efficientnet_b1_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model resnet18_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model resnet34_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model resnet50_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
    python3 time_inference.py --model resnext50_additional_dense_pretrained --floatx 32 16 --batch_size $BATCHSIZE
done