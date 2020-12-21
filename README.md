# Filter Grafting for Deep Neural Networks

## Introduction

This is the PyTorch implementation of our CVPR 2020 paper "[Filter Grafting for Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2020/html/Meng_Filter_Grafting_for_Deep_Neural_Networks_CVPR_2020_paper.html)". 

Invalid filters limit the potential of DNNs since they are identified as having little effect on the network. While filter pruning removes these invalid filters for efficiency consideration, Filter Grafting **re-activates** them from an accuracy boosting perspective. The activation is processed by grafting external information (weights) into invalid filters. 

![](./grafting.png)

## Prerequisites

Python 3.6+

PyTorch 1.0+

## CIFAR dataset

```
grafting.py [-h] [--lr LR] [--epochs EPOCHS] [--device DEVICE]
                   [--data DATA] [--s S] [--model MODEL] [--cifar CIFAR]
                   [--print_frequence PRINT_FREQUENCE] [--a A] [--c C]
                   [--num NUM] [--i I] [--cos] [--difflr]
PyTorch Grafting Training
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --epochs EPOCHS       total epochs for training
  --device DEVICE       cuda or cpu
  --data DATA           dataset path
  --s S                 checkpoint save path
  --model MODEL         Network used
  --cifar CIFAR         cifar10 or cifar100 dataset
  --print_frequence PRINT_FREQUENCE
                        test accuracy print frequency
  --a A                 hyperparameter a for calculate weighted average
                        coefficient
  --c C                 hyper parameter c for calculate weighted average
                        coefficient
  --num NUM             Number of Networks used for grafting
  --i I                 This program is the i th Network of all Networks
  --cos                 Use cosine annealing learning rate
  --difflr              Use different initial learning rate
```

### Execute example

#### Two models grafting

```shell
mkdir -pv checkpoint/grafting_cifar10_mobilenetv2;
CUDA_VISIBLE_DEVICES=0 nohup python grafting.py  --s checkpoint/grafting_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 --num 2 --i 1 >checkpoint/grafting_cifar10_mobilenetv2/1.log &
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py  --s checkpoint/grafting_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 --num 2 --i 2 >checkpoint/grafting_cifar10_mobilenetv2/2.log &
```

#### Three models grafting

```shell
mkdir -pv checkpoint/grafting_cifar10_mobilenetv2;
CUDA_VISIBLE_DEVICES=0 nohup python grafting.py  --s checkpoint/grafting_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 --num 3 --i 1 >checkpoint/grafting_cifar10_mobilenetv2/1.log &
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py  --s checkpoint/grafting_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 --num 3 --i 2 >checkpoint/grafting_cifar10_mobilenetv2/2.log &
CUDA_VISIBLE_DEVICES=2 nohup python grafting.py  --s checkpoint/grafting_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 --num 3 --i 3 >checkpoint/grafting_cifar10_mobilenetv2/3.log &
```

## Results

| model       | method        | cifar10   | cifar100  |
| ----------- | ------------- | --------- | --------- |
| ResNet32    | baseline      | 92.83     | 69.82     |
|             | grafting(slr) | 93.33     | 71.16     |
|             | grafting(dlr) | **93.94** | **71.28** |
| ResNet56    | baseline      | 93.50     | 71.55     |
|             | grafting(slr) | 94.28     | **73.09** |
|             | grafting(dlr) | **94.73** | 72.83     |
| ResNet110   | baseline      | 93.81     | 73.21     |
|             | grafting(slr) | 94.60     | 74.70     |
|             | grafting(dlr) | **94.96** | **75.27** |
| MobileNetv2 | baseline      | 92.42     | 71.44     |
|             | grafting(slr) | 93.53     | 73.26     |
|             | grafting(dlr) | **94.20** | **74.15** |

Grafting(slr) use the same learning rate with baseline that initial learning rate 0.1, and decay 0.1 at every 60 epochs.

While grafting(dlr) set different initial learning rate to increase two models' diversity, and use cosine annealing learning rate to make each batch of data have different importance to further increase the diversity.

| MobileNetV2       | CIFAR-10  | CIFAR-100 |
| ----------------- | --------- | --------- |
| baseline          | 92.42     | 71.44     |
| 6 models ensemble | 94.09     | 76.75     |
| 2 models grafting | 94.20     | 74.15     |
| 3 models grafting | 94.55     | 76.21     |
| 4 models grafting | 95.23     | 77.08     |
| 6 models grafting | **95.33** | **78.32** |
| 8 models grafting | 95.20     | 77.76     |

Comparison of the number of invalid filters

| model       | threshold | baseline(invlid/total) | grafting(invlid/total) |
| ----------- | --------- | ---------------------- | ---------------------- |
| ResNet32    | 0.1       | 36/1136                | 14/1136                |
|             | 0.01      | 35/1136                | 8/1136                 |
| MobileNetV2 | 0.1       | 10929/17088            | 9903/17088             |
|             | 0.01      | 9834/17088             | 8492/17088             |

Discusse the two hyper-pameters A and c

| MoblieNetV2 | A    | c    | cifar10 | cifar100 |
| ----------- | ---- | ---- | ------- | -------- |
|             |      | 1    | 93.19   | 73.3     |
|             |      | 5    | 92.76   | 72.69    |
|             | 0.4  | 10   | 93.31   | 73.26    |
|             |      | 50   | 93.24   | 73.05    |
|             |      | 500  | 92.79   | 72.38    |
| grafting    | 0    |      | 93.4    | 72.55    |
|             | 0.2  |      | 93.61   | 72.9     |
|             | 0.4  | 100  | 93.46   | 73.13    |
|             | 0.6  |      | 92.6    | 72.68    |
|             | 0.8  |      | 93.03   | 71.8     |
|             | 1    |      | 92.53   | 72.27    |

##  Citation

If you find this code useful, please cite the following paper:

```
@InProceedings{Meng_2020_CVPR,
author = {Meng, Fanxu and Cheng, Hao and Li, Ke and Xu, Zhixin and Ji, Rongrong and Sun, Xing and Lu, Guangming},
title = {Filter Grafting for Deep Neural Networks},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

## References

our code is based on https://github.com/kuangliu/pytorch-cifar.git