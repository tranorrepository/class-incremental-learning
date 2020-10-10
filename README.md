## Meta-Aggregating Networks for Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/class-incremental-learning/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8?style=flat-square)](https://pytorch.org/)

\[[arXiv preprint](https://arxiv.org/)\]

### Installation

In order to run this repository, we advise you to install python 3.6 and PyTorch 1.2.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name MANets-PyTorch python=3.6
conda activate MANets-PyTorch
conda install pytorch=1.2.0 
conda install torchvision -c pytorch
```

Install other requirements:
```bash
pip install tqdm tensorboardX Pillow==6.2.2
```

Clone this repository:

```bash
git clone https://github.com/yaoyao-liu/class-incremental-learning.git 
cd class-incremental-learning
```

### Running Experiments

Running experiments on CIFAR-100 (basline: iCaRL)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=icarl --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --ckpt_label=Exp_01
```

Running experiments on CIFAR-100 (basline: LUCIR)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=lucir --branch_mode=dual --branch_1=ss --branch_2=free --dataset=cifar100 --ckpt_label=Exp_01
```

Running experiments on ImageNet-Subset (basline: iCaRL)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=icarl --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=icarl --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=icarl --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 --ckpt_label=Exp_01
```

Running experiments on ImageNet-Subset (basline: LUCIR)
```bash
python3 main.py --nb_cl_fg=50 --nb_cl=10 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=lucir --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=5 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=lucir --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 --ckpt_label=Exp_01
python3 main.py --nb_cl_fg=50 --nb_cl=2 --gpu=0 --random_seed=1993 --fusion_lr=1e-05 --baseline=lucir --imgnet_backbone=resnet18 --branch_mode=dual --branch_1=ss --branch_2=fixed --dataset=imagenet_sub --data_dir=./seed_1993_subset_100_imagenet/data --test_batch_size=50 --epochs=90 --num_workers=16 --custom_weight_decay=1e-4 --test_batch_size=50 --ckpt_label=Exp_01
```
