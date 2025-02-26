# <p align=center>ACENet: Adaptive Context Enhancement Network for RGB-T Video Object Detection</p>

<p align=center>Zhengzheng Tu, Le Gu, Danyin Lin, and Zhicheng Zhao.</p>

## Introduction
This repository is the official implementation for "ACENet: Adaptive Context Enhancement Network for RGB-T Video Object Detection".

![image](ACENet.png)

## Abstract
RGB-thermal (RGB-T) video object detection (VOD) aims to leverage the complementary advantages of visible and thermal infrared sensors to achieve robust performance under various challenging conditions, such as low illumination and extreme illumination changes. However, existing multimodal VOD approaches face two critical challenges: accurate detection of objects at different scales and efficient fusion of temporal information from multimodal data. To address these issues, we propose an Adaptive Context Enhancement Network (ACENet) for RGB-T VOD. Firstly, we design an Adaptive Context Enhancement Module (ACEM) to adaptively enhance multi-scale context information. We introduce ACEM in the FPN section, where it can adaptively extract context information and incorporate it into the high-level feature maps. Secondly, we design a Multimodal Temporal Fusion Module (MTFM) to perform temporal and modal fusion using coordinate attention with atrous convolution at the early stage, significantly reducing the complexity of fusing temporal information from RGB and thermal data. Experimental results on the VT-VOD50 dataset show that our ACENet significantly outperforms other mainstream VOD methods. Our code is available at: https://github.com/bscs12/ACENet.

## Getting Started
### Prepare Environment
1. Clone repository
```
git clone https://github.com/bscs12/ACENet.git
cd ACENet
```

2. Create environment
```
conda create -n ACENet python=3.7
conda activate ACENet
```

3. Install PyTorch
```
pip install torch==1.7.0+cu101 torchvision==0.8.0+cu101 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```
or
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch
```
4. Install ACENet
```
pip install -U pip && pip install -r requirements.txt
pip install -v -e .  # or  python setup.py develop
```

5. Install APEX
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

6. Install pycocotools
```
pip install cython
pip install pycocotools
```

### Prepare Checkpoint
You can download the pretrained checkpoint from [https://github.com/Megvii-BaseDetection/YOLOX].

### Prepare Dataset
1. Download dataset

You can download the MMVOD2022 dataset from [https://pan.baidu.com/s/1S1onHrlVH8s6xF2uXlELYw?pwd=e2de] [Password: ```e2de```].

2. Prepare your own dataset

Make sure the dataset folder structure like this:
```
datasets
    ├── MMVOD2022
    │   ├── VOD2022
    │   │   ├── JPEGImages
    │   │   │   ├── L1
    │   │   │   │   ├── RGB
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── ...
    │   │   │   │   ├── T
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── ...
    │   │   │   ├── L2
    │   │   │   │   ├── RGB
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── ...
    │   │   │   │   ├── T
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── xxx.jpg
    │   │   │   │   │   ├── ...
    │   │   │   ├── ...
    │   │   ├── Annotations
    │   │   │   ├── L1
    │   │   │   │   ├── RGB
    │   │   │   │   │   ├── xxx.xml
    │   │   │   │   │   ├── xxx.xml
    │   │   │   │   │   ├── ...
    │   │   │   ├── ...
    │   │   ├── ImageSets
    │   │   │   ├── Main
    │   │   │   │   ├── train.txt
    │   │   │   │   ├── test.txt
```
Modify the classes in ```ACENet/yolox/data/datasets/voc_classes.py```
### Train

Modify the parameters in ```ACENet/yolox/exp/yolox_base.py``` and ```ACENet/train.py```
```
python train.py
```
### Evaluation
```
python eval.py
```
## Cite
If you find this work useful for your, please consider citing our paper. Thank you!