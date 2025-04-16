# NYCU-VRDL-HW2

Name: 李杰穎

Student ID: 110550088

## Introduction

This homework focuses on the digit recognition task using the Faster R-CNN object detection framework. The dataset consists of RGB images with multiple digits, each digit labeled with its bounding box and category. The task is divided into two subtasks: (1) detecting the bounding boxes and digit categories, and (2) predicting the entire number based on the digits detected. In order to make the Faster R-CNN framework more suitable for this task, I modify the anchor sizes and aspect ratios of the RPN. 
## Training

```bash
python src/train.py --data_dir ./data/ --batch_size 40 --lr 0.001 --epochs 10 --min_size 200 --max_size 400 --trainable_backbone_layers 3 --optimizer AdamW --scheduler cosine
```
