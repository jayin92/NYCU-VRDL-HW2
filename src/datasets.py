import os
import torch
import torchvision as tv

from pycocotools.coco import COCO
from natsort import natsorted
from PIL import Image


class COCODatasets(tv.datasets.CocoDetection):
    def __init__(self, root, mode='train', transform=None):
        assert mode in ['train', 'valid'], "mode should be either 'train' or 'valid'"

        self.path_img = os.path.join(root, mode)
        self.path_annot = os.path.join(root, f'{mode}.json')
        self.transform = transform
        self.coco = COCO(self.path_annot)
        self.ids = list(natsorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self.coco.imgs[img_id]
        path = os.path.join(self.path_img, img['file_name'])
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        anns = self.coco.imgToAnns[img_id]
        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])
        target = {}
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels, dtype=torch.int64)

        return image, target
    
    def __len__(self):
        return len(self.ids)


class COCOTestDatasets(tv.datasets.CocoDetection):
    def __init__(self, root, mode='test', transform=None):
        assert mode in ['test'], "mode should be 'test'"

        self.path_img = os.path.join(root, mode)
        self.transform = transform
        self.images = list(natsorted(os.listdir(self.path_img)))
        

    def __getitem__(self, index):
        path = os.path.join(self.path_img, self.images[index])
        image = Image.open(path)

        if self.transform:
            image = self.transform(image)

        return image, self.images[index].split('.')[0]
    
    def __len__(self):
        return len(self.images)
    
    def get_image_id(self, index):
        return int(self.images[index].split('.')[0])