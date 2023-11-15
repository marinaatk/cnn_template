import os
import pandas as pd

from PIL import Image

import torch # PyTorch lib
from torch import nn
from torch.utils.data import Dataset, DataLoader # Dataset, DataLoader class from here
from torch.autograd import Variable


class CustomDataset(Dataset):
  def __init__(self, img_dir, annotations, transform=None, target_transform=None):
    self.img_dir = img_dir
    self.img_labels = pd.read_csv(annotations)
    self.transform = transform
    self.target_transform = target_transform
    self.cls_idx = self.class_to_id(self.img_labels.iloc[:, 1].unique())
    self.idx_cls = self.id_to_class(self.img_labels.iloc[:, 1].unique())

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_name = self.img_labels.iloc[idx, 0]
    label = self.cls_idx[self.img_labels.iloc[idx,1]]
    img_path = os.path.join(self.img_dir, img_name)
    image = Image.open(img_path)
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)

    return image, label

  def class_to_id(self, classes):
    return {v:k for k,v in enumerate(classes)}

  def id_to_class(self, classes):
    return {k:v for k,v in enumerate(classes)}
