import torch
import os
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import sys


import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from torch.utils.data import Dataset
from PIL import Image
from models.chexnet.DensenetModels import DenseNet121
from models.models import ResNet18
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report, accuracy_score

class ChestXrayDataSet(Dataset):
    
    def convert_to_ones(self, df, disease):
        df[disease] = df[disease].replace([-1.0], 1.0)
    
    def convert_to_zeros(self, df, disease):
        df[disease] = df[disease].replace([-1.0], 0.0)
        
    def convert_to_multi(self, df, disease):
        df[disease] = df[disease].replace([-1.0], 2.0)

    def __init__(self, data_dir, image_list_file, diseases=['Atelectasis', 'Consolidation', 'Edema','Cardiomegaly', 'Pleural Effusion'], side='Frontal', transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        chex_df = pd.read_csv(image_list_file)
        chex_df = chex_df.fillna(0.0)
        chex_df = chex_df.loc[chex_df['Frontal/Lateral'] == side]
        self.convert_to_ones(chex_df, 'Atelectasis')
        self.convert_to_multi(chex_df, 'Consolidation')
        self.convert_to_ones(chex_df, 'Edema')
        self.convert_to_multi(chex_df, 'Cardiomegaly')
        self.convert_to_multi(chex_df, 'Pleural Effusion')

#         chex_df_diseases = chex_df[diseases]
                         
#         if 'train' in image_list_file:
#             chex_df = chex_df
#         if len(diseases) == 1:
#             chex_df = chex_df.loc[chex_df['Pleural Effusion'] != -1] #U-Ignore
#         print(chex_df)
        labels = chex_df.as_matrix(columns=diseases)
        labels = list(labels)

        image_names = chex_df.as_matrix(columns=['Path']).flatten()
        image_names = [os.path.join(data_dir, im_name) for im_name in image_names]

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = torch.FloatTensor(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_names)
