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

def computeAUROC (dataGT, dataPRED, classCount):

    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC

def computeClassMetrics(dataGT, dataPRED, classCount):
    classification_metrics = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        pred_to_category = datanpPRED[:, i].copy()
        pred_to_category[pred_to_category < 0.5]  = 0
        pred_to_category[pred_to_category != 0]  = 1
        classification_metrics.append(classification_report(datanpGT[:, i], pred_to_category))
    return classification_metrics

def computeAcc(dataGT, dataPRED, classCount):
    acc = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        pred_to_category = datanpPRED[:, i].copy()
        pred_to_category[pred_to_category < 0.5]  = 0
        pred_to_category[pred_to_category != 0]  = 1
        acc.append(accuracy_score(datanpGT[:, i], pred_to_category))
    return acc
