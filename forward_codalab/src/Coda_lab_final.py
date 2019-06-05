#!/usr/bin/env python
# coding: utf-8
import torch
import os
import torchvision.transforms as transforms
import numpy as np
import time
import sys
import csv

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
from PIL import Image
import PIL
import torch.nn.functional as func

from torch.utils.data import Dataset
from models.chexnet.DensenetModels import DenseNet121, DenseNet169


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


def computeAUROC (dataGT, dataPRED, classCount):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC


normalize = transforms.Normalize([0.50616586, 0.50616586, 0.50616586], [0.2879059, 0.2879059, 0.2879059]) 
        
transformList = []
transResize = (300, 300)
transformList.append(transforms.Resize(transResize))
transformList.append(transforms.ToTensor())
transformList.append(normalize)    
transform = transforms.Compose(transformList)

def load_and_resize_img(path):
    """
    Load and convert the full resolution images on CodaLab to
    low resolution used in the small dataset.
    """    
  #  img = cv2.imread(path, 0) 
    img = cv2.imread(path, 0)

    img_2 = Image.open(path)
    max_ind_2 = max(img_2.size)
    print(img_2.size)
    print(img.shape)
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)
    
    if max_ind == 1:
        # width fixed at 320
        wpercent = (320 / float(size[0]))
        hsize = int((size[1] * wpercent))
        new_size = (hsize, 320)
        
    else:
        # height fixed at 320
        hpercent = (320 / float(size[1]))
        wsize = int((size[0] * hpercent))
        new_size = (320, wsize)


    if max_ind_2 == 1:
        # width fixed at 320
        wpercent = (320 / float(size[0]))
        hsize = int((size[1] * wpercent))
        new_size_2 = (hsize, 320)
        
    else:
        # height fixed at 320
        hpercent = (320 / float(size[1]))
        wsize = int((size[0] * hpercent))
        new_size_2 = (320, wsize)

    resized_img = Image.fromarray(cv2.resize(img, new_size).astype('uint8'))
    resized_img_2 =  img_2.resize(new_size_2, PIL.Image.BILINEAR)
    print(resized_img.size)
    print(resized_img_2.size)

    return resized_img


def both_views(front_image_name, side_image_name):
    side_image = load_and_resize_img(side_image_name).convert('RGB')
    front_image =  load_and_resize_img(front_image_name).convert('RGB')
    
#    side_image = Image.fromarray(side_image_np.astype('uint8'), 'RGB') 
#    front_image = Image.fromarray(front_image_np.astype('uint8'), 'RGB') 
        
    if transform is not None:
        side_image = transform(side_image)
        front_image = transform(front_image)
    return front_image, side_image

def get_images(image_name):
    """
    Args:
        index: the index of item
    Returns:
        image
    """
    if 'frontal' in image_name:
        front_image_name = image_name
        side_image_name = image_name.replace('1_frontal', '2_lateral')
        if os.path.isfile(side_image_name):
            return both_views(front_image_name, side_image_name)

    if 'lateral' in image_name:
        side_image_name = image_name
        front_image_name = side_image_name.replace('2_lateral', '1_frontal')
        if os.path.isfile(front_image_name):
            return both_views(front_image_name, side_image_name)
        
 #   image_np = cv2.cvtColor(load_and_resize_img(image_name) , cv2.COLOR_GRAY2RGB) 
    image = load_and_resize_img(image_name).convert('RGB')
#    image = Image.fromarray(image_np.astype('uint8'), 'RGB') 
    if transform is not None:
        image = transform(image)
    return image, torch.ones((1,1,1))

def extract_study_path(path):
    paths = path.split('/')
    return '/'.join(paths[:-1])


class forward_side(nn.Module):
    def __init__(self, nnClassCount, forward_cp, side_cp):
        super(forward_side, self).__init__()
        self.forwardModel = DenseNet121(nnClassCount, False)
        self.sideModel = DenseNet121(nnClassCount, False)
        
        Fstate_dict = self.change_state_dict_keys(torch.load(forward_cp, map_location='cpu')['state_dict'])
        Cstate_dict = self.change_state_dict_keys(torch.load(side_cp, map_location='cpu')['state_dict'])
        self.forwardModel.load_state_dict(Fstate_dict)
        self.sideModel.load_state_dict(Cstate_dict)
        
        
        for param in self.forwardModel.parameters():
            param.requires_grad = False
        for param in self.sideModel.parameters():
            param.requires_grad = False
        
        self.FkernelCount = self.forwardModel.densenet121.classifier.in_features
        self.SkernelCount = self.sideModel.densenet121.classifier.in_features
        self.forwardModel.densenet121.classifier = nn.Identity()
        self.sideModel.densenet121.classifier = nn.Identity()
        
        self.fc1 = nn.Linear(self.FkernelCount+self.SkernelCount,500)
        self.fc2 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100,nnClassCount)
        
    def change_state_dict_keys(self,state_dict):
        keys = state_dict.keys()
        new_state_dict = {}
        for key in keys:
            clean_k = '.'.join(key.split('.')[1:]) # module.dense121.conv0.weight -> dense121.conv0.weight
            new_state_dict[clean_k] = state_dict[key]
        return new_state_dict
        
    def forward(self, xF,xS):
        xF = self.forwardModel(xF)
        xS = self.sideModel(xS)
        x = torch.cat((xF,xS),-1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def front_model_zoo():
    classes = 9
    
    model_zoo = []
    model = None
    model = DenseNet121(classes, False)#.cuda()
    model = torch.nn.DataParallel(model)#.cuda()
    checkpoint = torch.load('src/model_weights/m-37050_0-Copy1.897.pth.tar',  map_location='cpu' )
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model_zoo.append(model)

    model = None
    model = DenseNet121(classes, False)#.cuda()
    model = torch.nn.DataParallel(model)#.cuda()
    checkpoint = torch.load('src/model_weights/m-8370_0.893.pth.tar',  map_location='cpu' )
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model_zoo.append(model)

    model = None
    model = DenseNet169(classes, False)#.cuda()
    model = torch.nn.DataParallel(model)#.cuda()
    checkpoint = torch.load('src/model_weights/m-26280_0.892.pth.tar',  map_location='cpu' )
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model_zoo.append(model)

    model = None
    model = DenseNet169(classes, False)#.cuda()
    model = torch.nn.DataParallel(model)#.cuda()
    checkpoint = torch.load('src/model_weights/m-8370_0.887.pth.tar',  map_location='cpu' )
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model_zoo.append(model)
    return model_zoo

def hybrid_model_zoo():
    classes = 9
    model_zoo = []
    
    for f in os.listdir('src/model_weights/hybrid_zoo'):
        dud = 'src/model_weights/m-37050_0-Copy1.897.pth.tar'
        model = forward_side(classes, dud, dud)
        model = torch.nn.DataParallel(model)#.cuda()
        checkpoint_path = 'src/model_weights/hybrid_zoo/' + f
        checkpoint = torch.load(checkpoint_path,  map_location='cpu' )
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        model_zoo.append(model)
    return model_zoo
    
def to_probs(varOutput):
    varOutput[:,0] = torch.sigmoid(varOutput[:,0])
    varOutput[:,1] = torch.sigmoid(varOutput[:,1])
    varOutput[:,2] = torch.sigmoid(varOutput[:,2])            

    ### VAL Preds for AUROC
    bPRED = torch.zeros(varOutput.shape[0], 5)#.cuda()
    bPRED[:,0] = varOutput[:,0]
    bPRED[:,1] = varOutput[:,1]
    bPRED[:,2] = varOutput[:,2]

    soft_a = torch.nn.functional.softmax(varOutput[:,3:6], dim=-1).data

    a0, a1, a2 = soft_a[:, 0], soft_a[:, 1], soft_a[:, 2]
    bPRED[:, 3] = a1/(a0+a1)
    soft_b = torch.nn.functional.softmax(varOutput[:,6:9], dim=-1).data
    b0, b1, b2 = soft_b[:, 0], soft_b[:, 1], soft_b[:, 2]
    bPRED[:, 4] = b1/(b0+b1)
    
    return bPRED

def predict_front(front_image):
    model_zoo = front_model_zoo()
    for model in model_zoo:
        model.eval()
    varInput = torch.autograd.Variable(front_image)
    varInput = torch.unsqueeze(varInput, 0)
    predictions = []
    for model in model_zoo:
        probs = to_probs(model(varInput))
        predictions.append(probs.cpu().numpy().squeeze())
    
    predictions = np.array(predictions)
    del model_zoo
    
    return np.mean(predictions, axis=0)


def predict_hybrid(front_image, side_image):
    model_zoo = hybrid_model_zoo()
    for model in model_zoo:
        model.eval()
    varFrontInput = torch.autograd.Variable(front_image)
    varFrontInput = torch.unsqueeze(varFrontInput, 0)
    
    varSideInput = torch.autograd.Variable(side_image)
    varSideInput = torch.unsqueeze(varSideInput, 0)
    predictions = []
    for model in model_zoo:
        probs = to_probs(model(varFrontInput, varSideInput))
        predictions.append(probs.cpu().numpy().squeeze())
    
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0)    


def predict(im_name):
    (front_image, side_image) = get_images(im_name)
    c, h, w = side_image.shape

    if w == 1 and c == 1 and h == 1:
        # only front
        return predict_front(front_image)
    else:
        #return predict_hybrid(front_image, side_image)
        return predict_front(front_image)
    

def get_labels_dict(infile='valid_image_paths.csv'):
    study_path_to_labels = {}
    with torch.no_grad():
        with open (infile, 'r') as f:
            chex_df = csv.reader(f)
            next(chex_df, None)
            list_image_names = []
            for row in chex_df:
                list_image_names.append(row[0])
#        chex_df = pd.read_csv(infile)
#        list_image_names = list(chex_df['Path'])
        study_path_set = []
        for im_name in list_image_names:
            study_path = extract_study_path(im_name)
            if study_path in study_path_set:
                continue
            study_path_set.append(study_path)
            labels = predict(im_name)
            study_path_to_labels[study_path] = labels
    return study_path_to_labels

infile = sys.argv[1]  #sys.argv[1]
outfile = sys.argv[2]  #sys.argv[2]
labels_dict = get_labels_dict(infile)

fieldnames = ['Study', 'Atelectasis', 'Consolidation', 'Edema','Cardiomegaly', 'Pleural Effusion']
with open(outfile, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    for study in labels_dict:
        output_row = list(labels_dict[study])
        output_row.insert(0, study)
        writer.writerow(output_row)
