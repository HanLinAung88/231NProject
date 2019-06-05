#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


writer = SummaryWriter('./logs')


# In[3]:


"""
Read images and corresponding labels.
"""
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
        self.convert_to_ones(chex_df, 'Consolidation')
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


# In[4]:


class ChexnetTrainer():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    #--- classes - is the number of classes to predict (Note =/= final layer of Dense Net) -- Saj
    
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint,classes):
        #------------------  Special Loss 
        # Takes in Logits, except 0,1,2 --> logits => sigmoid
        # returns multi label loss
        def lossCriterion(varOutput,varTarget):
            CEloss =  torch.nn.CrossEntropyLoss()
            BCEloss = torch.nn.BCELoss()

            L1 = BCEloss(varOutput[:,0],varTarget[:,0]) 
            L2 = BCEloss(varOutput[:,1],varTarget[:,1])
            L3 = BCEloss(varOutput[:,2],varTarget[:,2])
            varTarget = varTarget.long()
            L4 = CEloss(varOutput[:,3:6],varTarget[:,3])
            L5 = CEloss(varOutput[:,6:9],varTarget[:,4])

            
            lossvalue = (L1 + L2 + L3 + L4 + L5)/5
            
            return lossvalue
        
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'RES-NET-18': model = ResNet18(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
       
        #-------------------- SETTINGS: DATA TRANSFORMS |TRAIN|
        normalize = transforms.Normalize([0.50616586, 0.50616586, 0.50616586], [0.2879059, 0.2879059, 0.2879059]) #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)    
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDER |TRAIN|
                    
        datasetTrain = ChestXrayDataSet(data_dir=pathDirData,image_list_file=pathFileTrain, transform=transformSequence)              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=0, pin_memory=False)
        
        
        
        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS |VAL|

        
        #-------------------- SETTINGS: DATASET BUILDERS |VAL|
        datasetVal =   ChestXrayDataSet(data_dir=pathDirData, image_list_file=pathFileVal, transform=transformSequence)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=0, pin_memory=False)
        
        
        
        
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        #-------------------- SETTINGS: LOSS
        loss = lossCriterion
       
        counter = 0
        
	#---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            counter = modelCheckpoint['counter']
        
        #---- TRAIN THE NETWORK
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            lossTrain, counter = ChexnetTrainer.epochTrain (model, dataLoaderTrain, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, counter,classes)
            lossVal, losstensor, __ = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, counter,classes)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            scheduler.step(losstensor.item())
            writer.add_scalar('logs/train_loss_epoch', lossTrain, epochID)
            writer.add_scalar('logs/val_loss_epoch', lossVal, epochID)
            if lossVal < lossMIN:

                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, './forward/m-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, dataLoaderVal, optimizer, scheduler, epochMax, classCount, loss, counter,classes):
        
        model.train()
        lossTrain = 0
        lossTrainNorm = 0
        
        avg_loss = 0.0

        for batchID, (input, target) in enumerate (dataLoader):

            target = target.cuda()
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)


            varOutput[:,0] = torch.sigmoid(varOutput[:,0])
            varOutput[:,1] = torch.sigmoid(varOutput[:,1])
            varOutput[:,2] = torch.sigmoid(varOutput[:,2])

            lossvalue = loss(varOutput,varTarget)

            avg_loss = avg_loss * (batchID)/(batchID+1) + lossvalue * 1.0/(batchID+ 1)
            lossTrain += lossvalue
            lossTrainNorm += 1

            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            writer.add_scalar('logs/train_loss', avg_loss, counter)
            if batchID % 200 == 0:
                ChexnetTrainer.epochVal(model, dataLoaderVal, optimizer, scheduler, epochMax, classCount, loss, counter,classes)
                print('Loss:' + str(avg_loss.item()))
            if batchID % 2400 == 0:
                __, __, aurocMean = ChexnetTrainer.epochVal(model, dataLoaderVal, optimizer, scheduler, epochMax, classCount, loss, counter,classes)
                torch.save({'counter' : counter, 'state_dict': model.state_dict(), 'valAUROC' : aurocMean , 'optimizer' : optimizer.state_dict()}, './forward/m-' + str(counter) + '_' + str(round(aurocMean, 3)) + '.pth.tar')

                
#             print(counter)
            counter += 1

        outLoss = lossTrain/lossTrainNorm
        return outLoss, counter

                        
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, counter,classes):
        
        print('epoc val')
        model.eval()
        
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoader):
                #Val code
                target = target.cuda()
                varInput = torch.autograd.Variable(input.cuda())
                varTarget = torch.autograd.Variable(target)
                varOutput = model(varInput)

                varOutput[:,0] = torch.sigmoid(varOutput[:,0])
                varOutput[:,1] = torch.sigmoid(varOutput[:,1])
                varOutput[:,2] = torch.sigmoid(varOutput[:,2])            


                ### VAL Preds for AUROC
                bPRED = torch.zeros(varOutput.shape[0], 5).cuda()
                bPRED[:,0] = varOutput[:,0]
                bPRED[:,1] = varOutput[:,1]
                bPRED[:,2] = varOutput[:,2]
                
                soft_a = torch.nn.functional.softmax(varOutput[:,3:6], dim=-1).data

                a0, a1, a2 = soft_a[:, 0], soft_a[:, 1], soft_a[:, 2]
                bPRED[:, 3] = a1/(a0+a1)
                soft_b = torch.nn.functional.softmax(varOutput[:,6:9], dim=-1).data
                b0, b1, b2 = soft_b[:, 0], soft_b[:, 1], soft_b[:, 2]
                bPRED[:, 4] = b1/(b0+b1)

                outPRED = torch.cat((outPRED, bPRED.data), 0)            
                outGT = torch.cat((outGT, target), 0)


                losstensor = loss(varOutput,varTarget)

                losstensorMean += losstensor
                lossVal += losstensor.item()
                lossValNorm += 1
                ##block comment was here

            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm

            aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, classes)
            aurocMean = np.array(aurocIndividual).mean()

            print("AUROC val", aurocMean)
            print("AUROC all", aurocIndividual)
            writer.add_scalar('logs/val_auroc', aurocMean, counter)

        return outLoss, losstensorMean, aurocMean            
#             outGT = torch.cat((outGT, target), 0)
# outPred = empty variable


##varMean to varOutput
#             outPRED = torch.zeros(out.shape[0], 5).cuda()
#             outPRED[:,0] = outMean[:,0]
#             outPRED[:,1] = outMean[:,1]
#             outPRED[:,2] = outMean[:,2]
#             outPRED[:,3] = torch.max(outMean[:,3:6],1)[0]
#             outPRED[:,4] = torch.max(outMean[:,6:9],1)[0]

# #             outPRED = torch.cat((outPRED, outMean.data), 0)





            
            
############            
#             outGT = torch.cat((outGT, target), 0)
            
#             bs, c, h, w = input.size()

#             varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda(), volatile=True)
            
#             out = model(varInput)
#             outMean = out.view(bs, -1)
    
#             outPRED = torch.zeros(out.shape[0], 5).cuda()
#             outPRED[:,0] = outMean[:,0]
#             outPRED[:,1] = outMean[:,1]
#             outPRED[:,2] = outMean[:,2]
#             outPRED[:,3] = torch.max(outMean[:,3:6],1)[0]
#             outPRED[:,4] = torch.max(outMean[:,6:9],1)[0]
            
            
# #             outPRED = torch.cat((outPRED, outMean.data), 0)
            
#             varOutput = outPRED
#             varTarget = outGT
            
# #             losstensor = loss(varOutput, varTarget)

#             CEloss =  torch.nn.CrossEntropyLoss()
#             BCEloss = torch.nn.BCELoss()

# #             varTarget = varTarget.type(torch.long)
#             L1 = BCEloss(varOutput[:,:1],varTarget[:,0]) 
#             L2 = BCEloss(varOutput[:,1:2],varTarget[:,1])
#             L3 = BCEloss(varOutput[:,2:3],varTarget[:,2])
#             varTarget = varTarget.long()
#             L4 = CEloss(varOutput[:,3:6],varTarget[:,3])
#             L5 = CEloss(varOutput[:,6:9],varTarget[:,4])

            
#             losstensor = L1 + L2 + L3 + L4 + L5
#             losstensor /= 5


#             losstensorMean += losstensor
#             lossVal += losstensor.item()
#             lossValNorm += 1
            

               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=0, shuffle=False, pin_memory=False)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (input, target) in enumerate(dataLoaderTest):
            
            target = target.cuda()
            outGT = torch.cat((outGT, target), 0)
            
            bs, n_crops, c, h, w = input.size()
            
            varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
            
            out = model(varInput)
            outMean = out.view(bs, n_crops, -1).mean(1)
            
            outPRED = torch.cat((outPRED, outMean.data), 0)

        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return
#-------------------------------------------------------------------------------- 


# In[ ]:


DATA_DIR = './data'
TRAIN_IMAGE_LIST = './data/CheXpert-v1.0-small/train.csv'
VAL_IMAGE_LIST = './data/CheXpert-v1.0-small/valid.csv'
valid_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                image_list_file=VAL_IMAGE_LIST)

nnIsTrained = True
nnArchitecture = 'DENSE-NET-121'

nnClassCount = 9
classes = 5

trBatchSize = 32
trMaxEpoch = 50
transResize = (300, 300)
transCrop = 224
launchTimestamp = ''
checkpoint = 'forward/m-8370_0.771.pth.tar'
ChexnetTrainer.train(DATA_DIR,TRAIN_IMAGE_LIST,VAL_IMAGE_LIST,nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint,classes)


# In[ ]:




