import json
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch import tensor
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse
import workspaceutils

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('--path', action ='store', default = './flowers/')

parser.add_argument('--gpu', dest='gpu', action ='store', default = 'gpu')

parser.add_argument('--learning_rate', dest='learning_rate', action='store',default=0.001,
                    help='input the learning rate')

parser.add_argument('--hidden_units', dest='hidden_units', action='store',default=4096,
                    help='input the number of hidden units')

parser.add_argument('--epochs', dest='epochs', type = int, action='store',default=4,
                    help='input the number of epochs')

parser.add_argument('--architecture', dest='architecture', type = str, action='store',default='vgg19',
                    help='input the model architecture')

parser.add_argument('--filepath', dest ='filepath', action ='store', default ="./checkpoint.pth")

args = parser.parse_args()

path = args.path
mode = args.gpu
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
architecture = args.architecture
filepath = args.filepath
                    
train_loader, valid_loader, test_loader = workspaceutils.loading_data(path)
model, criterion, optimizer, device = workspaceutils.nn_model(architecture, lr, hidden_units, epochs, mode)
                    
workspaceutils.train_model (model, criterion, optimizer, device, epochs)
                    
workspaceutils.save_checkpoint(filepath, architecture, hidden_units, lr, epochs)
                    
print("model trained I hope...")