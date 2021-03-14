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

import signal

from contextlib import contextmanager

import requests


DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}


def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler


@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import active session

    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)


def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:

    from workspace_utils import keep_awake

    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable

#fonction pour charger les données

def loading_data(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'        

# DONE: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# DONE: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = training_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
# DONE: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_loader, valid_loader, test_loader, train_data

    
#choose the model of the network
def nn_model(architecture ='vgg19', lr = 0.001, hidden_units = 4096, epochs =4, mode = 'gpu'):
    if architecture =='vgg19':
        model = models.vgg19(pretrained=True)
        input_layer = 25088
    elif architecture == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_layer = 1024
    else : 
        print('This is not a valid model. Choose between vgg19 or resnet18')


    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_layer, hidden_units)),
                          ('relu', nn.ReLU()),  
                          ('fc2', nn.Linear(hidden_units,102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier

    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and mode == 'gpu':
        model.cuda()
    
    return model, criterion, optimizer, classifier


#function to train the model
def train_model (model, criterion, optimizer, train_data, train_loader, test_loader, valid_loader,epochs = 4):
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                #with torch.no_grad():
                for inputs, labels in test_loader :
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
                running_loss = 0
                model.train()
                
    
#function to save the training of the network
def save_checkpoint(filepath, model,optimizer, classifier, train_data, architecture ='vgg19',hidden_units = 25088, lr = 0.001, epochs = 4):
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    checkpoint = {
              'hidden_units': hidden_units,
              'output_size': 102,
              'epochs': epochs,
              'batch_size': 64,
              'learning rate': lr,
              'model': model,
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
             }
   
    torch.save(checkpoint, 'checkpoint.pth')
 
#function to load the checkpoint
def load_checkpoint(filepath = 'checkpoint.pth'):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    hidden_units = checkpoint['hidden_units']
    lr = checkpoint['lr']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

#function to transform an image before introducing it in the network

def process_image(image_path):

    img_pil = Image.open(image)
    
    valid_transform= transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    image = valid_transforms(img_pil)
    
    return image

#function to predict

def predict(image_path, model, topk=5, mode = 'gpu'):
    ''' Predict the class of an image using a trained deep learning model.
    '''   
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0).float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    ps = F.softmax(output, dim = 1)
    
    return ps.topk(topk)