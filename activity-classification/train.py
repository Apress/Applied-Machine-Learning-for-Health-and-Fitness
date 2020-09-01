
import argparse
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import tempfile
from azureml.core import Dataset, Run
import azureml.contrib.dataset
from azureml.contrib.dataset import FileHandlingOption, LabeledDatasetTask

def load(f, size = .2):
    
    t = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225])])
        
    train = datasets.ImageFolder(f, transform=t)
    test = datasets.ImageFolder(f, transform=t)
    n = len(train)
    indices = list(range(n))
    split = int(np.floor(size * n))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train,sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test, sampler=test_sampler, batch_size=64)
    return trainloader, testloader

def get_mounting_path(labeled_dataset):
    
    mounted_path = tempfile.mkdtemp()
    mount_context = labeled_dataset.mount(mounted_path)
    mount_context.start()
    print(os.listdir(mounted_path))
    print (mounted_path)
    print(os.listdir(mounted_path+'/workspaceblobstore'))
    return mounted_path + '/workspaceblobstore/activities'

def start(output_folder, model_file_name):
    
    run = Run.get_context()
    labeled_dataset = run.input_datasets['activities']
    
    data_path =  get_mounting_path(labeled_dataset)

    trainloader, testloader = load(data_path, .2)
    print(trainloader.dataset.classes)
    images, labels = next(iter(trainloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False  

    features = model.fc.in_features
    model.fc = nn.Linear(features, len(labels))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # train the model
    print_every = 100
    train_losses, test_losses = [], []
    total_loss = 0
    i = 0
    epochs=3
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            i += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(total_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {total_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
    
    print('Finished training')
    os.makedirs(output_folder, exist_ok=True)
    torch.save(model, os.path.join(output_folder, model_file_name))
    print('Model saved:', model_file_name)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", default=None, type=str, dest='output_folder', required=True, help="Output folder for the model")    
    parser.add_argument("--model-file", default=None, type=str, dest='model_file_name', required=True, help="Output model file")
    args = parser.parse_args()
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
    output_folder = args.output_folder
    model_file_name = args.model_file_name
    print('Output folder:', output_folder)
    print('Model file:', model_file_name)
    start(output_folder, model_file_name)
    
