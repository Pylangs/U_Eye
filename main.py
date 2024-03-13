import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import argparse
import warnings
warnings.filterwarnings("ignore")

# Setup (Jupyter에서 하시려면, 변수명으로 지정하고 사용하시면 됩니다.)
def get_args_parser():
    parser = argparse.ArgumentParser("U-Eye Blue Semester Project", add_help = False)
    # SetUp
    parser.add_argument("--seed", type = int, required = False, default = 0)
    parser.add_argument("--gpu_ids", type = str)
    parser.add_argument("--img_size", type = int, default = 224)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--learing_rate", type = float, default = 1e-3, required = False)
    parser.add_argument("--chkpt", action="store_true")
    parser.add_argument("--chkpt_path", default = "./chkpt/", required = False)
    parser.add_argument("--store_chkpt", default = "./chkpt/", required = False)
    
    # Train
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type = int, default= 30, required = False)
    
    # Test
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return args

def get_model(args):
    model = models.resnet50(pretrained = True)
    num_fts = model.fc.in_features
    model.fc = nn.Linear(num_fts, 16)
    return model

def get_optimizer(args, model):
    optimizer = optim.Adam(model.parameters(), lr = args.learing_rate)
    return optimizer

def get_loss_fn(args):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn

def get_transform(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
        normalize,    
    ])
    
    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size)),
        normalize,
    ])
    return train_transforms, valid_transforms

def train(args, model, optimizer, loss_fn, train_dl):
    for _ in range(args.epochs):
        running_loss = 0
        for data in train_dl:
            input, label = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            outputs = model(input)
            loss = loss_fn(outputs, label)
            running_loss += loss
            loss.backward()
            optimizer.step() 
        print("Training : ", running_loss)
    return model

def infer(args, model, loss_fn, val_dl):
        accuracy = Accuracy(task="multiclass", num_classes=16)
        running_loss = 0
        pred_list, label_list = torch.Tensor([]), torch.Tensor([])
        model.eval()
        for data in val_dl:
            input, label = data[0].to(args.device), data[1].to(args.device)
            outputs = model(input)
            _, pred = torch.max(outputs.data, 1)
            pred_list = torch.cat([pred_list.cpu(), pred.cpu()], dim = 0)
            label_list = torch.cat([label_list.cpu(), label.cpu()], dim = 0)
            loss = loss_fn(outputs, label)
            running_loss += loss
        acc = accuracy(pred_list, label_list)    
        
        print("Eval Loss : ", running_loss)
        print("Eval Acc : ", acc * 100)
        
def get_dataloader(args, train_trans, valid_trans):
    train_ds = ImageFolder(root = "/home/mlmlab07/wongi/ueye/dataset/gaze/train",
                          transform = train_trans)
    test_ds = ImageFolder(root = "/home/mlmlab07/wongi/ueye/dataset/gaze/test",
                          transform = valid_trans)
    train_dl = DataLoader(train_ds, batch_size = args.batch_size, num_workers = 4, shuffle = True)
    test_dl = DataLoader(test_ds, batch_size = args.batch_size, num_workers = 4, shuffle = False)

    return train_dl, test_dl
    
if __name__ == "__main__":
    args = get_args_parser()
    train_trans, valid_trans = get_transform(args)
    model = get_model(args)
    model = model.to(args.device)
    loss_fn = get_loss_fn(args)
    train_dl, test_dl = get_dataloader(args, train_trans, valid_trans)
    
    if args.chkpt_path:
        # model.load_state_dict(torch.load("", map_location = args.device))
        pass
        
    if args.train:
    
        optimizer = get_optimizer(args, model)
        train(args, model, optimizer, loss_fn, train_dl)
        if args.chkpt:
            torch.save(model.state_dict(), os.path.join() + ".pt")
            torch.save(model.module.state_dict(), os.path.join()+ ".pt")
            
        if args.test:
            with torch.no_grad():
                infer(args, model, loss_fn, test_dl)
    if args.test:
        with torch.no_grad():
            infer(args, model, loss_fn, test_dl)