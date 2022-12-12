import yaml
import pandas as pd
import torchvision
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import mlflow 

import sys
sys.path.append('src/')
from data.CustomImageDataset import CustomImageDataset
from utils.mlflow_run_decorator import mlflow_run



# Define lightning dataset

class LightDataset(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

        self.train_T=T.Compose([
            T.AutoAugment(data_aug_policy),
            T.ToPILImage(),
            T.Resize(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
            ])

        self.val_T=T.Compose([
            T.ToPILImage(),
            T.Resize(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=mean_std['mean'], std=mean_std['std'])
            ])

    def train_dataloader(self):
        train_dataset = CustomImageDataset(annotations_file='./data/processed/train.csv', transform=self.train_T)
        train_loader = DataLoader(train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True, num_workers=8)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = CustomImageDataset(annotations_file='./data/processed/val.csv', transform=self.val_T)
        valid_loader = DataLoader(val_dataset,
                            batch_size=self.batch_size,
                            shuffle=False, num_workers=8)       
        return valid_loader
    
    def test_dataloader(self):
        test_dataset = CustomImageDataset(annotations_file='./data/processed/test.csv', transform=self.val_T)
        test_loader = DataLoader(test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False, num_workers=8)       
        return test_loader


# Define the LightningModule for the network

class mobilenetv3_fe(pl.LightningModule):

    def __init__(self, n_classes, learning_rate):
        super().__init__()

        # Parameters
        self.lr = learning_rate
        self.num_classes = n_classes

        # Init a pretrained mobilenetv3 model
        
        self.model = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1', pretrained=True, progress=True)
        
        # Feature extraction : freeze the network
        for param in self.model.parameters():
                param.requires_grad = False
        
        # Replacing last layer with new layer (not frozen)
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, self.num_classes)


    def forward(self, x):
        out = self.model(x)
        return out
    
    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out.view(-1, self.num_classes), target)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch
        label = labels.view(-1)
        img = images.view(-1, 3, images.size(2), images.size(2))
        out = self(img)
        loss = self.loss_fn(out, label)

        logits = nn.Softmax(-1)(out) 
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, label, task='multiclass', num_classes=self.num_classes)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def evaluate(self, batch, stage=None):
        images, labels = batch
        label = labels.view(-1)
        img = images.view(-1, 3, images.size(2), images.size(2))
        out = self(img)
        loss = self.loss_fn(out, label)
        logits = nn.Softmax(-1)(out) 
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, label, task='multiclass', num_classes=self.num_classes)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
        
    def validation_step(self,batch,batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        label = labels.view(-1)
        img = images.view(-1, 3, images.size(2), images.size(2))
        out = self(img)
        out = nn.Softmax(-1)(out) 
        return torch.argmax(out,dim=1)

@mlflow_run
def train_model():

    mlflow.pytorch.autolog()
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("batch_size", BATCH_SIZE)
    
    model = mobilenetv3_fe(learning_rate=LR, n_classes=num_classes)
    dx = LightDataset(batch_size=BATCH_SIZE)

    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_acc",
    min_delta=0.00, patience=3, verbose=False, mode="max")

    # Initialize a trainer
    trainer = pl.Trainer(accelerator='gpu', devices=-1,
    max_epochs=30, callbacks=[early_stop_callback])

    trainer.fit(model, dx)
    trainer.test(model, dx)


if __name__ == '__main__':
    # Parameters
    # Data aug
    params = yaml.safe_load(open('./params.yaml'))
    image_size = (params['prepare']['image_size'], params['prepare']['image_size'])
    policies = [T.AutoAugmentPolicy.CIFAR10, T.AutoAugmentPolicy.IMAGENET, T.AutoAugmentPolicy.SVHN]
    data_aug_policy = policies[params['train']['data_aug_policy']]


    BATCH_SIZE = params['train']['batch_size']
    LR = params['train']['learning_rate']
    label_map = yaml.safe_load(open('./data/processed/labels_map.yaml'))
    num_classes = len(label_map)

    # Image standardization
    mean_std = yaml.safe_load(open('./data/processed/image_mean_std.yaml'))


    train_model()