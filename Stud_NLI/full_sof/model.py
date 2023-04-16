import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import Accuracy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time
from matplotlib import pyplot as plt
import os
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel, AdamW, BertModel
from dataset import StudDataset
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import torchmetrics


class Classifier(LightningModule):
    def __init__(self, config):  # drop_prob를 활용하여 dropout 추가해볼 것.
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        # weight = torch.tensor([0.54743729, 6.00844647])
        weight = torch.tensor([1., 11.])
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        # self.criterion = nn.CrossEntropyLoss()

        self.train_logit_list = []
        self.train_label_list = []
        self.valid_logit_list = []
        self.valid_label_list = []
        self.test_logit_list = []
        self.test_label_list = []

        self.model = AutoModel.from_pretrained("klue/bert-base")

        self.f_mlp = nn.Sequential(
            nn.Linear(768, 2)
        )

    def forward(self, x):
        # model_input: [batch_size, 1952]
        # input_ids, attention_mask : [batch_size,  maxlen(512)]

        pooled_output = self.model(x[0], x[1], x[2]).pooler_output  # [batch_size, 768]
        return self.f_mlp(pooled_output)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        sch = self.lr_schedulers()
        sch.step()

    def training_step(self, train_batch, batch_idx):
        out = self.forward(
            [train_batch['input_ids'], train_batch['token_type_ids'], train_batch['attention_mask']])

        loss = self.criterion(out, train_batch['label'])
        output = self.train_metrics(torch.softmax(out, dim=1), train_batch['label'])

        self.train_logit_list.append(out.detach())
        self.train_label_list.append(train_batch['label'].detach())

        self.log_dict({'train_acc' : output['train_MulticlassAccuracy'],
                       'train_pre' : output['train_MulticlassPrecision'][1],
                       'train_rec' : output['train_MulticlassRecall'][1],
                       'train_f1' : output['train_MulticlassF1Score'][1],
                       'train_loss' : loss.item(),
                       'lr' : self.optimizers().param_groups[0]['lr']},
                       on_step=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        logits = torch.cat(self.train_logit_list, dim=0)
        labels = torch.cat(self.train_label_list, dim=0)
        output = self.train_epoch_metrics(torch.softmax(logits, dim=1), labels)
        self.log_dict({'train_epoch_acc' : output['train_epoch_MulticlassAccuracy'],
                       'train_epoch_pre' : output['train_epoch_MulticlassPrecision'][1],
                       'train_epoch_rec' : output['train_epoch_MulticlassRecall'][1],
                       'train_epoch_f1' : output['train_epoch_MulticlassF1Score'][1]},
                       on_epoch = True, prog_bar = True)
        self.train_logit_list = []
        self.train_label_list = []

    def validation_step(self, val_batch, batch_idx):
        out = self.forward([val_batch['input_ids'], val_batch['token_type_ids'], val_batch['attention_mask']])

        loss = self.criterion(out, val_batch['label'])
        output = self.valid_metrics(torch.softmax(out, dim=1), val_batch['label'])

        self.valid_logit_list.append(out.detach())
        self.valid_label_list.append(val_batch['label'].detach())

        self.log_dict({'val_acc' : output['val_MulticlassAccuracy'],
                       'val_pre' : output['val_MulticlassPrecision'][1],
                       'val_rec' : output['val_MulticlassRecall'][1],
                       'val_f1' : output['val_MulticlassF1Score'][1],
                       'val_loss' : loss.item()},
                       on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        logits = torch.cat(self.valid_logit_list, dim=0)
        labels = torch.cat(self.valid_label_list, dim=0)
        output = self.valid_epoch_metrics(torch.softmax(logits, dim=1), labels)
        self.log_dict({'val_epoch_acc' : output['val_epoch_MulticlassAccuracy'],
                       'val_epoch_pre' : output['val_epoch_MulticlassPrecision'][1],
                       'val_epoch_rec' : output['val_epoch_MulticlassRecall'][1],
                       'val_epoch_f1' : output['val_epoch_MulticlassF1Score'][1]},
                       on_epoch = True, prog_bar = True)

        self.valid_logit_list = []
        self.valid_label_list = []

    def test_step(self, test_batch, batch_idx):
        out = self.forward([test_batch['input_ids'], test_batch['token_type_ids'], test_batch['attention_mask']])
        loss = self.criterion(out, test_batch['label'])
        # output = self.test_metrics(torch.softmax(out, dim=1), test_batch['label'])

        self.test_logit_list.append(out.detach())
        self.test_label_list.append(test_batch['label'].detach())

        return loss

    def on_test_epoch_end(self):
        logits = torch.cat(self.test_logit_list, dim=0)
        labels = torch.cat(self.test_label_list, dim=0)
        output = self.test_metrics(torch.softmax(logits, dim=1), labels)
        self.log_dict({'test_acc' : output['test_MulticlassAccuracy'],
                       'test_pre' : output['test_MulticlassPrecision'][1],
                       'test_rec' : output['test_MulticlassRecall'][1],
                       'test_f1' : output['test_MulticlassF1Score'][1]},
                       on_epoch = True, prog_bar = True)
        self.test_logit_list = []
        self.test_label_list = []

    def prepare_data(self):
        self.train_dataset = StudDataset('../train.tsv', self.config['prompt'])
        self.val_dataset = StudDataset('../val.tsv', self.config['prompt'])
        self.test_dataset = StudDataset('../test.tsv', self.config['prompt'])

        metrics = torchmetrics.MetricCollection([
            Accuracy(task='multiclass', num_classes=2, average='micro'),
            Precision(task='multiclass', num_classes=2, average=None),
            Recall(task='multiclass', num_classes=2, average=None),
            F1Score(task='multiclass', num_classes=2, average=None)
        ])

        # metrics = torchmetrics.MetricCollection([
        #     Precision(task='binary', threshold=0.5),
        #     Recall(task='binary', threshold=0.5),
        #     F1Score(task='binary', threshold=0.5)
        # ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.train_epoch_metrics = metrics.clone(prefix='train_epoch_')
        self.valid_epoch_metrics = metrics.clone(prefix='val_epoch_')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"],
                          num_workers=self.config["num_workers"], collate_fn=self.train_dataset.custom_collate,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"],
                          num_workers=self.config["num_workers"], collate_fn=self.val_dataset.custom_collate,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config["batch_size"],
                          num_workers=self.config["num_workers"], collate_fn=self.test_dataset.custom_collate,
                          shuffle=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["learning_rate"],
            epochs=self.config['epoch'],
            steps_per_epoch=len(self.train_dataloader()) // self.config['accumulate'],
            anneal_strategy='linear',
            pct_start=0.1
        )


        return {"optimizer": optimizer, "lr_scheduler": scheduler}
