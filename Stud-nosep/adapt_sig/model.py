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
from transformers import AutoTokenizer, AutoModel, AdamW, BertConfig, BertModelWithHeads
from dataset import StudDataset
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import torchmetrics
from transformers.adapters import AdapterConfig

class Classifier(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        weight = torch.tensor([11.])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

        self.train_logit_list = []
        self.train_label_list = []
        self.valid_logit_list = []
        self.valid_label_list = []
        self.test_logit_list = []
        self.test_label_list = []

        bertconfig = BertConfig.from_pretrained(
            "klue/bert-base",
            num_labels=2,
        )
        self.model = BertModelWithHeads.from_pretrained(
            "klue/bert-base",
            config=bertconfig,
        )
        adapterconfig = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=self.config['reduction'], non_linearity="relu")

        self.model.add_adapter("dropout", config=adapterconfig)
        self.model.add_classification_head(
            "dropout",
            num_labels=1
        )
        self.model.train_adapter("dropout")

    def forward(self, x):
        # model_input: [batch_size, 1952]
        # input_ids, attention_mask : [batch_size,  maxlen(512)]

        out = self.model(x[0], x[1], x[2]).logits  # [batch_size, 768]
        return out

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        sch = self.lr_schedulers()
        sch.step()

    def training_step(self, train_batch, batch_idx):
        out = self.forward(
            [train_batch['input_ids'], train_batch['token_type_ids'], train_batch['attention_mask']])

        loss = self.criterion(out, train_batch['label'].unsqueeze(1).float())
        output = self.train_metrics(torch.sigmoid(out), train_batch['label'].unsqueeze(1).float())

        self.train_logit_list.append(out.detach())
        self.train_label_list.append(train_batch['label'].detach())

        # sch = self.lr_schedulers()
        # sch.step()

        self.log_dict({'train_acc' : output['train_BinaryAccuracy'],
                       'train_pre' : output['train_BinaryPrecision'],
                       'train_rec' : output['train_BinaryRecall'],
                       'train_f1' : output['train_BinaryF1Score'],
                       'train_loss' : loss.item(),
                       'lr' : self.optimizers().param_groups[0]['lr']},
                       on_step=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        logits = torch.cat(self.train_logit_list, dim=0)
        labels = torch.cat(self.train_label_list, dim=0)
        output = self.train_epoch_metrics(torch.sigmoid(logits), labels.unsqueeze(1).float())
        self.log_dict({'train_epoch_acc' : output['train_epoch_BinaryAccuracy'],
                       'train_epoch_pre' : output['train_epoch_BinaryPrecision'],
                       'train_epoch_rec' : output['train_epoch_BinaryRecall'],
                       'train_epoch_f1' : output['train_epoch_BinaryF1Score']},
                       on_epoch = True, prog_bar = True)
        self.train_logit_list = []
        self.train_label_list = []

    def validation_step(self, val_batch, batch_idx):
        out = self.forward([val_batch['input_ids'], val_batch['token_type_ids'], val_batch['attention_mask']])

        loss = self.criterion(out, val_batch['label'].unsqueeze(1).float())
        output = self.valid_metrics(torch.sigmoid(out), val_batch['label'].unsqueeze(1).float())

        self.valid_logit_list.append(out.detach())
        self.valid_label_list.append(val_batch['label'].detach())

        self.log_dict({'val_acc' : output['val_BinaryAccuracy'],
                       'val_pre' : output['val_BinaryPrecision'],
                       'val_rec' : output['val_BinaryRecall'],
                       'val_f1' : output['val_BinaryF1Score'],
                       'val_loss' : loss.item()},
                       on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        logits = torch.cat(self.valid_logit_list, dim=0)
        labels = torch.cat(self.valid_label_list, dim=0)
        output = self.valid_epoch_metrics(torch.sigmoid(logits), labels.unsqueeze(1).float())
        self.log_dict({'val_epoch_acc' : output['val_epoch_BinaryAccuracy'],
                       'val_epoch_pre' : output['val_epoch_BinaryPrecision'],
                       'val_epoch_rec' : output['val_epoch_BinaryRecall'],
                       'val_epoch_f1' : output['val_epoch_BinaryF1Score']},
                       on_epoch = True, prog_bar = True)

        self.valid_logit_list = []
        self.valid_label_list = []

    def test_step(self, test_batch, batch_idx):
        out = self.forward([test_batch['input_ids'], test_batch['token_type_ids'], test_batch['attention_mask']])
        loss = self.criterion(out, test_batch['label'].unsqueeze(1).float())
        # output = self.test_metrics(torch.sigmoid(out), test_batch['label'].unsqueeze(1).float())

        self.test_logit_list.append(out.detach())
        self.test_label_list.append(test_batch['label'].detach())

        return loss

    def on_test_epoch_end(self):
        logits = torch.cat(self.test_logit_list, dim=0)
        labels = torch.cat(self.test_label_list, dim=0)
        output = self.test_metrics(torch.sigmoid(logits), labels.unsqueeze(1).float())
        self.log_dict({'test_acc' : output['test_BinaryAccuracy'],
                       'test_pre' : output['test_BinaryPrecision'],
                       'test_rec' : output['test_BinaryRecall'],
                       'test_f1' : output['test_BinaryF1Score']},
                       on_epoch = True, prog_bar = True)

        self.test_logit_list = []
        self.test_label_list = []

    def prepare_data(self):
        self.train_dataset = StudDataset('../train.tsv')
        self.val_dataset = StudDataset('../val.tsv')
        self.test_dataset = StudDataset('../test.tsv')

        metrics = torchmetrics.MetricCollection([
            Accuracy(task='binary'),
            Precision(task='binary'),
            Recall(task='binary'),
            F1Score(task='binary')
        ])



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
            steps_per_epoch=int(len(self.train_dataloader()) / self.config['accumulate']),
            anneal_strategy='linear',
            pct_start=0.1
        )


        return {"optimizer": optimizer, "lr_scheduler": scheduler}
