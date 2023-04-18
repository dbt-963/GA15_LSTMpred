#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import os


def prediction(filename):

    generated1 = pd.read_excel(filename)
    pre_gene1 = generated1.iloc[:,1:]
    gene1 = pre_gene1.values

    y_gene1 = np.ones(len(gene1))

    gene1 = torch.from_numpy(gene1).type(torch.float32)
    y_gene1 = torch.from_numpy(y_gene1).type(torch.LongTensor)

    BATCH_SIZE = len(gene1)
    test_gene1 = DataLoader(TensorDataset(gene1, y_gene1), batch_size=BATCH_SIZE, shuffle=False)

    input_size = 90
    hidden_size = 256

    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.fc1 = nn.Linear(hidden_size, 128)
            self.fc2 = nn.Linear(128, 1)
            self.drop1 = nn.Dropout(0.7)


        def forward(self, x):
            x = x.view(x.size(0), -1, input_size)
            x = x.permute(1, 0, 2)
            x, _ = self.lstm(x)
            x = x[-1]
            x = self.bn1(x)
            x = torch.relu(self.fc1(x))
            x = self.drop1(x)
            x = torch.sigmoid(self.fc2(x))
            x = x.squeeze(-1)
            return x


    def predicting(model,test_gene1):

        val_pred_labels = []
        val_y_preds = []
        val_true_labels = []

        model.eval()
        with torch.no_grad():
            for x, y in test_gene1:
                y = y.to(torch.float)
                val_true_labels.extend(y.detach().numpy())
                y_pred = model(x)
                val_y_preds.extend(y_pred.detach().numpy())

                y_pred = y_pred.round()
                val_pred_labels.extend(y_pred.detach().numpy())

        return val_pred_labels,val_true_labels,val_y_preds

    basepath = os.path.dirname(__file__)
    modelpath = os.path.join(basepath, 'model')
    mymodel = LSTM() #调用前面训练好的模型
    mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 1 model.model')))
    gene_pred_labels1,gene_true_labels1,gene_y_preds1 = predicting(mymodel,test_gene1)

    mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 2 model.model')))
    gene_pred_labels2,gene_true_labels2,gene_y_preds2 = predicting(mymodel,test_gene1)

    mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 3 model.model')))
    gene_pred_labels3,gene_true_labels3,gene_y_preds3 = predicting(mymodel,test_gene1)

    mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 4 model.model')))
    gene_pred_labels4,gene_true_labels4,gene_y_preds4 = predicting(mymodel,test_gene1)

    mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 5 model.model')))
    gene_pred_labels5,gene_true_labels5,gene_y_preds5 = predicting(mymodel,test_gene1)

    pred_score1 = np.array(gene_y_preds1)
    pred_score2 = np.array(gene_y_preds2)
    pred_score3 = np.array(gene_y_preds3)
    pred_score4 = np.array(gene_y_preds4)
    pred_score5 = np.array(gene_y_preds5)
    total_pred_score = pred_score1+pred_score2+pred_score3+pred_score4+pred_score5
    pred_score = (total_pred_score/5)
    pred_score = np.round(pred_score,2)

    pred_label = pred_score.round()

    pred_label = list(pred_label)
    pred_score = list(pred_score)

    return {'predicted label': pred_label, 'predicted score': pred_score}





