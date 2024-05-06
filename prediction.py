#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import os


#定义预测函数接收输入文件名
def prediction1(filename):

    #读取Excel文件，跳过第一列并获取剩余的数据
    generated1 = pd.read_excel(filename)
    pre_gene1 = generated1.iloc[:,0:]
    gene1 = pre_gene1.values
    print(f"Input data shape (rows, columns): {gene1.shape}")
    #为每一行创建数据标签，设定为 1（预设目标值）
    y_gene1 = np.ones(len(gene1))

    # 转换数据为 PyTorch Tensor
    gene1 = torch.from_numpy(gene1).type(torch.float32)
    y_gene1 = torch.from_numpy(y_gene1).type(torch.LongTensor)

    # 输入特征维度
    input_size = 90

    # 打印数据形状
    print(f"Tensor shape before reshaping: {gene1.shape}")

    # 重塑数据形状
    batch_size = gene1.size(0)
    gene1 = gene1.view(batch_size, -1, input_size)
    print(f"Tensor shape after reshaping: {gene1}")

    # 定义批处理大小，将数据包装到 DataLoader 中以供模型评估使用
    BATCH_SIZE = len(gene1)
    test_gene1 = DataLoader(TensorDataset(gene1, y_gene1), batch_size=BATCH_SIZE, shuffle=False)

    # 输入特征维度和隐藏层维度
    input_size = 90
    hidden_size = 256

    # 定义 LSTM 模型
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.fc1 = nn.Linear(hidden_size, 128)
            self.fc2 = nn.Linear(128, 1)
            self.drop1 = nn.Dropout(0.7)

        #前向传播函数
        def forward(self, x):
            # 重塑输入以匹配 LSTM 的形状需求
            x = x.view(x.size(0), -1, input_size)
            # 转置数据，调整维度顺序
            x = x.permute(1, 0, 2)
            # LSTM 前向传播
            x, _ = self.lstm(x)
            # 取最后一个时间步的输出
            x = x[-1]
            # 通过 Batch Normalization
            x = self.bn1(x)
            # 激活函数 ReLU 和全连接层
            x = torch.relu(self.fc1(x))
            # 应用 Dropout
            x = self.drop1(x)
            # 最终输出通过 sigmoid 激活函数
            x = torch.sigmoid(self.fc2(x))
            # 移除最后一维度
            x = x.squeeze(-1)
            return x

    # 定义用于预测的函数
    def predicting(model,test_gene1):
        # 初始化预测标签、分数和真实标签的列表
        val_pred_labels = []
        val_y_preds = []
        val_true_labels = []

        # 进入评估模式
        model.eval()
        with torch.no_grad():
            for x, y in test_gene1:
                # 转换标签为浮点型
                y = y.to(torch.float)
                # 收集真实标签
                val_true_labels.extend(y.detach().numpy())
                # 预测输出
                y_pred = model(x)
                # 收集预测得分
                val_y_preds.extend(y_pred.detach().numpy())

                # 将输出得分四舍五入为标签
                y_pred = y_pred.round()
                # 收集预测标签
                val_pred_labels.extend(y_pred.detach().numpy())

        return val_pred_labels,val_true_labels,val_y_preds

    # 获取模型路径
    basepath = os.path.dirname(__file__)
    modelpath = os.path.join(basepath, 'model')
    print(f"Model path: {modelpath}")
    # 初始化模型并加载预训练权重
    mymodel = LSTM() #调用前面训练好的模型
    print("LSTM model instantiated successfully.")

    # 检查文件是否存在
    if not os.path.exists(modelpath):
        print(f"Model file does not exist at: {modelpath}")
    else:
        # 文件存在时加载模型
        mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 1 validation.model')))
        gene_pred_labels1,gene_true_labels1,gene_y_preds1 = predicting(mymodel,test_gene1)

        mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 2 validation.model')))
        gene_pred_labels2,gene_true_labels2,gene_y_preds2 = predicting(mymodel,test_gene1)

        mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 3 validation.model')))
        gene_pred_labels3,gene_true_labels3,gene_y_preds3 = predicting(mymodel,test_gene1)

        mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 4 validation.model')))
        gene_pred_labels4,gene_true_labels4,gene_y_preds4 = predicting(mymodel,test_gene1)

        mymodel.load_state_dict(torch.load(os.path.join(modelpath, 'LSTM 15v on 5 validation.model')))
        gene_pred_labels5,gene_true_labels5,gene_y_preds5 = predicting(mymodel,test_gene1)

        # 汇总五个模型的预测分数，并取平均值
        pred_score1 = np.array(gene_y_preds1)
        pred_score2 = np.array(gene_y_preds2)
        pred_score3 = np.array(gene_y_preds3)
        pred_score4 = np.array(gene_y_preds4)
        pred_score5 = np.array(gene_y_preds5)
        total_pred_score = pred_score1+pred_score2+pred_score3+pred_score4+pred_score5
        pred_score = (total_pred_score/5)
        pred_score = np.round(pred_score,2)

        # 将预测分数四舍五入为标签
        pred_label = pred_score.round()
        # 保存模型
        torch.save(mymodel.state_dict(), os.path.join(modelpath, 'your_model_name.pth'))
        # 将预测标签和分数转换为列表
        pred_label = list(pred_label)
        pred_score = list(pred_score)

        return {'predicted label': pred_label, 'predicted score': pred_score}

# 保存模型
# torch.save(mymodel.state_dict(), os.path.join(modelpath, 'your_model_name.pth'))



