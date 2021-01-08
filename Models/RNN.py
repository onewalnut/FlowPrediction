#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File   : RNN.py
# @Author : HT
# @Date   : 18-9-3
# @Desc   :


from classes import *
from paras import *


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(64, REGRESS_SIZE),
            # torch.nn.BatchNorm1d(REGRESS_SIZE),
            torch.nn.ReLU()
        )

        self.predict = torch.nn.Linear(REGRESS_SIZE, 1)

        for m in self.regression.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 10)
                m.bias.data.zero_()

        for m in self.predict.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 10)
                m.bias.data.zero_()


    def forward(self, x):
        feature, hidden = self.rnn(x)
        output = self.regression(feature[:, -1, :])
        output = self.predict(output)
        return output


class TemporalNet(torch.nn.Module):
    def __init__(self):
        super(TemporalNet, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.regression = torch.nn.Sequential(
            torch.nn.Linear(512, REGRESS_SIZE),
            # torch.nn.BatchNorm1d(REGRESS_SIZE),
            torch.nn.ReLU(inplace=True)
        )

        self.prediction = torch.nn.Linear(REGRESS_SIZE, 3)

        for m in self.regression.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        for m in self.prediction.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):
        feature, hidden_state = self.rnn(input)
        output = self.regression(feature[:, -1, :])
        output = self.prediction(output)

        return output


# short connection
class MRNN(torch.nn.Module):
    def __init__(self):
        super(MRNN, self).__init__()
        self.ModelName = 'MRNN'
        self.feature1 = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.feature2 = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.feature3 = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0.1
        )

        self.feature4 = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=256,
            num_layers=4,
            batch_first=True,
            dropout=0.1
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(256 * 4, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2)
        )

        self.predict = torch.nn.Sequential(
            torch.nn.Linear(4096, 2)
        )

    def forward(self, x):
        x1, (h_n1, h_c1) = self.feature1(x, None)
        x2, (h_n2, h_c2) = self.feature2(x, None)
        x3, (h_n3, h_c3) = self.feature3(x, None)
        x4, (h_n4, h_c4) = self.feature4(x, None)
        x = torch.cat((x1[:, -1, :], x2[:, -1, :], x3[:, -1, :], x4[:, -1, :]), -1)
        x = self.regressor(x)
        x = self.predict(x)
        return x
