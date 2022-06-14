import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial


class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(OctConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv_l2l =  nn.Conv1d(int(in_channels/2), int(out_channels/2),3, stride, padding, dilation)
        self.conv_l2h = nn.Conv1d(int(in_channels/2), out_channels, 1, stride, padding, dilation)
        self.conv_h2l = nn.Conv1d(int(in_channels/2), int(out_channels/2),kernel_size, stride, padding, dilation)
        self.conv_h2h = nn.Conv1d(in_channels, out_channels,kernel_size, stride, padding, dilation)

    def forward(self, x_h, x_l):

        x_h2h = self.conv_h2h(x_h)

        x_h2l = self.downsample(x_h)
        x_h2l = self.conv_h2l(x_h2l)

        x_l2l = self.conv_l2l(x_l)

        x_l2h = self.conv_l2h(x_l)
        x_l2h = self.upsample(x_l2h)

        x_h = x_l2h + x_h2h
        x_l = x_h2l + x_l2l
        return x_h, x_l



class LSTMFCN(nn.Module):
    def __init__(self,N_time,N_Features,N_LSTM_Out=128,N_LSTM_layers=1,Conv1_NF=128,Conv2_NF=256,Conv3_NF=128,isGRU=False,isOCT=False):
        super(LSTMFCN,self).__init__()
        self.N_time = N_time
        self.N_Features = N_Features
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.isGRU = isGRU
        self.isOCT = isOCT
        if self.isGRU==False:
            self.lstm = nn.LSTM(self.N_Features,self.N_LSTM_Out,self.N_LSTM_layers)
        else:
            self.lstm = nn.GRU(self.N_Features,self.N_LSTM_Out,self.N_LSTM_layers)
        if isOCT==False:
            self.C1 = nn.Conv1d(self.N_Features,self.Conv1_NF,8)
            self.C2 = nn.Conv1d(self.Conv1_NF,self.Conv2_NF,5)
            self.C3 = nn.Conv1d(self.Conv2_NF,self.Conv3_NF,3)
        else:
            self.process = nn.AvgPool2d(kernel_size=(2,5), stride=2)
            self.C1 = OctConv(self.N_Features,self.Conv1_NF,5)
            self.C2 = OctConv(self.Conv1_NF,self.Conv2_NF,5)
            self.C3 = OctConv(self.Conv2_NF,self.Conv3_NF,5)

        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN1_l = nn.BatchNorm1d(int(self.Conv1_NF/2))
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN2_l = nn.BatchNorm1d(int(self.Conv2_NF/2))
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.BN3_l = nn.BatchNorm1d(int(self.Conv3_NF/2))
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(0.8)
        self.ConvDrop = nn.Dropout(0.3)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out,2)
    
    def init_hidden(self):
        
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).cuda()
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).cuda()
        return h0,c0
    
    def forward(self,x):
        
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features
        
        h0,c0 = self.init_hidden()
        if self.isGRU == True:
            x1, ht = self.lstm(x, h0)
        else:
            x1, (ht,ct) = self.lstm(x, (h0, c0))
        x1 = x1[:,-1,:]
        
        x2 = x.transpose(2,1)

        if self.isOCT == False:
            x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
            x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
            x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
            x2 = torch.mean(x2,2)
        else:
            x2_h = x2
            x2_l = self.process(x2)
            x_h,x_l = self.C1(x2_h,x2_l)
            x_h = self.BN1(x_h)
            x_h = self.relu(x_h)
            x_h = self.ConvDrop(x_h)
            x_l = self.BN1_l(x_l)
            x_l = self.relu(x_l)
            x_l = self.ConvDrop(x_l)
            x_h1,x_l1 = self.C2(x_h,x_l)
            x_h1 = self.BN2(x_h1)
            x_h1 = self.relu(x_h1)
            x_h1 = self.ConvDrop(x_h1)
            x_l1 = self.BN2_l(x_l1)
            x_l1 = self.relu(x_l1)
            x_l1 = self.ConvDrop(x_l1)
            x_h2,x_l2 = self.C3(x_h1,x_l1)
            x_h2 = self.BN3(x_h2)
            x_h2 = self.relu(x_h2)
            x_h2 = self.ConvDrop(x_h2)
            x_l2 = self.BN3_l(x_l2)
            x_l2 = self.relu(x_l2)
            x_l2 = self.ConvDrop(x_l2)
            x2 = torch.mean(x_h2,2)

        
        
        x_all = torch.cat((x1,x2),dim=1)
        x_out = self.FC(x_all)
        return x_out