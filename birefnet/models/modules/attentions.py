import numpy as np
import torch
from torch import nn
from torch.nn import init


class SEWeightModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight


class PSA(nn.Module):

    def __init__(self, in_channels, S=4, reduction=4):
        super().__init__()
        self.S = S

        _convs = []
        for i in range(S):
            _convs.append(nn.Conv2d(in_channels//S, in_channels//S, kernel_size=2*(i+1)+1, padding=i+1))
        self.convs = nn.ModuleList(_convs)

        self.se_block = SEWeightModule(in_channels//S, reduction=S*reduction)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Step1: SPC module
        SPC_out = x.view(b, self.S, c//self.S, h, w) #bs,s,ci,h,w
        for idx, conv in enumerate(self.convs):
            SPC_out[:,idx,:,:,:] = conv(SPC_out[:,idx,:,:,:].clone())

        # Step2: SE weight
        se_out=[]
        for idx in range(self.S):
            se_out.append(self.se_block(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3: Softmax
        softmax_out = self.softmax(SE_out)

        # Step4: SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out


class SGE(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()

    def forward(self, x):
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x 

