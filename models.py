from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_inplanes():#输入通道
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class ChannelAttention(nn.Module):
    def __init__(self,inplanes,ratio=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(inplanes,inplanes//16,1,bias=False) # 本来是2的改成3了
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv3d(inplanes//16,inplanes,1,bias=False)

        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        avg_out = self.avg_pool(x)
        
        avg_out = self.fc1(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:#downsample??
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.Action=Action(in_planes ,in_planes )

    def forward(self, x):
        residual = x
        x=x.transpose(2,1)
        out = self.Action(x)
        x=x.transpose(2,1)
        out = self.conv1(x)#正常，x.shape=1,64,256,128,3
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se=SELayer(planes* self.expansion)# 0308 inplanes->planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)#正常，x.shape=1,64,256,128,3
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out=self.se(out)
        out += residual
        out = self.relu(out)

        return out


import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ResNet(nn.Module): 

    def __init__(self,
                 block1,
                 block2,
                 layers,
                 block_inplanes,  # 块数
                 n_input_channels=3,  # 输入RGB三个通道 
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()# 构造父类

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels,  # 通道数
                               self.in_planes,  # 通道数
                               kernel_size=(conv1_t_size, 7, 7),  #卷积核
                               stride=(conv1_t_stride, 2, 2),  # 步长
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block1, block2,block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block1,block2,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block1,block2,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block1,block2,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
         #在卷积层最后一层加入注意力机制
         
        self.ca1=ChannelAttention(64) 
        self.sa1 = SpatialAttention()
        self.ca2=ChannelAttention(2048) 

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  #三维的所以有三个参数
        # self.fc = nn.Linear(block_inplanes[3] * block1.expansion, n_classes)  #全连接层
        self.fc = nn.Linear(2048, n_classes)
        self.pro = nn.Softmax(dim = 1) 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block1,block2, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block1.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block1.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block1.expansion, stride),
                    nn.BatchNorm3d(planes * block1.expansion))

        layers = []
        layers.append(
            block1(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block1.expansion
        for i in range(1, blocks):
            if i%2==0:
                 layers.append(block1(self.in_planes, planes))
            else:
                layers.append(block2(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
         # 加入注意力模块
        x = self.ca1(x) * x
        x = self.sa1(x) * x


        if not self.no_max_pool:
            x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 加入注意力模块
        x = self.ca2(x) * x
        x = self.sa1(x) * x
        x = self.avgpool(x)  #平均池化

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.pro(x) # 归一化
        
        return x


def generate_model(model_depth, **kwargs):#向函数传递 可变参数
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, BasicBlock,[1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock,BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock,BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck1, Bottleneck2,[3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck1,Bottleneck2, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck1,Bottleneck2, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck1, Bottleneck2,[3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


import torch.nn as nn
class SELayer(nn.Module):
    # SE块首先为每个通道独立采用全局平均池化，然后使用两个全连接（FC）层以及非线性Sigmoid函数来生成通道权重。 
    # 两个FC层旨在捕获非线性跨通道交互，其中涉及降低维度以控制模型的复杂性
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Sequential(
        nn.Linear(channel, channel//reduction,bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel//reduction,channel, bias=False),
        nn.Sigmoid()
        )
    def forward(self, x):
        b,c,h,w ,t= x.size()
        y = self.avgpool(x).view(b,c)#1,256
        y = self.fc(y).view(b,c,1,1,1)#1,64,1,1,1
        return x * y.expand_as(x)

class Action(nn.Module):
    def __init__(self, in_channels,out_channels,n_segment=3, kernel_size=7,stride=1,padding=1,shift_div=8):#(self, net, n_segment=3, shift_div=8)
        super(Action, self).__init__()
        #self.net = net
        self.n_segment = n_segment
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.reduced_channels = self.in_channels//16
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fold = self.in_channels // shift_div

        # shifting
        self.action_shift = nn.Conv1d(
                                    self.in_channels, self.in_channels,
                                    kernel_size=3, padding=1, groups=self.in_channels,
                                    bias=False)      
        self.action_shift.weight.requires_grad = True
        self.action_shift.weight.data.zero_()
        self.action_shift.weight.data[:self.fold, 0, 2] = 1 # shift left
        self.action_shift.weight.data[self.fold: 2 * self.fold, 0, 0] = 1 # shift right  


        if 2*self.fold < self.in_channels:
            self.action_shift.weight.data[2 * self.fold:, 0, 1] = 1 # fixed


        # # spatial temporal excitation
        self.action_p1_conv1 = nn.Conv3d(1, 1, kernel_size=(3, 3, 3), 
                                    stride=(1, 1 ,1), bias=False, padding=(1, 1, 1))       


        # # channel excitation
        self.action_p2_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p2_conv1 = nn.Conv1d(self.reduced_channels, self.reduced_channels, kernel_size=3, stride=1, bias=False, padding=1, 
                                       groups=1)
        self.action_p2_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
    


        # motion excitation
        self.pad = (0,0,0,0,0,0,0,1)
        self.action_p3_squeeze = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))
        self.action_p3_bn1 = nn.BatchNorm2d(self.reduced_channels)
        self.action_p3_conv1 = nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 3), 
                                    stride=(1 ,1), bias=False, padding=(1, 1), groups=self.reduced_channels)
        self.action_p3_expand = nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=(1, 1), stride=(1 ,1), bias=False, padding=(0, 0))


    def forward(self, x):
        n,t,c,h, w = x.size()
        x_p1 = x.transpose(2,1).contiguous()
        x_p1 = x_p1.mean(1, keepdim=True)
        x_p1 = self.action_p1_conv1(x_p1)
        x_p1 = x_p1.transpose(2,1)
        x_p1 = self.sigmoid(x_p1)
        x_p1 = x * x_p1 + x


        return x_p1 



