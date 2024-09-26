import copy
import pdb
import time

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# from .config import InfoPro, InfoPro_balanced_memory
# from .auxiliary_nets_1 import Decoder, AuxClassifier

from config import InfoPro, InfoPro_balanced_memory
from auxiliary_nets_1 import Decoder, AuxClassifier

def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
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


class InfoProResNet_front(nn.Module):

    def __init__(self, block, layers, infopro_config, arch, batch_size, image_size=32, local_module_num=None, inplanes=None,
                 balanced_memory=False, dataset='cifar10', class_num=10,
                 wide_list=None, dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128,momentum = 0.999, device=None):
        super(InfoProResNet_front, self).__init__()

     
        
        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        
        self.widelist = wide_list
        self.inplanes = inplanes
        self.dropout_rate = dropout_rate
        self.feature_num = layers[-1][0]
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.momentum = momentum
        print(self.inplanes)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
         # 假设 block 是构建每层的基本模块，wide_list 是每层的宽度列表，layers 是每层的深度
        self.layerslist = nn.ModuleList()  # 用于存储所有层的 ModuleList

        # 创建 n 层
        curstage=1
        for i in range(len(layers)):
            if layers[i][1]!=curstage:
                curstage=layers[i][1]
                layer = self._make_layer(block, layers[i][0], 1 , 2)
            else:
                layer = self._make_layer(block, layers[i][0], 1 , 1)
                
            self.layerslist.append(layer)  # 将层添加到 nn.ModuleList 中
            
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.infopro_config = infopro_config
        self.criterion_ce = nn.CrossEntropyLoss()


        # for module_index in range(len(self.layerslist)):
        #     for layer_index in range(len(self.layerslist[module_index])):

        #         exec('self.decoder_' + str(module_index+1) + '_' + str(layer_index) +
        #              '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')

        #         exec('self.aux_classifier_' + str(module_index+1) + '_' + str(layer_index) +
        #              '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
        #              'loss_mode=local_loss_mode, class_num=class_num, '
        #              'widen=aux_net_widen, feature_dim=aux_net_feature_dim, device=device)')
                
        for layer_index in range(1,len(self.layerslist)):

            exec('self.decoder_' + str(layer_index) +
                    '= Decoder(layers[layer_index][0], image_size, widen=aux_net_widen)')

            exec('self.aux_classifier_' + str(layer_index) + 
                    '= AuxClassifier(layers[layer_index][0], net_config=aux_net_config, '
                    'loss_mode=local_loss_mode, class_num=class_num, '
                    'widen=aux_net_widen, feature_dim=aux_net_feature_dim, device=device)')

        
        # 创建LB,EMA_Net
        self.LB = nn.ModuleList([])
        self.EMA_Net = nn.ModuleList([])

        
        for layer_index in self.infopro_config:
    
            self.LB.append(copy.deepcopy(eval(f'self.layerslist[{layer_index}]')))
            self.EMA_Net.append(copy.deepcopy(eval(f'self.layerslist[{layer_index}]')))

        

        for i in range(len(self.EMA_Net)):
            for param in self.EMA_Net[i].parameters():
                param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
        
        if 'CIFAR' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)

        self.Encoder_Net = self._make_Encoder_Aux_Net()

        for net in self.Encoder_Net:
            net = net.cuda()

        for net1, net2 in zip(self.LB, self.EMA_Net):
            net1 = net1.cuda()
            net2 = net2.cuda()

        
        # print(self.Encoder_Net)
    #反归一化
    def _image_restore(self, normalized_image):
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_Encoder_Aux_Net(self):
        Encoder_Net = nn.ModuleList([])

        for layer in range(0,len(self.layers)):
                Encoder_Net.append(eval(f'self.layerslist[{layer}]'))
        return Encoder_Net

    # TODO 复现的时候有点问题，由于当前的local module 的最后一个ema_net需要下一个local module的第一个layer信息，但是在切分了网络之后，下一个local module的信息layer信息无法获得，目前只能舍弃local module的最后一个卷积层
    def forward_features(self, img, target=None,ixx_1=5, ixy_1=0.5,
                ixx_2=0, ixy_2=0):
        if self.training:
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)-1):
                
                x = self.Encoder_Net[i](x)
                y = self.LB[i](x) + self.EMA_Net[i](x)

                    
                lo = self.layers[i][1]
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                # print(f'x.shape{x.shape} i:{i} y.shape:{y.shape} self.decoder_' + str(i+1))
                # print('self.aux_classifier_' + str(lo) + '_' + str(la))
                
                loss_ixx = eval('self.decoder_' + str(i+1))(y,self._image_restore(img))
                loss_ixy = eval('self.aux_classifier_' + str(i+1))(y,target)
                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                loss.backward()

                x = x.detach()
          
           
     
        
                # cur_layer = eval(f'self.layerslist[{i+1}]')
                # # print(f'self.layerslist[{lo-1}][{la}]')
                
                # for paramEncoder, paramEMA in zip(cur_layer.parameters(),self.EMA_Net[i].parameters()):
                #     # print(f'paramEMA{paramEMA.data.shape} paramEncoder{paramEncoder.data.shape}')
                    
                #     paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)
            
            ### TODO
            # last local module       
            # x = self.Encoder_Net[-1](x)
            # x = self.avgpool(x)
            # x = x.view(x.size(0), -1)

            return x

        else:
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)-1):
                x = self.Encoder_Net[i](x)

            return x




class InfoProResNet_mid(nn.Module):

    def __init__(self, block, layers, infopro_config, arch, batch_size, image_size=32, local_module_num=None, inplanes=None,
                 balanced_memory=False, dataset='cifar10', class_num=10, 
                 wide_list=None, dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128, momentum = 0.999, device=None):
        super(InfoProResNet_back, self).__init__()

     
        
        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        self.widelist = wide_list
        self.inplanes = inplanes
        self.dropout_rate = dropout_rate
        self.feature_num = wide_list[-1]
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.momentum = momentum

        
         # 假设 block 是构建每层的基本模块，wide_list 是每层的宽度列表，layers 是每层的深度
        self.layerslist = nn.ModuleList()  # 用于存储所有层的 ModuleList

        # 创建 n 层
        curstage=layers[0][1]-1 if layers[0][1]>1 else 0

        for i in range(len(layers)):
            if layers[i][1]!=curstage:
                curstage=layers[i][1]
                layer = self._make_layer(block, layers[i][0], 1 , 2)
            else:
                layer = self._make_layer(block, layers[i][0], 1 , 1)
                
            self.layerslist.append(layer)  # 将层添加到 nn.ModuleList 中
            
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        # self.fc64 = nn.Linear(self.widelist[-1], self.class_num)
        self.head = nn.Linear(self.widelist[-1], self.class_num)
        
        self.Flatten = nn.Flatten()

        self.infopro_config = infopro_config
        self.criterion_ce = nn.CrossEntropyLoss()

        for module_index in range(len(self.layerslist)):
            for layer_index in range(len(self.layerslist[module_index])):

                exec('self.decoder_' + str(module_index+1) + '_' + str(layer_index) +
                     '= Decoder(wide_list[module_index], image_size, widen=aux_net_widen)')

                exec('self.aux_classifier_' + str(module_index+1) + '_' + str(layer_index) +
                     '= AuxClassifier(wide_list[module_index], net_config=aux_net_config, '
                     'loss_mode=local_loss_mode, class_num=class_num, '
                     'widen=aux_net_widen, feature_dim=aux_net_feature_dim, device=device)')

        self.LB = nn.ModuleList([])
        self.EMA_Net = nn.ModuleList([])

        for item in self.infopro_config[:-1]:
            module_index, layer_index = item
            if layer_index == len(self.layerslist[0]) - 1:
                mo,la = module_index + 1,0
            else:
                mo,la = module_index,layer_index + 1
                
            self.LB.append(copy.deepcopy(eval(f'self.layerslist[{mo-1}]')[la]))
            self.EMA_Net.append(copy.deepcopy(eval(f'self.layerslist[{mo-1}]')[la]))
                

        for i in range(len(self.EMA_Net)):
            for param in self.EMA_Net[i].parameters():
                param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
        
        if 'CIFAR' in dataset:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)

        self.Encoder_Net = self._make_Encoder_Aux_Net()

        for net in self.Encoder_Net:
            net = net.cuda()

        for net1, net2 in zip(self.LB, self.EMA_Net):
            net1 = net1.cuda()
            net2 = net2.cuda()
            

    def _image_restore(self, normalized_image):
        
        # print(f'normalized_image.shape: {normalized_image.shape}')
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_Encoder_Aux_Net(self):
        Encoder_Net = nn.ModuleList([])

        Encoder_temp = nn.ModuleList([])

        local_block_index = 0

        # Build Encoder_Net
        for blocks in range(len(self.layers)):
            for layers in range(self.layers[blocks]):
                Encoder_temp.append(eval(f'self.layerslist[{blocks}]')[layers])
                if blocks + 1 == self.infopro_config[local_block_index][0] \
                        and layers == self.infopro_config[local_block_index][1]:
                    Encoder_Net.append(nn.Sequential(*Encoder_temp))

                    Encoder_temp = nn.ModuleList([])
                    local_block_index += 1
        return Encoder_Net


    def forward(self, img, target=None, oriimg=None, ixx_1=0, ixy_1=0,
                ixx_2=0, ixy_2=0):
        if self.training:

            # x = self.conv1(img)
            # x = self.bn1(x)
            # x = self.relu(x)
            x=img
            for i in range(len(self.Encoder_Net)):
                # print(f'1x.shape: {x.shape}')
                x = self.Encoder_Net[i](x)
                # print(f'2x.shape: {x.shape}')
                
                y = self.LB[i](x) + self.EMA_Net[i](x)

                local_index,layer_index = self.infopro_config[i]
                # print(f'local_index {local_index}  layer_index {layer_index}')
                
                if layer_index == len(self.layerslist[0]) - 1:
                    lo,la = local_index + 1,0
                else:
                    lo,la = local_index,layer_index + 1
           
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                # print(f'lo {lo} la {la}  y.shape:{y.shape}')
                # tmp=self._image_restore(img)
                # print(f'tmp.shape: {tmp.shape}')
                
                
                
                loss_ixx = eval('self.decoder_' + str(lo) + '_' + str(la))(y,self._image_restore(oriimg))
           
                loss_ixy = eval('self.aux_classifier_' + str(lo) + '_' + str(la))(y,target)
                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                loss.backward()

                
                # print(f'3x.shape: {x.shape}')
                
                
                x = x.detach()
                cur_layer = eval(f'self.layerslist[{lo-1}]')[la]
                for paramEncoder, paramEMA in zip(cur_layer.parameters(),self.EMA_Net[i].parameters()):
                    paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)

            return x
      
        else:
            x = self.conv1(img)
            x = self.bn1(x)
            x = self.relu(x)

            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)


            return x



class InfoProResNet_back(nn.Module):

    def __init__(self, block, layers, infopro_config, arch, batch_size, image_size=32, local_module_num=None, inplanes=None,
                 balanced_memory=False, dataset='cifar10', class_num=10, 
                 wide_list=None, dropout_rate=0,
                 aux_net_config='1c2f', local_loss_mode='contrast',
                 aux_net_widen=1, aux_net_feature_dim=128, momentum = 0.999, device=None):
        super(InfoProResNet_back, self).__init__()

     
        
        assert arch in ['resnet32', 'resnet110'], "This repo supports resnet32 and resnet110 currently. " \
                                                  "For other networks, please set network configs in .configs."
        self.widelist = wide_list
        self.inplanes = inplanes
        self.feature_num = layers[-1][0]
        self.dropout_rate = dropout_rate
        self.class_num = class_num
        self.local_module_num = local_module_num
        self.layers = layers
        self.momentum = momentum

  
        
         # 假设 block 是构建每层的基本模块，wide_list 是每层的宽度列表，layers 是每层的深度
        self.layerslist = nn.ModuleList()  # 用于存储所有层的 ModuleList

        # 创建 n 层
        curstage=layers[0][1]-1 if layers[0][1]>1 else 0
        for i in range(len(layers)):
            if layers[i][1]!=curstage:
                curstage=layers[i][1]
                layer = self._make_layer(block, layers[i][0], 1 , 2)
            else:
                layer = self._make_layer(block, layers[i][0], 1 , 1)
                
            self.layerslist.append(layer)  # 将层添加到 nn.ModuleList 中
        
     
    
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_num, self.class_num)
        self.head = nn.Linear(self.feature_num, self.class_num)
        
        self.Flatten = nn.Flatten()

        self.infopro_config = infopro_config
        self.criterion_ce = nn.CrossEntropyLoss()

    
        for layer_index in range(1,len(self.layerslist)):

            exec('self.decoder_' + str(layer_index) +
                    '= Decoder(layers[layer_index][0], image_size, widen=aux_net_widen)')

            exec('self.aux_classifier_' + str(layer_index) + 
                    '= AuxClassifier(layers[layer_index][0], net_config=aux_net_config, '
                    'loss_mode=local_loss_mode, class_num=class_num, '
                    'widen=aux_net_widen, feature_dim=aux_net_feature_dim, device=device)')

        
        # 创建LB,EMA_Net
        self.LB = nn.ModuleList([])
        self.EMA_Net = nn.ModuleList([])

        
        for layer_index in self.infopro_config:
    
            self.LB.append(copy.deepcopy(eval(f'self.layerslist[{layer_index}]')))
            self.EMA_Net.append(copy.deepcopy(eval(f'self.layerslist[{layer_index}]')))

  
                

        for i in range(len(self.EMA_Net)):
            for param in self.EMA_Net[i].parameters():
                param.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        
        
        if 'CIFAR' in dataset:
            # print(f'device {device}')
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
        else:
            self.mask_train_mean = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)
            self.mask_train_std = torch.Tensor([x / 255.0 for x in [127.5, 127.5, 127.5]]).view(1, 3, 1, 1).expand(
                batch_size, 3, image_size, image_size
            ).to(device)

        self.Encoder_Net = self._make_Encoder_Aux_Net()

        for net in self.Encoder_Net:
            net = net.cuda()

        for net1, net2 in zip(self.LB, self.EMA_Net):
            net1 = net1.cuda()
            net2 = net2.cuda()
            

    def _image_restore(self, normalized_image):
        
        # print(f'normalized_image.shape: {normalized_image.shape}')
        return normalized_image.mul(self.mask_train_std[:normalized_image.size(0)]) \
               + self.mask_train_mean[:normalized_image.size(0)]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropout_rate=self.dropout_rate))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout_rate=self.dropout_rate))

        return nn.Sequential(*layers)

    def _make_Encoder_Aux_Net(self):
        Encoder_Net = nn.ModuleList([])

        for layer in range(0,len(self.layers)):
                Encoder_Net.append(eval(f'self.layerslist[{layer}]'))
        return Encoder_Net


    def forward(self, x, target=None, oriimg=None, ixx_1=5, ixy_1=0.5,
                ixx_2=0, ixy_2=0):
        if self.training:

            
            for i in range(len(self.Encoder_Net) - 1):
                
                
                
                x = self.Encoder_Net[i](x)
                y = self.LB[i](x) + self.EMA_Net[i](x)

                    
                lo = self.layers[i][1]
                ratio = lo / (self.local_module_num - 2) if self.local_module_num > 2 else 0
                ixx_r = ixx_1 * (1 - ratio) + ixx_2 * ratio
                ixy_r = ixy_1 * (1 - ratio) + ixy_2 * ratio
                # print(f'x.shape{x.shape} i:{i} y.shape:{y.shape} self.decoder_' + str(i+1))
                
                loss_ixx = eval('self.decoder_' + str(i+1))(y,self._image_restore(oriimg))
                loss_ixy = eval('self.aux_classifier_' + str(i+1))(y,target)
                loss = ixx_r * loss_ixx + ixy_r * loss_ixy
                loss.backward()

                x = x.detach()
          
           
     
        
                # cur_layer = eval(f'self.layerslist[{i+1}]')
                
                # for paramEncoder, paramEMA in zip(cur_layer.parameters(),self.EMA_Net[i].parameters()):
                    
                #     paramEMA.data = paramEMA.data * self.momentum + paramEncoder.data * (1 - self.momentum)
                
            

            # last local module
            x = self.Encoder_Net[-1](x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            logits = self.head(x)
            
            # return logits
            loss = self.criterion_ce(logits, target)
            loss.backward()
            return logits,loss
            
            # local_index, layer_index = self.infopro_config[-1]
            # cur_layer = eval(f'self.layerslist[{local_index-1}]')[layer_index]
            # return x,loss

        else:
 

            for i in range(len(self.Encoder_Net)):
                x = self.Encoder_Net[i](x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.head(x)
            # loss = self.criterion_ce(logits, target)
            # return x, loss
            
            return logits




def resnet32_front(**kwargs):
    # layers=kwargs['layers']
    # infopro_config=kwargs['infopro_config']
    # wide_list=kwargs['wide_list']
    print(f"pos1 {kwargs['local_module_num']}")
    model = InfoProResNet_front(BasicBlock, arch='resnet32', **kwargs)
    
    return model

def resnet32_mid(**kwargs):
    return 

def resnet32_back(**kwargs):
    model = InfoProResNet_back(BasicBlock, arch='resnet32', **kwargs)
    return model


# def resnet110(**kwargs):
#     model = InfoProResNet(BasicBlock, [18, 18, 18], arch='resnet110', **kwargs)
#     return model

import yaml

# if __name__ == "__main__":
    
#     cfg='configs/k2_resnet32_t.yaml'
    
#     with open(cfg, 'r') as file1:
#         cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
#     layers_f=cfg_data['front']['layers']
    
        
#     infopro_config_f=cfg_data['front']['infopro_config']
#     f_inplanes=cfg_data['front']['inplanes']
    
#     layers_b=cfg_data['back']['layers']
    
    
    
#     infopro_config_b=cfg_data['back']['infopro_config']
#     b_inplanes=cfg_data['back']['inplanes']
    
#     batch_size=cfg_data['common']['batch_size']
#     image_size=cfg_data['common']['image_size']
#     dropout_rate=cfg_data['common']['dropout_rate']
#     class_num=cfg_data['common']['class_num']
#     aux_net_config=cfg_data['common']['aux_net_config']
#     local_loss_mode=cfg_data['common']['local_loss_mode']
#     aux_net_widen=cfg_data['common']['aux_net_widen']
#     aux_net_feature_dim=cfg_data['common']['aux_net_feature_dim']
#     local_module_num=cfg_data['common']['local_module_num']
    
    
    
#     front= resnet32_front(layers=layers_f ,infopro_config=infopro_config_f, local_module_num=local_module_num, inplanes=f_inplanes, batch_size=256, image_size=32,
#                    dataset='cifar10', class_num=10,
#                     dropout_rate=0,
#                    aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
#                    aux_net_widen=1, aux_net_feature_dim=128, device='cuda:0')
#     front = front.cuda()
    
#     back= resnet32_back(layers=layers_b ,infopro_config=infopro_config_b, local_module_num=local_module_num, inplanes=b_inplanes, batch_size=256, image_size=32,
#                    dataset='cifar10', class_num=10,
#                    dropout_rate=0,
#                    aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
#                    aux_net_widen=1, aux_net_feature_dim=128,device='cuda:0')
    
#     back = back.cuda()
#     cache=[]
#     # print(f'front: {front}')
#     # exit(0)
    

        
#     x = torch.ones(4,3,32,32).cuda()
#     target = torch.zeros(4).long().cuda()
    
#     outfront=front.forward_features(x, target)
#     # print(f'front shape: {outfront.shape}')
    
#     # 打印更新后的参数及其对应的网络层名称
#     for name, param in front.named_parameters():
#         if param.grad is not None:  # 检查是否更新了
#             print(f"F Layer: {name}, Parameter: {param.shape}")
    
    
#     exit(0)
#     outend=back.forward(outfront,target,oriimg=x)
    
#     loss_function = torch.nn.CrossEntropyLoss()
    
#     loss = loss_function(outend, target)
#     # print(f'pred.requires_grad :{pred.requires_grad} loss.requires_grad:{loss.requires_grad}')
#     # exit(0)
    
    
#     ### 
#     loss.backward()

    
#     for name, param in back.named_parameters():
#         if param.grad is not None:  # 检查是否更新了
#             print(f"B Layer: {name}, Parameter : {param.shape}")
#     print(f'outend {outend}')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def train():
    
    epoches=400
    cfg='configs/k2_resnet32_t.yaml'
    
    with open(cfg, 'r') as file1:
        cfg_data=  yaml.load(file1,Loader=yaml.FullLoader)
    
    layers_f=cfg_data['front']['layers']
    
        
    infopro_config_f=cfg_data['front']['infopro_config']
    f_inplanes=cfg_data['front']['inplanes']
    
    layers_b=cfg_data['back']['layers']
    
    
    
    infopro_config_b=cfg_data['back']['infopro_config']
    b_inplanes=cfg_data['back']['inplanes']
    
    batch_size=cfg_data['common']['batch_size']
    image_size=cfg_data['common']['image_size']
    dropout_rate=cfg_data['common']['dropout_rate']
    class_num=cfg_data['common']['class_num']
    aux_net_config=cfg_data['common']['aux_net_config']
    local_loss_mode=cfg_data['common']['local_loss_mode']
    aux_net_widen=cfg_data['common']['aux_net_widen']
    aux_net_feature_dim=cfg_data['common']['aux_net_feature_dim']
    local_module_num=cfg_data['common']['local_module_num']
    
    
    
    front= resnet32_front(layers=layers_f ,infopro_config=infopro_config_f, local_module_num=local_module_num, inplanes=f_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10,
                    dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128, device='cuda:0')
    front = front.cuda()
    
    back= resnet32_back(layers=layers_b ,infopro_config=infopro_config_b, local_module_num=local_module_num, inplanes=b_inplanes, batch_size=1024, image_size=32,
                   dataset='cifar10', class_num=10,
                   dropout_rate=0,
                   aux_net_config=aux_net_config, local_loss_mode=local_loss_mode,
                   aux_net_widen=1, aux_net_feature_dim=128,device='cuda:0')
    
    back = back.cuda()
    cache=[]
   
    optimizef = torch.optim.SGD(front.parameters(),
                                lr=0.8,
                                momentum=0.995,
                                nesterov=True,
                                weight_decay=0.0001)
    
    
    
    optimizeb = torch.optim.SGD(back.parameters(),
                                lr=0.8,
                                momentum=0.995,
                                nesterov=True,
                                weight_decay=0.0001)
    
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    kwargs_dataset_train = {'train': True}
    kwargs_dataset_test = {'train': False}
    
    kwargs = {'num_workers': 8, 'pin_memory': False}
    
    transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10']('./data', download=True, transform=transform_train,
                                                **kwargs_dataset_train),
        batch_size=1024, shuffle=True, **kwargs)
    # val_loader = torch.utils.data.DataLoader(
    #     datasets.__dict__['CIFAR']('./data', transform=transform_test,
    #                                             **kwargs_dataset_test),
    #     batch_size=1024, shuffle=False, **kwargs)

    
    
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    front.train()
    back.train()
    
    
    print(f'train_batches_num {train_batches_num}')
    end = time.time()
    for epoch in range(epoches):
            
        for i, (x, target) in enumerate(tqdm(train_loader)):
            target = target.cuda()
            oriimg=x.cuda()
            x = x.cuda()
            optimizef.zero_grad()

            optimizeb.zero_grad()
            
            frontout=front.forward_features(img=x, target=target,ixx_1=5, ixy_1=0.5,
                    ixx_2=0, ixy_2=0)
            
        
            output,loss  = back(x=frontout,
                        target=target,
                        oriimg=oriimg,
                        ixx_1=5,
                        ixy_1=0.5,
                        ixx_2=0,
                        ixy_2=0)
            optimizef.step()
            optimizeb.step()
            

            
            # 打印更新后的参数及其对应的网络层名称
            for name, param in front.named_parameters():
                if param.grad is not None:  # 检查是否更新了
                    print(f"front Layer: {name}, Parameter: {param.shape}")
            for name, param in back.named_parameters():
                if param.grad is not None:  # 检查是否更新了
                    print(f"back Layer: {name}, Parameter: {param.shape}")
            exit(0)
            
            
            # measure accuracy and record loss
            # prec_x = accuracy(output_x.data, target, topk=(1,))[0]
            # prec_y = accuracy(output_y.data, target, topk=(1,))[0]
            # prec1 = (prec_x + prec_y) / 2
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.data.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 10 == 0:
                # print(discriminate_weights)
                string = ('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                        'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                        'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                        epoch, i+1, train_batches_num, batch_time=batch_time,
                        loss=losses, top1=top1))

                print(string)
            
            
            
            
            
if __name__ == "__main__":
    train()