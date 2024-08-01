from cv2 import fastNlMeansDenoising
import torch.nn as nn
import torch
import numpy as np
import math
from torch.nn.modules.pooling import AdaptiveMaxPool2d
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import copy
import timm
from models.coatnet import CoAtNet

BN = nn.BatchNorm2d



class Deepfas_Inclusion_CoAtNet(nn.Module):
    def __init__(self, cfg=None):
        super(Deepfas_Inclusion_CoAtNet, self).__init__()

        self.cfg = copy.deepcopy(cfg) if cfg is not None else None
        
        self.num_blocks = [2, 2, 3, 5, 2]
        self.channels = [64, 96, 192, 384, 768]
        self.branch1 = CoAtNet((224, 224), 3, self.num_blocks, self.channels, num_classes=2)


        self._init_weights()
        
    def _init_weights(self, init_type=''):
        if init_type == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (BN, nn.BatchNorm1d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, BN):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def _calc_loss(self, outputs, criterions, labels):
        loss_dict = {}

        if self.cfg.get('loss_weight_class') is None: self.cfg.loss_weight_class = 1.0
        loss_class = criterions['class'](outputs['prob'], labels['class'])
        loss_class = self.cfg.loss_weight_class * loss_class
        loss_dict['loss_class'] = loss_class
        loss_total = loss_class

        for k,v in outputs.items():
            if 'prob_aux' in k: 
                loss_class_aux = criterions['class_aux'](outputs['prob_aux'], labels['class_aux'])
                loss_class_aux = self.cfg.loss_weight_class_aux * loss_class_aux
                if loss_dict.get('loss_class_aux') is None:
                    loss_dict['loss_class_aux'] = loss_class_aux
                else:
                    loss_dict['loss_class_aux'] += loss_class_aux
                loss_total = loss_total+loss_class_aux

            elif k=='depth_map':
                loss_depth = criterions['depth'](outputs['depth_map'], labels['depth'])
                loss_depth = self.cfg.loss_weight_depth * loss_depth
                loss_dict['loss_depth'] = loss_depth
                loss_total = loss_total+loss_depth
            
            elif k=='recon_map':
                loss_recon = criterions['recon'](outputs['recon_map'], labels['recon'])
                loss_recon = self.cfg.loss_weight_recon * loss_recon
                loss_dict['loss_recon'] = loss_recon
                loss_total = loss_total+loss_recon
            
            elif k=='loss_faceid':
                loss_faceid = outputs['loss_faceid'].mean()  # reduction==mean
                loss_faceid = self.cfg.loss_weight_faceid * loss_faceid
                loss_dict['loss_faceid'] = loss_faceid
                loss_total = loss_total+loss_faceid
            
            elif k=='loss_exclud':
                loss_exclud = outputs['loss_exclud'].mean()  # reduction==mean
                loss_exclud = self.cfg.loss_weight_exclud * loss_exclud
                loss_dict['loss_exclud'] = loss_exclud
                loss_total = loss_total+loss_exclud

        loss_dict['loss_total'] = loss_total
        return loss_dict

    def forward(self, x, criterions=None, labels_dict=None, phase=""):
        if isinstance(x,dict):
            data1 = x['img_frame1']
        else:
            data1 = x[:,0:3]    # frame1

        x1 = self.branch1(data1)
        logits = x1
        logits_aux = x1

        if phase=="only_prediction":  # only inference prediction
            outputs = logits
            return outputs
        else:
            outputs = {}
            outputs['prob'] = logits
            outputs['prob_aux'] = logits_aux
            loss_dict = self._calc_loss(outputs, criterions, labels_dict)
            return outputs, loss_dict
