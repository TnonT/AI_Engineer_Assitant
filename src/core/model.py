# -*- coding: utf-8 -*-

"""
@project: AI_Engineer_Assitant
@author: heibai
@file: model.py
@ide: PyCharm
@time 2021/5/11 17:28
"""

import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):

    model_name = None

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplemented

    @abstractmethod
    def inference(self, *args, **kwargs):
        raise NotImplemented
