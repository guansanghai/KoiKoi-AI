# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 10:34:35 2021

@author: shguan3
"""

import torch # 1.8.1
import torch.nn as nn
import torch.nn.functional as F

# inputsize: (BATCH_SIZE,nFeature,48) (BATCH_SIZE,nFeature)
# conv1d: (BATCH_SIZE, nFeature, 48)
# multiheadattention: (48, BATCH_SIZE, nFeature)

# model.encoder_block.attn_encoder.layers[0].linear1

#%%
NetParameter = {
    'nHidden':512,
    'nEmb':192,
    'nFw':384,
    'nAttnHead':3,
    'nLayer':2}


class KoiKoiEncoderBlock(nn.Module):
    def __init__(self,nInput,nHidden,nEmb,nFw,nAttnHead,nLayer):
        super(KoiKoiEncoderBlock,self).__init__()
        self.f1 = nn.Conv1d(nInput, nHidden, 1)
        self.f2 = nn.Conv1d(nHidden, nEmb, 1)
        self.attn_layer = nn.TransformerEncoderLayer(nEmb, nAttnHead, nFw)
        self.attn_encoder = nn.TransformerEncoder(self.attn_layer, nLayer)
        
    def forward(self,x): 
        x = self.f2(F.relu(self.f1(x)))
        x = F.layer_norm(x,[x.size(-1)])
        x = x.permute(2,0,1)
        x = self.attn_encoder(x)
        x = x.permute(1,2,0)  
        return x


class DiscardModel(nn.Module):
    def __init__(self,nInput):
        super(DiscardModel,self).__init__()       
        self.encoder_block = KoiKoiEncoderBlock(nInput,**NetParameter)
        self.discard_out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self,x):       
        x = self.encoder_block(x)
        x = self.discard_out(x).squeeze(1)
        return x


class PickModel(nn.Module):
    def __init__(self,nInput):
        super(PickModel,self).__init__()
        self.encoder_block = KoiKoiEncoderBlock(nInput,**NetParameter)
        self.pick_out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self,x):
        x = self.encoder_block(x)
        x = self.pick_out(x).squeeze(1)
        return x
    

class KoiKoiModel(nn.Module):
    def __init__(self,nInput):
        super(KoiKoiModel,self).__init__()
        self.encoder_block = KoiKoiEncoderBlock(nInput,**NetParameter)
        self.koikoi_fc = nn.Conv1d(NetParameter['nEmb'], 8, 1)
        self.koikoi_out = nn.Linear(48*8, 2)
        
    def forward(self,x):
        x = self.encoder_block(x)
        x = F.relu(self.koikoi_fc(x))
        x = x.contiguous().view(x.size(0),-1)
        x = self.koikoi_out(x)
        return x
    
    

