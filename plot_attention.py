#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 22:22:12 2021

@author: guansanghai
"""

import pickle

import torch # 1.8.1
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import networkx as nx

from torch_text_mha import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct


NetParameter = {
    'nInput':300,
    'nEmb':256,
    'nFw':512,
    'nAttnHead':4,
    'nLayer':2}


class KoiKoiEncoderBlock0(nn.Module):
    def __init__(self, nInput, nEmb, nFw, nAttnHead, nLayer):
        super(KoiKoiEncoderBlock0,self).__init__()
        self.f1 = nn.Conv1d(nInput, nFw, 1)
        self.f2 = nn.Conv1d(nFw, nEmb, 1)
        
    def forward(self,x): 
        x = self.f2(F.relu(self.f1(x)))
        x = F.layer_norm(x,[x.size(-1)])
        x = x.permute(2,0,1)
        return x


class KoiKoiEncoderBlock1(nn.Module):
    def __init__(self, nInput, nEmb, nFw, nAttnHead, nLayer):
        super(KoiKoiEncoderBlock1,self).__init__()
        self.f1 = nn.Conv1d(nInput, nFw, 1)
        self.f2 = nn.Conv1d(nFw, nEmb, 1)
        attn_layer = nn.TransformerEncoderLayer(nEmb, nAttnHead, nFw)
        self.attn_encoder = nn.TransformerEncoder(attn_layer, 1)
        
    def forward(self,x): 
        x = self.f2(F.relu(self.f1(x)))
        x = F.layer_norm(x,[x.size(-1)])
        x = x.permute(2,0,1)
        x = self.attn_encoder(x) 
        return x


class AttnWeightModel(nn.Module):
    def __init__(self,n_layer):
        super(AttnWeightModel,self).__init__()
        if n_layer == 0:
            self.encoder_block = KoiKoiEncoderBlock0(**NetParameter)
        elif n_layer == 1:
            self.encoder_block = KoiKoiEncoderBlock1(**NetParameter)
        
    def forward(self,x):
        x = self.encoder_block(x)
        return x


def get_mha_container():
    in_proj_container = InProjContainer(torch.nn.Linear(256, 256),
                                    torch.nn.Linear(256, 256),
                                    torch.nn.Linear(256, 256))
    out_proj = torch.nn.Linear(256, 256)
    mha_container = MultiheadAttentionContainer(
        nhead = 4, in_proj_container = in_proj_container, 
        attention_layer = ScaledDotProduct(), out_proj = out_proj)
    return mha_container


def transfer_model(old_model, new_model, n_layer):
    new_model_state_dict = new_model.state_dict()
    old_model_state_dict = old_model.state_dict()
    update_state_dict = {k:v for k,v in old_model_state_dict.items() if k in new_model_state_dict.keys()}
    new_model_state_dict.update(update_state_dict)
    new_model.load_state_dict(new_model_state_dict)
    return new_model


def transfer_mha(model, mha_container, n_layer):
    od = model.state_dict()
    ud = mha_container.state_dict()
    s = f'encoder_block.attn_encoder.layers.{n_layer}.self_attn'
    ud['in_proj_container.query_proj.weight'] = od[f'{s}.in_proj_weight'][0:256,:]
    ud['in_proj_container.query_proj.bias'] = od[f'{s}.in_proj_bias'][0:256]
    ud['in_proj_container.key_proj.weight'] = od[f'{s}.in_proj_weight'][256:512,:]
    ud['in_proj_container.key_proj.bias'] = od[f'{s}.in_proj_bias'][256:512]
    ud['in_proj_container.value_proj.weight'] = od[f'{s}.in_proj_weight'][512:768,:]
    ud['in_proj_container.value_proj.bias'] = od[f'{s}.in_proj_bias'][512:768]
    ud['out_proj.weight'] = od[f'{s}.out_proj.weight']
    ud['out_proj.bias'] = od[f'{s}.out_proj.bias']
    mha_container.load_state_dict(ud)
    return mha_container


def draw_attn_bipartite(input_words, output_words, attentions, threshold, action_mask, head_color=0):
    # Github @liu-hz18/Visual-Attention
    # https://github.com/liu-hz18/Visual-Attention
    
    input_words = [word + '   ' for word in input_words]
    output_words = ['   ' + word for word in output_words]
    attn = attentions.detach().numpy().T
    left, right, bottom, top = .1, .9, .1, .9, 
    layer_sizes = [len(input_words), len(output_words)]
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)
    src_layer_left = left + h_spacing
    tgt_layer_left = left

    # add nodes and edges
    G = nx.Graph()
    for i in range(layer_sizes[0]):
        G.add_node(input_words[i], pos=(left + i*v_spacing, src_layer_left, ))
        for j in range(layer_sizes[1]):
            G.add_node(output_words[j], pos=(left + j*v_spacing, tgt_layer_left, ))
            if attn[i][j] > threshold and action_mask[i] > 0:
                G.add_edge(input_words[i], output_words[j], weight=attn[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = [edge[-1]['weight'] for edge in G.edges(data=True)]

    # draw graph
    plt.figure(figsize=(10,3))
    plt.box(on=None)
    plt.axis('off')
    color_map = [plt.cm.Blues, plt.cm.Purples, plt.cm.Oranges, plt.cm.Greens, ][head_color]
    nx.draw_networkx_nodes(G, pos, node_shape='o', alpha=0)
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=0.8, edge_cmap=color_map)
    show_label=False
    if show_label:
        nx.draw_networkx_labels(G, pos)
    edges.cmap = color_map

    return


if __name__ == '__main__':

    n_layer = 0 # 0 or 1
    model_path = 'model_agent/discard_rl_wp.pt'
    sample_path = 'dataset/discard/1_1_1_d.pickle'

    weight_filt_threshold = 0.2 # only draw edges of attention higher than threshold

    model = torch.load(model_path, map_location=torch.device('cpu'))
    encoder = AttnWeightModel(n_layer)
    encoder = transfer_model(model, encoder, n_layer)
    encoder.eval()
    
    mha_container = get_mha_container()
    mha_container = transfer_mha(model, mha_container, n_layer)
    mha_container.eval()
    
    with open(sample_path,'rb') as f:
        sample = pickle.load(f)
    feature = sample['feature'].unsqueeze(0)
    
    x = encoder(feature)
    x, w_all = mha_container(x,x,x)
    
    for head in [0,1,2,3]:
        w = w_all[head,:,:]
        src_label = [str(i) for i in range(48)]
        tgt_label = [str(i) for i in range(48)]
        # action_mask = sample['action_mask']
        action_mask = [1 for _ in range(48)]
        draw_attn_bipartite(src_label, tgt_label, w, weight_filt_threshold, action_mask, head)
        output = model(feature).squeeze(0)
    
    for ii in range(len(sample['action_mask'])):
        if sample['action_mask'][ii] > 0.1:
            print(f'{ii//4+1}-{ii%4+1} {output[ii]:.2f}')

