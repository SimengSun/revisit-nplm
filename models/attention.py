'''
A module which implements various attention mechanisms
'''
import pdb
import math
import torch
import time
import threading
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.multiprocessing as mp
from collections import defaultdict

from utils import same_tensor
import models.utils as utils

class Attention(nn.Module):
    ''' Implement a hard-coded attention module '''
    ATTN_TYPES = ['normal', 'learned']

    def __init__(self, config, embed_dim, num_heads=1, init_std=0.02):
        ''' Initialize the attention module '''
        super(Attention, self).__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'
        self.init_std = init_std

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = config.num_layers
        self.projection_dim = embed_dim // num_heads
        self.scale = self.projection_dim ** -0.5
        self.no_output_projection = config.attn_no_output_proj
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # hard-coded attn param
        self.attn_type = config.attn_type                 # learned / normal
        self.attn_offset = config.attn_offset             # -n represents center around left nth token from current; 0 current; n right nth token from current
        self.attn_std = config.attn_std                   # std of hard-coded gaussian
        self.attn_offset_window=config.attn_offset_window

        # conv param
        self.attn_threshold = config.attn_threshold
        self.attn_window = config.attn_window
        self.half_window = int((self.attn_window - 1) / 2)

        # attn implementation param
        self.attn_impl = config.attn_impl                  # full, conv, index
        
        # Combine projections for multiple heads into a single linear layer for efficiency
        self.attn_linear_transform = config.attn_weights  # 0: no weights; 1: no values
        self.input_weights = None
        if self.attn_linear_transform:
            if 'learned' in self.attn_type: # k,q,v all need linear transform
                self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
            else:                           # only v needs linear transform
                self.input_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        self.reset_parameters()
        self.attn_configs = list(self.load_attn_configs())

        
        if torch.cuda.device_count() > 1: # if using multi-gpu
            # self.max_qlen = math.ceil((config.batch_length*config.batch_size - config.bsz_gpu0*config.batch_size) / (torch.cuda.device_count() - 1) / config.batch_size)
            self.max_qlen = config.batch_length
        else:
            self.max_qlen = config.batch_length

        self.max_absolute_offset = max([abs(a) for a in config.attn_offset])
        self.max_uniform_pad = self.max_absolute_offset + max([abs(a) for a in config.attn_offset_window]) - 1
        self.attn_ofs_uniq = list(set(config.attn_offset))
        self.attn_std_uniq = list(set(config.attn_std))
        self.attn_ofs_wd_uniq = list(set(config.attn_offset_window))

    _attn_indices = threading.local()
    def get_attn_indices(self, qlen, attn_offset, device):

        attn_idx_store = Attention._attn_indices.__dict__

        if device not in attn_idx_store:
            indices_q = torch.arange(self.max_absolute_offset, self.max_qlen + self.max_absolute_offset).view(1, -1)
            attn_ofs_uniq = torch.tensor(self.attn_ofs_uniq).view(-1, 1)
            attn_idx_store[device] = (indices_q + attn_ofs_uniq).to(device)   
        offset_idx = [self.attn_ofs_uniq.index(i) for i in attn_offset]
        return attn_idx_store[device][offset_idx, :qlen][None, :, :, None] # 1, nh, qlen, 1    

    _attn_cache = threading.local()
    def get_attn_cache(self, attn_std, attn_offset, qlen, device, decoder_position=-1 ):

        attn_cache_store = Attention._attn_cache.__dict__

        if device not in attn_cache_store:
            num_std = len(self.attn_std_uniq)
            max_offset, min_offset = max(self.attn_ofs_uniq), min(self.attn_ofs_uniq)

            max_vlen = self.max_qlen + max_offset - min_offset
            attn_std_uniq = torch.tensor(self.attn_std_uniq).view(-1, 1, 1)
            indices_q = torch.arange(self.max_qlen).float().view(1, -1, 1)
            indices_v = torch.arange(-max_offset, self.max_qlen - min_offset).float().view(1, 1, -1) # -max_offset: focus on right most position, self.max_qlen - min_offset: leftmost
            
            distance_diff = indices_v - indices_q
            logits = (1 / (attn_std_uniq * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / attn_std_uniq) ** 2))
            if self.attn_threshold > 0:
                logits[logits < self.attn_threshold] = 0

            attn_cache_store[device] = logits.to(device) 

        std_idx = [self.attn_std_uniq.index(i) for i in attn_std]
        attn_ofs_l = np.array([-1 - a for a in attn_offset])
        attn_ofs_r = np.array([-1 - a + qlen for a in attn_offset]) 

        retrieved = attn_cache_store[device] # nh x qlen x vlen
        retrieved = retrieved[[[[a]] for a in std_idx], [[[b] for b in list(range(qlen))]], [[list(range(l, r))] for l, r in zip(attn_ofs_l, attn_ofs_r)]]
        
        if decoder_position == -1:
            return retrieved[:, :qlen, :qlen]

        else:
            return retrieved[:, decoder_position, :qlen].view(1, 1, 1, -1)

    _attn_uniform_cache = threading.local()
    def get_attn_uniform_cache(self, attn_offset, attn_offset_window, qlen, device, decoder_position=-1):

        attn_cache_store = Attention._attn_uniform_cache.__dict__
        if device not in attn_cache_store:

            # #(wd) x qlen x max_uf_pad
            logits = torch.zeros((len(self.attn_ofs_wd_uniq), self.max_qlen, self.max_qlen + self.max_uniform_pad), device=device)
            for i, wd in enumerate(self.attn_ofs_wd_uniq):
                logits[i, [[a] for a in range(self.max_qlen)], [list(range(s, s + wd)) for s in range(self.max_qlen)]] = 1. / wd

            attn_cache_store[device] = logits

        ofs_wd_idx = [self.attn_ofs_wd_uniq.index(i) for i in attn_offset_window]
        attn_ofs_l = np.array([- a for a in attn_offset])
        attn_ofs_r = np.array([- a + qlen for a in attn_offset]) 

        retrieved = attn_cache_store[device] # nh x qlen x vlen
        retrieved = retrieved[[[[a]] for a in ofs_wd_idx], [[[b] for b in list(range(qlen))]], [[list(range(l, r))] for l, r in zip(attn_ofs_l, attn_ofs_r)]]
        if decoder_position == -1:
            return retrieved[:, :qlen, :qlen]

        else:
            return retrieved[:, decoder_position, :qlen].view(1, 1, 1, -1)

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        if self.input_weights is not None:
            nn.init.normal_(self.input_weights, 0., self.init_std)
        nn.init.normal_(self.output_projection.weight, 0., self.init_std)


    def project(self, inputs, index=0, chunks=1, project=True):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim

        if project:
            projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1) 
        else:
            projections = [inputs]

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def project_learned(self, inputs, learned_idx):
        batch_size = int(inputs.shape[0] / self.num_heads)
        return inputs.view(batch_size,
                           self.num_heads,
                           -1,
                           self.projection_dim)[:, learned_idx].contiguous()\
            .view(batch_size * len(learned_idx),
                  -1,
                  self.projection_dim)

    def load_attn_configs(self):

        """ 
            expand attn_configs for each layer 

            c: input attn_configs; h: head; l: layer
            len(c) == 1 :         all attn heads are the same
            len(c) == #h :        each layer use the same head combinations
            len(c) == #l:         each layer uses same config for each head, different layers use different attn_configs
            len(c) == #h x #l :   all attn_configs specified, layer_0_head_0, layer_0_head_1, ..., layer_(l-1)_head_(h-1)
            #heads % len(c) == 0: repeat head attn_configs to #heads for each layer

        """

        for li in range(self.num_layers):

            attn_configs = {}

            for attr in self.__dict__:
                if not attr.startswith('attn_'):
                    continue

                c = getattr(self, attr)

                if attr == 'attn_impl': # only for attn_impl, expand to #layers instead of #layers x #heads
                    if len(c) == 1:
                        attn_configs[attr] = c[0]

                    else:
                        assert len(c) == self.num_layers
                        attn_configs[attr] = c[li]
                    continue

                if type(c) is not list:
                    continue

                if len(c) == 1:
                    c = c * self.num_heads

                elif len(c) == self.num_layers * self.num_heads:
                    c = c[li * self.num_heads : (li + 1) * self.num_heads]

                elif len(c) == self.num_heads:
                    c = c

                elif len(c) == self.num_layers:
                    c = [c[li]] * self.num_heads

                elif self.num_heads % len(c) == 0:
                    c *= self.num_heads // len(c)

                else:
                    raise ValueError('wrong head attn_configs')

                attn_configs[attr] = c

            if self.attn_window == -1:
                yield attn_configs, None
                continue

            with torch.no_grad():
                
                distance_diff = torch.arange(-self.half_window, self.half_window + 1, dtype=torch.float32, device=torch.device("cuda"))
                # conv_filter: (self.window_size,)

                conv_filters = {} # conv_filters[std][offset] stores conv filters

                attn_std, attn_offset = attn_configs['attn_std'], attn_configs['attn_offset']
                head_configs = defaultdict(list)
                for i, c in enumerate(zip(attn_std, attn_offset)):
                    head_configs[c].append(i)

                for hc, idx in head_configs.items():
                    attn_std, attn_offset = hc[0], hc[1]
                    conv_filter = (1 / (attn_std * math.sqrt(2 * math.pi)) * torch.exp(- 1 / 2 * (distance_diff / attn_std) ** 2)).view(1, 1, -1)
                    conv_filter[self.attn_window - (self.half_window + attn_offset):] = 0
                    conv_filters[attn_std][attn_offset] = conv_filter

            yield attn_configs, conv_filters

    def mha_reshape(self, tensor, batch_size):
        '''
            multi-headed attention reshape
            tensor.shape = B*H x L x proj_dim
            output tensor.shape = B x L x E
        '''

        return tensor.view(batch_size, self.num_heads, -1, 
                           self.projection_dim).transpose(2, 1).contiguous().view(
                            batch_size, -1, self.num_heads * self.projection_dim
                           )

    def gather_reshape(self, tensor, attn_indices, bsz, qlen, dp):
        '''
            used in `conv` and `indexing` implementation
            dp: decoder position
        '''
        if dp == -1:
            return torch.gather(tensor, 2, attn_indices.expand(
                    bsz, self.num_heads, qlen, self.projection_dim)
                    ).transpose(2,1).contiguous().view(bsz, -1, self.num_heads * self.projection_dim)
        else:
            return torch.gather(tensor, 2, attn_indices[:, :, dp:dp+1].expand(
                    bsz, self.num_heads, qlen, self.projection_dim)
                    ).transpose(2,1).contiguous().view(bsz, -1, self.num_heads * self.projection_dim)

    def attention_uniform(self, values, keys, queries, mask=None, layer_i=0, decoder_position=-1):

        queries_shape = queries.shape  # B*H x L x E
        values_shape = values.shape
        batch_size = queries_shape[0] // self.num_heads

        attn_configs, _ = self.attn_configs[layer_i]
        attn_offset, attn_ofs_window = attn_configs['attn_offset'], attn_configs['attn_offset_window']

        # uniform attention weight num_heads x qlen x vlen
        attn_weights = self.get_attn_uniform_cache(attn_offset, attn_ofs_window, queries_shape[1], values.device, decoder_position)

        # expand
        attn_weights = attn_weights.expand(batch_size, self.num_heads, queries_shape[1], values_shape[1]).contiguous().view(-1, queries_shape[1], values_shape[1])

        if mask is not None:
            attn_weights = attn_weights * (mask == 0).to(dtype=torch.float32)

        attended = torch.bmm(attn_weights, values)
        
        return self.mha_reshape(attended, batch_size)


    def attention_index(self, values, keys, queries, mask=None, layer_i=0, decoder_position=-1):
        queries_shape = queries.shape  # B*H x L x E
        values_shape = values.shape
        batch_size = queries_shape[0] // self.num_heads

        attn_configs, _ = self.attn_configs[layer_i]
        attn_type, attn_std, attn_offset = attn_configs['attn_type'], attn_configs['attn_std'], attn_configs['attn_offset']

        # bs x num_heads x vlen x proj_dim
        values = values.view(batch_size, self.num_heads, values_shape[1], values_shape[2])

        values = F.pad(values, (0, 0, self.max_absolute_offset, self.max_absolute_offset), "constant", 0)
        # recompute attended indices
        attn_indices = self.get_attn_indices(max(queries_shape[1], decoder_position+1), attn_offset, values.device)

        return self.gather_reshape(values, attn_indices, batch_size, queries_shape[1], decoder_position)

    def attention_conv(self, values, keys, queries, mask=None, layer_i=0, decoder_position=-1):

        queries_shape = queries.shape  # B*H x L x proj_dim
        values_shape = values.shape
        batch_size = queries_shape[0] // self.num_heads

        attn_configs, conv_filters = self.attn_configs[layer_i]
        attn_type, attn_std, attn_offset = attn_configs['attn_type'], attn_configs['attn_std'], attn_configs['attn_offset']
 
        curr_conv_filter = []
        for i in range(self.num_heads):
            curr_conv_filter.append(conv_filters[attn_std[i]][attn_offset[i]])
        curr_conv_filter = torch.cat(curr_conv_filter, dim=0)

        values = values.view(batch_size, self.num_heads, values_shape[1], values_shape[2])

        values = values.transpose(3,1).transpose(3,2).contiguous().view(batch_size * self.projection_dim, self.num_heads, -1)
        attended = F.conv1d(values, curr_conv_filter, padding=self.half_window + self.max_absolute_offset, groups=self.num_heads)
        attended = attended.view(batch_size, self.projection_dim, self.num_heads, -1).transpose(1,2).transpose(2,3).contiguous()

        # recompute attended indices
        attn_indices = self.get_attn_indices(max(queries_shape[1], decoder_position+1), attn_offset, values.device)

        return self.gather_reshape(attended, attn_indices, batch_size, queries_shape[1], decoder_position)

    def compute_together(self, attn_type, attn_std, attn_offset):

        return len(set(attn_type)) == 1 and \
               len(set(attn_std)) == 1 and \
               len(set(attn_offset)) == 1

    def attention(self, values, keys, queries, mask=None, layer_i=0, decoder_position=-1):

        queries_shape = queries.shape  # B*H x L x proj_dim
        values_shape = values.shape
        batch_size = queries_shape[0] // self.num_heads

        attn_configs, _ = self.attn_configs[layer_i]
        attn_type, attn_std, attn_offset = attn_configs['attn_type'], attn_configs['attn_std'], attn_configs['attn_offset']
        if all(a == 'learned' for a in attn_type): # all heads are learned
            logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))
            if mask is not None:
                logits += mask[:, :queries_shape[1], :queries_shape[1]]
            
            invalid_idx = (logits == float('-inf')).sum(dim=-1) == queries_shape[1]
            if invalid_idx.sum() != 0: # need to replace invalid with fixed vectors
                valid_idx = (logits == float('-inf')).sum(dim=-1) != queries_shape[1]
                attn_weights = logits.new_zeros(logits.shape)
                attn_weights[valid_idx] = F.softmax(logits[valid_idx], dim=-1)
                attended_ = torch.bmm(attn_weights, values)

                # z = torch.zeros_like(attended_)
                # fill = attended_.new_zeros((1, queries_shape[2]))
                # fill[:, :queries_shape[2]//2] = 0.1
                # fill[:, queries_shape[2]//2:] = -0.1

                # z[invalid_idx] = fill
                # attended = attended_ + z
                attended = attended_

            else:
                attn_weights = F.softmax(logits, dim=-1)
                attended = torch.bmm(attn_weights, values)

            batch_size = queries_shape[0] // self.num_heads

            return self.mha_reshape(attended, batch_size)

        elif 'learned' in attn_type:

            learned_indices = [i for i, x in enumerate(attn_type) if x == 'learned']
            queries_ = self.project_learned(queries, learned_indices)
            keys_ = self.project_learned(keys, learned_indices)
            values_ = self.project_learned(values, learned_indices)

            logits_ = self.scale * torch.bmm(queries_, keys_.transpose(2, 1))
            logits_shape_ = logits_.shape

            if mask is not None:
                logits_ += mask[:, :queries_shape[1], :queries_shape[1]]


            logits_ = F.softmax(logits_, dim=-1).view(batch_size,
                                                      len(learned_indices),
                                                      logits_shape_[-2],
                                                      logits_shape_[-1])
            logits_[logits_ != logits_] = 0
            learned_idx = 0 

        with torch.no_grad():

            # if config for all heads in the same layer is the same, compute them together
            if self.compute_together(attn_type, attn_std, attn_offset):

                attn_type = attn_type[0]
                attn_std = [attn_std[0]]
                attn_offset = [attn_offset[0]]

                logits = self.get_attn_cache(attn_std, attn_offset, queries_shape[1], values.device, decoder_position=decoder_position)

                # Copy the weights to each head
                attn_weights = logits.expand(batch_size, self.num_heads, queries_shape[1], values_shape[1])\
                    .contiguous().view(-1, queries_shape[1], values_shape[1])

            # if not all heads have the same config
            else:

                logits_list = []

                hc_indices = [i for i, t in enumerate(attn_type) if t != 'learned']
                attn_std = [x for i, x in enumerate(attn_std) if i in hc_indices]
                attn_offset = [x for i, x in enumerate(attn_offset) if i in hc_indices]
                logits = self.get_attn_cache(attn_std, attn_offset, queries_shape[1], values.device, decoder_position=decoder_position)
                attn_weights = values.new_zeros(batch_size, self.num_heads, queries_shape[1], values_shape[1])

                attn_weights[:, hc_indices] = logits.expand(batch_size, len(hc_indices), queries_shape[1], values_shape[1])

                if 'learned' in attn_type:
                    attn_weights[:, learned_indices] = logits_ # bs x learned_indices x L x L

                attn_weights = attn_weights.contiguous().view(-1, queries_shape[1], values_shape[1])
       
        if mask is not None:
            attn_weights = attn_weights * (mask == 0).to(dtype=torch.float32)

        attended = torch.bmm(attn_weights, values)

        return self.mha_reshape(attended, batch_size)

    def forward(self, values, keys, queries, attention_mask=None,layer_i=0, decoder_position=-1): 
        ''' Forward pass of the attention '''
        batch_size = values.shape[0]

        if 'learned' in self.attn_type:

            if self.attn_linear_transform: 

                if same_tensor(values, keys, queries):
                    values, keys, queries = self.project(values, chunks=3)

                elif same_tensor(values, keys):
                    values, keys = self.project(values, chunks=2)
                    queries, = self.project(queries, 2)

                else:
                    values, = self.project(values, 0)
                    keys, = self.project(keys, 1)
                    queries, = self.project(queries, 2)

            else:

                values, = self.project(values, 0, project=False)
                keys, = self.project(keys, 1, project=False)
                queries, = self.project(queries, 2, project=False)
                
        else:
            if self.attn_linear_transform:
                values = F.linear(values, self.input_weights)

            values, = self.project(values, 0)
            queries, = self.project(queries, 0, project=False) # BHxTxE , do we need this??

        if self.attn_configs[layer_i][0]['attn_impl'] == 'full' :
            attended = self.attention(values, keys, queries, 
                                  attention_mask, layer_i, decoder_position)

        elif self.attn_configs[layer_i][0]['attn_impl'] == 'conv':
            attended = self.attention_conv(values, keys, queries, 
                                  attention_mask, layer_i, decoder_position)

        elif self.attn_configs[layer_i][0]['attn_impl'] == 'index':
            attended = self.attention_index(values, keys, queries, 
                                  attention_mask, layer_i, decoder_position)

        elif self.attn_configs[layer_i][0]['attn_impl'] == 'uniform':
            attended = self.attention_uniform(values, keys, queries, 
                                  attention_mask, layer_i, decoder_position)

        else:
            raise ValueError("implementation method undefined")

        return self.output_projection(attended) if not self.no_output_projection else attended

