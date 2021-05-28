'''
A module which implements various types of embeddings
'''
import threading
import pdb
import torch
from torch import nn

from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    ''' An embedding layer used for the transformer '''
    def __init__(self, num_embeddings, embedding_dim, proj_dim, cutoffs, emb_std=0.01, proj_std=0.02, div_val=1, padding_idx=0, do_proj=False):
        super(TokenEmbedding, self).__init__()

        self.vocab_size = num_embeddings
        self.embed_dim = embedding_dim
        self.proj_dim = proj_dim
        self.cutoffs = [0] + cutoffs + [self.vocab_size]
        self.div_val = div_val

        self.emb_scale = self.proj_dim ** 0.5
        self.emb_std = emb_std
        self.proj_std = proj_std
        self.do_proj = do_proj

        self.padding_idx = padding_idx

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ModuleList() # changing to moduleList because current pytorch does not replicate parameterlist when using multi-gpu
        if self.div_val == 1:
            self.emb_layers.append(
                nn.Embedding(self.vocab_size, self.embed_dim)
            )
            if self.proj_dim != self.embed_dim and self.do_proj:
                self.emb_projs.append(nn.Linear(self.proj_dim, self.embed_dim))
        else:
            for i in range(len(self.cutoffs) - 1):
                l_idx, r_idx = self.cutoffs[i], self.cutoffs[i+1]
                d_emb_i = self.embed_dim // (self.div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Linear(self.proj_dim, d_emb_i))

        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset params'''
        for l in self.emb_layers:
            if self.emb_std is not None:
                nn.init.normal_(l.weight, mean=0, std=self.emb_std)
            else:
                nn.init.normal_(l.weight, mean=0, std=self.embed_dim ** -0.5)

        for p in self.emb_projs:
            if self.proj_std is not None:
                nn.init.normal_(p.weight, mean=0, std=self.proj_std)
            else:
                nn.init.normal_(p.weight, mean=0, std=self.embed_dim ** -0.5)
            nn.init.constant_(p.bias, 0.)

        nn.init.constant_(self.emb_layers[0].weight[self.padding_idx], 0)

    def forward(self, inputs, reverse=False): # pylint:disable=arguments-differ
        ''' Implement the forward pass of the embedding '''

        if reverse:
            return F.linear(inputs, self.emb_layers[0].weight) # bs x L x dim    dim x |V| ==> bs x L x |V|
        else:
            # inputs: bs x L
            if self.div_val == 1:
                embed = self.emb_layers[0](inputs)
                if self.proj_dim != self.embed_dim and self.do_proj:
                    embed  = F.linear(embed, self.emb_projs[0].weight)
            else:
                param = next(self.parameters())
                inp_flat = inputs.contiguous().view(-1)
                emb_flat = torch.zeros([inp_flat.size(0), self.proj_dim], 
                    dtype=param.dtype, device=param.device)

                for i in range(len(self.cutoffs)-1):
                    l_idx, r_idx = self.cutoffs[i], self.cutoffs[i + 1]

                    mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                    indices_i = mask_i.nonzero().squeeze()

                    if indices_i.numel() == 0:
                        continue

                    inp_i = inp_flat.index_select(0, indices_i) - l_idx
                    emb_i = self.emb_layers[i](inp_i)
                    emb_i = F.linear(emb_i, self.emb_projs[i].weight.t())

                    emb_flat.index_copy_(0, indices_i, emb_i)

                embed = emb_flat.view(*inputs.size(), self.proj_dim)

            embed.mul_(self.emb_scale)

            return embed


class PositionEmbedding(nn.Module):
    ''' Produce position embeddings '''
    def __init__(self, dim, freq=1e4):
        ''' Initialize the PositionEmbedding '''
        super(PositionEmbedding, self).__init__()

        self.dim = dim # require the number of dimension to be even
        self.freq = freq

    _embeddings = threading.local()
    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' Implement the forward pass of the embedding '''
        device = inputs.device
        max_length = inputs.shape[1]
        embedding_store = PositionEmbedding._embeddings.__dict__
        device_store = embedding_store.get(device, {})
        if (
                not device_store or
                self.dim not in device_store or
                device_store[self.dim].shape[0] < max_length
        ):
            positions = torch.arange(0., max_length, device=device).unsqueeze(1)

            # the tensor2tensor code is slightly different than described in the paper
            # dividing by (self.dim - 2) produces nearly identical results to their version
            # when comparing the tensorflow results to these torch results
            dims = torch.arange(0., self.dim, 2., device=device).unsqueeze(0) / (self.dim - 2)

            sin = torch.sin(positions / torch.pow(self.freq, dims))
            cos = torch.cos(positions / torch.pow(self.freq, dims))

            embeddings = torch.stack((sin, cos), 0)
            device_store[self.dim] = embeddings.transpose(0, 1).contiguous().view(-1, self.dim)

        embeddings = device_store[self.dim]
        embedding_store[device] = device_store
        return embeddings[:max_length].unsqueeze(0)
