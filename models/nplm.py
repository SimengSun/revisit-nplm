'''
A module which implements the NPLM
'''
import uuid
import threading
import pdb
import sys
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


from models.attention import Attention
from models.embeddings import TokenEmbedding, PositionEmbedding
from models.adaptive_softmax import AdaptiveSoftmax
from models.utils import LabelSmoothingLoss
from utils import left_shift, right_shift, triu


class NPLMSublayer(nn.Module):

    def __init__(self, sublayer, do_add, no_layernorm, sublayer_shape, dropout_p=0.1, init_std=0.02):
        ''' Initialize the NPLM sublayer '''
        super(NPLMSublayer, self).__init__()
        self.init_std = init_std
        self.sublayer = sublayer
        self.sublayer_shape = sublayer_shape
        self.do_add = do_add
        self.norm = nn.LayerNorm(sublayer_shape) if not no_layernorm else None
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        # ''' Reset parameters using xavier initialiation '''
        if self.norm is not None:
            nn.init.normal_(self.norm.weight, 1.0, self.init_std)

    def forward(self, inputs, *sublayer_args, **sublayer_kwargs): # pylint:disable=arguments-differ
        ''' The forward pass of the sublayer '''
        if self.do_add: 
            if inputs.size(2) != self.sublayer_shape:
                bsz, seq_len, dim = inputs.shape
                inputs = inputs.view(bsz, seq_len, -1, self.sublayer_shape).contiguous().sum(dim=-2)
            out = inputs + self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs))
            return out if self.norm is None else self.norm(out)
        else:
            out = self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs))
            return out if self.norm is None else self.norm(out)


class NPLMFF(nn.Module):
    ''' Implements the NPLM feed-forward network '''
    def __init__(self, input_dim, hidden_dim, init_std=0.02, output_proj=True, proj_dim=-1):
        super(NPLMFF, self).__init__()

        self.init_std = init_std
        self.relu = nn.ReLU()

        if proj_dim != -1: # if does intermediate projection
            self.hidden = nn.Linear(input_dim, proj_dim)
            self.output = nn.Linear(proj_dim, hidden_dim)

        else:
            self.hidden = nn.Linear(input_dim, hidden_dim)

            if output_proj:
                self.output = nn.Linear(hidden_dim, input_dim)

        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        nn.init.normal_(self.hidden.weight, 0., self.init_std)
        nn.init.constant_(self.hidden.bias, 0.)

        if hasattr(self, 'output'):
            nn.init.normal_(self.output.weight, 0., self.init_std)
            nn.init.constant_(self.output.bias, 0.)

    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' The forward pass of the feed-forward network '''
        if hasattr(self, 'output'):
            return self.output(self.relu(self.hidden(inputs)))
        else:
            return self.relu(self.hidden(inputs))

class NPLMLayer(nn.Module):
    ''' Implements a single decoder layer in a NPLM decoder stack '''
    def __init__(self, config, num_heads, dim, hidden_dim, layer_i, 
                 dropout_p=0.1):
        ''' Initialize the NPLM layer '''
        super(NPLMLayer, self).__init__()
        self.config = config
        self.uuid = uuid.uuid4() 

        # ngm: n tokens that concat with full embs
        # wsz: window size to average for long term context
        self.ngm, self.wsz = config.context_config    
        self.long_term_block = 0 if self.ngm > 0 and self.wsz == -1 else \
                                    (config.batch_length - self.ngm) // self.wsz
        self.long_term_block *= self.config.num_global_agg

        self.emb_dim = dim
        self.dim_concat_embs = self.ngm * dim + self.long_term_block * dim
        
        self.hidden_dim = hidden_dim
        self.num_layers = config.num_layers
        
        if layer_i in config.concat_layers:

            if self.config.global_aggregate == 'kernel':
                for i in range(self.long_term_block):
                    setattr(self, f'learned_global_kernels_l{layer_i}_b{i}', 
                            nn.Parameter(torch.tensor(1./self.wsz).repeat(self.wsz)[None, None, :, None]\
                                .repeat(self.config.num_global_agg, 1, 1, 1),requires_grad=True))


            self.ffn_nplm = NPLMSublayer(
                        NPLMFF(self.dim_concat_embs, 
                               self.emb_dim, 
                               output_proj=False,
                               proj_dim=self.config.mid_dim), 
                        True,
                        config.no_layernorm,
                        self.emb_dim, dropout_p)

            if self.config.TFN: # if transformer-N, include self-attention

                self.self_attention = NPLMSublayer(
                    Attention(config, dim, num_heads), 
                            True, 
                            config.no_layernorm,
                            self.emb_dim, dropout_p)

                self.ffn = NPLMSublayer(NPLMFF(self.emb_dim, self.hidden_dim), 
                        True, 
                        config.no_layernorm,
                        self.emb_dim, dropout_p)

        else:
            if self.config.TFN:
                self.self_attention = NPLMSublayer(
                    Attention(config, dim, num_heads),
                            True, 
                            config.no_layernorm,
                            dim, dropout_p)

            self.ffn = NPLMSublayer(
                        NPLMFF(self.emb_dim, self.hidden_dim), 
                        True,
                        config.no_layernorm,
                        self.emb_dim, dropout_p)
    
    _kernels = threading.local()
    def _get_kernel(self, device):
        kernel_store = NPLMLayer._kernels.__dict__
        if device not in kernel_store:
            kernel_store[device] = torch.tensor(1./self.wsz).repeat(self.wsz)[None, None, :, None].to(device)

        return kernel_store[device]

    _masks = threading.local()
    def mask(self, inputs):
        dim = inputs.shape[1]
        device = inputs.device
        mask_store = NPLMLayer._masks.__dict__
        if device not in mask_store:
            mask = inputs.new_full((dim, dim), float('-inf'))
            mask_store[device] = triu(mask, 1, 1, 1)

        mask = mask_store[device]
        return mask[None, :dim, :dim]

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        if hasattr(self, 'ffn'):
            self.ffn.reset_parameters()
        if hasattr(self, 'ffn_nplm'):
            self.ffn_nplm.reset_parameters()
        if hasattr(self, 'self_attention'):
            self.self_attention.reset_parameters()

    def forward(self, inputs, layer_i=0, global_mem=0): # pylint:disable=arguments-differ
        ''' The forward pass '''

        state = inputs['state']             # bsz x L x emb_dim
        cache = inputs.get('cache')
        decoder_position = state.shape[1] - 1
        ngm, wsz = self.ngm, self.wsz
        dim = self.emb_dim

        # embedding concatenation layer
        if layer_i in self.config.concat_layers: 
            bsz, L, emb_dim = state.shape

            state_ = state.new_full((bsz, L, self.dim_concat_embs), 0.)
            for i in range(ngm):
                state_[:, i:, i*emb_dim : (i+1)*emb_dim] = state[:, : L-i, :]

            ltb = min((L - ngm) // wsz, self.long_term_block // self.config.num_global_agg)
            ltb = min((L - ngm) // wsz * self.config.num_global_agg, self.long_term_block) \
                    if self.config.global_aggregate == 'average' else ltb

            # distant context is chunked to multiple blocks
            # for each block, apply learned kernel or simply uniform average with conv1d
            # then paste the output to the `state' variable
            for  i in range(ltb):
                if self.config.global_aggregate == 'average':
                    conv_tmp = F.conv1d(state[:, None, : - ngm - i*wsz], 
                                        self._get_kernel(state.device),
                                        padding=(wsz-1,0))[:, :, :-wsz+1].squeeze(1)
                    state_[:, ngm + i * wsz:, (ngm+i) * dim: (ngm+i+1) * dim] = conv_tmp

                elif self.config.global_aggregate == 'kernel':
                    conv_tmp = F.conv1d(state[:, None, : - ngm - i*wsz],  
                                        getattr(self, f'learned_global_kernels_l{layer_i}_b{i}'),
                                        padding=(wsz-1,0))[:, :, :-wsz+1].squeeze(1)

                    conv_tmp = conv_tmp.transpose(2, 1).contiguous().view(bsz, -1, dim * ltb * self.config.num_global_agg)
                    state_[:, ngm + i * wsz:, (ngm + i) * dim: ] = conv_tmp

            # store global representation 
            _, global_l, global_dim = state.shape
            self.global_mem = state_[:, :, ngm * dim:].view(bsz, global_l, -1, emb_dim).contiguous()
            self.global_mem = self.global_mem.sum(dim=-2)

            state = self.ffn_nplm(state_, state_)
            
            # Transformer-N
            if self.config.TFN:
                assert hasattr(self, 'self_attention')
                kwargs = {'layer_i': layer_i}
                if cache is None:                                       
                    residual = state
                    kwargs['attention_mask'] = self.mask(state)       
                else:
                    residual = state[:, -1:]
                    kwargs['decoder_position'] = decoder_position

                state = self.self_attention(residual,                                           
                                            state, state, state, **kwargs)
                state = self.ffn(state, state)

        else:
            # regular Transformer layer
            if self.config.TFN:
                assert hasattr(self, 'self_attention')
                if hasattr(self, 'self_attention'):
                    kwargs = {'layer_i': layer_i}
                    if cache is None:                                       
                        residual = state
                        kwargs['attention_mask'] = self.mask(state)       
                    else:
                        residual = state[:, -1:]
                        kwargs['decoder_position'] = decoder_position

                    state = self.self_attention(residual,                                           
                                                state, state, state, **kwargs)
                state = self.ffn(state, state)

            # regular NPLM layer
            else:
                state = self.ffn(state, state)
                state = state + global_mem # add the global representation in the concat layer

        if cache is not None:
            cached = cache.get(self.uuid)
            state = cache[self.uuid] = torch.cat((cached, state), 1)

        return {'state': state, 'cache': cache}


class NPLM(nn.Module):
    ''' The neural proababilistic LM module '''
    def __init__(self, config, dataset):
        ''' Initialize'''
        super(NPLM, self).__init__()

        self.dataset = dataset
        
        self.adaptive = config.adaptive
        # ngm: n tokens that concat with full emb
        # wsz: window size to average for long term context
        self.ngm, self.wsz = config.context_config                  
        self.long_term_block = 0 if self.ngm > 0 and self.wsz == -1 else \
                                    (config.batch_length - self.ngm) // self.wsz

        self.dim_concat_embs = self.ngm * config.embedding_size + self.long_term_block * config.embedding_size

        self.embedding = TokenEmbedding(
                dataset.vocab_size,
                config.embedding_size,
                config.model_size, 
                config.cutoffs,
                emb_std=config.emb_std,
                proj_std = config.proj_std,
                div_val=config.div_val,
                padding_idx=self.padding_idx,
                do_proj=config.do_proj
            )

        if self.adaptive:
            self.adaptive_softmax = AdaptiveSoftmax(self.dataset.vocab_size, config.embedding_size, config.embedding_size, 
                                                    config.cutoffs, div_val=config.div_val)

            self.tie_weights = config.tie_weights
            self.tie_projs = config.tie_projs

            if self.tie_weights:
                for i in range(len(self.adaptive_softmax.out_layers)):
                    self.adaptive_softmax.out_layers[i].weight = self.embedding.emb_layers[i].weight

            if self.tie_projs:
                for i in range(1, len(self.adaptive_softmax.out_projs)):
                    if config.div_val == 1 and config.model_size != config.embedding_size:
                        self.adaptive_softmax.out_projs[i] = self.embedding.emb_projs[0]
                    elif config.div_val != 1:
                        self.adaptive_softmax.out_projs[i] = self.embedding.emb_projs[i]

        self.layers = self.create_layers(config)
        self.position_embedding = PositionEmbedding(config.model_size) # only used in transformer-N
        self.label_smoothing = LabelSmoothingLoss(
            config.label_smoothing or 0,
            ignore_index=self.padding_idx,
            reduction='none'
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            reduction='none'
        )

        self.dropout = nn.Dropout(config.dropout_p, inplace=True)

        self.config = config


    @classmethod
    def create_layers(self, config):
        ''' Create the NPLM decoders '''
        kwargs = {'dropout_p': config.dropout_p}                    # sublayer kwargs

        args = [config, config.num_heads, config.embedding_size, config.hidden_dim]

        layers = nn.ModuleList([
            NPLMLayer(*args, layer_i, **kwargs)
            for layer_i in range(config.num_layers)
        ])

        return layers

    @property
    def padding_idx(self):
        return self.dataset.padding_idx

    @property
    def eos_idx(self):
        return  self.dataset.eos_idx

    def reset_named_parameters(self, modules):

        if 'layers' in modules:
            for layer in self.layers:
                layer.reset_parameters()

        if 'embeddings' in modules:
            self.embedding.reset_parameters()

    def forward(self, batch): # pylint:disable=arguments-differ

        batch = batch.t()
        targets = left_shift(batch)
        decoded = self.decode(right_shift(batch))

        state = decoded['state']

        if not self.adaptive:
            logits = self.embedding(state, reverse=True).transpose(2, 1)
            dims = list(range(1, logits.dim()))
            nll = self.cross_entropy(logits, targets).view(-1)
            smoothed_nll = self.label_smoothing(logits, targets).sum(dims)

            if not self.config.return_rank:
                return smoothed_nll, nll

            else:
                logits = logits.transpose(2, 1)
                assert targets.shape[0] == 1
                targets = targets.squeeze(0)
                target_logits = logits[:, range(targets.shape[0]), targets]
                rank = (logits > target_logits.unsqueeze(-1)).sum(dim=-1)
                return rank, nll

        else:
            state = state.view(-1, state.shape[-1]) # (bsz*L, embed_dim)
            targets = targets.contiguous().view(-1) # (bsz*L, )

            if not self.config.return_rank:
                nll = self.adaptive_softmax(state, targets, keep_order=True)
                smoothed_nll = nll
                return smoothed_nll, nll

            else:
                nll, rank = self.adaptive_softmax(state, targets, keep_order=True, return_rank=True)
                return rank, nll

        return smoothed_nll, nll

    def decode(self, batch, cache=None):
        ''' if targest is not None,  '''
        word_embedding = self.embed(batch, self.embedding)

        decoded = {
            'cache': cache,
            'state': word_embedding,
        }

        # concat layer
        decoded = self.layers[0](decoded, layer_i=0)
        global_mem = self.layers[0].global_mem

        # regular layers
        for i, decoder in enumerate(self.layers[1:]):
            decoded = decoder(decoded, layer_i=i+1, global_mem=global_mem)

        # compute projection to the vocabulary
        state = decoded['state']
        if cache is not None:
            state = state[:, -1:]       # fetch newly generated tok

        return {
            'cache': decoded.get('cache'),
            'state': state,            # bs x L x dim_emb or bs x L x hidden_dim
        }

    def embed(self, inputs, token_embedding):
        ''' Embed the given inputs, no position embedding '''
        if self.config.TFN:
            return self.dropout(token_embedding(inputs) + self.position_embedding(inputs))
        else:
            return self.dropout(token_embedding(inputs))

