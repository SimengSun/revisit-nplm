'''
A module which implements the basic Transformer
'''
import uuid
import threading
import pdb
import sys
import torch
from torch import nn
import numpy as np

from models.attention import Attention
from models.embeddings import PositionEmbedding, TokenEmbedding
from models.adaptive_softmax import AdaptiveSoftmax
from models.utils import LabelSmoothingLoss
from utils import left_shift, right_shift, triu


class TransformerSublayer(nn.Module):
    '''
    Implements a sub layer of the transformer model, which consists of:
    1) A sub layer module
    2) Followed by dropout
    3) Plus a residual connection
    4) With layer normalization
    '''
    def __init__(self, sublayer, sublayer_shape, dropout_p=0.1, init_std=0.02):
        ''' Initialize the transformer sublayer '''
        super(TransformerSublayer, self).__init__()
        self.init_std = init_std
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(sublayer_shape)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        # ''' Reset parameters using xavier initialiation '''
        # self.norm.reset_parameters()
        nn.init.normal_(self.norm.weight, 1.0, self.init_std)

    def forward(self, inputs, *sublayer_args, **sublayer_kwargs): # pylint:disable=arguments-differ
        ''' The forward pass of the sublayer '''
        return self.norm(inputs + self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs)))


class TransformerFFN(nn.Module):
    ''' Implements the Transformer feed-forward network '''
    def __init__(self, embedding_size, hidden_dim, init_std=0.02):
        super(TransformerFFN, self).__init__()

        self.init_std = init_std
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(embedding_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, embedding_size)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        nn.init.normal_(self.hidden.weight, 0., self.init_std)
        nn.init.normal_(self.output.weight, 0., self.init_std)
        nn.init.constant_(self.hidden.bias, 0.)
        nn.init.constant_(self.output.bias, 0.)

    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' The forward pass of the feed-forward network '''
        return self.output(self.relu(self.hidden(inputs)))


class TransformerLayer(nn.Module):
    ''' Implements a single decoder layer in a transformer decoder stack '''
    def __init__(self, config, num_heads, dim, hidden_dim, layer_i, causal=True, 
                 dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerLayer, self).__init__()

        self.no_attention = config.no_attention[layer_i]
        self.FF2 = config.FF2

        self.uuid = uuid.uuid4() 

        self.ffn = TransformerSublayer(
            TransformerFFN(dim, hidden_dim),
            dim, dropout_p)

        if config.FF2:
            self.ffn2 = TransformerSublayer(
            TransformerFFN(dim, hidden_dim),
            dim, dropout_p)

        self.num_heads = num_heads
        self.self_attention = TransformerSublayer(
                Attention(config, dim, num_heads),
                dim, dropout_p) if not self.no_attention else None

        # unidirectional lm
        self.causal = True

        # enforce learned heads to look at local windows
        self.context_mask = config.context_mask * config.num_layers if len(config.context_mask) == 1 else config.context_mask
        assert len(self.context_mask) == config.num_layers

        # add constraint on learned attention
        if len(config.context_mask_left) == 1 and config.context_mask_left[0] == -1 \
            and len(config.context_mask_right) == 1 and config.context_mask_right[0] == -1:
            self.context_mask_left = [-1]
            self.context_mask_right = [-1]
        else:
            assert all([a <= 0 for a in config.context_mask_left]), "mask left bound invalid"
            assert all([a <= 0 for a in config.context_mask_right]), "mask right bound invalid"
            self.context_mask_left  = config.context_mask_left if len(config.context_mask_left) != 1 \
                                        else config.context_mask_left * config.num_heads
            self.context_mask_right = config.context_mask_right if len(config.context_mask_right) != 1 \
                                        else config.context_mask_right * config.num_heads
            assert len(config.context_mask_left) == config.num_heads, "num masks must equals to num heads"

        self.non_contiguous_mask = config.non_contiguous_mask
        if self.non_contiguous_mask:
            assert len(self.context_mask_left) != 1 and self.context_mask[0] != -1

        self.config = config

        self.sample_idx = None

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()
        if self.FF2:
            self.ffn2.reset_parameters()

    def forward(self, inputs, layer_i=0, global_mask=None): # pylint:disable=arguments-differ
        ''' The forward pass '''

        state = inputs['state']
        cache = inputs.get('cache')
        decoder_position = state.shape[1] - 1
        L = state.shape[1]
        
        kwargs = {'layer_i': layer_i}                           # each layer might have different config
        if cache is None:                                       # train/eval
            residual = state
            if self.context_mask[layer_i] == -1:
                if len(self.context_mask_left) != 1 and len(self.context_mask_right) != 1:
                    kwargs['attention_mask'] = self.constrained_masks(state) # causal mask + flexible context mask
                else:
                    kwargs['attention_mask'] = self.mask(state)    # just causal mask
            else:
                if self.non_contiguous_mask:
                    kwargs['attention_mask'] = self.non_contiguous_masks(state, 
                                                                         local_window_size=self.context_mask[layer_i])
                else:                                           # just local mask + causal mask
                    kwargs['attention_mask'] = self.masks(state,    
                                                self.context_mask[layer_i])     # causal mask + local context mask
        else:
            residual = state[:, -1:]
            kwargs['decoder_position'] = decoder_position       # test time
            
        state = self.self_attention(
            residual,                                           # residual
            state, state, state, **kwargs                       # passed to attention
        ) if not self.no_attention else state

        state = self.ffn(
            state,                                              # residual
            state                                               # passed to FF layer
        )

        state = self.ffn2(state, state) if self.FF2 else state

        if cache is not None:
            cached = cache.get(self.uuid)
            state = cache[self.uuid] = torch.cat((cached, state), 1)

        return {'state': state, 'cache': cache}

    _masks = threading.local()
    def mask(self, inputs):
        '''
        Get a self-attention mask
        The mask will be of shape [T x T] containing elements from the set {0, -inf}
        Input shape:  (B x T x E)
        Output shape: (T x T)
        '''
        if not self.causal:
            return None

        dim = inputs.shape[1]
        device = inputs.device
        mask_store = TransformerLayer._masks.__dict__
        if device not in mask_store or (device in mask_store and mask_store[device].shape[1] < dim):
            mask = inputs.new_full((dim, dim), float('-inf'))
            mask_store[device] = triu(mask, 1, 1, 1)

        mask = mask_store[device]
        return mask[None, :dim, :dim]

    _all_masks = threading.local()
    def masks(self, inputs, local_window_size):
        '''
        Stores the sum of causal mask and context mask
        '''
        dim = inputs.shape[1]
        device = inputs.device

        masks_store = TransformerLayer._all_masks.__dict__
        if device not in masks_store:
            causal_mask = self.mask(inputs)[0]
            context_mask = inputs.new_full((dim, dim), float('-inf'))
            context_mask = triu(context_mask, local_window_size, 1, 1).t()
            masks_store[device] = causal_mask + context_mask

        mask = masks_store[device]
        return mask[None, :dim, :dim]

    _flex_masks = threading.local()
    def constrained_masks(self, inputs):
        '''
        stores causal masks plus flexible masks, masked part == -inf 
        because the constraints are per heads, masks should be of shape bsz*nh x L x L
        '''
        
        device = inputs.device

        masks_store = TransformerLayer._flex_masks.__dict__
        if device not in masks_store:
            dim = inputs.shape[1]
            bsz = inputs.shape[0]
            causal_mask = self.mask(inputs)                                             # 1 x L x L
            # ct_mask = inputs.new_full((self.num_heads, dim, dim), -sys.maxsize*1.)
            ct_mask = inputs.new_full((self.num_heads, dim, dim), float('-inf'))        # nh x L x L
            for i, (l, r) in enumerate(zip(self.context_mask_left, self.context_mask_right)):
                ct_mask[i] = torch.tril(ct_mask[i], diagonal=-r-1).t() + torch.tril(ct_mask[i], diagonal=l-1)
            all_masks = causal_mask + ct_mask.repeat(bsz, 1, 1)
            masks_store[device] = all_masks.contiguous().view(bsz * self.num_heads, dim, dim)
            
        mask = masks_store[device]
        return mask

    _ncontig_masks = threading.local()
    def non_contiguous_masks(self, inputs, local_window_size=None):
        device = inputs.device
        local_mask = self.masks(inputs, local_window_size)
        flex_mask = self.constrained_masks(inputs)

        masks_store = TransformerLayer._ncontig_masks.__dict__
        if (device, local_window_size) not in masks_store:
            new_mask = flex_mask.clone()
            local_mask = local_mask.repeat(flex_mask.shape[0], 1, 1)
            new_mask[local_mask == 0] = 0
            masks_store[(device, local_window_size)] = new_mask

        mask = masks_store[(device, local_window_size)]
        return mask

class Transformer(nn.Module):
    ''' The Transformer LM module '''
    def __init__(self, config, dataset):
        ''' Initialize the Transformer '''
        super(Transformer, self).__init__()

        self.dataset = dataset
        self.config = config
        
        self.adaptive = config.adaptive

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
            self.adaptive_softmax = AdaptiveSoftmax(self.dataset.vocab_size, config.embedding_size, config.model_size, 
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

        self.position_embedding = PositionEmbedding(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout_p, inplace=True)

        if len(config.no_attention) == 1:
            config.no_attention = config.no_attention * config.num_layers
        assert len(config.no_attention) == config.num_layers

        self.layers = self.create_layers(config)

        self.label_smoothing = LabelSmoothingLoss(
            config.label_smoothing or 0,
            ignore_index=self.padding_idx,
            reduction='none'
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            reduction='none'
        )
    @classmethod
    def create_layers(self, config):
        ''' Create the transformer decoders '''
        kwargs = {'dropout_p': config.dropout_p}                    # sublayer kwargs

        args = [config, config.num_heads, config.model_size, config.hidden_dim]

        layers = nn.ModuleList([
            TransformerLayer(*args, layer_i, **kwargs)
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

    def forward(self, batch, global_mask=None): # pylint:disable=arguments-differ
        ''' batch: length x bsz'''

        batch = batch.transpose(1, 0)
        targets = left_shift(batch)
        decoded = self.decode(right_shift(batch), global_mask=global_mask)

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
            if self.config.batch_length < state.size(1):
                state = state[:, -self.config.batch_length:].contiguous()
                targets = targets[:, -self.config.batch_length:].contiguous()

            state = state.view(-1, state.shape[-1]) # (bsz*L, embed_dim)
            targets = targets.contiguous().view(-1) # (bsz*L, )

            if not self.config.return_rank:
                nll = self.adaptive_softmax(state, targets, keep_order=True)
                smoothed_nll = nll
                return smoothed_nll, nll

            else:
                nll, rank = self.adaptive_softmax(state, targets, keep_order=True, return_rank=True)
                return rank, nll

    def decode(self, batch, cache=None, global_mask=None):
        ''' if targest is not None,  '''

        bsz, L = batch.shape
        word_embedding = self.embed(batch, self.embedding)

        decoded = {
            'state': word_embedding,
        }
        
        decoded['state'][batch == self.padding_idx] = 0

        for i, decoder in enumerate(self.layers):
            decoded = decoder(decoded, layer_i=i, global_mask=global_mask)

        return {
            'state': decoded['state'],          # bs x L x hidden_dim
        }

    def embed(self, inputs, token_embedding):
        ''' Embed the given inputs '''
        return self.dropout(token_embedding(inputs) + self.position_embedding(inputs))


