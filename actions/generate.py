from __future__ import print_function

import os
import pdb
import sys
import timeit
import numpy as np
from contextlib import ExitStack

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import tqdm_wrap_stdout
from utils import left_shift


class Generator(object):
    ''' An object that encapsulates model evaluation '''
    CURRENT = None

    def __init__(self, config, model, dataloader, device):
        self.config = config
        self.dataloader = dataloader

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        self.modules = {
            'model': model
        }

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    @property
    def padding_idx(self):
        ''' Get the padding index '''
        return self.dataset.padding_idx

    def generate(self, output_path, epoch, experiment, verbose=0):
        ''' Generate all predictions from the dataset '''
        def get_description():
            description = f'Generate #{epoch}'
            if verbose > 1:
                description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        # read prompt from stdin
        with open(self.config.prompt_file, "r") as f:
            prompts = f.readlines()

        # encode prompts
        encoded_prompts = [self.dataset.encode_line(line.split()[-512:])[:-1] for line in prompts]

        # generate and store to output_file
        all_decoded = []
        for encoded_prompt in tqdm(encoded_prompts):

            # add batch dimension (only support one example at a time)
            encoded_prompt = encoded_prompt[None, :].cuda()

            decode_output = []
            while len(decode_output) < self.config.max_length and len(decode_output) + encoded_prompt.shape[1] < 512:
                decoded = torch.tensor([[decode_output[-1]]]).cuda() if len(decode_output) != 0 else torch.tensor([[]]).cuda()
                encoded_prompt = torch.cat([encoded_prompt, decoded], dim=1).long()

                # decode
                decoded = self.model.module.decode(encoded_prompt) # bs x L x hidden_dim
                state = decoded['state'][:, -1]             # only need the last token for generation

                state = state.contiguous().view(-1, state.shape[-1]) # (bsz*1, embed_dim)
                targets = encoded_prompt[:,-1:].contiguous().view(-1) # (bsz*1)
                all_probs = self.model.module.adaptive_softmax(state, targets, keep_order=True, return_all_logprobs=True)
                
                if self.config.decoding_algorithm == "greedy":
                    decode_output.append(torch.argmax(all_probs))

                else:
                    raise NotImplementedError

            all_decoded.append(decode_output)

        with open(output_path, 'w') as f:
            for idx, (prompt, decoded) in enumerate(zip(prompts, all_decoded)):
                f.write(f"======{idx}======:\n")
                f.write(prompt )
                f.write(self.dataset.decode_tokids(decoded))
                f.write('\n\n')

    def __call__(self, epoch, experiment, verbose=0):
        ''' Generate from the model '''
        enter_mode = experiment.validate
        if self.dataset.split == 'test':
            enter_mode = experiment.test

        with ExitStack() as stack:
            stack.enter_context(enter_mode())
            stack.enter_context(torch.no_grad())

            if not os.path.isdir(self.config.output_directory):
                os.makedirs(self.config.output_directory)

            step = experiment.curr_step
            output_filename = self.config.output_filename or f'generated_{step}.txt'
            output_path = os.path.join(self.config.output_directory, output_filename)
            # output_file = stack.enter_context(open(output_path, 'wt'))

            if verbose:
                print(f'Outputting to {output_path}')

            self.generate(output_path, epoch, experiment, verbose)


