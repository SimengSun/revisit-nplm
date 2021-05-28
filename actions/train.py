'''
stupidLM

--
Main entry point for training stupidLM
'''

from __future__ import print_function

import os
import sys
import time
import shutil
from contextlib import ExitStack

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, CosineAnnealingLR, CyclicLR
from tqdm import tqdm
import numpy as np

import args
import metrics
from actions.evaluate import Evaluator
from data.parallel import NewDataParallel
from models.utils import LinearLRSchedule, WarmupLRSchedule, checkpoint
from utils import tqdm_wrap_stdout, tqdm_unwrap_stdout
import pdb

class Trainer(object):
    ''' An object that encapsulates model training '''
    def __init__(self, config, model, dataloader, device, valid_dataloader=None, clip=0.25):
        self.model = model
        self.config = config
        self.device = device
        self.stopped_early = False
        self.clip = clip
        self.dataloader = dataloader
        self.validation_dataloader = valid_dataloader
        self.last_checkpoint_time = time.time()

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda()) if torch.cuda.device_count() == 1 else NewDataParallel(config.bsz_gpu0, model.cuda())

        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(model.parameters(), config.base_lr, betas=(config.beta_1, config.beta_2), eps=1e-08) # for transformer

            if config.lr_scheduler == 'warmup':
                self.lr_scheduler = LambdaLR(
                    self.optimizer,
                    WarmupLRSchedule( config.warmup_steps )
                )

            elif config.lr_scheduler == 'linear':
                self.lr_scheduler = LambdaLR(
                    self.optimizer,
                    LinearLRSchedule(
                        config.base_lr,
                        config.final_lr,
                        config.max_steps 
                    ) 
                )

            elif config.lr_scheduler == "cosine":
                self.lr_scheduler = CosineAnnealingLR(self.optimizer, config.max_steps, eta_min=config.final_lr)

            elif config.lr_scheduler == 'cyclic':
                self.lr_scheduler = CyclicLR(self.optimizer, cycle_momentum=False, base_lr=1e-7, max_lr=config.base_lr, step_size_up=4000, step_size_down=12000)

            elif config.lr_scheduler == 'customize':
                self.lr_scheduler = CosineAnnealingLR(self.optimizer, config.max_steps, eta_min=config.final_lr)

            else:
                raise ValueError('Unknown learning rate scheduler!')

        elif self.config.optimizer == "sgd":
            print("using sgd optimizer")
            self.optimizer = optim.SGD(model.parameters(), lr=config.base_lr, momentum=0.99)
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, config.max_steps, eta_min=config.final_lr)

        else:
            raise ValueError('Unknown optimizer!') 


        # Initialize the metrics
        metrics_path = os.path.join(self.config.checkpoint_directory, 'train_metrics.pt')
        self.metric_store = metrics.MetricStore(metrics_path)
        self.metric_store.add(metrics.Metric('oom', metrics.format_int, 't'))
        self.metric_store.add(metrics.Metric('nll', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('ppl', metrics.format_float, max_history=1000))
        self.metric_store.add(metrics.Metric('lr', metrics.format_scientific, 'g', max_history=1))
        self.metric_store.add(metrics.Metric('num_tok', metrics.format_int, 'a', max_history=1000))

        if self.config.early_stopping:
            self.metric_store.add(metrics.Metric('vnll', metrics.format_float, 'g'))

        self.modules = {
            'model': model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }

        self.step = 0

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    def train_epoch(self, epoch, experiment, verbose=0):
        ''' Run one training epoch '''
        oom = self.metric_store['oom']
        learning_rate = self.metric_store['lr']
        num_tokens = self.metric_store['num_tok']
        neg_log_likelihood = self.metric_store['nll']
        perplexity = self.metric_store['ppl']

        def try_optimize(i, curr_step, last=False):
            # optimize if:
            #  1) last and remainder
            #  2) not last and not remainder
            remainder = bool(i % self.config.accumulate_steps)
            if not last ^ remainder:
                next_lr = self.optimize(curr_step)

                learning_rate.update(next_lr)
                experiment.log_metric('learning_rate', next_lr)
                return True

            return False

        def get_description():
            description = f'Train #{epoch}'
            if verbose > 0:
                description += f' {self.metric_store}'
            if verbose > 1:
                description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        batches = tqdm(
            self.dataloader,
            unit='batch',
            dynamic_ncols=True,
            desc=get_description(),
            file=sys.stdout 
        )
        with tqdm_wrap_stdout():
            i = 1
            nll_per_update = 0.
            length_per_update = 0
            num_tokens_per_update = 0
            cnter = 0
            for i, batch in enumerate(batches, 1):

                if type(batch) is not torch.Tensor:
                    # concatenated dataset
                    batch = ([b.squeeze(0) for b in batch])
                else:
                    batch = (batch,)

                try:
                    # if batch.shape[0] == self.config.batch_length:
                    if True:
                        
                        nll, length = self.calculate_gradient(batch)
                        did_optimize = try_optimize(i, experiment.curr_step )

                        # record the effective number of tokens
                        num_tokens_per_update += length

                        if length:
                            # record length and nll
                            nll_per_update += nll
                            length_per_update += length

                        if did_optimize:
                            # advance the experiment step
                            experiment.set_step(experiment.curr_step + 1)

                            num_tokens.update(num_tokens_per_update)
                            nll = nll_per_update / length_per_update
                            neg_log_likelihood.update(nll)
                            perplexity.update(np.exp(nll))

                            experiment.log_metric('num_tokens', num_tokens_per_update)
                            experiment.log_metric('nll', neg_log_likelihood.last_value)
                            experiment.log_metric('ppl', perplexity.last_value)

                            nll_per_update = 0.
                            length_per_update = 0
                            num_tokens_per_update = 0

                except RuntimeError as rte:
                    if 'out of memory' in str(rte):
                        torch.cuda.empty_cache()

                        oom.update(1)
                        experiment.log_metric('oom', oom.total)
                        #exit(-1)
                        raise rte
                    else:
                        batches.close()
                        raise rte

                if self.should_checkpoint():
                    new_best = False
                    if self.config.early_stopping:
                        with tqdm_unwrap_stdout():
                            new_best = self.evaluate(experiment, epoch, verbose)

                    self.checkpoint(epoch, experiment.curr_step, new_best)

                batches.set_description_str(get_description())
                if self.is_done(experiment, epoch):
                    batches.close()
                    break

            try_optimize(i, experiment.curr_step, last=True)

    def should_checkpoint(self):
        ''' Function which determines if a new checkpoint should be saved '''
        return time.time() - self.last_checkpoint_time > self.config.checkpoint_interval

    def checkpoint(self, epoch, step, best=False):
        ''' Save a checkpoint '''
        checkpoint_path = checkpoint(
            epoch, step, self.modules,
            self.config.checkpoint_directory,
            max_checkpoints=self.config.max_checkpoints
        )

        if best:
            dirname = os.path.dirname(checkpoint_path)
            basename = os.path.basename(checkpoint_path)
            best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
            shutil.copy2(checkpoint_path, best_checkpoint_path)

        self.metric_store.save()
        self.last_checkpoint_time = time.time()

    def evaluate(self, experiment, epoch, verbose=0):
        ''' Evaluate the current model and determine if it is a new best '''
        model = self.modules['model']
        evaluator = Evaluator(self.config, model, self.validation_dataloader, self.device)
        vnll = evaluator(epoch, experiment, verbose)
        metric = self.metric_store['vnll']
        full_history = metric.values
        metric.update(vnll)
        self.metric_store.save()

        return all(vnll < nll for nll in full_history[:-1])

    def is_done(self, experiment, epoch):
        ''' Has training completed '''
        if self.config.max_steps and experiment.curr_step >= self.config.max_steps:
            return True

        if self.config.max_epochs and epoch >= self.config.max_epochs:
            return True

        if self.config.early_stopping:
            history = self.metric_store['vnll'].values[-self.config.early_stopping - 1:]
            if len(history) == self.config.early_stopping + 1:
                self.stopped_early = all(history[-1] > nll for nll in history[:-1])
                return self.stopped_early

        return False

    def optimize(self, i):
        ''' Calculate an optimization step '''

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.config.lr_scheduler not in ["cosine", "customize"]:
            self.lr_scheduler.step()
            return self.lr_scheduler.get_lr()[0]

        elif self.config.lr_scheduler == 'customize':

            mid_ms = 15000
            mid_lr = 5e-5

            if i < self.config.warmup_steps:
                self.optimizer.param_groups[0]['lr'] = self.config.base_lr * i / self.config.warmup_steps
                return self.optimizer.param_groups[0]['lr']

            elif i >= self.config.warmup_steps and i < mid_ms:
                self.optimizer.param_groups[0]['lr'] = self.config.base_lr - (self.config.base_lr - mid_lr) / (mid_ms - self.config.warmup_steps) * (i - self.config.warmup_steps)
                return self.optimizer.param_groups[0]['lr']
            else:
                self.optimizer.param_groups[0]['lr'] = mid_lr - (mid_lr - self.config.final_lr) / (self.config.max_steps - mid_ms) * (i - mid_ms)
                return self.optimizer.param_groups[0]['lr']

        else:
            if i < self.config.warmup_steps:
                self.optimizer.param_groups[0]['lr'] = self.config.base_lr * i / self.config.warmup_steps
                return self.optimizer.param_groups[0]['lr']
            else:
                self.lr_scheduler.step()
                return self.lr_scheduler.get_lr()[0]

    def calculate_gradient(self, batch):
        ''' Runs one step of optimization '''
        # run the data through the model
        self.model.train()
        
        num_tok = batch[0].shape[0]*batch[0].shape[1]
        batch = torch.cat(batch, dim=-2)
        loss, nll = self.model(batch)
        # length = nll.shape[0]

        # nn.DataParallel wants to gather rather than doing a reduce_add, so the output here
        # will be a tensor of values that must be summed
        nll = nll.sum()
        loss = loss.sum()

        loss.backward()
        return nll.item(), num_tok

    def __call__(self, start_epoch, experiment, verbose=0):
        ''' Execute training '''
        with ExitStack() as stack:
            # stack.enter_context(chunked_scattering())
            stack.enter_context(experiment.train())

            if start_epoch > 0 or experiment.curr_step > 0:
                # TODO: Hacky approach to decide if the metric store should be loaded. Revisit later
                self.metric_store = self.metric_store.load()

            epoch = start_epoch
            experiment.log_current_epoch(epoch)
            while not self.is_done(experiment, epoch):
                experiment.log_current_epoch(epoch)
                self.train_epoch(epoch, experiment, verbose)
                experiment.log_epoch_end(epoch)
                epoch += 1

            if self.stopped_early:
                print('Stopping early!')
            else:
                new_best = False
                if self.config.early_stopping:
                    new_best = self.evaluate(experiment, epoch, verbose)

                self.checkpoint(epoch, experiment.curr_step, new_best)

