'''
Analyze transformer: importance of global tokens
'''

from __future__ import print_function

import os
import sys
import signal
import time
import atexit
from contextlib import ExitStack
import pdb
import numpy as np
import pickle
import torch
import itertools
from torch import nn
from tqdm import tqdm
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

import metrics
from models.utils import restore
from utils import tqdm_wrap_stdout


class CheckpointEventHandler(FileSystemEventHandler):
    ''' A filesystem event handler for new checkpoints '''
    def __init__(self, handler, experiment, verbose=0):
        ''' Initialize the CheckpointEventHandler '''
        super(CheckpointEventHandler, self).__init__()
        self.watches = set()
        self.handler = handler
        self.verbose = verbose
        self.experiment = experiment

    def on_created(self, event):
        ''' Watcher for a new file '''
        root, ext = os.path.splitext(event.src_path)
        basename = os.path.basename(root)
        if ext == '.incomplete' and basename == 'checkpoint.pt':
            self.watches.add(event.src_path)

            if self.verbose > 1:
                print(f'Waiting for {event.src_path}')

    def on_moved(self, event):
        ''' Handle when a file has been modified '''
        if event.src_path in self.watches:
            self.watches.remove(event.src_path)
            self.handler(event.dest_path, self.experiment, self.verbose)


class LambadaAcc(object):
    ''' An object that encapsulates model evaluation '''
    def __init__(self, config, model, dataloader, device):
        self.model = model
        self.config = config
        self.device = device
        self.dataloader = dataloader

        self.should_exit = False
        signal.signal(signal.SIGHUP, self.on_training_complete)

        self.observer = None

        if 'cuda' in device.type:
            self.model = nn.DataParallel(model.cuda())

        self.modules = {
            'model': model
        }

    @property
    def dataset(self):
        ''' Get the dataset '''
        return self.dataloader.dataset

    def evaluate(self, batch):
        ''' Runs one evaluation step '''
        with torch.no_grad():
            self.model.eval()
            L = batch.shape[0]
            
            if self.config.return_rank:
                rank, nll = self.model(batch)
                rank = rank.contiguous().view(self.config.batch_size, -1).reshape(-1)
                length = nll.shape[0]
            else:
                _, nll = self.model(batch)
                length = nll.shape[0]

            nll = nll.sum()
            return nll.item(), length, None if not self.config.return_rank else rank

    def evaluate_epoch(self, epoch, experiment, verbose=0):
        ''' Evaluate a single epoch '''
        neg_log_likelihood = metrics.Metric('nll', metrics.format_float)
        def get_description():
            mode_name = 'Test' if self.config.split == 'test' else 'Validate'
            description = f'{mode_name} #{epoch}'
            if verbose > 0:
                description += f' {neg_log_likelihood}'
            if verbose > 1:
                description += f' [{profile.mem_stat_string(["allocated"])}]'
            return description

        assert self.config.return_rank is True
        data_dir = self.dataloader.dataset.data_dir
        fold = self.dataloader.dataset.split
        with open(os.path.join(data_dir, f'{fold}.txt'), 'r') as f:
             lines = f.readlines()

        max_len = max(len(l.split()) for l in lines)
        batches = []
        tok2id = self.dataloader.dataset.tok2id
        for line in lines:
            l_sp = line.strip().split()
            data = torch.tensor([tok2id[w] if w in tok2id else tok2id['<unk>'] 
                                for w in l_sp + ['<pad>']*(max_len - len(l_sp))])
            batches.append((data, len(l_sp)))

        res_dict = {}
        nll_sum, num_tok_sum = 0, 0
        acc_tok_sum = 0
        num_example = 0

        pred = []
        for bi, (batch, b_len) in enumerate(tqdm(batches)):

            batch = batch[:, None]
            nll, num_tok, rank = self.evaluate(batch)
            nll_sum += nll
            num_tok_sum += num_tok
            if rank[b_len-2] == 0:
                acc_tok_sum += 1
                pred.append(1)
            else:
                pred.append(0)
            num_example += 1

        nll = nll_sum / num_tok_sum
        ppl = np.exp(nll)
        print(acc_tok_sum / num_example)
        print(ppl)
        with open(os.path.join(self.config.checkpoint_directory, f'lambada-acc-{fold}.txt'), 'w') as f:
            for p in pred:
                f.write(f'{p}\n')

    def on_new_checkpoint(self, path, experiment, verbose=0):
        ''' Upon receiving a new checkpoint path '''
        epoch, step = restore(
            path,
            self.modules,
            num_checkpoints=self.config.average_checkpoints,
            map_location=self.device.type
        )
        experiment.set_step(step)
        self.evaluate_epoch(epoch, experiment, verbose)

    def on_training_complete(self, signum, frame): # pylint:disable=unused-argument
        ''' Received a SIGHUP indicating the training session has ended '''
        self.should_exit = True

    def shutdown(self):
        ''' Shutdown the current observer '''
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def watch(self, experiment, verbose=0):
        ''' Watch for a new checkpoint and run an evaluation step '''
        # Use a polling observer because slurm doesn't seem to correctly handle inotify events :/
        self.observer = PollingObserver() if self.config.polling else Observer()
        event_handler = CheckpointEventHandler(self.on_new_checkpoint, experiment, verbose)
        self.observer.schedule(event_handler, path=self.config.watch_directory)
        self.observer.start()

        while not self.should_exit:
            time.sleep(1)

        atexit.register(self.shutdown)

    def __call__(self, epoch, experiment, verbose=0):
        ''' Validate the model and store off the stats '''
        enter_mode = experiment.validate
        if self.config.split == 'test':
            enter_mode = experiment.test

        with ExitStack() as stack:
            stack.enter_context(enter_mode())
            stack.enter_context(torch.no_grad())

            return self.evaluate_epoch(epoch, experiment, verbose)
