from __future__ import print_function

import os
import sys
import signal
import time
import atexit
from contextlib import ExitStack
import pdb
import numpy as np
import torch
import math
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


class Evaluator(object):
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

    def evaluate(self, batch, tgt_len=-1):
        ''' Runs one evaluation step '''
        with torch.no_grad():
            self.model.eval()
            batch = torch.cat(batch, dim=-2)
            _, nll = self.model(batch)
            
            if tgt_len != -1:
                nll = nll.contiguous().view(self.config.batch_size, -1)[:, 
                                    -tgt_len:].reshape(-1)
            length = nll.shape[0]
            nll = nll.sum()
            return nll.item(), length

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

        batches = tqdm(
            self.dataloader,
            unit='batch',
            dynamic_ncols=True,
            desc=get_description(),
            file=sys.stdout # needed to make tqdm_wrap_stdout work
        )

        batches = [batch for batch in batches]
        batches_ = torch.cat(batches, dim=0)
        batches = []

        total_len = batches_.shape[0]
        seq_len = self.config.batch_length
        tgt_len = self.config.target_length

        lid, rid = 0, seq_len
        while True:
            batches.append((lid, rid))
            lid += tgt_len
            rid += tgt_len
            if rid >= total_len:
                break

        nll_sum, num_tok_sum = 0, 0
        for batch in tqdm(batches):

            lid, rid = batch[0], batch[1]
            batch = batches_[lid:rid]
            if type(batch) is not torch.Tensor:
                batch = ([b.squeeze(0) for b in batch])
            else:
                batch = (batch,)

            nll, num_tok = self.evaluate(batch, tgt_len=tgt_len)
            nll_sum += nll
            num_tok_sum += num_tok

        nll = nll_sum / num_tok_sum
        ppl = np.exp(nll)
        bpc = nll / math.log(2)
        print("nll {:5} ppl {:5} bpc: {:5}".format(nll, ppl, bpc))
        return nll_sum / num_tok_sum

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

