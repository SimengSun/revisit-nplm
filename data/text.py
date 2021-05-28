"""
	Implement torch iterable dataset
		- build vocab ordered by freq for 
"""
from tqdm import tqdm
import torch
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
import os
import sys
import pickle
import math
from collections import defaultdict

SPLITS = ['train', 'valid', 'test']
EOS = '<eos>'
PAD = '<pad>'

class Dataset(torch.utils.data.IterableDataset):

	def __init__(self, data_dir, batch_size, split):

		self.data_dir = data_dir
		if not self.data_exist():
			self.build_vocab()
			for s in SPLITS:
				self.binarize(s)

		self.load_vocab()
		self.data = self.load_data(split, batch_size) # bsz x (len(data)/bsz)
		self.start = 0
		self.end = self.data.size(1)
		self.split = split

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:  # single-process data loading, return the full iterator
			iter_start = self.start
			iter_end = self.end
		else:  					# in a worker process split workload
			per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
			worker_id = worker_info.id
			iter_start = self.start + worker_id * per_worker
			iter_end = min(iter_start + per_worker, self.end)
		return iter(self.data.transpose(1,0)[iter_start:iter_end])

	@property
	def eos_idx(self):
		return self.tok2id[EOS]
	
	@property
	def padding_idx(self):
		return self.tok2id[PAD]

	@property
	def size(self):
		return len(self.id2tok)
	

	def build_vocab(self, min_freq=0, max_freq=sys.maxsize):
		"""
		build vocab + add eos
		encode sentence
		"""
		with open(os.path.join(self.data_dir, 'train.txt'), 'r') as fn:
			data = fn.readlines()

		if 'lambada' in self.data_dir:
			with open(os.path.join(self.data_dir, 'test.txt'), 'r') as fn:
				data.extend(fn.readlines())

			with open(os.path.join(self.data_dir, 'valid.txt'), 'r') as fn:
				data.extend(fn.readlines())

		print('building vocab ...')
		self.vocab = defaultdict(int)
		self.tok2id = {}
		self.id2tok = []

		for line in tqdm(data):
			line = line.strip().split()
			for tok in line:
				self.vocab[tok] += 1
		
		self.vocab = {a : self.vocab[a] for a in self.vocab if self.vocab[a] >= min_freq and self.vocab[a] <= max_freq}
		# sort vocab in case of using adaptive softmax
		self.vocab = list(sorted(self.vocab.items(), key=lambda a: a[1], reverse=True))
		print(self.vocab[:10])

		if 'lambada' in self.data_dir:
			self.vocab = self.vocab[:60000]
			self.vocab.append(('<unk>', 0))

		self.id2tok = ['<pad>'] + ['<eos>'] + [a[0] for a in self.vocab] 
		self.tok2id = {a : i for i, a in enumerate(self.id2tok)}
		self.vocab_size = len(self.id2tok)

		print('end building vocab ...')
		print('vocab size', len(self.tok2id))
		with open(os.path.join(self.data_dir, 'vocab.pkl'), 'wb') as fn: 
			pickle.dump({'id2tok': self.id2tok, 'tok2id': self.tok2id, 'vocab_size':self.vocab_size}, fn)

	def encode_line(self, line):

		if 'lambada' not in self.data_dir:
			return torch.tensor([self.tok2id[tok] for tok in line+['<eos>']])
		else:
			return torch.tensor([self.tok2id[tok] if tok in self.tok2id else self.tok2id['<unk>'] for tok in line])

	def binarize(self, split):
		"""binarize data to torch.tensor shape (doc_len, )"""
		with open(os.path.join(self.data_dir, f"{split}.txt"), "r") as fn:
			data = [line.strip().split() for line in fn.readlines()]

		print('binarizing data ...')
		doc = []
		for line in tqdm(data):
			if line != '':
				doc.append(self.encode_line(line))

		doc = torch.cat(doc)

		print('end binarizing data ...')
		print('doc shape', doc.shape)
		print([self.id2tok[i] for i in doc[:100]])
		with open(os.path.join(self.data_dir, f"{split}.bin"), "wb") as fout:
			pickle.dump({"data": doc}, fout, protocol=pickle.HIGHEST_PROTOCOL)

	def load_vocab(self):
		
		with open(os.path.join(self.data_dir, 'vocab.pkl'), 'rb') as fn: 
			data = pickle.load(fn)
		print('loading vocab...')
		self.id2tok = data['id2tok']
		self.tok2id = data['tok2id']
		self.vocab_size = data['vocab_size']
		# self.id2freq = data['id2freq']
		print(f'vocab size {self.vocab_size}')

	def data_exist(self):
		return all([os.path.exists(os.path.join(self.data_dir, f"{fn}.bin")) \
			for fn in ['train', 'valid', 'test'] ] + [os.path.exists(os.path.join(self.data_dir, "vocab.pkl"))])

	def load_data(self, split, bsz):

		with open(os.path.join(self.data_dir, f"{split}.bin"), "rb") as fin:
			data = pickle.load(fin)['data']

		nstep = data.size(0) // bsz
		return data[ : nstep * bsz].view(bsz, -1)
