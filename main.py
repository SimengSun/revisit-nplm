import comet_ml
import torch
import numpy as np
from args import parse_args
from data.text import Dataset
from actions.train import Trainer
from actions.evaluate import Evaluator
from actions.generate import Generator
from actions.lambada_acc import LambadaAcc
from models.transformer import Transformer
from models.nplm import NPLM
from models.utils import restore
from torch.utils.data.dataloader import DataLoader

def build_model(configs, dataset):
	if configs.model == 'transformer':
		model = Transformer(configs, dataset)

	elif configs.model == 'nplm':
		model = NPLM(configs, dataset)

	return model


def	main():
	
	configs = parse_args()

	print('=====================================')
	print('All configs:')
	v_configs = vars(configs)
	for k in v_configs:
		print('\t{:20s} {:50s}'.format(k, str(v_configs[k])))
	print('=====================================')

	if configs.model == 'transformer':
		configs.do_proj = True

	num_worker = 0
	
	ds = Dataset(configs.data_directory, configs.batch_size, configs.split)
	dl = DataLoader(ds, num_workers=num_worker, batch_size=configs.batch_length)
	
	model = build_model(configs, ds)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print(model)

	all_params = [v for k,v in model.named_parameters()]
	non_emb_params = [v for k, v in model.named_parameters() if 'embedding' not in k]
	num_params = sum([np.prod(p.size()) for p in all_params])
	num_no_emb_params = sum([np.prod(p.size()) for p in non_emb_params])
	print(f'total number of parameters {num_params}')
	print(f'total number of non-embedding parameters {num_no_emb_params}')
	
	if configs.action == "preprocess":
		train_dataset = Dataset(configs.data_directory, configs.batch_size, "valid")
		train_dataloader = DataLoader(train_dataset, num_workers=num_worker, batch_size=configs.batch_length)
		exit()

	if configs.action == "train":
		vds = Dataset(configs.data_directory, configs.batch_size * 8, 'valid')
		vdl = DataLoader(vds, num_workers=num_worker, batch_size=configs.batch_length)
		actioner = Trainer(configs, model, dl, device, clip=configs.clip, valid_dataloader=vdl)

	elif configs.action == "evaluate":	
		actioner = Evaluator(configs, model, dl, device)

	elif configs.action == "generate":
		actioner = Generator(configs, model, dl, device)

	elif configs.action == "acc":
		actioner = LambadaAcc(configs, model, dl, device)

	else:
		raise Exception("action not implemented")


	step = 0
	epoch = 0
	if configs.restore:
		restore_modules = {
			module_name: module
			for module_name, module in actioner.modules.items()
			if module_name not in configs.reset_parameters
		}

		epoch, step = restore(
			configs.restore,
			restore_modules,
			num_checkpoints=configs.average_checkpoints,
			map_location=device,
			strict=not configs.reset_parameters
		)

		model.reset_named_parameters(configs.reset_parameters)
		if 'step' in configs.reset_parameters:
			step = 0
			epoch = 0

	configs.experiment.set_step(step)
	actioner(epoch, configs.experiment, configs.verbose)


if __name__ == "__main__":
	main()

