import argparse
import os
from utils import get_version_string
from comet_ml import Experiment
from comet_ml import Experiment, ExistingExperiment

DATASETS = ['wt103', 'wt102', 'enwik8', 'lambada']
MODELS = ['transformer', 'nplm']
ACTIONS = ['preprocess', 'train', 'evaluate', 'acc']

def _integer_geq(value=0):
	'''
	Return a function which when evaluated returns an integer if a passed in string when converted
	to an int is greater than or equal to the specified constant. Otherwise it raises an error.
	'''
	def convert(string):
		''' Check if the string is an integer that is greater than value '''
		integer = int(string)
		if integer < value:
			raise argparse.ArgumentTypeError(f'{string} should be >= {value}')
		return integer

	return convert

def parse_args(argv=None):

	parser = argparse.ArgumentParser(description="config parser")


	# action
	parser.add_argument(
		'--action',
		type=str,
		choices=ACTIONS,
		default='train',
		help='action type'
	)

	# shared configs
	parser.add_argument(
		'-m',
		'--model',
		type=str,
		default='transformer',
		choices=MODELS,
		help='Which model to instantiate'
	)
	parser.add_argument(
		'-r',
		'--restore',
		type=str,
		default=None,
		help='Location of the checkpoint to restore'
	)
	parser.add_argument(
		'--reset-parameters',
		type=str,
		nargs='*',
		default=[],
		choices=['encoder', 'decoder', 'embeddings', 'step', 'optimizer', 'lr_scheduler'],
		help='What parameters to reset when restoring a checkpoint.'
	)
	parser.add_argument(
		'--average-checkpoints',
		type=int,
		default=1,
		help='How many checkpoints to average over when restoring'
	)
	parser.add_argument(
		'-s',
		'--seed',
		type=int,
		default=None,
		help='Set random seed for deterministic evaluation'
	)
	parser.add_argument(
		'--track',
		default=False,
		const=True,
		nargs='?',
		help='Whether to track this experiment. If an experiment id is provided, it will track \
		the existing experiment. If a filename ending with guid it is provided, it will wait \
		until the file exists, then start tracking that experiment.'
	)

	parser.add_argument(
		'--project-name',
		default='transformer-attn',
		type=str,
		help='Specify where to store in comet'
	)
	parser.add_argument(
		'-v',
		'--verbose',
		default=0,
		action='count',
		help='Increase the verbosity level'
	)

	# model configs
	parser.add_argument(
		'--num-layers',
		type=int,
		default=6,
		help='Number of layers in each stack'
	)
	parser.add_argument(
		'--FF2',
		default=False,
		action='store_true',
		help='if true, each Transformer layer has two FF sublayers'
	)
	parser.add_argument(
		'--no-attention',
		default=[0],
		type=int,
		nargs='+',
		help='if 0, no no-attention, if 1, no attention in that layer, should be of 1 or num-layers'
	)
	parser.add_argument(
		'--num-heads',
		type=int,
		default=8,
		help='Number of heads in each Transformer layer for multi-headed attention'
	)
	parser.add_argument(
		'--tie-weights',
		default=False,
		action='store_true',
		help='if true, tie word embedding and corresponding weight in adaptive softmax'
	)
	parser.add_argument(
		'--tie-projs',
		default=False,
		action='store_true',
		help='if true, tie projection of word embedding and corresponding weight in adaptive softmax'
	)
	parser.add_argument(
		'--emb-std',
		default=0.02,
		type=float,
		help='std of embedding initialization'
	)
	parser.add_argument(
		'--proj-std',
		default=0.01,
		type=float,
		help='std of embedding projection initialization'
	)
	parser.add_argument(
		'--do-proj',
		default=False,
		action='store_true',
		help='whether to do emb_dim -> model_dim projection, used for adaptive softmax'
	)
	parser.add_argument(
		'--embedding-size',
		type=int,
		default=512,
		help='The size of the model dimension, if use adaptive input, embdding dim for each cluster is different'
	)
	parser.add_argument(
		'--hidden-dim',
		type=int,
		default=2048,
		help='The size of the feed-forward hidden layer'
	)
	parser.add_argument(
		'--model-size',
		type=int,
		default=512,
		help="the size of model dimension "
	)

	# attention mask config
	parser.add_argument(
		'--context-mask',
		type=int,
		nargs='+',
		default=[-1],
		help='the local window that learned attention should be constrained; if input a list, the size should be equal to num-layers'
	)
	parser.add_argument(
		'--context-mask-left', # context-mask == 5, <==> left: -4, right: 0
		type=int,
		nargs='+',
		default=[-1],
		help='if not constraining learned attention to only local attention , needs to provide both left and right of the window'
	)
	parser.add_argument(
		'--context-mask-right', # context-mask == 5, <==> left: -4, right: 0
		type=int,
		nargs='+',
		default=[-1],
		help='if not constraining learned attention to only local attention , needs to provide both left and right of the window'
	)
	parser.add_argument(
		'--non-contiguous-mask', 
		action="store_true",
		default=False,
		help='if non-contiguous-mask, context-mask(local mask) and context-mask-left, context-mask-right should all be specified, both masks will be added together to have both local and global windows'
	)

	# attention config, `normal': hard-coded attention as in https://www.aclweb.org/anthology/2020.acl-main.687.pdf
	parser.add_argument(
		'--attn-type',
		type=str,
		nargs='+',
		default=['learned'],
		choices=['normal', 'learned'],
		help='What type of attention we are using for the rules'
	)
	parser.add_argument(
		'--attn-std',
		type=float,
		nargs='+',
		default=[1],
		help='when attention type is normal. The standard deviation.'
			 'when attention type is uniform. The number of tokens to focus on to each direction.'
			 'If window size is 2, then we have uniform distribution on 5 words.'
	)
	parser.add_argument(
		 '--attn-threshold',
		 type=float,
		 default=-1,
		 help='when attention type is normal, '
			  'Only attend to places where distribution is above or equal to the attn-threshold'
	)
	parser.add_argument(
		'--attn-window',
		type=int,
		default=-1,
		help='The window to do convolution at'
	)
	parser.add_argument(
		'--attn-offset',
		type=int,
		nargs='+',
		default=[-1],
		help='Only works when corresponding attn-position is left or right.'
			 'The number of steps to take to that direction.'
			 'When attn-position is bin, this number has to be between 1 and number of attn-bins'
	)
	parser.add_argument(
		'--attn-offset-window',
		type=int,
		nargs='+',
		default=[1],
		help='only for attn-indexing, if offset of a head is -1, window-size=3, then this head stores average of [-3, -2, -1] '
	)
	parser.add_argument(
		'--attn-weights',
		type=int,
		default=[1],
		choices=[0, 1],
		help='Whether or not use weights in non-learned attention. 0 no weights, 1 all with weights, '
	)
	parser.add_argument(
		'--attn-impl',
		type=str,
		nargs='+',
		default=['full'],
		choices=['full', 'conv', 'index', 'uniform'],
		help='choose which implementation to use'
	)
	parser.add_argument(
		'--attn-no-output-proj',
		default=False,
		action='store_true',
		help='if true, no final output projection in attention block'
	)

	# NPLM config
	parser.add_argument(
		'--context-config',
		default=[5, 100],
		type=int, 
		nargs='+',
		help="config for context control for NPLM, the first is n-gram of local context, the second is the window size to be averaged for long term context"
	)
	parser.add_argument(
		'--no-layernorm',
		action="store_true",
		default=False,
		help="if add this arg, sublayer will not have layernorm"
	)
	parser.add_argument(
		'--concat-layers',
		default=[0],
		type=int, 
		nargs='+',
		help="layers that concat embeddings and project from concat_emb to emb_dim"
	)
	parser.add_argument(
		'--global-aggregate',
		type=str,
		default='kernel',
		choices=['average', 'kernel'],
		help='how to aggregate global tokens, average: uniform average; kernel: learned kernel'
	)
	parser.add_argument(
		'--pre-norm',
		action="store_true",
		default=False,
		help="whether to put layer norm before adding residual"
	)
	parser.add_argument(
		'--num-global-agg',
		default=1,
		type=int, 
		help="number of learned kernels for global aggregation"
	)
	parser.add_argument(
		'--mid-dim',
		default=-1,
		type=int, 
		help="num dimension of middle projection layer"
	)
	parser.add_argument(
		'--concat-hidden-dims',
		type=int,
		nargs='+',
		default=[410],
		help='hidden dimension list for nplm concatenation layer'
	)
	parser.add_argument(
		'--TFN',
		default=False,
		action='store_true',
		help='if true, use Transformer-N architecture, base model should be NPLM'
	)
	parser.add_argument(
		'--return-rank',
		default=False,
		action='store_true',
		help='if true, adaptive softmax also returns rank of the target token in the vocabulary'
	)


	# data configs
	parser.add_argument(
		'-b',
		'--batch-size',
		type=int,
		default=60, 
		help='number of examples per batch'
	)
	parser.add_argument(
		'--bsz-gpu0',
		type=int,
		default=150,
		help='batch length on gpu0, gpu0 needs to have smaller batch size than the rest gpus'
	) 
	parser.add_argument(
		'--batch-length',
		type=int,
		default=150, 
		help='length of an example'
	)
	parser.add_argument(
		'--target-length',
		type=int,
		default=128, 
		help='target length during evaluation'
	)
	parser.add_argument(
		'-d',
		'--data-directory',
		type=str,
		default='/tmp/stupidlm/data',
		help='Location of the data'
	)
	parser.add_argument(
		'-D',
		'--dataset',
		type=str,
		default='wt103',
		choices=DATASETS,
		help='Name of the dataset to load.'
	)
	parser.add_argument(
		'--split',
		type=str,
		default='train',
		choices=['train', 'valid', 'test', 'test-control'],
		help='Location for the preprocessed data; for LAMBADA dataset, the rename the control set to test-control.txt'
	)

	# train configs
	parser.add_argument(
		'-A',
		'--accumulate-steps',
		type=int,
		default=1,
		help='How many batches of data to accumulate gradients over'
	)
	parser.add_argument(
		'--dropout-p',
		type=float,
		default=0.1,
		help='The dropout percentage during training'
	)
	parser.add_argument(
		'--clip',
		default=0.25,
		type=float, 
		help="clip gradient"
	)
	parser.add_argument(
		'--early-stopping',
		type=_integer_geq(),
		default=0,
		help='If > 0, stop training after this many checkpoints of increasing nll on the validation'
		' set. This also implies storing of the best_checkpoint.'
	)
	parser.add_argument(
		'--label-smoothing',
		type=float,
		default=0.1,
		help='The amount of label smoothing'
	)
	parser.add_argument(
		'-c',
		'--checkpoint-directory',
		type=str,
		default='/tmp/stupidlm/checkpoints',
		help='Where to store model checkpoints'
	)
	parser.add_argument(
		'--checkpoint-interval',
		type=int,
		default=10*60,
		help='Generate a checkpoint every `n` seconds'
	)
	parser.add_argument(
		'--max-checkpoints',
		type=int,
		default=5,
		help='The maximum number of checkpoints to keep'
	)
	parser.add_argument(
		'-e',
		'--max-epochs',
		type=int,
		default=0,
		help='Maximum number of epochs for training the model'
	)
	parser.add_argument(
		'--max-steps',
		type=int,
		default=100000,
		help='Maximum number of steps for training the model'
	)
	parser.add_argument(
		'-l',
		'--learning-rate',
		dest='base_lr',
		type=float,
		default=None,
		help='The initial learning rate of the optimizer. Defaults to embedding_size ** -0.5'
	)
	parser.add_argument(
		'--init-lr',
		type=float,
		default=None,
		help='initial learning rate for warm up'
	)
	parser.add_argument(
		'--adam-beta1',
		dest='beta_1',
		type=float,
		default=0.9,
		help='adam optimizer beta1'
	)
	parser.add_argument(
		'--adam-beta2',
		dest='beta_2',
		type=float,
		default=0.999,
		help='adam optimizer beta2'
	)
	parser.add_argument(
		'-L',
		'--learning-rate-decay',
		dest='lr_decay',
		type=float,
		default=.999995,
		help='The learning rate decay of the optimizer'
	)
	parser.add_argument(
		'--final-learning-rate',
		dest='final_lr',
		type=float,
		default=1e-5,
		help='For the linear annealing schedule'
	)
	parser.add_argument(
		'--learning-rate-scheduler',
		dest='lr_scheduler',
		type=str,
		default='cosine',
		choices=['warmup', 'linear', 'cosine', 'cyclic'],
		help='The learning rate schedule of the optimizer'
	)
	parser.add_argument(
		'-w',
		'--warmup-steps',
		type=int,
		default=4000,
		help='Number of warmup steps for the learning rate'
	)
	parser.add_argument(
		'--optimizer',
		type=str,
		default='adam',
		choices=['adam', 'sgd'],
		help='add optimizer'
	)
	parser.add_argument(
		'--adaptive',
		default=False,
		action='store_true',
		help='whether to use projected adaptive softmax'
	)
	parser.add_argument(
		'--cutoffs',
		default='2e4,4e4,2e5',
		type=str,
		help='cutoffs of the adaptive softmax'
	)
	parser.add_argument(
		'--div-val',
		default=1,
		type=int,
		help='divided value for adaptive softmax'
	)

	# test configs
	parser.add_argument(
		'--output-directory',
		type=str,
		default='/tmp/stupidlm/output',
		help='Where to store translated strings'
	)
	parser.add_argument(
		'--output-filename',
		type=str,
		default=None,
		help='Default output filename is translated_{step}.txt'
	)
	parser.add_argument(
		'--order-output',
		default=False,
		action='store_true',
		help='Whether to print the translated strings in the original dataset ordering'
	)
	parser.add_argument(
		'--timed',
		type=int,
		default=0,
		const=1,
		nargs='?',
		help='How many times to run translation to gauge the translation speed'
	)

	args = parser.parse_args()

	args.version = get_version_string()
	if args.track and '-dirty' in args.version:
		raise RuntimeError(
				'''
					Trying to track an experiment, but the workspace is dirty!
					Commit your changes first, then try again.
				''')

	args.cutoffs = [int(float(a)) for a in args.cutoffs.split(',')]

	api_key = None if args.track else ''
	experiment_type = Experiment
	experiment_args = [api_key]
	if isinstance(args.track, str):
		experiment_type = ExistingExperiment
		if args.track.endswith('.guid'):
			wait_count = 0
			while not os.path.exists(args.track):
				wait_string = '...'[:wait_count % 4]
				wait_count += 1

				print(f'\r\033[KWaiting for experiment: {args.track} {wait_string}', end='')
				time.sleep(1)

			print(f'\r\033[KLoading experiment: {args.track}')
			with open(args.track, 'rt') as guid_file:
				experiment_args.append(guid_file.readline().strip())
		else:
			experiment_args.append(args.track)

	args.experiment = experiment_type(
		*experiment_args,
		project_name=args.project_name,
		workspace='umass-nlp',
		disabled=not args.track,
		auto_metric_logging=False,
		auto_output_logging=None,
		auto_param_logging=False,
		log_git_metadata=False,
		log_git_patch=False,
		log_env_details=False,
		log_graph=False,
		log_code=False,
		parse_args=False,
	)

	if hasattr(args, 'base_lr') and not args.base_lr:
		args.base_lr = args.embedding_size ** -0.5

	if args.track and experiment_type == Experiment and args.action == 'train':
		with open(os.path.join(args.checkpoint_directory, 'experiment.guid'), 'wt') as guid_file:
			guid_file.write(args.experiment.id)

	# This needs to be called separately to disable monkey patching of the ML frameworks which is on
	# by default :(
	args.experiment.disable_mp()

	if experiment_type is Experiment:
		args.experiment.log_parameter('version', args.version)
		args.experiment.log_parameters(vars(args))
		
	if args.action != "train":
		args.dropout_p = 0
		
	return args




