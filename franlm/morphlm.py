import os, sys

import shutil
import logging
#import coloredlogs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from model import CharRNN


class MorphLM:

	def __init__(self, num_layers=2, emb_size=50, hidden_size=256, dropout=0.5):
		self.num_layers = num_layers
		self.emb_size = emb_size
		self.hidden_size = hidden_size 
		self.dropout = dropout
		self.idx_to_token = {}
		self.token_to_idx = {}
		self.rnn_type = 'lstm'
		self.START = '#'
		self.BOUNDARY = '>'
		self.PADDING = '*'

	def sequences_to_tensors(self, sequences):
		""" Casts a list of sequences into rnn-digestable padded tensor """
		seq_idx = []
		for seq in sequences:
			seq_idx.append([self.token_to_idx[token] for token in seq])
		sequences = [torch.LongTensor(x) for x in seq_idx]
		return nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.token_to_idx[self.PADDING])
	
	def preprocess(self, file_name):
		with open(file_name) as file:
			self.sequences = file.read()[:-1].split('\n')
			self.sequences = [self.START + seq.lower() for seq in self.sequences]
	
		logging.info('number of sequences: {}'.format(len(self.sequences)))
		for seq in self.sequences[::1000]:
			print(seq[1:])
	
		MAX_LENGTH = max(map(len, self.sequences))
		logging.info('max length: {}'.format(MAX_LENGTH))
	
		self.idx_to_token = list(set([token for seq in self.sequences for token in seq]))
		self.idx_to_token.append(self.PADDING)
		n_tokens = len(self.idx_to_token)
		logging.info('number of unique tokens: {}'.format(n_tokens))
	
		self.token_to_idx = {token: self.idx_to_token.index(token) for token in self.idx_to_token}
		assert len(self.idx_to_token) ==  len(self.token_to_idx), 'dicts must have same lengths'
	
		logging.debug('processing tokens')
		self.sequences = self.sequences_to_tensors(self.sequences)

		#return sequences, token_to_idx, idx_to_token

	def iterate_minibatches(self, inputs, batchsize, shuffle=False):
		logging.info('iterate_minibatches {}'.format(batchsize))
		if shuffle:
			indices = np.random.permutation(inputs.size(0))
		for start_idx in range(0, inputs.size(0) - batchsize + 1, batchsize):
#			logging.info('start_idx {}'.format(start_idx))
			if shuffle:
				excerpt = indices[start_idx:start_idx + batchsize]
			else:
				excerpt = slice(start_idx, start_idx + batchsize)
#			logging.info('yielding {}'.format(excerpt))
			yield inputs[excerpt]

	def train(self, filename, checkpoint_path, num_epochs, batch_size, learning_rate=0.001, dropout=0.5):
		""" 
			Trains a character-level Recurrent Neural Network in PyTorch.
		"""
		logging.info('reading `{}` for character sequences'.format(filename))

		self.preprocess(file_name=filename)
	
		logging.info('Alphabet:')
		logging.info(self.idx_to_token)
		logging.info(self.token_to_idx)

		inputs = self.sequences

		logging.info('inputs: {}'.format(len(inputs)))
	
		n_tokens = len(self.idx_to_token)
		max_length = inputs.size(1)
		
		logging.debug('creating char-level RNN model')
		model = CharRNN(num_layers=self.num_layers, rnn_type=self.rnn_type, 
						dropout=self.dropout, n_tokens=n_tokens,
						emb_size=self.emb_size, hidden_size=self.hidden_size, 
						pad_id=self.token_to_idx[self.PADDING])

		if torch.cuda.is_available():
			model = model.cuda()
		
		logging.debug('defining model training operations')
		# define training procedures and operations for training the model
		criterion = nn.NLLLoss(reduction='mean')
		optimiser = optim.Adam(model.parameters(), lr=learning_rate)
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', min_lr=1e-6, factor=0.1, patience=7, verbose=True)
	
		# train-val-test split of the dataset
		split_index = int(0.9 * inputs.size(0))
		train_tensors, inputs = inputs[: split_index], inputs[split_index: ]
		split_index = int(0.5 * inputs.size(0))
		val_tensors, test_tensors = inputs[: split_index], inputs[split_index: ]
		#del inputs
		logging.info('train tensors: {}'.format(train_tensors.size()))
		logging.info('val tensors: {}'.format(val_tensors.size()))
		logging.info('test tensors: {}'.format(test_tensors.size()))
	
		logging.debug('training char-level RNN model')
		# loop over epochs
		for epoch in range(1, num_epochs + 1):
			logging.debug('Epoch {}'.format(epoch))
			epoch_loss, n_iter = 0.0, 0
			# loop over batches
			for tensors in tqdm(self.iterate_minibatches(train_tensors, batchsize=batch_size),
							desc='Epoch[{}/{}]'.format(epoch, num_epochs), leave=False,
							total=train_tensors.size(0) // batch_size):
				# optimize model parameters
				epoch_loss += self.optimise(model, tensors, max_length, n_tokens, criterion, optimiser)
				n_iter += 1
			# evaluate model after every epoch
			val_loss = self.evaluate(model, val_tensors, max_length, n_tokens, criterion)
			# lr_scheduler decreases lr when stuck at local minima 
			scheduler.step(val_loss)
			# log epoch status info
			logging.info('Epoch[{}/{}]: train_loss - {:.4f}   val_loss - {:.4f}'.format(epoch, num_epochs, epoch_loss / n_iter, val_loss))
			
			# sample from the model every few epochs
			if epoch % 5 == 0:
				print('Epoch[{}/{}]: train_loss - {:.4f}   val_loss - {:.4f}'.format(epoch, num_epochs, epoch_loss / n_iter, val_loss))
				for _ in range(5):
					sample = self.sample(model, max_length, n_tokens)
					logging.debug(sample)
	
			checkpoint = {
				'epoch': epoch + 1,
				'valid_loss_min': val_loss,
				'state_dict': model.state_dict(),
				'optimiser': optimiser.state_dict(),
				't2i': self.token_to_idx,
				'num_layers': self.num_layers,
				'rnn_type': self.rnn_type,
				'dropout': dropout,
				'lr': learning_rate,
				'emb_size': self.emb_size,
				'hidden_size': self.hidden_size,
				'i2t': self.idx_to_token,
			}
			# save checkpoint
			best_model_path = checkpoint_path
			self.save(checkpoint, False, checkpoint_path, best_model_path)

	def save(self, state, is_best, checkpoint_path, best_model_path):
		"""
		state: checkpoint we want to save
		is_best: is this the best checkpoint; min validation loss
		checkpoint_path: path to save checkpoint
		best_model_path: path to save best model
		"""
		f_path = checkpoint_path
		# save checkpoint data to the path given, checkpoint_path
		torch.save(state, f_path)
		# if it is a best model, min validation loss
		if is_best:
			best_fpath = best_model_path
			# copy that checkpoint file to best path given, best_model_path
			shutil.copyfile(f_path, best_fpath)
	

	def load(self,checkpoint_fpath):
		"""
		checkpoint_path: path to save checkpoint
		model: model that we want to load checkpoint parameters into	   
		optimiser: optimiser we defined in previous training
		"""
		# load check point
		checkpoint = torch.load(checkpoint_fpath)

		self.token_to_idx = checkpoint['t2i']
		self.idx_to_token = checkpoint['i2t']
		n_tokens = len(self.idx_to_token)

		model = CharRNN(num_layers=checkpoint['num_layers'], rnn_type=checkpoint['rnn_type'], 
						dropout=checkpoint['dropout'], n_tokens=n_tokens,
						emb_size=checkpoint['emb_size'], hidden_size=checkpoint['hidden_size'], 
						pad_id=self.token_to_idx[self.PADDING])

		optimiser = optim.Adam(model.parameters(), lr=checkpoint['lr'])

		# initialise state_dict from checkpoint to model
		model.load_state_dict(checkpoint['state_dict'])
		# initialise optimiser from checkpoint to optimiser
		optimiser.load_state_dict(checkpoint['optimiser'])
		# initialise valid_loss_min from checkpoint to valid_loss_min
		valid_loss_min = checkpoint['valid_loss_min']
		# return model, optimiser, epoch value, min validation loss 
		logging.info('loaded model')

		self.model = model
		self.optimiser = optimiser

		#return model, optimiser #.item()
	
	
	def optimise(self, model, inputs, max_length, n_tokens, criterion, optimiser):
		model.train()
		optimiser.zero_grad()
		# compute outputs after one forward pass
		outputs = self.forward(model, inputs, max_length, n_tokens)
		# ignore the first timestep since we don't have prev input for it
		# (timesteps, batches, 1) -> (timesteps x batches x 1)
		targets = inputs[:, 1: ].contiguous().view(-1)
		# compute loss wrt targets
		loss = criterion(outputs, targets)
		# backpropagate error
		loss.backward()
		_ = torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
		# update model parameters
		optimiser.step()
		return loss.item()
	
	def evaluate(self, model, inputs, max_length, n_tokens, criterion):
		model.eval()
		# compute outputs after one forward pass
		outputs = self.forward(model, inputs, max_length, n_tokens)
		# ignore the first timestep since we don't have prev input for it
		# (timesteps, batches, 1) -> (timesteps x batches x 1)
		targets = inputs[:, 1: ].contiguous().view(-1)
		# compute loss wrt targets
		loss = criterion(outputs, targets)
		return loss.item()
	
	def forward(self, model, inputs, max_length, n_tokens):
		hidden = model.initHidden(inputs.size(0))
		if torch.cuda.is_available():
			inputs = inputs.cuda()
			if type(hidden) == tuple:
				hidden = tuple([x.cuda() for x in hidden])
			else:
				hidden = hidden.cuda()
		# tensor for storing outputs of each time-step
		outputs = torch.Tensor(max_length, inputs.size(0), n_tokens)
		# loop over time-steps
		for t in range(max_length):
			# t-th time-step input
			input_t = inputs[:, t]
			outputs[t], hidden = model(input_t, hidden)
		# (timesteps, batches, n_tokens) -> (batches, timesteps, n_tokens)
		outputs = outputs.permute(1, 0, 2)
		# ignore the last time-step since we don't have a target for it.
		outputs = outputs[:, :-1, :]
		# (batches, timesteps, n_tokens) -> (batches x timesteps, n_tokens)
		outputs = outputs.contiguous().view(-1, n_tokens)
		return outputs
	
	def sample(self, model, n_tokens, max_length=20):
		""" Generates samples using seed phrase.
	
		Args:
			model (nn.Module): the character-level RNN model to use for sampling.
			token_to_idx (dict of `str`: `int`): character to token_id mapping dictionary (vocab).
			idx_to_token (list of `str`): index (token_id) to character mapping list (vocab).
			max_length (int): max length of a sequence to sample using model.
			seed_phrase (str): the initial seed characters to feed the model. If unspecified, defaults to `START`.
		
		Returns:
			str: generated sample from the model using the seed_phrase.
		"""
		model.eval()
		seed_phrase = self.START 
		try:
			# convert to token ids for model
			sequence = [self.token_to_idx[token] for token in seed_phrase]
		except KeyError as e:
			logging.error('unknown token: {}'.format(e))
			exit(0)
		input_tensor = torch.LongTensor([sequence])
	
		hidden = model.initHidden(1)
		if torch.cuda.is_available():
			input_tensor = input_tensor.cuda()
			if type(hidden) == tuple:
				hidden = tuple([x.cuda() for x in hidden])
			else:
				hidden = hidden.cuda()
	
		# feed the seed phrase to manipulate rnn hidden states
		for t in range(len(sequence) - 1):
			_, hidden = model(input_tensor[:, t], hidden)
		
		# self.START generating
		for _ in range(max_length - len(seed_phrase)):
			# sample char from previous time-step
			input_tensor = torch.LongTensor([sequence[-1]])
			if torch.cuda.is_available():
				input_tensor = input_tensor.cuda()
			probs, hidden = model(input_tensor, hidden)
	
			# need to use `exp` as output is `LogSoftmax`
			probs = list(np.exp(np.array(probs.data[0].cpu())))
			# normalise probabilities to ensure sum = 1
			probs /= sum(probs)
			# sample char randomly based on probabilities
			sequence.append(np.random.choice(len(self.idx_to_token), p=probs))
		# format the string to ignore `pad_token` and `self.START_token` and return
		return str(''.join([self.idx_to_token[ix] for ix in sequence 
					if self.idx_to_token[ix] != self.PADDING and self.idx_to_token[ix] != self.START]))

def main():
	logging.root.setLevel(logging.NOTSET)
	try:
		lm = MorphLM()
		lm.train(filename=sys.argv[1], checkpoint_path=sys.argv[2], batch_size=32, num_epochs=100)
	except KeyboardInterrupt:
		print('Aborted!')

if __name__ == '__main__':
	main()

