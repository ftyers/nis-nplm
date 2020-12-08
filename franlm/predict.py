import os, sys

import shutil
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from model import CharRNN

from morphlm import MorphLM

def predict(lm, line):
	model = lm.model
	model.eval()
	line = lm.START + line.lower()

	try:
		# convert to token ids for model
		sequence = [lm.token_to_idx[token] for token in line]
	except KeyError as e:
		logging.error('unknown symbol: {}'.format(e))
		exit(0)


	hidden = model.initHidden(1)
	if torch.cuda.is_available():
		input_tensor = input_tensor.cuda()
		if type(hidden) == tuple:
			hidden = tuple([x.cuda() for x in hidden])
		else:
			hidden = hidden.cuda()

	# feed the seed phrase to manipulate rnn hidden states

	# loop over the input

#	input_tensor = torch.LongTensor([sequence])
	ctx = [sequence[0]]
	_, hidden = model(torch.LongTensor([ctx[0]]), hidden)
	predictions = [sequence[0]]
	hits = 0 # number of correct guesses
	clicks = 0 #
	clickstxt = []
	t = 1
	while t < len(sequence):
#	for t in range(1,len(sequence)):
#		model.eval()
#		hidden = model.initHidden(1)
		suggestions = []	
		print('# t:', t, '||| ctx:', ''.join([lm.idx_to_token[i] for i in ctx]), '|||', predictions, '|||pred:', ''.join([lm.idx_to_token[i] for i in predictions]))
#		for j in ctx:
#			print(t, j)
#			_, hidden = model(torch.LongTensor([j]), hidden)

#		print('~')
		suggestions = []	
		
		input_tensor = torch.LongTensor([ctx[-1]])
	
		if torch.cuda.is_available():
			input_tensor = input_tensor.cuda()

		# step the model and normalise the probs
		probs, hidden = model(input_tensor, hidden)
		probs = list(np.exp(np.array(probs.data[0].cpu())))
		probs /= sum(probs)

		# this is the top list of predicted characters 
		topN = []
		mass = 0 # prob mass
		pprobs = [(j, i) for (i, j) in enumerate(probs)]
		pprobs.sort(reverse=True)
		for (j, i) in pprobs:
			print('pp:', j, i)
			mass += j
			if mass >= 0.95: # break if we already found the top best, probably also break with >= 3
				topN.append(i)
				break	
			topN.append(i)
		
		print('topN:', topN)
		# for each of the top chars
		for char in topN:
			tmphidden = hidden
			prediction = lm.idx_to_token[char]
			cchar = lm.idx_to_token[char]
			print('  char:', lm.idx_to_token[char], '|||', char,'|||', probs[char])
			tmp = ctx + [char]
			p2prob = 1.0
			# unroll till we hit either:
			#   '>' (morph boundary), 
			#   ' ' (word boundary)
			#   '*' (packing)
			# also hard quit if the pred is longer than 10
			while cchar != '>' and cchar!= ' ' and cchar !='*' and len(prediction) <= 10:
				p2, tmphidden = model(torch.LongTensor([char]), tmphidden)
				p2= list(np.exp(np.array(p2.data[0].cpu())))
				p2/= sum(p2)
				p2pred = np.argmax(p2)
				p2prob *= p2[p2pred]
				cchar = lm.idx_to_token[p2pred]
				prediction += cchar
				tmp = tmp + [p2pred]
				print('       ', cchar, '|||', p2pred,'|||', p2prob)
			suggestions.append((prediction, p2prob))
			print('  suggestion:', prediction, p2prob)

		print('suggestions:', suggestions)

		found = False
		for s2 in suggestions:
			print('%', s2, '|||',line[t:t+len(s2[0])], '|||',line)
			if line[t:t+len(s2[0])] == s2[0]:
				print('HIT!', s2)
				ctx += [lm.token_to_idx[k] for k in s2[0]]
				predictions += [lm.token_to_idx[k] for k in s2[0]]
				found = True
				hits += len(s2[0])
				t += len(s2[0])
				clickstxt.append(s2[0])
				break

		pred = np.argmax(probs)
		cpred = lm.idx_to_token[pred]

		if found:
			print('CLICK', ctx)
			clicks += 1
			continue

		ctx.append(sequence[t])
		predictions.append(pred)

		if pred != sequence[t]:
			print('!',end=' ')
		else:
			print('@',end=' ')
			hits += 1
		clicks += 1
		clickstxt.append(lm.idx_to_token[sequence[t]])
		print(lm.idx_to_token[sequence[t]], '|||', cpred, '|||', probs[pred])

		t += 1
#		print(cpred , '|||', pred, '|||', lm.idx_to_token[sequence[t]], '|||', sequence[t], '|||', probs[pred])


	print('seq:',sequence, ''.join([lm.idx_to_token[i] for i in sequence]))
	print('ctx:',ctx)
	print('pred:',predictions, ''.join([lm.idx_to_token[i] for i in predictions]))
	print('hits:',hits)
	print('clicks:',clicks, clickstxt)
	print('len:', len(line.replace('>','')))

def main():
	logging.root.setLevel(logging.NOTSET)
	try:
		lm = MorphLM()
		lm.load(sys.argv[1])
		#predict(lm, 'чама ны>ръиле>ӄинэ>т')
		#predict(lm, 'алымъым чит нин нрычвойгымъым')
		#predict(lm, 'энмэн га>тва>лен валв>ийӈы>н')
		predict(lm, 'she is run>ning')

	except KeyboardInterrupt:
		print('Aborted!')

if __name__ == '__main__':
	main()

