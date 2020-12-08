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

from morphlm import MorphLM

def main():
	logging.root.setLevel(logging.NOTSET)
	try:
		lm = MorphLM(num_layers=2,hidden_size=256,emb_size=50)
		lm.train(filename=sys.argv[1], checkpoint_path=sys.argv[2], batch_size=32, learning_rate=0.001, num_epochs=100)
	except KeyboardInterrupt:
		print('Aborted!')

if __name__ == '__main__':
	main()

