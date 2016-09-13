import tensorflow as tf
import numpy as np
import argparse

def main():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('--training_data', type=str, default='', help='Historical price data file to be parsed')
	parser.add_argument('--save_dir', type=str, default='')
	parser.add_argument('--batch_size')
	parser.add_argument('--num_units')
	parser.add_argument('--num_layers')
	parser.add_argument('--input_length')