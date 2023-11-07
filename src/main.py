#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse, logging
import numpy as np
import hypers2v
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from time import time
import inspect
import os.path
import graph
import config
# from memory_profiler import profile

def parse_args():
	'''
	Parses the HyperS2V arguments.
	'''
	parser = argparse.ArgumentParser(description="Run HyperS2V.")

	parser.add_argument('--input', nargs='?', default='karate-mirrored.edgelist',
	                    help='Input graph path')
	parser.add_argument('--output', nargs='?', default=None,
	                    help='Output emb path, if Not given, follow input file name')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=20,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--until-layer', type=int, default=6,
                    	help='Calculation until the layer.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	# parser.add_argument('--weighted', dest='weighted', action='store_true',
	#                     help='Boolean specifying (un)weighted. Default is unweighted.')
	# parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	# parser.set_defaults(weighted=False)

	# parser.add_argument('--directed', dest='directed', action='store_true',
	#                     help='Graph is (un)directed. Default is undirected.')
	# parser.add_argument('--undirected', dest='undirected', action='store_false')
	# parser.set_defaults(directed=False)

	parser.add_argument('--OPT1', action='store_true',
                      help='optimization 1: using CMPD and CNCMPD to calculate similarity. Recommended.')
	parser.add_argument('--OPT2', action='store_true',
                      help='optimization 2: thin out node pair to make multilayer network sparse. Recommended for large networks. Try turning off first.')
	parser.add_argument('--OPT3', action='store_true',
                      help='optimization 3: activate the until-layer argument. Recommended.')
	# parser.add_argument('--OPT4', action='store_true',
    #                   help='optimization 4 kmeans for degree list')
	# parser.add_argument('--OPT5', action='store_true',
    #                   help='optimization 5 kmeans for degree list on each axis')
	# parser.add_argument('--scalefree', action='store_true',
    #                   help='scale free flag')
	parser.add_argument('--suffix', nargs='?', default='',
	                    help='log file and pickles folder suffix')
	return parser.parse_args()

def read_graph_hyper():
	'''
	Reads the input hyper network.
	'''
	logging.info(" - Loading hyper graph...")
	G = graph.load_edgelist_hyper(args.input,undirected=False)
	logging.info(" - Hyper Graph loaded.")
	return G

def learn_embeddings(basename, suffix=None):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	logging.info("Initializing creation of the representations...")
	if suffix is None:
		walks = LineSentence('random_walks.txt')
	else:
		walks = LineSentence('pickles_{}/random_walks.txt'.format(suffix))
# 	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=1, sg=1, workers=args.workers, iter=args.iter)
#	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, hs=0, sg=1,negative=5, ns_exponent=0.75, workers=args.workers, iter=args.iter)
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, hs=0, sg=1,negative=5, ns_exponent=0.75, workers=args.workers, epochs=args.iter)
	model.wv.save_word2vec_format("emb/{}.emb".format(basename))
	logging.info("Representations created.")

	return

def exec_hypers2v(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	if(args.OPT3):
		until_layer = args.until_layer
	else:
		until_layer = None

	# G = read_graph()
	G = read_graph_hyper()
	print('read complete')
	G = hypers2v.Graph_hyper(G, args.workers, untilLayer = until_layer, opt4 = None, opt1 = args.OPT1)
	print('HyperS2V graph complete')
	
	if(args.OPT1):
		G.preprocess_neighbors_with_bfs_compact() # TODO Later
		pass
	else:
		G.preprocess_neighbors_with_bfs()

	logging.info("preprocess_neighbors_XXX end")
	if(args.OPT2):
		# if(args.OPT5):
		# 	G.create_vectors_complex_directed_shrink() # TODO		
		# 	logging.info("create_vectors_complex_directed_shrink end ")
		# 	G.calc_distances_complex_directed_shrink(compactDegree = args.OPT1, scale_free = args.scalefree) # TODO
		# else:
		G.create_vectors_hyper()
		logging.info("create_vectors_hyper end")
		G.calc_distances_hyper(compactDegree = args.OPT1, scale_free = False)
		pass
	else:
		G.calc_distances_all_vertices(compactDegree = args.OPT1)
	logging.info("calc_distances_XXX end")
	G.create_distances_network()
	logging.info("create_distances_network end")
	G.preprocess_parameters_random_walk()
	print('multi-layer network generation complete')
	logging.info("multi-layer network generation complete")
	G.simulate_walks(args.num_walks, args.walk_length, args.suffix)

	print('random walk complete')
	return G

# @profile
def main(args):
	print('Process start')
	G = exec_hypers2v(args)
	print('complete network generations.')
	if (args.output is not None):
		basename = args.output
	else:
		basename = args.input.split("/")[1].split(".")[0]
	learn_embeddings(basename, args.suffix)

if __name__ == "__main__":
	args = parse_args()
	print (args)
	dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	config.folder_pickles = dir_f+"/../pickles_{}/".format(args.suffix)
	os.makedirs(config.folder_pickles, exist_ok=True)
	logging.basicConfig(filename='pickles_{}/hypers2v_{}.log'.format(args.suffix, args.suffix),filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')
	logging.info("Args={}".format(args))

	# print (config.folder_pickles)
	main(args)