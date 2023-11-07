# -*- coding: utf-8 -*-

import numpy as np
import random,sys,logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from time import time
from collections import deque, defaultdict

from utils import *
from algorithm import *
from algorithm_distance import *
import graph
import itertools
# from memory_profiler import profile
from sklearn.preprocessing import MinMaxScaler

class Graph_hyper():
	def __init__(self, g, workers, untilLayer = None, opt4=False, opt1=False):

		logging.info(" - Converting graph to dict...")
		self.H = defaultdict(mylambda)
		self.H.update(g.gToDict())
		self.nG = defaultdict(mylambda)
		self.nG.update(g.normalGraph())

		logging.info("Graph converted.")
		self.vertices = g.nodes()
		self.degreesVec = g.degreesVec()
		self.degrees = g.degrees()
		if opt1 :
			self.degreesVecCompact = g.degreesVecCompact()
			# print (self.H, self.degreesVecCompact)
			# print (g.degreesVecCompactMap())
			self.degreesVecCompactMap, self.reversedMap = g.degreesVecCompactMap()
			saveVariableOnDisk(self.degreesVecCompactMap,'degreesVecCompactMap')
			saveVariableOnDisk(self.reversedMap,'reversedMap')
			logging.info('HyperGraph - Number of degreeVec: {}'.format(max(self.reversedMap.values())+1))

		self.maxNodeOrder = g.maxNodeOrder()
		self.maxEdgeOrder = g.maxEdgeOrder()
		self.num_vertices = len(self.vertices)
		# self.num_edges = g1.number_of_edges() + g2.number_of_edges() + g3.number_of_edges() + g4.number_of_edges()
		self.workers = workers
		self.calcUntilLayer = untilLayer
		self.opt4 = opt4
		self.opt1 = opt1
		logging.info('Graph - Number of vertices: {}'.format(self.num_vertices))
		# logging.info('Graph - Number of edges: {}'.format(self.num_edges))

	def preprocess_neighbors_with_bfs(self):
		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs_hyper,self.H,self.degreesVec,self.workers,self.calcUntilLayer)

			job.result()

		return

	def preprocess_neighbors_with_bfs_compact(self): # TODO
		# exec_bfs_compact_complex_directed(self.G_pi,self.G_ni,self.G_po,self.G_no,self.workers,self.calcUntilLayer)
		with ProcessPoolExecutor(max_workers=self.workers) as executor:
			job = executor.submit(exec_bfs_compact_hyper, self.H, self.degreesVecCompact,self.degreesVecCompactMap, self.reversedMap, self.workers, self.calcUntilLayer, self.opt4)

			job.result()

		return

	def create_vectors_hyper(self):
		'''
		cal the top log(n) linked nodes by sum, max, and length
		'''
		logging.info("Creating degree vectors [hyper] ...")
		degrees_len = {} # matrix
		degrees_max = {} # matrix
		degrees_sum = {} # matrix
		degrees = {} # matrix
		degrees_sorted_len = set()
		degrees_sorted_max = set()
		degrees_sorted_sum = set()
		G = self.H

		for v in G.keys():
			len_ = len(G[v])
			max_ = max([len(x) + 1 for x in G[v]])
			sum_ = sum([len(x) + 1 for x in G[v]])
			degrees_sorted_len.add(len_)
			degrees_sorted_max.add(max_)
			degrees_sorted_sum.add(sum_)
			if len_ not in degrees_len:
				degrees_len[len_] = deque()
			degrees_len[len_].append(v)
			if max_ not in degrees_max:
				degrees_max[max_] = deque()
			degrees_max[max_].append(v)
			if sum_ not in degrees_sum:
				degrees_sum[sum_] = deque()
			degrees_sum[sum_].append(v)
		
		degrees_sorted_len = np.array(list(degrees_sorted_len),dtype='int')
		degrees_sorted_len = np.sort(degrees_sorted_len)
		degrees_sorted_max = np.array(list(degrees_sorted_max),dtype='int')
		degrees_sorted_max = np.sort(degrees_sorted_max)
		degrees_sorted_sum = np.array(list(degrees_sorted_sum),dtype='int')
		degrees_sorted_sum = np.sort(degrees_sorted_sum)

		degrees_sorted_len = list(degrees_sorted_len)
		degrees_sorted_max = list(degrees_sorted_max)
		degrees_sorted_sum = list(degrees_sorted_sum)

		saveVariableOnDisk(degrees_sorted_len,'degrees_vector_lenList')
		saveVariableOnDisk(degrees_sorted_max,'degrees_vector_maxList')
		saveVariableOnDisk(degrees_sorted_sum,'degrees_vector_sumList')
		saveVariableOnDisk(degrees_len,'degrees_vector_node_lenList')
		saveVariableOnDisk(degrees_max,'degrees_vector_node_maxList')
		saveVariableOnDisk(degrees_sum,'degrees_vector_node_sumList')

		logging.info("Degree vectors numbers = len:{}, max:{}, sum:{}.".format(len(degrees_len),len(degrees_max),len(degrees_sum)))

	def create_vectors_complex_directed_shrink(self):
		'''
		cal the top log(n) linked nodes, node degree shrinked by kmeans.
		'''
		logging.info("Creating degree vectors...")
		degrees_pi = {} # matrix
		degrees_ni = {} # matrix
		degrees_po = {} # matrix
		degrees_no = {} # matrix
		degrees = {} # matrix
		degrees_sorted_pi = set()
		degrees_sorted_ni = set()
		degrees_sorted_po = set()
		degrees_sorted_no = set()
		
		G = self.G_pi
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted_pi.add(degree)
			if(degree not in degrees_pi):
				degrees_pi[degree] = deque()
				# degrees_pi[degree]['vertices'] = deque() 
			degrees_pi[degree].append(v)
		degrees_sorted_pi = np.array(list(degrees_sorted_pi),dtype='int')
		degrees_sorted_pi = np.sort(degrees_sorted_pi)

		kmeans = KMeans(n_clusters=int(np.sqrt(len(degrees_sorted_pi))), random_state=0).fit(np.array(degrees_sorted_pi.reshape(-1,1)))
		bb = kmeans.labels_
		aa = kmeans.cluster_centers_
		mapping_AD2node_pi = {0:list(set(self.vertices) - set(G.keys()))}
		mapping_degree2AD_pi = {0:0}
		# mapping_AD2node_pi = {}
		# mapping_degree2AD_pi = {}
		for index_ in range(len(aa)):
			# print (aa[index_][0])
			mapping_AD2node_pi[int(aa[index_][0].round())]=list(itertools.chain.from_iterable([degrees_pi[x] for x in degrees_sorted_pi[bb==index_]]))
			mapping_degree2AD_pi.update(dict(zip(degrees_sorted_pi[bb==index_], [int(aa[index_][0].round())]*len(degrees_sorted_pi[bb==index_]))))
		mapping_AD2node_pi = dict(sorted(mapping_AD2node_pi.items()))

		G = self.G_ni
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted_ni.add(degree)
			if(degree not in degrees_ni):
				degrees_ni[degree] = deque()
				# degrees_ni[degree]['vertices'] = deque() 
			degrees_ni[degree].append(v)
		degrees_sorted_ni = np.array(list(degrees_sorted_ni),dtype='int')
		degrees_sorted_ni = np.sort(degrees_sorted_ni)

		kmeans = KMeans(n_clusters=int(np.sqrt(len(degrees_sorted_ni))), random_state=0).fit(np.array(degrees_sorted_ni.reshape(-1,1)))
		bb = kmeans.labels_
		aa = kmeans.cluster_centers_
		mapping_AD2node_ni = {0:list(set(self.vertices) - set(G.keys()))}
		mapping_degree2AD_ni = {0:0}
		# mapping_AD2node_ni = {}
		# mapping_degree2AD_ni = {}
		for index_ in range(len(aa)):
			# print (aa[index_][0])
			mapping_AD2node_ni[int(aa[index_][0].round())]=list(itertools.chain.from_iterable([degrees_ni[x] for x in degrees_sorted_ni[bb==index_]]))
			mapping_degree2AD_ni.update(dict(zip(degrees_sorted_ni[bb==index_], [int(aa[index_][0].round())]*len(degrees_sorted_ni[bb==index_]))))
		mapping_AD2node_ni = dict(sorted(mapping_AD2node_ni.items()))
			
		G = self.G_po
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted_po.add(degree)
			if(degree not in degrees_po):
				degrees_po[degree] = deque()
				# degrees_po[degree]['vertices'] = deque() 
			degrees_po[degree].append(v)
		degrees_sorted_po = np.array(list(degrees_sorted_po),dtype='int')
		degrees_sorted_po = np.sort(degrees_sorted_po)

		kmeans = KMeans(n_clusters=int(np.sqrt(len(degrees_sorted_po))), random_state=0).fit(np.array(degrees_sorted_po.reshape(-1,1)))
		bb = kmeans.labels_
		aa = kmeans.cluster_centers_
		mapping_AD2node_po = {0:list(set(self.vertices) - set(G.keys()))}
		mapping_degree2AD_po = {0:0}
		# mapping_AD2node_po = {}
		# mapping_degree2AD_po = {}
		for index_ in range(len(aa)):
			print (aa[index_][0])
			mapping_AD2node_po[int(aa[index_][0].round())]=list(itertools.chain.from_iterable([degrees_po[x] for x in degrees_sorted_po[bb==index_]]))
			mapping_degree2AD_po.update(dict(zip(degrees_sorted_po[bb==index_], [int(aa[index_][0].round())]*len(degrees_sorted_po[bb==index_]))))
		mapping_AD2node_po = dict(sorted(mapping_AD2node_po.items()))
			
		G = self.G_no
		for v in G.keys():
			degree = len(G[v])
			degrees_sorted_no.add(degree)
			if(degree not in degrees_no):
				degrees_no[degree] = deque() 
				# degrees_no[degree]['vertices'] = deque() 
			degrees_no[degree].append(v)
		degrees_sorted_no = np.array(list(degrees_sorted_no),dtype='int')
		degrees_sorted_no = np.sort(degrees_sorted_no)

		kmeans = KMeans(n_clusters=int(np.sqrt(len(degrees_sorted_no))), random_state=0).fit(np.array(degrees_sorted_no.reshape(-1,1)))
		bb = kmeans.labels_
		aa = kmeans.cluster_centers_
		mapping_AD2node_no = {0:list(set(self.vertices) - set(G.keys()))}
		mapping_degree2AD_no = {0:0}
		# mapping_AD2node_no = {}
		# mapping_degree2AD_no = {}
		for index_ in range(len(aa)):
			# print (aa[index_][0])
			mapping_AD2node_no[int(aa[index_][0].round())]=list(itertools.chain.from_iterable([degrees_no[x] for x in degrees_sorted_no[bb==index_]]))
			mapping_degree2AD_no.update(dict(zip(degrees_sorted_no[bb==index_], [int(aa[index_][0].round())]*len(degrees_sorted_no[bb==index_]))))
		mapping_AD2node_no = dict(sorted(mapping_AD2node_no.items()))
			
		saveVariableOnDisk(mapping_degree2AD_ni,'degrees_vector_negativeInList')
		saveVariableOnDisk(mapping_degree2AD_pi,'degrees_vector_positiveInList')
		saveVariableOnDisk(mapping_degree2AD_no,'degrees_vector_negativeOutList')
		saveVariableOnDisk(mapping_degree2AD_po,'degrees_vector_positiveOutList')
		saveVariableOnDisk(mapping_AD2node_ni,'degrees_vector_node_negativeInList')
		saveVariableOnDisk(mapping_AD2node_pi,'degrees_vector_node_positiveInList')
		saveVariableOnDisk(mapping_AD2node_no,'degrees_vector_node_negativeOutList')
		saveVariableOnDisk(mapping_AD2node_po,'degrees_vector_node_positiveOutList')

		logging.info("Degree vectors numbers = {},{},{},{}.".format(len(mapping_AD2node_no),len(mapping_AD2node_ni),len(mapping_AD2node_po),len(mapping_AD2node_pi)))
		# print (len(mapping_AD2node_no), len(mapping_degree2AD_no))
		# print (len(mapping_AD2node_ni), len(mapping_degree2AD_ni))
		# print (len(mapping_AD2node_po), len(mapping_degree2AD_po))
		# print (len(mapping_AD2node_pi), len(mapping_degree2AD_pi))
		for degree_pi in mapping_AD2node_pi.keys():
			# print (degree_pi)
			logging.info("Degree vectors numbers process: {}".format(degree_pi))
			for degree_ni in mapping_AD2node_ni.keys():
				# print (degree_pi,degree_ni)
				for degree_po in mapping_AD2node_po.keys():
					for degree_no in mapping_AD2node_no.keys():
						# degree = np.quaternion(degree_pi, degree_ni, degree_po, degree_no)
						degree = (degree_pi, degree_ni, degree_po, degree_no)
						if(degree not in degrees):
							degrees[degree] = []
						degrees[degree] = list(set(mapping_AD2node_ni[degree_ni]) & set(mapping_AD2node_pi[degree_pi]) & set(mapping_AD2node_no[degree_no]) & set(mapping_AD2node_po[degree_po]))

		logging.info("Degree vectors created.")
		logging.info("Saving degree vectors...")
		saveVariableOnDisk(degrees,'degrees_vector')

	def pre_calc_distances_all_vertices(self,compactDegree = False):
		futures = {}
		vertices = self.vertices
		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()
		distances_l0 = {}
		with ProcessPoolExecutor(max_workers = self.workers) as executor:
			part = 1
			for c in chunks:
				logging.info("[Level 0] Executing part {}...".format(part)+str(c))
				list_v = []
				for v in c:
					list_v += [vd for vd in vertices if vd > v]
				job = executor.submit(calc_distances_l0_hyper, c, list_v, self.degreesVec,part, compactDegree = compactDegree)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")

			for job in as_completed(futures):
				dl = job.result()
				r = futures[job]
				distances_l0.update(dl)
				logging.info("Part {} Completed.".format(r))
		saveVariableOnDisk(distances_l0,'distances-l0')
		logging.info('Distances level 0 calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))
		return

	def calc_distances_all_vertices(self,compactDegree = False):
		# maxA = np.log(np.sqrt(self.n_max_degree**2+self.p_max_degree**2)+1)
		# set_maxA(maxA)
		# logging.info("Using maxA: {}".format(maxA))
		logging.info("Using compactDegree: {}".format(compactDegree))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		count_calc = 0
		vertices = self.vertices

		if(compactDegree):
			logging.info("Recovering compactDegreeList from disk...")
			degreeList = restoreVariableFromDisk('compactDegreeList')
			reversedMap = restoreVariableFromDisk('reversedMap')
			degreesVecCompactMap = restoreVariableFromDisk('degreesVecCompactMap')
		else:
			logging.info("Recovering degreeList from disk...")
			degreeList = restoreVariableFromDisk('degreeList')
			reversedMap = None
			degreesVecCompactMap = None
		# print (degreeList)
		parts = self.workers
		chunks = partition(vertices,parts)

		t0 = time()
		# logging.info("calculate mean dist of 0 hops Start...")
		# theta_layer0 = calc_distances_0hop_hyper(degreeList)
		# logging.info("calculate mean dist of 0 hops End  ...")
		
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part)+str(c))
				list_v = []
				for v in c:
					list_v.append([vd for vd in degreeList.keys() if vd > v])
				# job = executor.submit(calc_distances_all_hyper, c, list_v, degreeList,part,theta_layer0, compactDegree = compactDegree)
				job = executor.submit(calc_distances_all_hyper_euclid2, c, list_v, degreeList, reversedMap, degreesVecCompactMap, part, self.calcUntilLayer, compactDegree = compactDegree)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")

			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} Completed.".format(r))
# end debug
		logging.info('Distances calculated.')
		t1 = time()
		logging.info('Time : {}m'.format((t1-t0)/60))

		return

	def calc_distances_hyper(self, compactDegree = False, scale_free = True):
    
		logging.info("Using compactDegree: {}".format(compactDegree))
		logging.info("Scale free flag: {}".format(scale_free))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		#distances = {}

		count_calc = 0

		# G = self.G
		# vertices = list(set(list(self.G_p.keys()) + list(self.G_n.keys())))
		vertices = self.vertices
		parts = self.workers
		chunks = partition(vertices,parts)

		# part = 1
		# for c in chunks:
		# 	# print ('debug part = '+str(part)+ ' chunks'+str(chunks))
		# 	splitDegreeList_hyper(part,c,vertices,self.H,compactDegree)
		# 	# print ('debug part = '+str(part))
		# 	part += 1
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			logging.info("Split degree List...")
			part = 1
			for c in chunks:
				# job = executor.submit(splitDegreeList_hyper,part,c,self.G_p,self.G_n,compactDegree)
				job = executor.submit(splitDegreeList_hyper,part,c,vertices,self.H,compactDegree)
				futures[job] = part
				part += 1
				logging.info("degreeList {} completed.".format(part))
				# job.result()
				# part += 1

			logging.info("[splitDegreeList_hyper] Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))

		# if(compactDegree):
		# 	logging.info("Recovering compactDegreeList from disk...")
		# 	degreeList = restoreVariableFromDisk('compactDegreeList')
		# else:
		# 	logging.info("Recovering degreeList from disk...")
		# 	degreeList = restoreVariableFromDisk('degreeList')
		# distance_list = []
		# futures = {}
		# with ProcessPoolExecutor(max_workers = self.workers) as executor:
		# 	logging.info("calculate dist list of 0 hops Start...")
		# 	part = 1
		# 	for c in chunks:
		# 		# job = executor.submit(splitDegreeList_hyper,part,c,self.G_p,self.G_n,compactDegree)
		# 		job = executor.submit(calc_distances_0hop_hyper_opt2, degreeList, part)
		# 		futures[job] = part
		# 		part += 1
		# 		logging.info("dist list {} completed.".format(part))
		# 		# job.result()
		# 		# part += 1
		# 	logging.info("[calc_distances_0hop_hyper_opt2] Receiving results...")
		# 	for job in as_completed(futures):
		# 		dl = job.result()
		# 		# print (job)
		# 		r = futures[job]
		# 		distance_list += dl
		# 		logging.info("Part {} completed.".format(r))
		# # theta_layer0 = calc_distances_0hop_hyper(distance_list)
		# scaler = MinMaxScaler()
		# distance_list #TODO!!!0612
		# theta_layer0 = gradient_decent(np.array(distance_list), L=np.e)
		# logging.info("calculate dist of 0 hops End  ...")
		

		# part = 1
		
		# for c in chunks:
		# 	logging.info("Executing part {}...".format(part))
		# 	calc_distances_hyper_r2_euclide2( part, self.calcUntilLayer, compactDegree = compactDegree, scale_free = scale_free)
		# 	part += 1

		futures = {}
		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				# job = executor.submit(calc_distances_hyper_r2, part, theta_layer0, compactDegree = compactDegree, scale_free = scale_free)
				job = executor.submit(calc_distances_hyper_r2_euclide2, part, self.calcUntilLayer, compactDegree = compactDegree, scale_free = scale_free)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))


		return

	def calc_distances_complex_directed_shrink(self, compactDegree = False, scale_free = True):
    
		logging.info("Using compactDegree: {}".format(compactDegree))
		logging.info("Scale free flag: {}".format(scale_free))
		if(self.calcUntilLayer):
			logging.info("Calculations until layer: {}".format(self.calcUntilLayer))

		futures = {}
		#distances = {}

		count_calc = 0

		# G = self.G
		# vertices = list(set(list(self.G_p.keys()) + list(self.G_n.keys())))
		vertices = list(reversed(sorted(list(set(list(self.G_pi.keys()) + list(self.G_ni.keys()) + list(self.G_po.keys()) + list(self.G_no.keys()))))))

		parts = self.workers
		chunks = partition(vertices,parts)
#S
		part = 1
		for c in chunks:
			# print ('debug part = '+str(part)+ ' chunks'+str(chunks))
			splitDegreeList_complex_directed_shrink(part,c,self.G_pi,self.G_ni,self.G_po,self.G_no,compactDegree)
			# print ('debug part = '+str(part))
			part += 1
#E

		# with ProcessPoolExecutor(max_workers = 1) as executor:

		# 	logging.info("Split degree List...")
		# 	part = 1
		# 	for c in chunks:
		# 		job = executor.submit(splitDegreeList_complex,part,c,self.G_p,self.G_n,compactDegree)
		# 		job.result()
		# 		logging.info("degreeList {} completed.".format(part))
		# 		part += 1

		# part = 1
		
		# for c in chunks:
		# 	logging.info("Executing part {}...".format(part))
		# 	calc_distances_complex( part, compactDegree = compactDegree)
		# 	part += 1

		with ProcessPoolExecutor(max_workers = self.workers) as executor:

			part = 1
			for c in chunks:
				logging.info("Executing part {}...".format(part))
				job = executor.submit(calc_distances_complex_directed, part, compactDegree = compactDegree, scale_free = scale_free)
				futures[job] = part
				part += 1

			logging.info("Receiving results...")
			for job in as_completed(futures):
				job.result()
				r = futures[job]
				logging.info("Part {} completed.".format(r))


		return

	def consolide_distances(self):

		distances = {}

		parts = self.workers
		for part in range(1,parts + 1):
			d = restoreVariableFromDisk('distances-'+str(part))
			preprocess_consolides_distances(distances)
			distances.update(d)


		preprocess_consolides_distances(distances)
		saveVariableOnDisk(distances,'distances')


	def create_distances_network(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_distances_network,self.workers)

			job.result()

		return

	def preprocess_parameters_random_walk(self):

		with ProcessPoolExecutor(max_workers=1) as executor:
			job = executor.submit(generate_parameters_random_walk,self.workers)

			job.result()

		return


	def simulate_walks(self,num_walks,walk_length,suffix):

		# for large graphs, it is serially executed, because of memory use.
		if(self.num_vertices > 500000):

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks_large_graphs,num_walks,walk_length,self.workers, self.vertices, suffix)

				job.result()

		else:

			with ProcessPoolExecutor(max_workers=1) as executor:
				job = executor.submit(generate_random_walks,num_walks,walk_length,self.workers, self.vertices, suffix)
				job.result()

		return	
