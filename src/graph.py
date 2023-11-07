#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
import math
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations,chain
import collections

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count

#novas importações
import numpy as np
import operator

class HyperGraph(defaultdict):
  """Efficient basic implementation of Hyper graph"""  
  def __init__(self):
    super(HyperGraph, self).__init__(list)

  def nodes(self):
    return list(self.keys())

  def normalGraph(self):
    d = {}
    for k,v in self.items():
      d[k] = set(chain.from_iterable(list(v)))
    return d    

  def degreesVec(self):
    res = {}
    for node in self.keys():
      res[node] = sorted([len(x) + 1 for x in self[node]], reverse=True)
    return res
  
  def degreesVecCompact(self):
    res = {}
    for node in self.keys():
      base = [len(x) + 1 for x in self[node]]
      res[node] = np.array(sorted(dict(collections.Counter(base)).items(), reverse=True))
    return res

  def degreesVecCompactMap(self):
    mappingDegreeVec = {}
    tmp_mapping = {}
    reversedMap = {}
    for node in self.keys():
      base = sorted([len(x) + 1 for x in self[node]], reverse=True)
      if str(base) not in tmp_mapping:
        tmp_mapping[str(base)] = len(tmp_mapping)
        mappingDegreeVec[tmp_mapping[str(base)]] = {'vector':base, 'compact':np.array(sorted(dict(collections.Counter(base)).items(), reverse=True)), 'node':[]}
      mappingDegreeVec[tmp_mapping[str(base)]]['node'].append(node)
    for id in mappingDegreeVec.keys():
      reversedMap.update(dict(zip(mappingDegreeVec[id]['node'], [id]*len(mappingDegreeVec[id]['node']))))
    return mappingDegreeVec, reversedMap

  def degrees(self):
    res = {}
    for node in self.keys():
      res[node] = len(self[node])
    return res
    
  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(self[k], key=lambda x:(len(x)), reverse=True))
    t1 = time()
    # logger.info('make_consistent: made consistent in {}s'.format(t1-t0))
    return self

  def maxNodeOrder(self):
    '''get the max order of node (node with largerest number of hyperedges)'''
    return max([len(self[v]) for v in list(self.keys())])

  def maxEdgeOrder(self):
    '''get the max order of edge (edge with largerest number of node)'''
    return max([len(x) for x in chain.from_iterable(list(self.values()))]) + 1

  # def number_of_edges(self):
  #   "Returns the number of nodes in the graph"
  #   return len(self)

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return len(self)

  def gToDict(self):
    d = {}
    for k,v in self.items():
      d[k] = v
    return d

def load_edgelist_hyper(file_, undirected=False):
  G = HyperGraph() # hyper graph
  with open(file_) as f:
    for line in f:
        x= line.strip().split()
        x = set(int(xx) for xx in x[1:])
        for xx in x:
          G[xx].append(x - {xx})

  G.make_consistent()
  return G

