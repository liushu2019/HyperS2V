# -*- coding: utf-8 -*-
from ast import While
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os
# import quaternion
from sklearn.cluster import KMeans
import collections
import warnings
import random
# import cmath

limiteDist = 20
maxA = 2.0 # for calculating the distance

def getDegreeListsVertices_hyper(g,degreeVec,vertices,calcUntilLayer):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getDegreeLists_hyper(g,degreeVec,v,calcUntilLayer)

    return degreeList

def getCompactDegreeListsVertices_hyper(H,degreesVecCompact,degreesCecCompactMap, reversedMap,vertices,calcUntilLayer,opt4):
    degreeList = {}

    logging.info('TEST getCompactDegreeListsVertices_complex in') # DEBUG
    if opt4:
        for v in vertices:
            degreeList[v] = getDegreeLists_complex_directed_kmeans(Gpi,Gni,Gpo,Gno,v,calcUntilLayer) # TODO
    else:
        for v in vertices:
            degreeList[v] = getCompactDegreeLists_hyper(H,degreesVecCompact,degreesCecCompactMap, reversedMap,v,calcUntilLayer)
    return degreeList

def getCompactDegreeLists_hyper(H, degreesVecCompact,degreesCecCompactMap, reversedMap, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(list(H.keys())) + 1)
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    list_tmp = []

    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    logging.info('BFS vertex {}. in !'.format(root))
    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1
        list_tmp.append(reversedMap[vertex])
        for v_list in H[vertex]:
            for v in v_list:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):
            listas[depth] = tuple(sorted(dict(collections.Counter(list_tmp)).items(), reverse=True))
            list_tmp = []

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0

    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas

def getDegreeLists_complex_directed_kmeans(gpi,gni,gpo,gno, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(max(gpi), max(gni), max(gpo), max(gno)) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1


    l = {}

    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = np.quaternion(len(gpi[vertex]), len(gni[vertex]), len(gpo[vertex]), len(gno[vertex]))
        if(d not in l):
            l[d] = 0
        l[d] += 1

        for v in gpi[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   
        for v in gni[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1  
        for v in gpo[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   
        for v in gno[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1  

        if(timeToDepthIncrease == 0):
            list_d = []
            for degree,freq in l.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0].real+x[0].imag.sum())
            # listas[depth] = np.array(list_d)#,dtype=np.int32)
            listas[depth] = list_d

            l = {}

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0
            if(2 == depth):
                break
    if calcUntilLayer > 1:
        l=deque()
        while queue:
            vertex = queue.popleft()
            timeToDepthIncrease -= 1

            # l.append(len(g[vertex]))
            l.append([len(gpi[vertex]), len(gni[vertex]), len(gpo[vertex]), len(gno[vertex])])

            for v in gpi[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1    
            for v in gni[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1   
            for v in gpo[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1    
            for v in gno[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1   

            if(timeToDepthIncrease == 0 and depth > 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=min(16, len(l)), random_state=0).fit(np.array(l))
                    bb = kmeans.labels_
                    aa = kmeans.cluster_centers_
                    cc = collections.Counter(bb)
                    cc = dict(sorted(cc.items(), key=lambda x:x[1], reverse=True))
                    lp = []
                    for ix,x in cc.items():
                        lp.append((np.quaternion(aa[ix][0], aa[ix][1], aa[ix][2], aa[ix][3]), x))
                    listas[depth] = lp
                    l = deque()

                if(calcUntilLayer == depth):
                    break

                depth += 1
                timeToDepthIncrease = pendingDepthIncrease
                pendingDepthIncrease = 0
    
    t1 = time()
    logging.info('BFS vertex kmeans {}. Time: {}s'.format(root,(t1-t0)))

    return listas

def getDegreeLists_hyper(g, degreeVec, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1


    l = deque()

    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1
        l.append(degreeVec[vertex])

        for v1 in g[vertex]:
            for v in v1:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1

        if(timeToDepthIncrease == 0):

            # lp = np.array(sorted(l, key=len, reverse=True))
            lp = sorted(l, key=lambda x: (len(x), max(x), sum(x), np.median(x)),reverse=True)
            listas[depth] = lp
            l = deque()

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0

    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas

def cost(a,b):
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return ((m/mi) - 1)
def cost2d(a_,b_):
    # print (a_,b_)
    a = a_[0]
    b = b_[0]
    ep = 0.5
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return np.sqrt(((m/mi) - 1)**2 + (a_[1]-b_[1])**2)
def cost_euc(a,b):
    return (abs(a-b))

def cost_max(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * max(a[1],b[1])

def cost_complex_vector_2area_logscale(a,b):
    a_ = np.log(np.array(a)+1)
    b_ = np.log(np.array(b)+1)    
    return abs(a_[0]*b_[1]-a_[1]*b_[0]) + (np.linalg.norm(a_) - np.linalg.norm(b_))**2

def cost_complex_vector_Euclidean_logscale(a,b):
    a_ = np.log(np.array(a)+1)
    b_ = np.log(np.array(b)+1)    
    return np.linalg.norm(a_-b_)

def cost_complex_vector_Euclidean_logscale_max(a,b):
    a_ = np.log(np.array(a[0])+1)
    b_ = np.log(np.array(b[0])+1)    
    # return np.linalg.norm(a_-b_)*max(a[1],b[1])
    return np.linalg.norm(a_-b_)*max(a[1],b[1])

def cost_complex_vector_Euclidean(a,b):
    # a_ = np.log(np.array(a)+1)
    # b_ = np.log(np.array(b)+1)    
    return np.linalg.norm(a-b)

def cost_complex_vector_Euclidean_max(a,b):
    # a_ = np.log(np.array(a[0])+1)
    # b_ = np.log(np.array(b[0])+1)    
    # return np.linalg.norm(a_-b_)*max(a[1],b[1])
    return np.linalg.norm(a[0]-b[0])*max(a[1],b[1])
    # return np.linalg.norm(a-b)*max(a[1],b[1]) #BUG!!! Not used in CNA2022 -> Fine

def cost_complex_vector_Euclidean_logscale_directed(a,b):
    a_ = np.log(np.array(a)+1)
    b_ = np.log(np.array(b)+1)    
    return np.linalg.norm(a_-b_)

def cost_complex_vector_Euclidean_logscale_max_directed(a,b):
    a_ = np.log(np.array(a[0])+1)
    b_ = np.log(np.array(b[0])+1)    
    # return np.linalg.norm(a_-b_)*max(a[1],b[1])
    return np.linalg.norm(a_-b_)*max(a[1][0],b[1][0])

def cost_complex_vector_Euclidean_directed(a,b):
    # print (a,b, np.linalg.norm(a-b))
    # a_ = np.log(np.array(a)+1)
    # b_ = np.log(np.array(b)+1)    
    return np.linalg.norm(a-b)

def cost_complex_vector_Euclidean_max_directed(a,b):
    # a_ = np.log(np.array(a[0])+1)
    # b_ = np.log(np.array(b[0])+1)    
    # return np.linalg.norm(a_-b_)*max(a[1],b[1])
    return np.linalg.norm(a[0]-b[0])*max(a[1][0],b[1][0])

def cost_complex_sinusoidalWave_logscale_vector(a,b): # complex numbers NG for fastDTW
    maxA = get_maxA()
    # print (maxA)
    if (a==b).all():
        return 0
    if maxA == 1:
        return 0
    a1 = np.log(np.sqrt(a[0]**2+a[1]**2)+1)
    a1 = (a1-1)/(maxA-1) + 1
    a2 = np.log(np.sqrt(b[0]**2+b[1]**2)+1)
    a2 = (a2-1)/(maxA-1) + 1
    # a1 = np.sqrt(np.log(a[0]+1)**2+np.log(a[1]+1)**2)
    # a2 = np.sqrt(np.log(b[0]+1)**2+np.log(b[1]+1)**2)
    # print ('000 111')
    alpha1 = math.atan2(a[1],a[0])
    alpha2 = math.atan2(b[1],b[0])
    # print ('000 112')
    p1 = np.arctan((a2*np.sin(alpha2)-a1*np.sin(alpha1))/(a1*np.cos(alpha1)-a2*np.cos(alpha2)))
    # print ('000 113')
    return abs(4*a2*np.cos(p1+alpha2)-4*a1*np.cos(p1+alpha1))

def cost_min(a,b):
    ep = 0.5
    m = max(a[0],b[0]) + ep
    mi = min(a[0],b[0]) + ep
    return ((m/mi) - 1) * min(a[1],b[1])

# def preprocess_degreeLists():

#     logging.info("Recovering degreeList from disk...")
#     degreeList = restoreVariableFromDisk('degreeList')

#     logging.info("Creating compactDegreeList...")

#     dList = {}
#     dFrequency = {}
#     for v,layers in degreeList.items():
#         dFrequency[v] = {}
#         for layer,degreeListLayer in layers.items():
#             dFrequency[v][layer] = {}
#             for degree in degreeListLayer:
#                 if(degree not in dFrequency[v][layer]):
#                     dFrequency[v][layer][degree] = 0
#                 dFrequency[v][layer][degree] += 1
#     for v,layers in dFrequency.items():
#         dList[v] = {}
#         for layer,frequencyList in layers.items():
#             list_d = []
#             for degree,freq in frequencyList.items():
#                 list_d.append((degree,freq))
#             list_d.sort(key=lambda x: x[0])
#             dList[v][layer] = np.array(list_d,dtype='float')

#     logging.info("compactDegreeList created!")

#     saveVariableOnDisk(dList,'compactDegreeList')

def logiFunc(x, theta=0.02, L=np.e**4, d=0):
    # L=np.e
    return  2*L/(1+np.exp(-theta*(x-d))) - L

def derivative_variance_lf(x, theta, L=np.e**4):
    
    # print ('deri 1 ', max(x), min(x), theta, len(x))
    e_thetax =np.exp(theta*x)
    # x = x[np.isfinite(e_thetax)] # TODO check the feasibility
    # e_thetax = e_thetax[np.isfinite(e_thetax)] # TODO check the feasibility
    # print ('deri 1 ', max(x), min(x), theta, len(x), len(e_thetax))
    lf = logiFunc(x, theta, L, 0)

    derivativeA = x*L*e_thetax / (1 + e_thetax)**2
    derivativeB = (derivativeA).mean()
    derivativeAll = 2* ((lf - lf.mean())*(derivativeA - derivativeB)).mean()
    # print ('deri 2 ', max(x), min(x), theta)
    return derivativeAll

def variance_lf(x, theta, L=np.e**4):
    lf = logiFunc(x, theta, L, 0)
    return ((lf - lf.mean())**2).mean()

def gradient_decent(input_, eta=0.1, max_iteration=1000, x0=0.00001, L=np.e**4):
    # for i in range(max_iteration):
    #     df = derivative_variance_lf(input_ , x0, L)
    #     if x0 + eta * df <= 0:
    #         x0 = x0 * (1 - random.random())
    #     else:
    #         x0 = x0 + eta * df
    # L = np.e
    # print (max(input_))
    rst = x0
    for i in range(max_iteration):
        if np.isnan(x0) or not np.isfinite(x0):
            print ('GD Nan detect, early stopped!')
            return x0
        df = derivative_variance_lf(input_ , x0, L)
        # print ('logi= {} ; x0 = {}'.format(logiFunc(input_.mean(), x0, L), x0))
        if x0 + eta * df <= 0:
            # print ('Minus True')
            x0 = x0 * (1 - random.random())
            continue
        if logiFunc(input_.mean(), x0 + eta * df, L) > 0.99*L:
            # print ('Over True')
            eta = eta * 0.1
            continue
        
        x0 = x0 + eta * df
    return x0

def verifyDegrees(degrees,degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def get_vertices(v,degree_v,degrees,a_vertices):
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        if('before' not in degrees[degree_v]):
            degree_b = -1
        else:
            degree_b = degrees[degree_v]['before']
        if('after' not in degrees[degree_v]):
            degree_a = -1
        else:
            degree_a = degrees[degree_v]['after']
        if(degree_b == -1 and degree_a == -1):
            raise StopIteration
        degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

        while True:
            for v2 in degrees[degree_now]['vertices']:
                if(v != v2):
                    vertices.append(v2)
                    c_v += 1
                    if(c_v > a_vertices_selected):
                        raise StopIteration

            if(degree_now == degree_b):
                if('before' not in degrees[degree_b]):
                    degree_b = -1
                else:
                    degree_b = degrees[degree_b]['before']
            else:
                if('after' not in degrees[degree_a]):
                    degree_a = -1
                else:
                    degree_a = degrees[degree_a]['after']

            if(degree_b == -1 and degree_a == -1):
                raise StopIteration

            degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)

def verifyDegrees_matrix(degrees,degree_v_root, degree_list_p, degree_list_n,degree_left_bottom, degree_right_top ):#,degree_a_p,degree_b_p,degree_a_n,degree_b_n):
    # print (degree_v_root, degree_left_bottom, degree_right_top)
    pass_flag_p = False
    pass_flag_n = False
    if (degree_list_p.index(degree_left_bottom.real) == 0) and (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)) and (degree_list_n.index(degree_left_bottom.imag) == 0) and (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        raise StopIteration
    if (degree_list_p.index(degree_left_bottom.real) == 0) and (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)):
        pass_flag_p = True
        pass
    elif (degree_list_p.index(degree_left_bottom.real) == 0):
        degree_now_p = degree_list_p[degree_list_p.index(degree_right_top.real)+1]
    elif (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)):
        degree_now_p = degree_list_p[degree_list_p.index(degree_left_bottom.real)-1]
    elif (abs(degree_list_p[degree_list_p.index(degree_right_top.real)+1] - degree_v_root.real) < abs(degree_list_p[degree_list_p.index(degree_left_bottom.real)-1] - degree_v_root.real)):
        degree_now_p = degree_list_p[degree_list_p.index(degree_right_top.real)+1]
    else:
        degree_now_p = degree_list_p[degree_list_p.index(degree_left_bottom.real)-1]
    
    if (degree_list_n.index(degree_left_bottom.imag) == 0) and (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        pass_flag_n = True
        pass
    elif (degree_list_n.index(degree_left_bottom.imag) == 0):
        degree_now_n = degree_list_n[degree_list_n.index(degree_right_top.imag)+1]
    elif (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        degree_now_n = degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1]
    elif (abs(degree_list_n[degree_list_n.index(degree_right_top.imag)+1] - degree_v_root.imag) < abs(degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1] - degree_v_root.imag)):
        degree_now_n = degree_list_n[degree_list_n.index(degree_right_top.imag)+1]
    else:
        degree_now_n = degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1]
    # print (degree_now_n, degree_now_p)
    # if(degree_b_p.real == -1):
    #     degree_now_p = degree_a_p.real
    # elif(degree_a_p.real == -1):
    #     degree_now_p = degree_b_p.real
    # elif(abs(degree_b_p.real - degree_v_root.real) < abs(degree_a_p.real - degree_v_root.real)):
    #     degree_now_p = degree_b_p.real
    # else:
    #     degree_now_p = degree_a_p.real
    

    # if(degree_b_n.imag == -1):
    #     degree_now_n = degree_a_n.imag
    # elif(degree_a_n.imag == -1):
    #     degree_now_n = degree_b_n.imag
    # elif(abs(degree_b_n.imag - degree_v_root.imag) < abs(degree_a_n.imag - degree_v_root.imag)):
    #     degree_now_n = degree_b_n.imag
    # else:
    #     degree_now_n = degree_a_n.imag
    
    if ((not pass_flag_n) and (not pass_flag_p) and (abs(degree_now_n - degree_v_root.imag) < abs(degree_now_p - degree_v_root.real))) or (pass_flag_p):
        # print ('N')
        assert( (degree_now_n >= degree_right_top.imag) or (degree_now_n <= degree_left_bottom.imag) ), 'Search ERROR in verifyDegrees_matrix'
        degree_now = [complex(x, degree_now_n) for x in degree_list_p[degree_list_p.index(degree_left_bottom.real): degree_list_p.index(degree_right_top.real)+1 ]]
        if (degree_now_n > degree_right_top.imag):
            degree_right_top = complex(degree_right_top.real, degree_now_n)
        else:
            degree_left_bottom = complex(degree_left_bottom.real, degree_now_n)
    else:
        # print ('P',degree_right_top)
        assert( (degree_now_p >= degree_right_top.real) or (degree_now_p <= degree_left_bottom.real) ), 'Search ERROR in verifyDegrees_matrix'
        degree_now = [complex(degree_now_p, x) for x in degree_list_n[degree_list_n.index(degree_left_bottom.imag): degree_list_n.index(degree_right_top.imag)+1 ]]
        if (degree_now_p > degree_right_top.real):
            degree_right_top = complex(degree_now_p, degree_right_top.imag)
            # print ('P',degree_right_top, )
        else:
            degree_left_bottom = complex(degree_now_p, degree_left_bottom.imag)
    degree_now.sort(key=lambda x:abs(x - degree_v_root))

    # print (degree_now, degree_left_bottom, degree_right_top)
    return degree_now, degree_left_bottom, degree_right_top

def verifyDegrees_4axis_directed(degrees,degree_v_root, degree_list_p, degree_list_n,degree_left_bottom, degree_right_top ):#,degree_a_p,degree_b_p,degree_a_n,degree_b_n):
    # print (degree_v_root, degree_left_bottom, degree_right_top)
    pass_flag_p = False
    pass_flag_n = False
    if (degree_list_p.index(degree_left_bottom.real) == 0) and (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)) and (degree_list_n.index(degree_left_bottom.imag) == 0) and (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        raise StopIteration
    if (degree_list_p.index(degree_left_bottom.real) == 0) and (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)):
        pass_flag_p = True
        pass
    elif (degree_list_p.index(degree_left_bottom.real) == 0):
        degree_now_p = degree_list_p[degree_list_p.index(degree_right_top.real)+1]
    elif (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)):
        degree_now_p = degree_list_p[degree_list_p.index(degree_left_bottom.real)-1]
    elif (abs(degree_list_p[degree_list_p.index(degree_right_top.real)+1] - degree_v_root.real) < abs(degree_list_p[degree_list_p.index(degree_left_bottom.real)-1] - degree_v_root.real)):
        degree_now_p = degree_list_p[degree_list_p.index(degree_right_top.real)+1]
    else:
        degree_now_p = degree_list_p[degree_list_p.index(degree_left_bottom.real)-1]
    
    if (degree_list_n.index(degree_left_bottom.imag) == 0) and (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        pass_flag_n = True
        pass
    elif (degree_list_n.index(degree_left_bottom.imag) == 0):
        degree_now_n = degree_list_n[degree_list_n.index(degree_right_top.imag)+1]
    elif (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        degree_now_n = degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1]
    elif (abs(degree_list_n[degree_list_n.index(degree_right_top.imag)+1] - degree_v_root.imag) < abs(degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1] - degree_v_root.imag)):
        degree_now_n = degree_list_n[degree_list_n.index(degree_right_top.imag)+1]
    else:
        degree_now_n = degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1]
    # print (degree_now_n, degree_now_p)
    # if(degree_b_p.real == -1):
    #     degree_now_p = degree_a_p.real
    # elif(degree_a_p.real == -1):
    #     degree_now_p = degree_b_p.real
    # elif(abs(degree_b_p.real - degree_v_root.real) < abs(degree_a_p.real - degree_v_root.real)):
    #     degree_now_p = degree_b_p.real
    # else:
    #     degree_now_p = degree_a_p.real
    

    # if(degree_b_n.imag == -1):
    #     degree_now_n = degree_a_n.imag
    # elif(degree_a_n.imag == -1):
    #     degree_now_n = degree_b_n.imag
    # elif(abs(degree_b_n.imag - degree_v_root.imag) < abs(degree_a_n.imag - degree_v_root.imag)):
    #     degree_now_n = degree_b_n.imag
    # else:
    #     degree_now_n = degree_a_n.imag
    
    if ((not pass_flag_n) and (not pass_flag_p) and (abs(degree_now_n - degree_v_root.imag) < abs(degree_now_p - degree_v_root.real))) or (pass_flag_p):
        # print ('N')
        assert( (degree_now_n >= degree_right_top.imag) or (degree_now_n <= degree_left_bottom.imag) ), 'Search ERROR in verifyDegrees_matrix'
        degree_now = [complex(x, degree_now_n) for x in degree_list_p[degree_list_p.index(degree_left_bottom.real): degree_list_p.index(degree_right_top.real)+1 ]]
        if (degree_now_n > degree_right_top.imag):
            degree_right_top = complex(degree_right_top.real, degree_now_n)
        else:
            degree_left_bottom = complex(degree_left_bottom.real, degree_now_n)
    else:
        # print ('P',degree_right_top)
        assert( (degree_now_p >= degree_right_top.real) or (degree_now_p <= degree_left_bottom.real) ), 'Search ERROR in verifyDegrees_matrix'
        degree_now = [complex(degree_now_p, x) for x in degree_list_n[degree_list_n.index(degree_left_bottom.imag): degree_list_n.index(degree_right_top.imag)+1 ]]
        if (degree_now_p > degree_right_top.real):
            degree_right_top = complex(degree_now_p, degree_right_top.imag)
            # print ('P',degree_right_top, )
        else:
            degree_left_bottom = complex(degree_now_p, degree_left_bottom.imag)
    degree_now.sort(key=lambda x:abs(x - degree_v_root))

    # print (degree_now, degree_left_bottom, degree_right_top)
    return degree_now, degree_left_bottom, degree_right_top

def get_vertices_matrix(v,degree_v,degrees,a_vertices,degrees_sorted_n,degrees_sorted_p):
    '''
    degree_v: v's degree+ and v's degree- complex style
    degrees: matrix style
    a_vertices: # nodes
    '''
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]['vertices']:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration

        # if('before_p' not in degrees[degree_v]):
        #     degree_b_p = -1
        # else:
        #     degree_b_p = degrees[degree_v]['before_p']
        # if('after_p' not in degrees[degree_v]):
        #     degree_a_p = -1
        # else:
        #     degree_a_p = degrees[degree_v]['after_p']

        # if('before_n' not in degrees[degree_v]):
        #     degree_b_n = -1
        # else:
        #     degree_b_n = degrees[degree_v]['before_n']
        # if('after_n' not in degrees[degree_v]):
        #     degree_a_n = -1
        # else:
        #     degree_a_n = degrees[degree_v]['after_n']

        # if(degree_b_p == -1 and degree_a_p == -1 and degree_b_n == -1 and degree_a_n == -1):
        #     raise StopIteration
        degree_left_bottom = degree_v
        degree_right_top = degree_v
        degree_now_list, degree_left_bottom, degree_right_top = verifyDegrees_matrix(degrees,degree_v, degrees_sorted_p, degrees_sorted_n,degree_left_bottom, degree_right_top )#,degree_a_p,degree_b_p,degree_a_n,degree_b_n)

        while True:
            # print (v, c_v, a_vertices_selected, degree_now_list, degree_left_bottom, degree_right_top)
            for degree_now in degree_now_list:
                for v2 in degrees[degree_now]['vertices']:
                    if(v != v2):
                        vertices.append(v2)
                        c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
            degree_now_list, degree_left_bottom, degree_right_top = verifyDegrees_matrix(degrees,degree_v, degrees_sorted_p, degrees_sorted_n,degree_left_bottom, degree_right_top )#,degree_a_p,degree_b_p,degree_a_n,degree_b_n)

            # if(degree_now == degree_b):
            #     if('before' not in degrees[degree_b]):
            #         degree_b = -1
            #     else:
            #         degree_b = degrees[degree_b]['before']
            # else:
            #     if('after' not in degrees[degree_a]):
            #         degree_a = -1
            #     else:
            #         degree_a = degrees[degree_a]['after']

            # if(degree_b == -1 and degree_a == -1):
            #     raise StopIteration

            # degree_now = verifyDegrees(degrees,degree_v,degree_a,degree_b)

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)

def get_vertices_4axis_directed(v,degree_v,degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po): # 
    '''
    degree_v: v's degree+i, v's degree-i, v's degree+o, v's degree-o quaternion style
    degrees: 4D matrix style
    a_vertices: # nodes
    '''
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration
                a_vertices_selected -= c_v
                a_vertices_selected = a_vertices_selected//4 + 1
        minDegree = degree_v[0]
        maxDegree = degree_v[0]
        centerDegree = degree_v[0]
        c_v = 0  
        try:
            while True:
                if (degrees_sorted_pi.index(minDegree) == 0) and (degrees_sorted_pi.index(maxDegree) < len(degrees_sorted_pi) - 1) :
                    maxDegree = degrees_sorted_pi[degrees_sorted_pi.index(maxDegree)+1]
                    for v2 in degrees_pi[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_pi.index(minDegree) > 0) and (degrees_sorted_pi.index(maxDegree) == len(degrees_sorted_pi) - 1) :
                    minDegree = degrees_sorted_pi[degrees_sorted_pi.index(minDegree)-1]
                    for v2 in degrees_pi[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_pi.index(minDegree) == 0) and (degrees_sorted_pi.index(maxDegree) == len(degrees_sorted_pi) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_pi[degrees_sorted_pi.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_pi[degrees_sorted_pi.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_pi[degrees_sorted_pi.index(minDegree)-1]
                    for v2 in degrees_pi[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_pi[degrees_sorted_pi.index(maxDegree)+1]
                    for v2 in degrees_pi[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            minDegree = degree_v[1]
            maxDegree = degree_v[1]
            centerDegree = degree_v[1]
            c_v = 0  
        try:
            while True:
                if (degrees_sorted_ni.index(minDegree) == 0) and (degrees_sorted_ni.index(maxDegree) < len(degrees_sorted_ni) - 1) :
                    maxDegree = degrees_sorted_ni[degrees_sorted_ni.index(maxDegree)+1]
                    for v2 in degrees_ni[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_ni.index(minDegree) > 0) and (degrees_sorted_ni.index(maxDegree) == len(degrees_sorted_ni) - 1) :
                    # print (minDegree, maxDegree, degrees_ni, degrees_sorted_ni)
                    minDegree = degrees_sorted_ni[degrees_sorted_ni.index(minDegree)-1]
                    for v2 in degrees_ni[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_ni.index(minDegree) == 0) and (degrees_sorted_ni.index(maxDegree) == len(degrees_sorted_ni) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_ni[degrees_sorted_ni.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_ni[degrees_sorted_ni.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_ni[degrees_sorted_ni.index(minDegree)-1]
                    for v2 in degrees_ni[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_ni[degrees_sorted_ni.index(maxDegree)+1]
                    for v2 in degrees_ni[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            minDegree = degree_v[2]
            maxDegree = degree_v[2]
            centerDegree = degree_v[2]
            c_v = 0  
        try:
            while True:
                if (degrees_sorted_po.index(minDegree) == 0) and (degrees_sorted_po.index(maxDegree) < len(degrees_sorted_po) - 1) :
                    maxDegree = degrees_sorted_po[degrees_sorted_po.index(maxDegree)+1]
                    for v2 in degrees_po[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_po.index(minDegree) > 0) and (degrees_sorted_po.index(maxDegree) == len(degrees_sorted_po) - 1) :
                    minDegree = degrees_sorted_po[degrees_sorted_po.index(minDegree)-1]
                    for v2 in degrees_po[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_po.index(minDegree) == 0) and (degrees_sorted_po.index(maxDegree) == len(degrees_sorted_po) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_po[degrees_sorted_po.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_po[degrees_sorted_po.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_po[degrees_sorted_po.index(minDegree)-1]
                    for v2 in degrees_po[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_po[degrees_sorted_po.index(maxDegree)+1]
                    for v2 in degrees_po[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            minDegree = degree_v[3]
            maxDegree = degree_v[3]
            centerDegree = degree_v[3]
            c_v = 0  
        try:
            while True:
                if (degrees_sorted_no.index(minDegree) == 0) and (degrees_sorted_no.index(maxDegree) < len(degrees_sorted_no) - 1) :
                    maxDegree = degrees_sorted_no[degrees_sorted_no.index(maxDegree)+1]
                    for v2 in degrees_no[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_no.index(minDegree) > 0) and (degrees_sorted_no.index(maxDegree) == len(degrees_sorted_no) - 1) :
                    minDegree = degrees_sorted_no[degrees_sorted_no.index(minDegree)-1]
                    for v2 in degrees_no[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_no.index(minDegree) == 0) and (degrees_sorted_no.index(maxDegree) == len(degrees_sorted_no) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_no[degrees_sorted_no.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_no[degrees_sorted_no.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_no[degrees_sorted_no.index(minDegree)-1]
                    for v2 in degrees_no[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_no[degrees_sorted_no.index(maxDegree)+1]
                    for v2 in degrees_no[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            pass

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)


def get_vertices_3axis_hyper(v,degree_v,a_vertices,degrees_sorted_len,degrees_sorted_max,degrees_sorted_sum,degrees_len,degrees_max,degrees_sum): # 
    '''
    degree_v: v's len, sum, max
    degrees: 3D matrix style
    a_vertices: # nodes
    '''
    a_vertices_selected = int(math.log(a_vertices,2) + 1)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        minDegree = degree_v[0]
        maxDegree = degree_v[0]
        centerDegree = degree_v[0]
        vertices += degrees_len[minDegree]
        c_v = 0
        try:
            while True:
                # tmp_list = list(mapping_degree2AD_pi.keys())
                if (degrees_sorted_len.index(minDegree) == 0) and (degrees_sorted_len.index(maxDegree) < len(degrees_sorted_len) - 1) :
                    maxDegree = degrees_sorted_len[degrees_sorted_len.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (degrees_sorted_len.index(minDegree) > 0) and (degrees_sorted_len.index(maxDegree) == len(degrees_sorted_len) - 1) :
                    minDegree = degrees_sorted_len[degrees_sorted_len.index(minDegree)-1]
                    tmp_index = minDegree
                elif (degrees_sorted_len.index(minDegree) == 0) and (degrees_sorted_len.index(maxDegree) == len(degrees_sorted_len) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_len[degrees_sorted_len.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_len[degrees_sorted_len.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_len[degrees_sorted_len.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = degrees_sorted_len[degrees_sorted_len.index(maxDegree)+1]
                    tmp_index = maxDegree

                if (c_v + len(degrees_len[tmp_index]) <= a_vertices_selected):
                    vertices += degrees_len[tmp_index]
                    c_v += len(degrees_len[tmp_index])
                else:
                    vertices += random.sample(degrees_len[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[1]
            maxDegree = degree_v[1]
            centerDegree = degree_v[1]
            vertices += degrees_max[minDegree]
            c_v = 0
        try:
            while True:
                if (degrees_sorted_max.index(minDegree) == 0) and (degrees_sorted_max.index(maxDegree) < len(degrees_sorted_max) - 1) :
                    maxDegree = degrees_sorted_max[degrees_sorted_max.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (degrees_sorted_max.index(minDegree) > 0) and (degrees_sorted_max.index(maxDegree) == len(degrees_sorted_max) - 1) :
                    minDegree = degrees_sorted_max[degrees_sorted_max.index(minDegree)-1]
                    tmp_index = minDegree
                elif (degrees_sorted_max.index(minDegree) == 0) and (degrees_sorted_max.index(maxDegree) == len(degrees_sorted_max) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_max[degrees_sorted_max.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_max[degrees_sorted_max.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_max[degrees_sorted_max.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = degrees_sorted_max[degrees_sorted_max.index(maxDegree)+1]
                    tmp_index = maxDegree

                if (c_v + len(degrees_max[tmp_index]) <= a_vertices_selected):
                    vertices += degrees_max[tmp_index]
                    c_v += len(degrees_max[tmp_index])
                else:
                    vertices += random.sample(degrees_max[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[2]
            maxDegree = degree_v[2]
            centerDegree = degree_v[2]
            vertices += degrees_sum[minDegree]
            c_v = 0  
        try:
            while True:
                if (degrees_sorted_sum.index(minDegree) == 0) and (degrees_sorted_sum.index(maxDegree) < len(degrees_sorted_sum) - 1) :
                    maxDegree = degrees_sorted_sum[degrees_sorted_sum.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (degrees_sorted_sum.index(minDegree) > 0) and (degrees_sorted_sum.index(maxDegree) == len(degrees_sorted_sum) - 1) :
                    minDegree = degrees_sorted_sum[degrees_sorted_sum.index(minDegree)-1]
                    tmp_index = minDegree
                elif (degrees_sorted_sum.index(minDegree) == 0) and (degrees_sorted_sum.index(maxDegree) == len(degrees_sorted_sum) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_sum[degrees_sorted_sum.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_sum[degrees_sorted_sum.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_sum[degrees_sorted_sum.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = degrees_sorted_sum[degrees_sorted_sum.index(maxDegree)+1]
                    tmp_index = maxDegree
                
                if (c_v + len(degrees_sum[tmp_index]) <= a_vertices_selected):
                    vertices += degrees_sum[tmp_index]
                    c_v += len(degrees_sum[tmp_index])
                else:
                    vertices += random.sample(degrees_sum[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            pass

    except StopIteration:
        # print (minDegree, maxDegree, centerDegree, vertices, degrees_sorted_len, degrees_len, tmp_index)

        return list(set(vertices) - set([v]))

    return list(set(vertices) - set([v]))

def get_vertices_4axis_directed_shrink(v,degree_v,degrees,a_vertices,mapping_degree2AD_ni,mapping_degree2AD_pi,mapping_degree2AD_no,mapping_degree2AD_po): # 
    '''
    degree_v: v's degree+i, v's degree-i, v's degree+o, v's degree-o quaternion style
    degrees: 4D matrix style
    a_vertices: # nodes
    !!! Note: var name different with outside!!!
    '''
    a_vertices_selected = int(2 * math.log(a_vertices,2) + 1)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()
    try:
        c_v = 0  
        if (c_v + len(degrees[degree_v]) <= a_vertices_selected):
            vertices += degrees[degree_v]
            c_v = len(degrees[degree_v])
        else:
            vertices += random.sample(degrees[degree_v], a_vertices_selected - c_v)
            c_v = a_vertices_selected
        #     raise StopIteration
        # if(c_v >= a_vertices_selected):
        #     raise StopIteration
        a_vertices_selected = 2*a_vertices_selected - c_v
        a_vertices_selected = int(a_vertices_selected//4 + 1)
        minDegree = degree_v[0]
        maxDegree = degree_v[0]
        centerDegree = degree_v[0]
        c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_pi.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_pi) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_pi) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_pi) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_pi[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_pi[tmp_index]
                    c_v += len(mapping_degree2AD_pi[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_pi[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[1]
            maxDegree = degree_v[1]
            centerDegree = degree_v[1]
            c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_ni.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_ni) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_ni) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_ni) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_ni[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_ni[tmp_index]
                    c_v += len(mapping_degree2AD_ni[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_ni[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[2]
            maxDegree = degree_v[2]
            centerDegree = degree_v[2]
            c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_po.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_po) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_po) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_po) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_po[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_po[tmp_index]
                    c_v += len(mapping_degree2AD_po[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_po[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[3]
            maxDegree = degree_v[3]
            centerDegree = degree_v[3]
            c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_no.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_no) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_no) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_no) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_no[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_no[tmp_index]
                    c_v += len(mapping_degree2AD_no[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_no[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            pass

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)


def splitDegreeList(part,c,G,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')

    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(G)

    for v in c:
        nbs = get_vertices(v,len(G[v]),degrees,a_vertices)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))


def splitDegreeList_complex(part,c,Gp,Gn,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')
    degrees_sorted_n = restoreVariableFromDisk('degrees_vector_negativeList')
    degrees_sorted_p = restoreVariableFromDisk('degrees_vector_positiveList')

    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(list(set(list(Gp.keys()) + list(Gn.keys()))))
    for v in c:
        nbs = get_vertices_matrix(v,complex(len(Gp[v]), len(Gn[v])),degrees,a_vertices,degrees_sorted_n,degrees_sorted_p) # TODO
        # print (v, nbs)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))

def splitDegreeList_complex_directed(part,c,Gpi,Gni,Gpo,Gno,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')
    degrees_sorted_ni = restoreVariableFromDisk('degrees_vector_negativeInList')
    degrees_sorted_pi = restoreVariableFromDisk('degrees_vector_positiveInList')
    degrees_sorted_no = restoreVariableFromDisk('degrees_vector_negativeOutList')
    degrees_sorted_po = restoreVariableFromDisk('degrees_vector_positiveOutList')
    degrees_ni = restoreVariableFromDisk('degrees_vector_node_negativeInList')
    degrees_pi = restoreVariableFromDisk('degrees_vector_node_positiveInList')
    degrees_no = restoreVariableFromDisk('degrees_vector_node_negativeOutList')
    degrees_po = restoreVariableFromDisk('degrees_vector_node_positiveOutList')
    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(list(set(list(Gpi.keys()) + list(Gni.keys()) + list(Gpo.keys()) + list(Gno.keys()))))
    for v in c:
        # nbs = get_vertices_4axis_directed(v,np.quaternion(len(Gpi[v]), len(Gni[v]), len(Gpo[v]), len(Gno[v])),degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po) # 
        nbs = get_vertices_4axis_directed(v,(len(Gpi[v]), len(Gni[v]), len(Gpo[v]), len(Gno[v])),degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po) # 
        # print (v, nbs)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))

def splitDegreeList_hyper(part,c,a_vertices,G,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    # degrees = restoreVariableFromDisk('degrees_vector')
    degrees_sorted_len = restoreVariableFromDisk('degrees_vector_lenList')
    degrees_sorted_max = restoreVariableFromDisk('degrees_vector_maxList')
    degrees_sorted_sum = restoreVariableFromDisk('degrees_vector_sumList')
    degrees_len = restoreVariableFromDisk('degrees_vector_node_lenList')
    degrees_max = restoreVariableFromDisk('degrees_vector_node_maxList')
    degrees_sum = restoreVariableFromDisk('degrees_vector_node_sumList')
    degreeListsSelected = {}
    vertices = {}
    for v in c:
        # nbs = get_vertices_4axis_directed(v,np.quaternion(len(Gpi[v]), len(Gni[v]), len(Gpo[v]), len(Gno[v])),degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po) # 
        nbs = get_vertices_3axis_hyper(v,(len(G[v]), max([len(x)+1 for x in G[v]]), sum([len(x)+1 for x in G[v]])),len(a_vertices),degrees_sorted_len,degrees_sorted_max,degrees_sorted_sum,degrees_len,degrees_max,degrees_sum) # 
        # print (v, nbs)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))


def splitDegreeList_complex_directed_shrink(part,c,Gpi,Gni,Gpo,Gno,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')
    mapping_degree2AD_ni = restoreVariableFromDisk('degrees_vector_negativeInList')
    mapping_degree2AD_pi = restoreVariableFromDisk('degrees_vector_positiveInList')
    mapping_degree2AD_no = restoreVariableFromDisk('degrees_vector_negativeOutList')
    mapping_degree2AD_po = restoreVariableFromDisk('degrees_vector_positiveOutList')
    mapping_AD2node_ni = restoreVariableFromDisk('degrees_vector_node_negativeInList')
    mapping_AD2node_pi = restoreVariableFromDisk('degrees_vector_node_positiveInList')
    mapping_AD2node_no = restoreVariableFromDisk('degrees_vector_node_negativeOutList')
    mapping_AD2node_po = restoreVariableFromDisk('degrees_vector_node_positiveOutList')
    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(list(set(list(Gpi.keys()) + list(Gni.keys()) + list(Gpo.keys()) + list(Gno.keys()))))
    for v in c:
        # nbs = get_vertices_4axis_directed(v,np.quaternion(len(Gpi[v]), len(Gni[v]), len(Gpo[v]), len(Gno[v])),degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po) # 
        nbs = get_vertices_4axis_directed_shrink(v,(mapping_degree2AD_pi[len(Gpi[v])], mapping_degree2AD_ni[len(Gni[v])], mapping_degree2AD_po[len(Gpo[v])], mapping_degree2AD_no[len(Gno[v])]),degrees,a_vertices,mapping_AD2node_ni,mapping_AD2node_pi,mapping_AD2node_no,mapping_AD2node_po)
                
        # print (v, nbs)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))

def splitDegreeList_signed(part,c,G,compactDegree,type):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk(type+'compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk(type+'degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk(type+'degrees_vector')

    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(G)

    for v in c:
        nbs = get_vertices(v,len(G[v]),degrees,a_vertices)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,type+'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,type+'split-degreeList-'+str(part))


def calc_distances(part, compactDegree = False):

    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

    distances = {}

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1,nbs in vertices.items():
        lists_v1 = degreeList[v1]

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# start dist_minus calculation

#  end  dist_minus calculation
                distances[v1,v2][layer] = dist

            t11 = time()
            logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))


    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_complex(part, compactDegree = False, scale_free = True):
    # scale_free = False
    # scale_free = True
    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

    distances = {}

    if compactDegree:
        dist_func = cost_complex_vector_Euclidean_logscale_max if scale_free else cost_complex_vector_Euclidean_max
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[((x[0].real, x[0].imag), x[1].real) for x in value_]}) # TODO
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[((x[0].real, x[0].imag), x[1].real) for x in value_]}) # TODO

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    # distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e
    else:
        dist_func = cost_complex_vector_Euclidean_logscale if scale_free else cost_complex_vector_Euclidean
        # dist_func = cost_complex_vector_2area_logscale
        # dist_func = cost_complex_sinusoidalWave_logscale_vector
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[(x.real, x.imag) for x in value_]})
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[(x.real, x.imag) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    # distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e

#     if compactDegree:
#         dist_func = cost_max
#     else:
#         dist_func = cost

#     for v1,nbs in vertices.items():
#         lists_v1 = degreeList[v1]

#         for v2 in nbs:
#             t00 = time()
#             lists_v2 = degreeList[v2]

#             max_layer = min(len(lists_v1),len(lists_v2))
#             distances[v1,v2] = {}

#             for layer in range(0,max_layer):
#                 dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# # start dist_minus calculation

# #  end  dist_minus calculation
#                 distances[v1,v2][layer] = dist

            # t11 = time()
            # logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))


    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_complex_directed(part, compactDegree = False, scale_free = True):
    # scale_free = False
    # scale_free = True
    print (part)
    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))
    print (part)

    distances = {}

    if compactDegree:
        dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z),  (x[1], 0, 0, 0)) for x in value_]})
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    # distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e
    else:
        dist_func = cost_complex_vector_Euclidean_logscale_directed if scale_free else cost_complex_vector_Euclidean_directed
        # dist_func = cost_complex_vector_2area_logscale
        # dist_func = cost_complex_sinusoidalWave_logscale_vector
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    # distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e

#     if compactDegree:
#         dist_func = cost_max
#     else:
#         dist_func = cost

#     for v1,nbs in vertices.items():
#         lists_v1 = degreeList[v1]

#         for v2 in nbs:
#             t00 = time()
#             lists_v2 = degreeList[v2]

#             max_layer = min(len(lists_v1),len(lists_v2))
#             distances[v1,v2] = {}

#             for layer in range(0,max_layer):
#                 dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# # start dist_minus calculation

# #  end  dist_minus calculation
#                 distances[v1,v2][layer] = dist

            # t11 = time()
            # logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))


    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

# def calc_distances_hyper(part, compactDegree = False, scale_free = True):

#     vertices = restoreVariableFromDisk('split-vertices-'+str(part))
#     degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

#     distances = {}
#     dtw_distance_layer1 = {}

#     def cost_hyper_base_dtw1(a,b):
#         a_ = sorted(cost_hyper_base_dtw1_in_a[int(a)], reverse=True)
#         b_ = sorted(cost_hyper_base_dtw1_in_b[int(b)], reverse=True)
#         size = max(max(a_,b_)) - min(min(a_,b_))
#         min_digit = min(len(a_), len(b_))
#         a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min_digit] ), reverse=True)
#         if (str((a_,b_)) not in dtw_distance_layer1.keys()):
#             dist_inner, tmp_inner = fastdtw(list(zip(a_, np.array(range(len(a_)))/(len(a_)-1)*size)) if len(a_)>1 else list(zip(a_, [0])), list(zip(b_, np.array(range(len(b_)))/(len(b_)-1)*size)) if len(b_)>1 else list(zip(b_, [0])), radius=1,dist=cost2d)
#             dtw_distance_layer1[str((a_,b_))] = dist_inner #+ cost(len(a_), len(b_)) # Here!
#         return dtw_distance_layer1[str((a_,b_))]

#     if compactDegree:
#         pass
#     #     dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
#     #     for v1 in vertices:
#     #         lists_v1 = degreeList[v1]
#     #         lists_v1_new = {}
#     #         for key_,value_ in lists_v1.items():
#     #             lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})
#     #         for v2 in list_vertices[cont]:
#     #             lists_v2 = degreeList[v2]
#     #             lists_v2_new = {}
#     #             for key_,value_ in lists_v2.items():
#     #                 lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

#     #             max_layer = min(len(lists_v1),len(lists_v2))
#     #             distances[v1,v2] = {}
#     #             for layer in range(0,max_layer):
#     #                 #t0 = time()
                    
#     #                 # print ('101 111')
#     #                 # print (layer,lists_v1_new[layer],lists_v2_new[layer])
#     #                 dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
#     #                 # print ('100 111')
#     #                 # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
#     # # start dist_minus calculation

#     # #  end  dist_minus calculation
#     #                 #t1 = time()
#     #                 #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
#     #                 distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
#     #                 # print (v1,v2, layer, np.exp(dist))
#     #                 # distances[v1,v2][layer] = dist 
#     #                 # distances[v1,v2][layer] = np.exp(dist)**np.e


#     #         cont += 1

#     else:
#         dist_func = cost_hyper_base_dtw1
#         # dist_func = cost_complex_vector_2area_logscale
#         # dist_func = cost_complex_sinusoidalWave_logscale_vector
#         for v1,nbs in vertices.items():
#             lists_v1 = degreeList[v1]
#             # lists_v1_new = {}
#             # for key_,value_ in lists_v1.items():
#             #     lists_v1_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})
#             for v2 in nbs:
#                 lists_v2 = degreeList[v2]
#                 # lists_v2_new = {}
#                 # for key_,value_ in lists_v2.items():
#                 #     lists_v2_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})

#                 max_layer = min(len(lists_v1),len(lists_v2))
#                 distances[v1,v2] = {}
#                 for layer in range(0,max_layer):
#                     #t0 = time()
                    
#                     # print ('101 111')
#                     # print (layer,lists_v1_new[layer],lists_v2_new[layer])
#                     # dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
#                     # print (lists_v1[layer],lists_v2[layer])
#                     cost_hyper_base_dtw1_in_a = lists_v1[layer]
#                     cost_hyper_base_dtw1_in_b = lists_v2[layer]
#                     dist, path = fastdtw(range(len(lists_v1[layer])),range(len(lists_v2[layer])),radius=1,dist=dist_func)
#                     # if layer == 0:
#                     #     dist += cost(len(lists_v1[layer][0]), len(lists_v2[layer][0]))
#                     # else:
#                     #     dist += cost(len(lists_v1[layer]), len(lists_v2[layer]))
#                     # print ('** ',v1,v2,layer,dist,len(lists_v1[layer]),len(lists_v1[layer][0]),cost(len(lists_v1[layer]), len(lists_v2[layer])),cost_hyper_base_dtw1_in_a,cost_hyper_base_dtw1_in_b)
#                     # print (layer, len(lists_v1[layer]), len(lists_v2[layer]), dist, lists_v1[layer], lists_v2[layer])
#                     # distances [v1,v2][layer] = np.exp(dist)
#                     distances[v1,v2][layer] = np.exp(dist)
#     preprocess_consolides_distances(distances)
#     saveVariableOnDisk(distances,'distances-'+str(part))
#     return
def prob_rev(in_):
    tmp = 1/(in_.max() -in_ + 1)
    return (tmp / tmp.sum())
def prob_exp(in_):
    in_ = np.array(in_)
    tmp = np.exp(in_ - in_.max())
    return (tmp / tmp.sum())

def collapse_prob(base): # rev
    tup = np.array(sorted(dict(collections.Counter(base)).items(), reverse=True)) # TODO
    tmp = 1/(tup[:,0].max() -tup[:,0] +1)
    # prob = tmp/(tmp*tup[:,1]).sum() # all connected
    prob = tmp/tup[:,1] # only self item
    return np.concatenate([tup, prob.reshape((-1,1))], 1)

def collapsed_prob(base): # rev
    # tup = np.array(sorted(dict(collections.Counter(base)).items(), reverse=True)) # TODO
    tmp = 1/(base[:,0].max() -base[:,0] +1)
    # prob = tmp/(tmp*tup[:,1]).sum() # all connected
    prob = tmp/base[:,1] # only self item
    return np.concatenate([base, prob.reshape((-1,1))], 1)

def cost2d_s2v(a_,b_, ep=0.01):
    # print (a_,b_)
    a = a_[0]
    b = b_[0]
    # ep = 0
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return np.sqrt(((m/mi) - 1)**2 + (a_[1]-b_[1])**2)
def cost2d_euclid(a_,b_, ep=0.01):
    return np.sqrt(((max(a_[0] , b_[0])+ep)/(min(a_[0] , b_[0]) + ep) - 1)**2 + (a_[1]-b_[1])**2)
def cost2d_euclid2(a_,b_): #scale first element
    # return np.exp((1 - (min(a_[0] , b_[0])+ep)/(max(a_[0] , b_[0]) + ep))*(abs(a_[1]-b_[1])+1)) # bad
    return np.exp(np.sqrt((1 - min(a_[0] , b_[0])/max(a_[0] , b_[0]))**2 + (a_[1]-b_[1])**2))-1 # good

def cost2d_euclid2_opt1(a_,b_): #scale first element
    # return np.sqrt(a_[1]*b_[1])*(np.exp(np.sqrt((1 - (min(a_[0] , b_[0]))/(max(a_[0] , b_[0])))**2 + (a_[2]-b_[2])**2))-1)
    return max(a_[1],b_[1])*(np.exp(np.sqrt((1 - min(a_[0] , b_[0])/max(a_[0] , b_[0]))**2 + (a_[2]-b_[2])**2))-1)


def cost2d_multip(a_,b_, ep=0.01):
    return ((max(a_[0] , b_[0])+ep)/(min(a_[0] , b_[0]) + ep)) * (abs(a_[1]-b_[1]) +1)

def my_logistic_func_m(x, mean=100):
    L=100
    k=2/mean
    d=0
    return 2*L/(1 + np.exp(-k*(x-d)))-L

def my_logistic_func_out(x, mean = 100):
    if mean == 0:
        return x
    L=100
    k=0.04/mean
    d=0
    return 2*L/(1 + np.exp(-k*(x-d)))-L

def calc_distances_0hop_hyper(degreeList, L=np.e**4):
    list_ = []
    for v1 in degreeList.keys():
        for v2 in degreeList.keys():
            if v1 > v2:
                a_ = sorted(degreeList[v1][0][0], reverse=True)
                b_ = sorted(degreeList[v2][0][0], reverse=True)
                a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
                a_ = np.array(a_)
                b_ = np.array(b_)
                a_ = a_ - min(a_.min(), b_.min()) + 1
                b_ = b_ - min(a_.min(), b_.min()) + 1
                a_2d = tuple(zip(a_, prob_rev(a_)))
                b_2d = tuple(zip(b_, prob_rev(b_)))
                dist_inner, tmp_inner = fastdtw(a_2d, b_2d, radius=1, dist=cost2d_euclid)
                list_.append(dist_inner)
    # search theta with maximum variance
    theta = gradient_decent(np.array(list_), L=L)
    return theta

def calc_distances_0hop_hyper_opt2(degreeList, part):
    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    list_ = []
    for v1 in vertices.keys():
        for v2 in vertices[v1]:
            a_ = sorted(degreeList[v1][0][0], reverse=True)
            b_ = sorted(degreeList[v2][0][0], reverse=True)
            a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
            a_ = np.array(a_)
            b_ = np.array(b_)
            a_ = a_ - min(a_.min(), b_.min()) + 1
            b_ = b_ - min(a_.min(), b_.min()) + 1
            a_2d = tuple(zip(a_, prob_rev(a_)))
            b_2d = tuple(zip(b_, prob_rev(b_)))
            dist_inner, tmp_inner = fastdtw(a_2d, b_2d, radius=1, dist=cost2d_euclid)
            list_.append(dist_inner)
    # print (part, len(list_))
    return list_

def calc_distances_hyper_r2(part, theta_layer0, compactDegree = False, scale_free = True): # 20230427

    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

    distances = {}
    dtw_distance_layer1 = {}



    def cost_hyper_base_dtw1_opt2(a,b):
        a_ = sorted(cost_hyper_base_dtw1_in_a[int(a)], reverse=True)
        b_ = sorted(cost_hyper_base_dtw1_in_b[int(b)], reverse=True)
        a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
        a_ = np.array(a_)
        b_ = np.array(b_)
        a_ = a_ - min(a_.min(), b_.min()) + 1
        b_ = b_ - min(a_.min(), b_.min()) + 1
        if (str((a_,b_)) not in dtw_distance_layer1.keys()):
            a_2d = tuple(zip(a_, prob_rev(a_)))
            b_2d = tuple(zip(b_, prob_rev(b_)))
            dist_inner, tmp_inner = fastdtw(a_2d, b_2d, radius=1, dist=cost2d_euclid)
            # dtw_distance_layer1[str((a_,b_))] = logiFunc(dist_inner, theta_layer0, np.e**4) #+ cost(len(a_), len(b_)) # Here!
            dtw_distance_layer1[str((a_,b_))] = logiFunc(dist_inner, theta_layer0, np.e) #+ cost(len(a_), len(b_)) # Here!
            
        return dtw_distance_layer1[str((a_,b_))]
    
    # def cost_hyper_base_dtw1(a,b):
    #     a_ = sorted(cost_hyper_base_dtw1_in_a[int(a)], reverse=True)
    #     b_ = sorted(cost_hyper_base_dtw1_in_b[int(b)], reverse=True)
    #     # size = max(max(a_,b_)) - min(min(a_,b_))
    #     # min_digit = 
    #     a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
    #     a_ = np.array(a_)
    #     b_ = np.array(b_)
    #     a_ = a_ - min(a_.min(), b_.min()) + 1
    #     b_ = b_ - min(a_.min(), b_.min()) + 1
    #     if (str((a_,b_)) not in dtw_distance_layer1.keys()):
    #         # a_2d = tuple(zip(a_, prob_exp(a_)))
    #         # b_2d = tuple(zip(b_, prob_exp(b_)))
    #         a_2d = tuple(zip(a_, prob_rev(a_)))
    #         b_2d = tuple(zip(b_, prob_rev(b_)))
    #         # print ('inner ... ', len(a_2d), len(b_2d), a_2d, b_2d)
    #         dist_inner, tmp_inner = fastdtw(a_2d, b_2d, radius=1, dist=cost2d_euclid)
    #         # dist_inner, tmp_inner = fastdtw(list(zip(a_, np.array(range(len(a_)))/(len(a_)-1)*size)) if len(a_)>1 else list(zip(a_, [0])), list(zip(b_, np.array(range(len(b_)))/(len(b_)-1)*size)) if len(b_)>1 else list(zip(b_, [0])), radius=1,dist=cost2d)
    #         # dtw_distance_layer1[str((a_,b_))] = np.exp(dist_inner) #+ cost(len(a_), len(b_)) # Here!
    #         dtw_distance_layer1[str((a_,b_))] = dist_inner #+ cost(len(a_), len(b_)) # Here!
    #     # del(a_, b_, min_digit, a_2d, b_2d)
    #     return dtw_distance_layer1[str((a_,b_))]

    if compactDegree:
        pass
    #     dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
    #     for v1 in vertices:
    #         lists_v1 = degreeList[v1]
    #         lists_v1_new = {}
    #         for key_,value_ in lists_v1.items():
    #             lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})
    #         for v2 in list_vertices[cont]:
    #             lists_v2 = degreeList[v2]
    #             lists_v2_new = {}
    #             for key_,value_ in lists_v2.items():
    #                 lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

    #             max_layer = min(len(lists_v1),len(lists_v2))
    #             distances[v1,v2] = {}
    #             for layer in range(0,max_layer):
    #                 #t0 = time()
                    
    #                 # print ('101 111')
    #                 # print (layer,lists_v1_new[layer],lists_v2_new[layer])
    #                 dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
    #                 # print ('100 111')
    #                 # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # # start dist_minus calculation

    # #  end  dist_minus calculation
    #                 #t1 = time()
    #                 #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
    #                 distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
    #                 # print (v1,v2, layer, np.exp(dist))
    #                 # distances[v1,v2][layer] = dist 
    #                 # distances[v1,v2][layer] = np.exp(dist)**np.e


    #         cont += 1

    else:
        dist_func = cost_hyper_base_dtw1_opt2
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    cost_hyper_base_dtw1_in_a = lists_v1[layer]
                    cost_hyper_base_dtw1_in_b = lists_v2[layer]
                    dist, path = fastdtw(range(len(lists_v1[layer])),range(len(lists_v2[layer])),radius=1,dist=dist_func)
                    distances[v1,v2][layer] = dist
    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_hyper_r2_euclide2(part, untilLayer, compactDegree = False, scale_free = True): # 20230427
    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))
    if compactDegree:
        degreesVecCompactMap = restoreVariableFromDisk('degreesVecCompactMap')
    distances = {}
    dtw_distance_layer1 = {}
    def cost_hyper_base_dtw1_opt2_opt4_compact(a,b):
        # print (a,b, cost_hyper_base_dtw1_in_a[int(a)],cost_hyper_base_dtw1_in_b[int(b)])
        a = int(a)
        b = int(b)
        a_ = degreesVecCompactMap[cost_hyper_base_dtw1_in_a[a][0]]['compact']
        b_ = degreesVecCompactMap[cost_hyper_base_dtw1_in_b[b][0]]['compact']
        str_tmp = str((cost_hyper_base_dtw1_in_a[a],cost_hyper_base_dtw1_in_b[b]))
        if (str_tmp) not in dtw_distance_layer1.keys():
            dist_inner, tmp_inner = fastdtw(collapsed_prob(a_), collapsed_prob(b_), radius=1, dist=cost2d_euclid2_opt1)
            # dtw_distance_layer1[str_tmp] = dist_inner * np.sqrt(cost_hyper_base_dtw1_in_a[a][1]*cost_hyper_base_dtw1_in_b[b][1])
            dtw_distance_layer1[str_tmp] = dist_inner * max(cost_hyper_base_dtw1_in_a[a][1], cost_hyper_base_dtw1_in_b[b][1])
        return dtw_distance_layer1[str_tmp]
    def cost_hyper_base_dtw1_opt2_Opt4(a,b):
        a_ = sorted(cost_hyper_base_dtw1_in_a[int(a)], reverse=True)
        b_ = sorted(cost_hyper_base_dtw1_in_b[int(b)], reverse=True)
        a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
        a_ = np.array(a_)
        b_ = np.array(b_)
        # diff = - min(a_.min(), b_.min()) + 1
        # a_ = a_ + diff
        # b_ = b_ + diff
        if (str((a_,b_)) not in dtw_distance_layer1.keys()):
            dist_inner, tmp_inner = fastdtw(collapse_prob(a_), collapse_prob(b_), radius=1, dist=cost2d_euclid2)
            # dtw_distance_layer1[str((a_,b_))] = logiFunc(dist_inner, theta_layer0, np.e**4) #+ cost(len(a_), len(b_)) # Here!
            dtw_distance_layer1[str((a_,b_))] = dist_inner
        return dtw_distance_layer1[str((a_,b_))]  
    def cost_hyper_base_dtw1_opt2_notOpt4(a,b):
        a_ = sorted(cost_hyper_base_dtw1_in_a[int(a)], reverse=True)
        b_ = sorted(cost_hyper_base_dtw1_in_b[int(b)], reverse=True)
        a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
        a_ = np.array(a_)
        b_ = np.array(b_)
        # diff = - min(a_.min(), b_.min()) + 1
        # a_ = a_ + diff
        # b_ = b_ + diff
        if (str((a_,b_)) not in dtw_distance_layer1.keys()):
            a_2d = tuple(zip(a_, prob_rev(a_)))
            b_2d = tuple(zip(b_, prob_rev(b_)))
            dist_inner, tmp_inner = fastdtw(a_2d, b_2d, radius=1, dist=cost2d_euclid2)
            # dtw_distance_layer1[str((a_,b_))] = logiFunc(dist_inner, theta_layer0, np.e**4) #+ cost(len(a_), len(b_)) # Here!
            dtw_distance_layer1[str((a_,b_))] = dist_inner
        return dtw_distance_layer1[str((a_,b_))]

    if compactDegree:
        dist_func = cost_hyper_base_dtw1_opt2_opt4_compact
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                if untilLayer is None:
                    max_layer = min(len(lists_v1),len(lists_v2))
                else:
                    max_layer = min(len(lists_v1),len(lists_v2), untilLayer)
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    cost_hyper_base_dtw1_in_a = lists_v1[layer]
                    cost_hyper_base_dtw1_in_b = lists_v2[layer]
                    dist, path = fastdtw(range(len(lists_v1[layer])),range(len(lists_v2[layer])),radius=1,dist=dist_func)
                    distances[v1,v2][layer] = dist
    else:
        dist_func = cost_hyper_base_dtw1_opt2_Opt4
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                if untilLayer is None:
                    max_layer = min(len(lists_v1),len(lists_v2))
                else:
                    max_layer = min(len(lists_v1),len(lists_v2), untilLayer)
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    cost_hyper_base_dtw1_in_a = lists_v1[layer]
                    cost_hyper_base_dtw1_in_b = lists_v2[layer]
                    dist, path = fastdtw(range(len(lists_v1[layer])),range(len(lists_v2[layer])),radius=1,dist=dist_func)
                    distances[v1,v2][layer] = dist
    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_complex_directed_shrink(part, compactDegree = False, scale_free = True): # Not used
    # scale_free = False
    # scale_free = True
    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))

    distances = {}

    if compactDegree:
        dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z),  (x[1], 0, 0, 0)) for x in value_]})
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = []
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    distances[v1,v2].append(np.exp(dist)) # Link to # edges might be better!!! TODO
                    # distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e
    else:
        dist_func = cost_complex_vector_Euclidean_logscale_directed if scale_free else cost_complex_vector_Euclidean_directed
        # dist_func = cost_complex_vector_2area_logscale
        # dist_func = cost_complex_sinusoidalWave_logscale_vector
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = []
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    # distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    distances[v1,v2].append(dist)
                    # distances[v1,v2][layer] = np.exp(dist)**np.e

#     if compactDegree:
#         dist_func = cost_max
#     else:
#         dist_func = cost

#     for v1,nbs in vertices.items():
#         lists_v1 = degreeList[v1]

#         for v2 in nbs:
#             t00 = time()
#             lists_v2 = degreeList[v2]

#             max_layer = min(len(lists_v1),len(lists_v2))
#             distances[v1,v2] = {}

#             for layer in range(0,max_layer):
#                 dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# # start dist_minus calculation

# #  end  dist_minus calculation
#                 distances[v1,v2][layer] = dist

            # t11 = time()
            # logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))


    preprocess_consolides_distances_shrink(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_signed(part,type, compactDegree = False):

    vertices = restoreVariableFromDisk(type+'split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk(type+'split-degreeList-'+str(part))

    distances = {}

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1,nbs in vertices.items():
        lists_v1 = degreeList[v1]

        for v2 in nbs:
            t00 = time()
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# start dist_minus calculation

#  end  dist_minus calculation
                distances[v1,v2][layer] = dist

            t11 = time()
            logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))


    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,type+'distances-'+str(part))
    return

def calc_distances_all(vertices,list_vertices,degreeList,part, compactDegree = False):

    distances = {}
    cont = 0

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1 in vertices:
        lists_v1 = degreeList[v1]

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                #t0 = time()
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# start dist_minus calculation

#  end  dist_minus calculation
                #t1 = time()
                #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                distances[v1,v2][layer] = dist


        cont += 1

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def get_maxA():
    global maxA
    return maxA

def set_maxA(value):
    global maxA
    maxA = value
    return

def calc_distances_all_complex(vertices,list_vertices,degreeList,part, compactDegree = False, scale_free = True):
    # scale_free = False
    # scale_free = True
    distances = {}
    cont = 0

    # TODO
    if compactDegree:
        dist_func = cost_complex_vector_Euclidean_logscale_max if scale_free else cost_complex_vector_Euclidean_max
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[((x[0].real, x[0].imag), x[1].real) for x in value_]}) # BUG!... fixed
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[((x[0].real, x[0].imag), x[1].real) for x in value_]}) # BUG!... fixed

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    # distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e


            cont += 1

    else:
        dist_func = cost_complex_vector_Euclidean_logscale if scale_free else cost_complex_vector_Euclidean
        # dist_func = cost_complex_vector_2area_logscale
        # dist_func = cost_complex_sinusoidalWave_logscale_vector
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[(x.real, x.imag) for x in value_]})
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[(x.real, x.imag) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    # distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    distances[v1,v2][layer] = dist #BUG
                    # distances[v1,v2][layer] = np.exp(dist)**np.e


            cont += 1

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_all_complex_directed(vertices,list_vertices,degreeList,part, compactDegree = False, scale_free = True):
    # scale_free = False
    # scale_free = True
    distances = {}
    cont = 0

    # TODO
    if compactDegree:
        dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    # print (v1,v2, layer, np.exp(dist))
                    # distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e


            cont += 1

    else:
        dist_func = cost_complex_vector_Euclidean_logscale_directed if scale_free else cost_complex_vector_Euclidean_directed
        # dist_func = cost_complex_vector_2area_logscale
        # dist_func = cost_complex_sinusoidalWave_logscale_vector
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    # distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    distances[v1,v2][layer] = np.exp(dist)
                    # print (v1,v2, layer, np.exp(dist), dist)
                    # distances[v1,v2][layer] = np.exp(dist)**np.e


            cont += 1

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_all_hyper(vertices,list_vertices,degreeList,part, theta_layer0, compactDegree = False):

    distances = {}
    cont = 0
    dtw_distance_layer1 = {}

    def cost_hyper_base_dtw1(a,b):
        a_ = sorted(cost_hyper_base_dtw1_in_a[int(a)], reverse=True)
        b_ = sorted(cost_hyper_base_dtw1_in_b[int(b)], reverse=True)
        a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
        a_ = np.array(a_)
        b_ = np.array(b_)
        a_ = a_ - min(a_.min(), b_.min()) + 1
        b_ = b_ - min(a_.min(), b_.min()) + 1
        if (str((a_,b_)) not in dtw_distance_layer1.keys()):
            a_2d = tuple(zip(a_, prob_rev(a_)))
            b_2d = tuple(zip(b_, prob_rev(b_)))
            dist_inner, tmp_inner = fastdtw(a_2d, b_2d, radius=1, dist=cost2d_euclid)
            dtw_distance_layer1[str((a_,b_))] = logiFunc(dist_inner, theta_layer0, np.e**4) #+ cost(len(a_), len(b_)) # Here!
            
        return dtw_distance_layer1[str((a_,b_))]

    if compactDegree:
        pass
    #     dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
    #     for v1 in vertices:
    #         lists_v1 = degreeList[v1]
    #         lists_v1_new = {}
    #         for key_,value_ in lists_v1.items():
    #             lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})
    #         for v2 in list_vertices[cont]:
    #             lists_v2 = degreeList[v2]
    #             lists_v2_new = {}
    #             for key_,value_ in lists_v2.items():
    #                 lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

    #             max_layer = min(len(lists_v1),len(lists_v2))
    #             distances[v1,v2] = {}
    #             for layer in range(0,max_layer):
    #                 #t0 = time()
                    
    #                 # print ('101 111')
    #                 # print (layer,lists_v1_new[layer],lists_v2_new[layer])
    #                 dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
    #                 # print ('100 111')
    #                 # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # # start dist_minus calculation

    # #  end  dist_minus calculation
    #                 #t1 = time()
    #                 #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
    #                 distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
    #                 # print (v1,v2, layer, np.exp(dist))
    #                 # distances[v1,v2][layer] = dist 
    #                 # distances[v1,v2][layer] = np.exp(dist)**np.e


    #         cont += 1

    else:
        dist_func = cost_hyper_base_dtw1
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    cost_hyper_base_dtw1_in_a = lists_v1[layer]
                    cost_hyper_base_dtw1_in_b = lists_v2[layer]
                    dist, path = fastdtw(range(len(lists_v1[layer])),range(len(lists_v2[layer])),radius=1,dist=dist_func)
                    distances[v1,v2][layer] = dist
            cont += 1
    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_all_hyper_euclid2(vertices,list_vertices,degreeList, reversedMap, degreesVecCompactMap,part, untilLayer, compactDegree = False):
    distances = {}
    cont = 0
    dtw_distance_layer1 = {}
    def cost_hyper_base_dtw1(a,b):
        a_ = sorted(cost_hyper_base_dtw1_in_a[int(a)], reverse=True)
        b_ = sorted(cost_hyper_base_dtw1_in_b[int(b)], reverse=True)
        a_, b_ = sorted([a_,b_], key=lambda x: (len(x), sum(x), x[0:min(len(a_), len(b_))] ), reverse=True)
        a_ = np.array(a_)
        b_ = np.array(b_)
        # diff = - min(a_.min(), b_.min()) + 1
        # a_ = a_ + diff
        # b_ = b_ + diff
        if (str((a_,b_)) not in dtw_distance_layer1.keys()):
            # a_2d = tuple(zip(a_, prob_rev(a_)))
            # b_2d = tuple(zip(b_, prob_rev(b_)))
            dist_inner, tmp_inner = fastdtw(collapse_prob(a_), collapse_prob(b_), radius=1, dist=cost2d_euclid2_opt1)
            dtw_distance_layer1[str((a_,b_))] = dist_inner
            # dtw_distance_layer1[str((a_,b_))] = np.exp(dist_inner)
            
        return dtw_distance_layer1[str((a_,b_))]

    def cost_hyper_base_dtw1_compact(a,b):
        # print (a,b, cost_hyper_base_dtw1_in_a[int(a)],cost_hyper_base_dtw1_in_b[int(b)])
        a = int(a)
        b = int(b)
        a_ = degreesVecCompactMap[cost_hyper_base_dtw1_in_a[a][0]]['compact']
        b_ = degreesVecCompactMap[cost_hyper_base_dtw1_in_b[b][0]]['compact']
        str_tmp = str((cost_hyper_base_dtw1_in_a[a],cost_hyper_base_dtw1_in_b[b]))
        if (str_tmp) not in dtw_distance_layer1.keys():
            dist_inner, tmp_inner = fastdtw(collapsed_prob(a_), collapsed_prob(b_), radius=1, dist=cost2d_euclid2_opt1)
            # dtw_distance_layer1[str_tmp] = dist_inner * np.sqrt(cost_hyper_base_dtw1_in_a[a][1]*cost_hyper_base_dtw1_in_b[b][1])
            dtw_distance_layer1[str_tmp] = dist_inner * max(cost_hyper_base_dtw1_in_a[a][1], cost_hyper_base_dtw1_in_b[b][1])
        return dtw_distance_layer1[str_tmp]
    
    if compactDegree:
        dist_func = cost_hyper_base_dtw1_compact
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                if untilLayer is not None:
                    max_layer = min(len(lists_v1),len(lists_v2), untilLayer)
                else:
                    max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    cost_hyper_base_dtw1_in_a = lists_v1[layer]
                    cost_hyper_base_dtw1_in_b = lists_v2[layer]
                    dist, path = fastdtw(range(len(lists_v1[layer])),range(len(lists_v2[layer])),radius=1,dist=dist_func)
                    distances[v1,v2][layer] = dist
                    # distances[v1,v2][layer] = np.exp(dist)
            cont += 1
    else:
        dist_func = cost_hyper_base_dtw1
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                if untilLayer is not None:
                    max_layer = min(len(lists_v1),len(lists_v2), untilLayer)
                else:
                    max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    cost_hyper_base_dtw1_in_a = lists_v1[layer]
                    cost_hyper_base_dtw1_in_b = lists_v2[layer]
                    dist, path = fastdtw(range(len(lists_v1[layer])),range(len(lists_v2[layer])),radius=1,dist=dist_func)
                    distances[v1,v2][layer] = dist
                    # distances[v1,v2][layer] = np.exp(dist)
            cont += 1
    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_l0_hyper(vertices,list_vertices,degreesVec,part, compactDegree = False):
    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost
    distances = {}
    for v1 in vertices:
        for v2 in list_vertices:
            distances[v1,v2] = {}
            dist, path = fastdtw(sorted(degreesVec[v1], reverse=True),sorted(degreesVec[v2], reverse=True),radius=1,dist=dist_func)
            distances[v1,v2] = dist
    # saveVariableOnDisk(distances,'distances-l0-'+str(part))
    return distances

def calc_distances_all_signed(vertices,list_vertices,degreeList,part,type, compactDegree = False):

    distances = {}
    cont = 0

    if compactDegree:
        dist_func = cost_max
    else:
        dist_func = cost

    for v1 in vertices:
        lists_v1 = degreeList[v1]

        for v2 in list_vertices[cont]:
            lists_v2 = degreeList[v2]

            max_layer = min(len(lists_v1),len(lists_v2))
            distances[v1,v2] = {}

            for layer in range(0,max_layer):
                #t0 = time()
                dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# start dist_minus calculation

#  end  dist_minus calculation
                #t1 = time()
                #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                distances[v1,v2][layer] = dist


        cont += 1

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,type+'distances-'+str(part))
    return

def selectVertices(layer,fractionCalcDists):
    previousLayer = layer - 1

    logging.info("Recovering distances from disk...")
    distances = restoreVariableFromDisk('distances')

    threshold = calcThresholdDistance(previousLayer,distances,fractionCalcDists)

    logging.info('Selecting vertices...')

    vertices_selected = deque()

    for vertices,layers in distances.items():
        if(previousLayer not in layers):
            continue
        if(layers[previousLayer] <= threshold):
            vertices_selected.append(vertices)

    distances = {}

    logging.info('Vertices selected.')

    return vertices_selected


def preprocess_consolides_distances(distances, startLayer = 1):

    logging.info('Consolidating distances...')

    for vertices,layers in distances.items():
        keys_layers = sorted(list(layers.keys()))
        startLayer = min(len(keys_layers),startLayer)
        for layer in range(0,startLayer):
            keys_layers.pop(0)


        for layer in keys_layers:
            layers[layer] += layers[layer - 1]

    logging.info('Distances consolidated.')

def preprocess_consolides_distances_shrink(distances, startLayer = 1):

    logging.info('Consolidating distances shrink...')

    for vertices,layers in distances.items():
        for layer in range(startLayer, len(layers)):
            layers[layer] += layers[layer - 1]

    logging.info('Distances consolidated shrink.')

def exec_bfs_compact_hyper(H,degreesVecCompact,degreesCecCompactMap, reversedMap, workers,calcUntilLayer,opt4):
    futures = {}
    degreeList = {}

    t0 = time()
    vertices = list(sorted(list(H.keys())))
    parts = workers
    chunks = partition(vertices,parts)

    # logging.info('Capturing larger degree...')
    # for v in vertices:
    #     if(len(Gpi[v])+len(Gni[v])+len(Gpo[v])+len(Gno[v]) > maxDegree):
    #         maxDegree = len(Gpi[v])+len(Gni[v])+len(Gpo[v])+len(Gno[v])
    # logging.info('Larger degree captured')

    # part = 1
    
    # print (chunks) #DEBUG
    # for c in chunks:
    #     dl = getCompactDegreeListsVertices_complex_directed(Gpi,Gni,Gpo,Gno,c,maxDegree,calcUntilLayer)
    #     degreeList.update(dl)
    # print (degreeList) #DEBUG
    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getCompactDegreeListsVertices_hyper,H,degreesVecCompact,degreesCecCompactMap, reversedMap,c,calcUntilLayer,opt4)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'compactDegreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))
    return

def exec_bfs_hyper(G,degreeVec,workers,calcUntilLayer):
    futures = {}
    degreeList = {}

    t0 = time()
    vertices = list(sorted(set(list(G.keys()))))
    parts = workers
    chunks = partition(vertices,parts)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        part = 1
        for c in chunks:
            job = executor.submit(getDegreeListsVertices_hyper,G,degreeVec,c,calcUntilLayer)
            futures[job] = part
            part += 1
        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))

    return


def generate_distances_network_part1(workers):
    parts = workers
    weights_distances = {}
    for part in range(1,parts + 1):    

        logging.info('Executing part {}...'.format(part))
        distances = restoreVariableFromDisk('distances-'+str(part))
        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances):
                    weights_distances[layer] = {}
                weights_distances[layer][vx,vy] = distance

        logging.info('Part {} executed.'.format(part))

    for layer,values in weights_distances.items():
        saveVariableOnDisk(values,'weights_distances-layer-'+str(layer))
        # print (layer, values)
    return

def generate_distances_network_part2(workers):
    parts = workers
    graphs = {}
    for part in range(1,parts + 1):

        logging.info('Executing part {}...'.format(part))
        distances = restoreVariableFromDisk('distances-'+str(part))

        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in graphs):
                    graphs[layer] = {}
                if(vx not in graphs[layer]):
                   graphs[layer][vx] = [] 
                if(vy not in graphs[layer]):
                   graphs[layer][vy] = [] 
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
        logging.info('Part {} executed.'.format(part))

    for layer,values in graphs.items():
        saveVariableOnDisk(values,'graphs-layer-'+str(layer))

    return

def generate_distances_network_part3():

    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        weights_distances = restoreVariableFromDisk('weights_distances-layer-'+str(layer))

        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}

        for v,neighbors in graphs.items():
            e_list = deque()
            sum_w = 0.0


            for n in neighbors:
                if (v,n) in weights_distances:
                    wd = weights_distances[v,n]
                else:
                    wd = weights_distances[n,v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q

        saveVariableOnDisk(weights,'distances_nets_weights-layer-'+str(layer))
        saveVariableOnDisk(alias_method_j,'alias_method_j-layer-'+str(layer))
        saveVariableOnDisk(alias_method_q,'alias_method_q-layer-'+str(layer))
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info('Weights created.')

    return


def generate_distances_network_part4():
    logging.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1


    logging.info("Saving distancesNets on disk...")
    saveVariableOnDisk(graphs_c,'distances_nets_graphs')
    logging.info('Graphs consolidated.')
    return

def generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while(isPickle('alias_method_j-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_j = restoreVariableFromDisk('alias_method_j-layer-'+str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    saveVariableOnDisk(alias_method_j_c,'nets_weights_alias_method_j')

    return

def generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while(isPickle('alias_method_q-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_q = restoreVariableFromDisk('alias_method_q-layer-'+str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    saveVariableOnDisk(alias_method_q_c,'nets_weights_alias_method_q')

    return

def generate_distances_network(workers):
    t0 = time()
    logging.info('Creating distance network...')

    os.system("rm "+returnPathPickles()+"weights_distances-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time()
    os.system("rm "+returnPathPickles()+"graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 2: {}s'.format(t))
    logging.info('distance network created.')

    logging.info('Transforming distances into weights...')

    t0 = time()
    os.system("rm "+returnPathPickles()+"distances_nets_weights-layer-*.pickle")
    os.system("rm "+returnPathPickles()+"alias_method_j-layer-*.pickle")
    os.system("rm "+returnPathPickles()+"alias_method_q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part3)
        # job = executor.submit(generate_distances_network_part3_r1_panorama)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 3: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 4: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 5: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 6: {}s'.format(t))

    return

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    # J = np.zeros(K, dtype=np.int)
    J = np.zeros(K, dtype=int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def mylambda():
    return []