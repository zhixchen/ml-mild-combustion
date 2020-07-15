#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:10:33 2020

@author: Zhi X. Chen, Cambridge University, zc252@cam.ac.uk
"""

import numpy as np
import os
import os.path
from scipy.stats import beta
import numpy.matlib as npm
from scipy import interpolate

def KLDiv(P,Q):
    #  dist = KLDiv(P,Q) Kullback-Leibler divergence of two discrete probability
    #  distributions
    #  P and Q  are automatically normalised to have the sum of one on rows
    # have the length of one at each 
    # P =  n x nbins
    # Q =  1 x nbins or n x nbins(one to one)
    # dist = n x 1
    P = P + 1e-16
    Q = Q + 1e-16
    if np.size(P,1)!=np.size(Q,1):
        print('The number of columns in P and Q shoulb be the same')
    #normalizing P and Q
    if np.size(Q,0) == 1:
        Q = Q/np.sum(Q)
        P = P/npm.repmat(np.sum(P,1), 1, np.size(P,1))
        dist = np.sum(P*np.log(P/npm.repmat(Q,np.size(P,1),1)),axis = 1)

    elif np.size(Q,0) == np.size(P,0):
        Q = Q/npm.repmat(np.sum(Q,1),1,np.size(Q,1))
        P = P/npm.repmat(np.sum(P,1),1,np.size(P,1))
        dist = np.sum(P*np.log(P / Q), axis=1)
    
    np.nan_to_num(dist,nan=0)
    
    return(dist)

def JSDiv(P,Q):
    # Jensen-Shannon divergence of two probability distributions
    #  dist = JSD(P,Q) Kullback-Leibler divergence of two discrete probability
    #  distributions
    #  P and Q  are automatically normalised to have the sum of one on rows
    # have the length of one at each 
    # P =  n x nbins
    # Q =  1 x nbins
    # dist = n x 1
    if np.size(P,1)!= np.size(Q,1):
        print('The number of columns in P and Q shoulb be the same')
        
    Q = Q/np.sum(Q)
    Q = npm.repmat(Q, np.size(P,0), 1)
    P = P/(npm.repmat(np.sum(P,1), 1, np.size(P,1)))
    
    M = 0.5*(P+Q)
    return((0.5*KLDiv(P,M)) + (0.5*KLDiv(Q,M)))

def copulaPDF(i,j,k,centres,edges,input_zc):
    # start_time = time.time()
    cbar = input_zc[0]
    gc = input_zc[3]/(cbar*(1-cbar)+1e-8)
    aa = cbar*(1/gc-1) 
    bb = (1-cbar)*(1/gc-1)
    c_space = np.linspace(0,1,101)
    betaPDF = beta.pdf(c_space,aa,bb)
    if sum(np.isinf(betaPDF)) > 0:
        betaPDF[np.isinf(betaPDF).nonzero()] = 0.
    betaPDF = betaPDF/np.trapz(betaPDF,x=c_space)
    betaCDF = np.zeros(betaPDF.shape)
    betaCDF[0] = betaPDF[0]
    for ii in range(1,len(c_space)):
        betaCDF[ii] = np.trapz(betaPDF[0:ii],x=c_space[0:ii])

    zbar = input_zc[1]
    gz = input_zc[2]/(zbar*(1-zbar)+1e-8)
    aa = zbar*(1/gz-1) 
    bb = (1-zbar)*(1/gz-1)
    z_space = np.logspace(-3,np.log10(max(edges[1])/2),98)
    z_space = np.insert(z_space,0,[0.],axis=0)
    z_space = np.insert(z_space,len(z_space),[max(edges[1])],axis=0)
    zbetaPDF = beta.pdf(z_space,aa,bb)
    if sum(np.isinf(zbetaPDF)) > 0:
        zbetaPDF[np.isinf(zbetaPDF).nonzero()] = 0.
    zbetaPDF = zbetaPDF/np.trapz(zbetaPDF,x=z_space)
    zbetaCDF = np.zeros(zbetaPDF.shape)
    zbetaCDF[0] = zbetaPDF[0]
    for ii in range(1,len(z_space)):
        zbetaCDF[ii] = np.trapz(zbetaPDF[0:ii],x=z_space[0:ii])
    
    cov = input_zc[4]
    tmp = np.array([gz*zbar*(1-zbar),gc*cbar*(1-cbar),cov,max(edges[1])])
    cplDir = 'copula_v3'
    with open('copulaDir.txt','w') as strfile:
        strfile.write(cplDir)
    np.savetxt(cplDir + '/copula_data.txt',tmp[np.newaxis,:],
               fmt='%.5e %.5e %.5e %.5e')
    np.savetxt(cplDir + '/betaPDFs.txt',
                np.concatenate((betaPDF[np.newaxis,:],zbetaPDF[np.newaxis,:],
                                betaCDF[np.newaxis,:],zbetaCDF[np.newaxis,:])
                              ,axis=1).T,fmt='%.5e')
    os.system(cplDir + '/copula')
    fln = (cplDir + '/copulaJPDF_.dat')
    data = np.loadtxt(fln)
    cplPDF = data.reshape(len(c_space),len(z_space),order='F')
    f = interpolate.interp2d(z_space,c_space,cplPDF)
    cplPDF_intp = f(centres[1],centres[0])
    # print("\n---Completed in %s s ---" % ((time.time() - start_time)))

    return cplPDF_intp