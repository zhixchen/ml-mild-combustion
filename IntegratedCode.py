#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:10:33 2020

@author: Zhi X. Chen, Cambridge University, zc252@cam.ac.uk
"""
# %%
from scipy.io import loadmat
import numpy as np
import time
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import beta
import os

import main_ANN
import func

newDirName = "./model_zc"
if not os.path.exists(newDirName):
    os.mkdir(newDirName)
            
start_time = time.time()
# copula code not included, contact author if interested
switch_copula = 0 
# switch for four- (0) or five-input (1) training
switch_cov = 1

## GLOBAL VARIABLES
#number of pdf bins in c space                                               
N_cBins = 32 
#number of pdf bins in Z space
N_ZBins = 36
# bin edges 2D
edges = [] 
# bin centres 2D
centres = []
# bin areas 2D
binAreas = np.array((N_cBins-1,N_ZBins-1))
# half filter width 
band = 0
# number of dices in 1D
inputSize1D = 0
# number of time stamps
No_ts = 0

#%% # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#   CASE SETUP AND TRAINING DATA EXTRACTION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def data_extract(filterSize):
    global edges, jump, band, inputSize1D, centres, binAreas, No_ts
    band = int(filterSize/2) 
    jump = int(filterSize/2)
    edges = [];
    edges.append(np.linspace(0,1,N_cBins))
    if case == 'BZ1':
        Zmax = 0.025
    else:
        Zmax = 0.05
    edges.append(np.logspace(-3,np.log10(Zmax/4),N_ZBins-3))
    edges[1] = np.insert(edges[1],0,[0.],axis=0)
    edges[1] = np.insert(edges[1],len(edges[1]),Zmax/2,axis=0)
    edges[1] = np.insert(edges[1],len(edges[1]),Zmax,axis=0)
    binWidth0 = np.diff(edges[0])[:,np.newaxis]
    binWidth1 = np.diff(edges[1])[np.newaxis,:]
    binAreas = np.matmul(binWidth0,binWidth1);
    centres = [];
    centres.append(edges[0][0:-1]/2 + edges[0][1:]/2)
    centres.append(edges[1][0:-1]/2 + edges[1][1:]/2)

    #Adjusting jump and No.TS
    if case == 'AZ1':
        ts_start = 399
        ts_end = 501
        ts_jumpAll = [ts_end-ts_start,6,3]
        ts_jump = ts_jumpAll[filterIdx-1]
        if filterSize == 120:
            jump = int(jump/1.3)
    elif case == 'AZ2':
        ts_start = 500
        ts_end = 670
        ts_jumpAll = [ts_end-ts_start,10,5]
        ts_jump = ts_jumpAll[filterIdx-1]
        if filterSize == 120:
            jump = int(jump/1.3)
    else:
        ts_start = 336
        ts_end = 501
        ts_jumpAll = [12,3]
        ts_jump = ts_jumpAll[filterIdx-1]
        if filterSize == 148:
            jump = int(jump/1.5)
            
    #obtain number of dices in one direction (for training)
    #first and last half width is excluded
    inputSize1D = 0;
    for i1 in range (band, 512-band, jump):
        inputSize1D = inputSize1D +1
    inputSize = inputSize1D**3
    No_ts = len(range(ts_start,ts_end+1,ts_jump))
    print("\nNo. snapshots = %d" % No_ts)
    inputSizeAll = inputSize*No_ts
    print("\nNo. training sets = %d" %inputSizeAll)
    
    MX_zc = np.empty((inputSize1D,inputSize1D,inputSize1D,No_ts),dtype=object)
    Mtarget_zc = np.empty(MX_zc.shape, dtype=object)
    betaPDF3D = np.zeros(MX_zc.shape, dtype=object)
    cplPDF3D = np.zeros(MX_zc.shape, dtype=object)
    dnsPDF3D = np.zeros(MX_zc.shape, dtype=object)
    annPDF3D = np.zeros(MX_zc.shape, dtype=object)
    WcT_flt = np.zeros(np.array(MX_zc).shape)
    RHO_flt = np.zeros(np.array(MX_zc).shape)
    Cbar3D = np.zeros(np.array(MX_zc).shape)
    Zbar3D = np.zeros(np.array(MX_zc).shape)
    Zvar3D = np.zeros(np.array(MX_zc).shape)
    Cvar3D = np.zeros(np.array(MX_zc).shape)
    cov3D = np.zeros(np.array(MX_zc).shape)
    X_zc = [] #learning input vector 
    target_zc = [] #learning target vector
    count = 0
    countCpl = 0
    
    print("\nBinning Data")
    t = -1
    for ts in range(ts_start,ts_end+1,ts_jump):
        t += 1
        print("\nReading " + case + "_ts%d_FS%s data..." % (ts,filterSize))
        mat_data = loadmat(fileDir + 'ts' + str(ts) + '_' + case + '_FS'
                            + str(filterSize) + '_Fav.mat')
        mat_data['Z'][(mat_data['Z']>Zmax).nonzero()[0],
                      (mat_data['Z']>Zmax).nonzero()[1],
                      (mat_data['Z']>Zmax).nonzero()[2]] = Zmax
                
        #obtain pdf for each traning dices 
        a = -1
        for i in range (band, 512-band, jump):
            a += 1
            b = -1
            for j in range (band, 512-band, jump):
                b += 1
                c = -1
                for k in range (band, 512-band, jump):
                    c += 1
    
                    count = count + 1
                    #Z and c fields for the current dice
                    Cb = (mat_data['C'][i-band:i+band,j-band:j+band,k-band:k+band]*
                          mat_data['RHO'][i-band:i+band,j-band:j+band,k-band:k+band]
                          /np.mean(mat_data['RHO'][i-band:i+band,j-band:j+band,k-band:k+band]))
                    Zb = (mat_data['Z'][i-band:i+band,j-band:j+band,k-band:k+band]*
                          mat_data['RHO'][i-band:i+band,j-band:j+band,k-band:k+band]
                          /np.mean(mat_data['RHO'][i-band:i+band,j-band:j+band,k-band:k+band]))
                    
                    #mean and variances for the current dice 
                    Cbar3D[a,b,c,t] = mat_data['Ctld'][i,j,k]
                    Zbar3D[a,b,c,t] = mat_data['Ztld'][i,j,k]
                    Zvar3D[a,b,c,t] = mat_data['ZvarFa'][i,j,k]
                    gZ = mat_data['ZvarFa'][i,j,k]/(Zbar3D[a,b,c,t]*(1-Zbar3D[a,b,c,t])+1e-8)
                    Cvar3D[a,b,c,t] = mat_data['CvarFa'][i,j,k]
                    gc = mat_data['CvarFa'][i,j,k]/(Cbar3D[a,b,c,t]*(1-Cbar3D[a,b,c,t])+1e-8)
                    cov3D[a,b,c,t]  = mat_data['covFa'][i,j,k]
                    # gzc = mat_data['covFa'][i,j,k]/np.sqrt(mat_data['ZvarFa'][i,j,k]
                    #             *mat_data['CvarFa'][i,j,k]+1e-8)
                    
                    input_zc = [Cbar3D[a,b,c,t],Zbar3D[a,b,c,t],
                                Zvar3D[a,b,c,t],Cvar3D[a,b,c,t]]
                    if switch_cov:
                        input_zc = np.insert(input_zc,len(input_zc),cov3D[a,b,c,t],axis=0)
                    MX_zc[a,b,c,t] = input_zc
                    X_zc.append(input_zc)
                    NRe,edges[0],edges[1] = np.histogram2d(Cb.flatten(),Zb.flatten(),
                                                            (edges[0],edges[1]))
                    NRe_norm = NRe / np.sum(NRe)
                    output = NRe_norm.reshape(1,(N_cBins-1)*(N_ZBins-1),order='F')
                    Mtarget_zc[a,b,c,t] = NRe_norm
                    target_zc.append(np.concatenate(output))
    
                    #1D PDF for c
                    betaPDF = np.zeros(centres[0].shape)
                    if Cbar3D[a,b,c,t] > edges[0][-2]:
                        betaPDF[-1] = 1./np.concatenate(binWidth0)[-1]
                    elif Cbar3D[a,b,c,t] < edges[0][1]:
                        betaPDF[0] = 1./np.concatenate(binWidth0)[0]
                    else:
                        aa = Cbar3D[a,b,c,t]*(1/gc-1)
                        bb = (1-Cbar3D[a,b,c,t])*(1/gc-1)
                        betaPDF = beta.pdf(centres[0],aa,bb)
                        betaPDF = betaPDF/np.trapz(betaPDF,x=centres[0])
                        if sum(np.isnan(betaPDF)) > 0:
                            betaPDF = 0.
                            indx = np.concatenate(np.histogram(Cbar3D[a,b,c,t],
                                                                edges[0])[0].nonzero())
                            betaPDF[indx[0]] = 1./binWidth0[indx[0]]
    
                    #1D PDF for Z
                    zbetaPDF = np.zeros(centres[1].shape)
                    if Zbar3D[a,b,c,t] > edges[1][-1]:
                        zbetaPDF[-1] = 1./np.concatenate(binWidth1)[-1]
                    elif Zbar3D[a,b,c,t] < edges[1][1]:
                        zbetaPDF[0] = 1./np.concatenate(binWidth1)[0]
                    else:
                        aa = Zbar3D[a,b,c,t]*(1/gZ-1)
                        bb = (1-Zbar3D[a,b,c,t])*(1/gZ-1)
                        zbetaPDF = beta.pdf(centres[1],aa,bb)
                        zbetaPDF = zbetaPDF/np.trapz(zbetaPDF,x=centres[1])
                        if sum(np.isnan(zbetaPDF)) > 0:
                            zbetaPDF = 0.
                            indx = np.concatenate(np.histogram(Zbar3D[a,b,c,t],
                                                                edges[1])[0].nonzero())
                            zbetaPDF[indx[0]] = 1./binWidth1[indx[0]]
                            
                    if switch_copula:
                        threCov = 4e-5; 
                        if(case == 'BZ1'): threCov = 0.5*threCov
                        if(abs(mat_data['covFa'][i,j,k]) < threCov):
                            cplPDF3D[a,b,c,t] = np.atleast_2d(betaPDF).T*zbetaPDF
                        else:
                            countCpl = countCpl + 1
                            cplPDF3D[a,b,c,t] = func.copulaPDF(i,j,k,centres,edges,
                                                                input_zc)
    
                    betaPDF3D[a,b,c,t] = np.atleast_2d(betaPDF).T*zbetaPDF
                    dnsPDF3D[a,b,c,t] = Mtarget_zc[a,b,c,t]/binAreas
    
                    WcT_flt[a,b,c,t] = mat_data['WcTbar'][i,j,k]
                    RHO_flt[a,b,c,t] = mat_data['RHObar'][i,j,k]
                    if(np.remainder(count,100) == 0):
                        print("Progress --> %s%% || copula %s%%"
                          % (str('{:.1f}'.format(count/inputSizeAll*100)),
                              str('{:.1f}'.format(countCpl/inputSizeAll*100))))
    
    dataDict = {
             'MX_zc': MX_zc
            ,'Mtarget_zc': Mtarget_zc
            ,'betaPDF3D': betaPDF3D
            ,'cplPDF3D': cplPDF3D
            ,'dnsPDF3D': dnsPDF3D
            ,'annPDF3D': annPDF3D
            ,'WcT_flt': WcT_flt
            ,'RHO_flt': RHO_flt
            ,'Zbar3D': Zbar3D
            ,'Cbar3D': Cbar3D
            ,'Zvar3D': Zvar3D
            ,'Cvar3D': Cvar3D
            ,'cov3D': cov3D
            ,'X_zc': X_zc
            ,'target_zc': target_zc
              }
    
    return dataDict

# %%
def DNN(X_zc,target_zc):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   TRAINING DATA MANIPULATION 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    print("\nPre-processing")
    X_zc = np.array(X_zc)
    target_zc = np.array(target_zc)
    main_ANN.preprocessing_zc(X_zc, target_zc,'model_zc/')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #   LEARNING AND VALIDATION
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         
    print("\nLearning")
    main_ANN.Learning()
        
#%%   POST-PROCESSING 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def Post_processing(fileDir,case,dataDict):      
    print("\nPost processing")
    predictions_zc = np.load('model_zc/Results-ANN/predictions_zc_newInput.npy')
    outlier_index  = np.load('model_zc/Results-ANN/outlier_index.npy')
    not_outlier = 1 - outlier_index
    
    n_elements = np.size(predictions_zc,0)
    n_variables = np.size(predictions_zc,1)
    PDFs_zc = np.empty(n_elements, dtype=object)    
    PDF3D_zc = np.empty((inputSize1D,inputSize1D,inputSize1D,No_ts), dtype=object)
    unscaled_PDF_vectors = np.zeros((n_elements, n_variables))
    
    for i in range(n_elements):
        unscaled_PDF_vectors[i,:] = predictions_zc[i,:]
        
    for i in range(n_elements):
        PDFs_zc[i] = unscaled_PDF_vectors[i,:].reshape((N_cBins-1),(N_ZBins-1)
                                                        ,order='F') / binAreas
    
    PDF3D_zc = (PDFs_zc.reshape(inputSize1D,inputSize1D,inputSize1D,No_ts,order='F'))
    annPDF3D = np.transpose(PDF3D_zc,(2,1,0,3)) 
    not_outlier3D = (not_outlier.reshape(inputSize1D,inputSize1D,inputSize1D,No_ts,order='F'))
    not_outlier3D = np.transpose(not_outlier3D,(2,1,0,3)) 
    
    '''######################################################
    ### Choosing a test dice 
    ######################################################'''
    
    a = 5; b= 5; c = 7; t = 0
    
    betaPDF = np.trapz(dataDict['betaPDF3D'][a,b,c,t],x=centres[1],axis=1)
    zbetaPDF = np.trapz(dataDict['betaPDF3D'][a,b,c,t],x=centres[0],axis=0)
    
    #compute copula joint PDF
    if switch_copula:
        cplPDF = dataDict['cplPDF3D'][a,b,c,t]
    
    print("Ctld = %s" % dataDict['Cbar3D'][a,b,c,t])
    print("Ztld = %s" % dataDict['Zbar3D'][a,b,c,t])
    print("Zvar = %s" % dataDict['Zvar3D'][a,b,c,t])
    print("Cvar = %s" % dataDict['Cvar3D'][a,b,c,t])
    print("cov = %s" % dataDict['cov3D'][a,b,c,t])
    
    '''######################################################
    ### Joint PDF contours
    ######################################################'''
    plt.figure()
    #Joint PDF -DNS
    plt.contourf(centres[1],centres[0],dataDict['dnsPDF3D'][a,b,c,t],20)
    plt.colorbar()
    plt.show()
    
    #Joint PDF - 2 beta
    jbetaPDF =np.atleast_2d(betaPDF).T*zbetaPDF
    plt.contourf(centres[1],centres[0],jbetaPDF)    
    plt.colorbar()
    plt.show()
    
    #Joint PDF - copula
    if switch_copula:
        plt.contourf(centres[1],centres[0],cplPDF)
        plt.colorbar()
        plt.show()
        
    #Joint PDF-ANN
    plt.contourf(centres[1],centres[0],annPDF3D[a,b,c,t])
    plt.colorbar()
    plt.show()        
    
    '''######################################################
    ### Marginal PDF by integrating joint PDF plots
    ######################################################'''
    #1D PDF for c
    dnsPDF_c = np.trapz(dataDict['dnsPDF3D'][a,b,c,t],x=centres[1],axis=1)
    dnsPDF_c = dnsPDF_c/np.trapz(dnsPDF_c,x=centres[0]);
    plt.plot(centres[0],dnsPDF_c,label='DNS')
    
    plt.plot(centres[0],betaPDF,label='beta')
    
    if switch_copula:
        cplPDF_c = np.trapz(cplPDF,x=centres[1],axis=1)
        cplPDF_c = cplPDF_c/np.trapz(cplPDF_c,x=centres[0]);
        plt.plot(centres[0],cplPDF_c,label='copula')
    
    PDF_c = np.trapz(annPDF3D[a,b,c,t],x=centres[1],axis=1)
    PDF_c = PDF_c/np.trapz(PDF_c,x=centres[0])
    plt.plot(centres[0],PDF_c,label='ANN')
    
    plt.legend()
    plt.show()
    print('JSD-beta_c = %s' %np.mean(func.JSDiv(np.atleast_2d(dnsPDF_c),np.atleast_2d(betaPDF))),
          ' | JSD-ANN_c = %s' %np.mean(func.JSDiv(np.atleast_2d(dnsPDF_c),np.atleast_2d(PDF_c))))
    if switch_copula:
        print(' | JSD-copula_z = %s' %np.mean(func.JSDiv(np.atleast_2d(dnsPDF_c),np.atleast_2d(cplPDF_c))))
        
    #1D PDF for Z
    dnsPDF_z = np.trapz(dataDict['dnsPDF3D'][a,b,c,t],x=centres[0],axis=0)
    dnsPDF_z = dnsPDF_z/np.trapz(dnsPDF_z,x=centres[1]);
    plt.plot(centres[1],dnsPDF_z,label='DNS')
    
    plt.plot(centres[1],zbetaPDF,label='beta')
    
    if switch_copula:
        cplPDF_z = np.trapz(cplPDF,x=centres[0],axis=0)
        cplPDF_z = cplPDF_z/np.trapz(cplPDF_z,x=centres[1]);
        plt.plot(centres[1],cplPDF_z,label='copula')
    
    PDF_z = np.trapz(annPDF3D[a,b,c,t],x=centres[0],axis=0)
    PDF_z = PDF_z/np.trapz(PDF_z,x=centres[1])
    plt.plot(centres[1],PDF_z,label='ANN')
    
    plt.legend()
    plt.show()
    print('JSD-beta_z = %s' %np.mean(func.JSDiv(np.atleast_2d(dnsPDF_z),np.atleast_2d(zbetaPDF))),
          ' | JSD-ANN_z = %s' %np.mean(func.JSDiv(np.atleast_2d(dnsPDF_z),np.atleast_2d(PDF_z))))  
    if switch_copula:
        print(' | JSD-copula_z = %s' %np.mean(func.JSDiv(np.atleast_2d(dnsPDF_z),np.atleast_2d(cplPDF_z))))
    
    '''######################################################
    ###JSD stats
    ######################################################'''
    
    JSD_c_beta3D = np.zeros(annPDF3D.shape)
    JSD_c_cpl3D = np.zeros(annPDF3D.shape)
    JSD_c_ann3D = np.zeros(annPDF3D.shape)
    JSD_z_beta3D = np.zeros(annPDF3D.shape)
    JSD_z_cpl3D = np.zeros(annPDF3D.shape)
    JSD_z_ann3D = np.zeros(annPDF3D.shape)
    
    for t in range(No_ts):
        for a in range(inputSize1D):
            for b in range(inputSize1D):
                for c in range(inputSize1D):
                    
                    if(b+c == 0):
                        print(str(int((a+1)/inputSize1D*100)) + '%')
                                        
                    if switch_copula:
                        cplPDF = dataDict['cplPDF3D'][a,b,c,t]
                    
                    #1D PDF for c
                    dnsPDF_c = np.trapz(dataDict['dnsPDF3D'][a,b,c,t],x=centres[1],axis=1)
                    dnsPDF_c = dnsPDF_c/np.trapz(dnsPDF_c,x=centres[0])
                    betaPDF = np.trapz(dataDict['betaPDF3D'][a,b,c,t],x=centres[1],axis=1)
                    JSD_c_beta3D[a,b,c,t] = np.mean(func.JSDiv(np.atleast_2d(dnsPDF_c),np.atleast_2d(betaPDF)))
                    if switch_copula:
                        cplPDF_c = np.trapz(cplPDF,x=centres[1],axis=1)
                        cplPDF_c = cplPDF_c/np.trapz(cplPDF_c,x=centres[0])
                        JSD_c_cpl3D[a,b,c,t] = np.mean(func.JSDiv(np.atleast_2d(dnsPDF_c), np.atleast_2d(cplPDF_c)))
                    PDF_c = np.trapz(annPDF3D[a,b,c,t],x=centres[1],axis=1)
                    PDF_c = PDF_c/np.trapz(PDF_c,x=centres[0])
                    JSD_c_ann3D[a,b,c,t] = np.mean(func.JSDiv(np.atleast_2d(dnsPDF_c), np.atleast_2d(PDF_c)))
                    
                    #1D PDF for Z
                    dnsPDF_z = np.trapz(dataDict['dnsPDF3D'][a,b,c,t],x=centres[0],axis=0)
                    dnsPDF_z = dnsPDF_z/np.trapz(dnsPDF_z,x=centres[1])
                    zbetaPDF = np.trapz(dataDict['betaPDF3D'][a,b,c,t],x=centres[0],axis=0)
                    JSD_z_beta3D[a,b,c,t] = np.mean(func.JSDiv(np.atleast_2d(dnsPDF_z),np.atleast_2d(zbetaPDF)))
                    if switch_copula:
                        cplPDF_z = np.trapz(cplPDF,x=centres[0],axis=0)
                        cplPDF_z = cplPDF_z/np.trapz(cplPDF_z,x=centres[1])
                        JSD_z_cpl3D[a,b,c,t] = np.mean(func.JSDiv(np.atleast_2d(dnsPDF_z),np.atleast_2d(cplPDF_z)))
                    PDF_z = np.trapz(annPDF3D[a,b,c,t],x=centres[0],axis=0)
                    PDF_z = PDF_z/np.trapz(PDF_z,x=centres[1])
                    JSD_z_ann3D[a,b,c,t] = np.mean(func.JSDiv(np.atleast_2d(dnsPDF_z),np.atleast_2d(PDF_z)))
                    
    print('Javg-beta_c = %s' % (np.mean(JSD_c_beta3D*not_outlier3D)),
          ' \ Javg-ANN_c = %s' % (np.mean(JSD_c_ann3D*not_outlier3D)))
    print('J90-beta_c = %s' % (np.percentile(JSD_c_beta3D*not_outlier3D,90)),
          ' \ J90-ANN_c = %s' % (np.percentile(JSD_c_ann3D*not_outlier3D,90)))
    if switch_copula:
        print(' \ Javg-coula_c = %s' % (np.mean(JSD_c_cpl3D*not_outlier3D)),
              ' \ J90-coula_c = %s' % (np.percentile(JSD_c_cpl3D*not_outlier3D,90)))
    plt.hist((JSD_c_beta3D*not_outlier3D).flatten(),bins=np.linspace(0,0.5,20),label='beta')
    plt.hist((JSD_c_ann3D*not_outlier3D).flatten(),bins=np.linspace(0,0.5,20),label='ann')
    if switch_copula:
        plt.hist((JSD_c_cpl3D*not_outlier3D).flatten(),bins=np.linspace(0,0.5,20),label='copula')
    plt.legend()
    plt.show()
    
    print('Javg-beta_z = %s' % (np.mean(JSD_z_beta3D*not_outlier3D)),
          ' \ Javg-ANN_z = %s' % (np.mean(JSD_z_ann3D*not_outlier3D)) )
    print('J90-beta_z = %s' % (np.percentile(JSD_z_beta3D*not_outlier3D,90)),
          ' \ J90-ANN_z = %s' % (np.percentile(JSD_z_ann3D*not_outlier3D,90)) )
    if switch_copula:
        print(' \ Javg-coula_z = %s' % (np.mean(JSD_z_cpl3D*not_outlier3D)),
              ' \ J90-coula_z = %s' % (np.percentile(JSD_z_cpl3D*not_outlier3D,90)))
    plt.hist((JSD_z_beta3D*not_outlier3D).flatten(),bins=np.linspace(0,0.5,20),label='beta')
    plt.hist((JSD_z_ann3D*not_outlier3D).flatten(),bins=np.linspace(0,0.5,20),label='ann')
    if switch_copula:
        plt.hist((JSD_z_cpl3D*not_outlier3D).flatten(),bins=np.linspace(0,0.5,20),label='copula')
    plt.legend()
    plt.show()
    
    '''######################################################
    ### Reaction rate closure
    ######################################################'''
    conAvg_WcT = loadmat(fileDir + 'conAvg_WcT_all_' + case + '.mat')['conAvg_WcT_all']
    conAvg_RHO = loadmat(fileDir + 'conAvg_WcT_all_' + case + '.mat')['conAvg_RHO_all']
    WcT_dns = np.zeros(annPDF3D.shape)
    WcT_beta = np.zeros(annPDF3D.shape)
    WcT_cpl = np.zeros(annPDF3D.shape)
    WcT_ann = np.zeros(annPDF3D.shape)
    
    for t in range(No_ts):
        for a in range(inputSize1D):
            for b in range(inputSize1D):
                for c in range(inputSize1D):

                    if switch_copula:
                        integrad = np.multiply(conAvg_WcT/(conAvg_RHO+1e-4),t,dataDict['cplPDF3D'][a,b,c,t])
                        tmp = np.trapz(integrad,x=centres[0],axis=0)
                        WcT_cpl[a,b,c,t] = np.trapz(tmp,x=centres[1])*dataDict['RHO_flt'][a,b,c,t]
                    integrad = np.multiply(conAvg_WcT/(conAvg_RHO+1e-4),dataDict['betaPDF3D'][a,b,c,t])
                    tmp = np.trapz(integrad,x=centres[0],axis=0)
                    WcT_beta[a,b,c,t] = np.trapz(tmp,x=centres[1])*dataDict['RHO_flt'][a,b,c,t]
                    integrad = np.multiply(conAvg_WcT/(conAvg_RHO+1e-4),annPDF3D[a,b,c,t])
                    tmp = np.trapz(integrad,x=centres[0],axis=0)
                    WcT_ann[a,b,c,t] = np.trapz(tmp,centres[1])*dataDict['RHO_flt'][a,b,c,t]
                    integrad = np.multiply(conAvg_WcT/(conAvg_RHO+1e-4),dataDict['dnsPDF3D'][a,b,c,t])
                    tmp = np.trapz(integrad,x=centres[0],axis=0)
                    WcT_dns[a,b,c,t] = np.trapz(tmp,x=centres[1])*dataDict['RHO_flt'][a,b,c,t]
                    
    RMSE1 = 0.;
    if switch_copula:
        RMSE1 = sqrt(np.mean(((WcT_cpl - WcT_dns)/(WcT_dns+10.))**2*not_outlier3D))
        plt.scatter(WcT_dns.flatten(),WcT_cpl.flatten(),1,label='copula')
    RMSE2 = sqrt(np.mean(((WcT_ann - WcT_dns)/(WcT_dns+10.))**2*not_outlier3D))
    plt.scatter((WcT_dns*not_outlier3D).flatten(),(WcT_ann*not_outlier3D).flatten(),1,label='ann')
    RMSE3 = sqrt(np.mean(((WcT_beta - WcT_dns)/(WcT_dns+10.))**2*not_outlier3D))
    plt.scatter((WcT_dns*not_outlier3D).flatten(),(WcT_beta*not_outlier3D).flatten(),1,label='beta')
    plt.title('RMSE ANN:' + str('{:.2f}'.format(RMSE2)) 
                + ' | beta:' + str('{:.2f}'.format(RMSE3)))
    plt.legend()
    plt.show()
    
    #contour plots
    plt.contourf(np.squeeze(dataDict['WcT_flt'][int(inputSize1D/2),:,:,0]))
    plt.title('flt')
    plt.clim(0,1000)
    plt.show()
    
    plt.contourf(np.squeeze(WcT_dns[int(inputSize1D/2),:,:,0]))
    plt.title('dns')
    plt.clim(0,1000)
    plt.show()
    
    plt.contourf(np.squeeze(WcT_beta[int(inputSize1D/2),:,:,0]))
    plt.title('beta')
    plt.clim(0,1000)
    plt.show()
    
    if switch_copula:
        plt.contourf(np.squeeze(WcT_cpl[int(inputSize1D/2),:,:,0]))
        plt.title('copula')
        plt.clim(0,1000)
        plt.show()
    
    plt.contourf(np.squeeze(WcT_ann[int(inputSize1D/2),:,:,0]))
    plt.title('ann')
    plt.clim(0,1000)
    plt.show()    
    
    
# %%
if __name__ == "__main__":
    for caseIdx in [1]:
    # for caseIdx in range(3+1)[1:]:
        caseAll = ['AZ1','AZ2','BZ1']
        case = caseAll[caseIdx-1]
        fileDir = './DNS_data/'
        filterSizeAll = np.array([[40,80,120],[40,80,120],[74,148]])
        for filterIdx in [1]:
        # for filterIdx in range(3+1)[1:]:
            filterSize = filterSizeAll[caseIdx-1][filterIdx-1]
            
            datDic = data_extract(filterSize)
            
            DNN(datDic['X_zc'],datDic['target_zc'])
            
            Post_processing(fileDir,case,datDic)