#!/usr/bin/python3
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from forecaster import forecaster3 as mr
import glob
plt.rcParams['lines.linewidth']=0.4
plt.rcParams.update({'font.size': 8})
import os

def MX_offsets(rv_list,bjd,c):
    '''
    Apply offsets as calculated by Ritvik Basant.

    Args:
        rv_list (array): array of RV data points
        bjd (array): array of dates in bjd format
        c (int): instrument indicator (blue = 0, red = 1)
    '''
    if c<2:
        for dd in bjd:
            ll = np.where(bjd == dd)
            # 09/2020: 
            if 2458900<dd<2459149:
                if c==1: # red
                    rv_list[ll]+=13.28
                elif c==0: # blue
                    rv_list[ll]+=10.94
            # 11/2020: 
            elif 2459150<dd<2459240:
                if c==1:  
                    rv_list[ll]+=10.4
                elif c==0: 
                    rv_list[ll]+=9.09  
            # 02/2021: 
            elif 2459246<dd<2459300: 
                if c==1:  
                    rv_list[ll]+=16.08
                elif c==0: 
                    rv_list[ll]+=12.92
            # 04/2021: 
            elif 2459305<dd<2459335: 
                if c==1:  
                    rv_list[ll]+=8.92 
                elif c==0: 
                    rv_list[ll]+=7.4 
            # 05/2021: 
            elif 2459350<dd<2459380: 
                if c==1:  
                    rv_list[ll]+= 7.4 
                elif c==0: 
                    rv_list[ll]+= 6.5 
            # 08/2021: 
            elif 2459430<dd<2459460: 
                if c==1:  
                    rv_list[ll]+=4.61 
                elif c==0: 
                    rv_list[ll]+=3.47 
            # 10/2021: 
            elif 2459510<dd<2459550: 
                if c==1:  
                    rv_list[ll]+=3.2 
                elif c==0: 
                    rv_list[ll]+=1.61 
            # 03/2022: 
            elif 2459655<dd<2459700:
                if c==1:  
                    rv_list[ll]-=1.28 
                elif c==0: 
                    rv_list[ll]-=2.53 
            # 05/2022: 
            elif 2459720<dd<2459750:
                if c==1:  
                    rv_list[ll]-= 2.67 
                elif c==0: 
                    rv_list[ll]-= 1.29 
            # 07/2022: 
            elif 2459765<dd<2459810:
                if c==1:  
                    rv_list[ll]-=3.74 
                elif c==0: 
                    rv_list[ll]-=2.43 
            # 06/2023: 
            elif 2460115<dd<2460165: 
                if c==1:  
                    rv_list[ll]-=4.08 
                elif c==0: 
                    rv_list[ll]-=2.48 
            # 10/2023: 
            elif 2460218<dd<2460248: 
                if c==1:  
                    rv_list[ll]-=11.98
                elif c==0: 
                    rv_list[ll]-=10.83
            # 11/2023: 
            elif 2460250<dd<2460279: 
                if c==1:  
                    rv_list[ll]-=12.02
                elif c==0: 
                    rv_list[ll]-=11.11
            # 12/2023: 
            elif 2460280<dd<2460320: 
                if c==1:  
                    rv_list[ll]-=12.32
                elif c==0: 
                    rv_list[ll]-=11.7
            else:
                print('error: date %s out of range'%dd) 
    return rv_list    

def loadparams(TOI = '', others={}):
    '''
    Load in planet parameters from ExoFOP.

    Args:
        TOI (float): TOI (target of interest) ID as named by TESS
        others (dict): input for additional planet parameters, or alternate planet parameters. Requires P (period [d]), t0 (epoch [BJD]), r (planet radius [earth rad]) OR m (planet mass [earth mass]), and m_s (stellar mass [sol mass]), with planet specified as '_pn'.
            ex: others = {'P_p1':10., 't0_p1':5., 'r_p1':1., 'm_s':15.}
        
    Returns:
        params (dict): the final set of parameters that will be used 
    '''

    params = {}
    
    # Load ExoFOP data if available
    if TOI != '': 
        print('TOI not empty, loading in ExoFOP data...')
        exofop_table=pd.read_csv('https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe', delimiter='|',index_col=1)
        n = 1
        # Iterate over number of planets:
        while True:
            print(f'Attempting TOI-{TOI+(0.01*n)}...')
            try:
                exofop_p=exofop_table.loc[TOI+(0.01*n)]
                params['P_p'+str(n)] = exofop_p['Period (days)']
                params['t0_p'+str(n)] = exofop_p['Epoch (BJD)']
                params['r_p'+str(n)] = exofop_p['Planet Radius (R_Earth)'] # ExoFOP doesn't list planet masses -- even confirmed masses :(
                params['m_s'] = exofop_p['Stellar Mass (M_Sun)'] # rewrites stellar mass for each planet -- probably not very efficient, but it works!
                n += 1
            except:
                print(f'Loaded {n-1} planet(s).')
                break
    
    # If there are planet parameters provided:
    if bool(others): 
        # Works whether or not a TOI was provided. Will write over any ExoFOP-sourced values. 
        print('Using user-uploaded parameters...')

        for param_others in others:
            params[param_others] = int(others[param_others])
        
        # Count up number of planets:
        n = 0
        for param in params:
            if 'P_p' in param:
                n += 1
        
        # Remove radius values if both mass and radius provided:
        for nn in range(n):
            nn +=1
            try:
                if params['m_p'+str(nn)] and params['r_p'+str(nn)]: 
                    del params['r_p'+str(nn)]
                    n += 1
            except:
                pass
            
    if (len(params.keys())-1)%3 != 0:
        print("Missing parameter in dictionary 'params'. Requires: P [d], epoch [BJD], r [r_E] OR m [m_E], and m_s [m_S]. \n Please review:",params)
        sys.exit()
            
    return params

def loaddata(TOI = '', tn = '', servalfolder = ''):
    # set tn if TOI target is used -- simplifies things a bit:
    if tn == '' and TOI != '':
        tn = 'TOI-'+str(TOI)

    remotehost = input('Remote host: ')
    localfile_r = './'+tn+'_r.csv'
    localfile_b = './'+tn+'_b.csv'
    if tn == '' and TOI!='':
        remotefile_r = '/home/maroonx/serval3/'+servalfolder+'/TOI?'+str(TOI)+'/MAROONXredcoadd/TOI?'+str(TOI)+'_rv_unbin.csv'
        remotefile_b = '/home/maroonx/serval3/'+servalfolder+'/TOI?'+str(TOI)+'/MAROONXbluecoadd/TOI?'+str(TOI)+'_rv_unbin.csv'
    else:
        remotefile_r = '/home/maroonx/serval3/'+servalfolder+'/'+tn+'/MAROONXredcoadd/'+tn+'_rv_unbin.csv'
        remotefile_b = '/home/maroonx/serval3/'+servalfolder+'/'+tn+'/MAROONXbluecoadd/'+tn+'_rv_unbin.csv'
    os.system('scp "%s:%s" "%s"' % (remotehost, remotefile_r, localfile_r) )
    os.system('scp "%s:%s" "%s"' % (remotehost, remotefile_b, localfile_b) )