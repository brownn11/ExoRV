#!/usr/bin/python3
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from forecaster import forecaster3 as mr
plt.rcParams['lines.linewidth']=0.4
plt.rcParams.update({'font.size': 8})
import os

def MX_offsets(rv_list,bjd,c):
    '''
    Apply offsets as calculated in Basant et al. (2025).

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
                if c==0: # red
                    rv_list[ll]+=12.08
                elif c==1: # blue
                    rv_list[ll]+=10.71
            # 11/2020: 
            elif 2459150<dd<2459240:
                if c==0:  
                    rv_list[ll]+=8.56
                elif c==1: 
                    rv_list[ll]+=8.68  
            # 02/2021: 
            elif 2459246<dd<2459300: 
                if c==0:  
                    rv_list[ll]+=10.2
                elif c==1: 
                    rv_list[ll]+=9.38
            # 04/2021: 
            elif 2459305<dd<2459335: 
                if c==0:  
                    rv_list[ll]+=8.44 
                elif c==1: 
                    rv_list[ll]+=7.27 
            # 05/2021: 
            elif 2459350<dd<2459380: 
                if c==0:  
                    rv_list[ll]+= 6.74 
                elif c==1: 
                    rv_list[ll]+= 6.04 
            # 08/2021: 
            elif 2459430<dd<2459460: 
                if c==0:  
                    rv_list[ll]+=3.98 
                elif c==1: 
                    rv_list[ll]+=3.04 
            # 10/2021: 
            elif 2459510<dd<2459550: 
                if c==0:  
                    rv_list[ll]+=1.96 
                elif c==1: 
                    rv_list[ll]+=1.17 
            # 03/2022: 
            elif 2459655<dd<2459700:
                if c==0:  
                    rv_list[ll]-=1.94 
                elif c==1: 
                    rv_list[ll]-=1.95 
            # 05/2022: 
            elif 2459720<dd<2459750:
                if c==0:  
                    rv_list[ll]-= 2.95 
                elif c==1: 
                    rv_list[ll]-= 1.67 
            # 07/2022: 
            elif 2459765<dd<2459810:
                if c==0:  
                    rv_list[ll]-=4.56 
                elif c==1: 
                    rv_list[ll]-=2.57 
            # 06/2023: 
            elif 2460115<dd<2460165: 
                if c==0:  
                    rv_list[ll]-=4.33 
                elif c==1: 
                    rv_list[ll]-=2.73 
            # 10/2023: 
            elif 2460218<dd<2460248: 
                if c==0:  
                    rv_list[ll]-=12.29
                elif c==1: 
                    rv_list[ll]-=10.52
            # 11/2023: 
            elif 2460250<dd<2460279: 
                if c==0:  
                    rv_list[ll]-=13.21
                elif c==1: 
                    rv_list[ll]-=10.96
            # 12/2023: 
            elif 2460280<dd<2460320: 
                if c==0:  
                    rv_list[ll]-=13.62
                elif c==1: 
                    rv_list[ll]-=11.15
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
    if TOI != 0: 
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
                params['e_p'+str(n)] = 0
                params['w_p'+str(n)] = 90
                n += 1
            except:
                print(f'Loaded {n-1} planet(s).')
                break
    
    # If there are planet parameters provided:
    if bool(others): 
        # Works whether or not a TOI was provided. Will write over any ExoFOP-sourced values. 
        print('Using user-uploaded parameters...')

        for param_others in others:
            params[param_others] = float(others[param_others])
        
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
            except:
                pass

        # Add eccentricity and/or omega if not provided:
        for nn in range(n):
            nn+=1
            try:
                if params['e_p'+str(nn)]:
                    pass
            except:
                params['e_p'+str(nn)] = 0
            try:
                if params['w_p'+str(nn)]:
                    pass
            except:
                params['w_p'+str(nn)] = 0

            
    if (len(params.keys())-1)%5 != 0:
        print("Missing parameter in dictionary 'params'. Requires: P [d], epoch [BJD], r [r_E] OR m [m_E], e, w, and m_s [m_S]. \n Please review:",params)
        sys.exit()
            
    print('Usings params:',params)
    return params

def loaddata_mx(TOI = '', tn = '', servalfolder = ''):
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

def loaddata(path = ''):
    # set tn if TOI target is used -- simplifies things a bit:

    remotehost = input('Remote host: ')
    localfile = './'+path.split('/')[-1]

    os.system('scp "%s:%s" "%s"' % (remotehost, path, localfile) )
