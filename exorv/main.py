#!/usr/bin/python3
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from forecaster import forecaster3 as mr
plt.rcParams['lines.linewidth']=0.2

def MX_offsets(rv_list,bjd,c):
    if c<2:
        for dd in bjd:
            ll = np.where(bjd == dd)
            # 09/2020: 
            if 2458900<dd<2459149:
                if c==0: # red
                    rv_list[ll]+=13.28
                elif c==1: # blue
                    rv_list[ll]+=10.94
            # 11/2020: 
            elif 2459150<dd<2459240:
                if c==0:  
                    rv_list[ll]+=10.4
                elif c==1: 
                    rv_list[ll]+=9.09  
            # 02/2021: 
            elif 2459246<dd<2459300: 
                if c==0:  
                    rv_list[ll]+=16.08
                elif c==1: 
                    rv_list[ll]+=12.92
            # 04/2021: 
            elif 2459305<dd<2459335: 
                if c==0:  
                    rv_list[ll]+=8.92 
                    rv_list[ll]+= 0.239 # additional fit offset error from Juliet
                elif c==1: 
                    rv_list[ll]+=7.77 
                    rv_list[ll]+= 0.649
            # 05/2021: 
            elif 2459350<dd<2459380: 
                if c==0:  
                    rv_list[ll]+= 9.21 
                    rv_list[ll]-= 3.316
                elif c==1: 
                    rv_list[ll]+= 8.4 
                    rv_list[ll]-= 3.063
            # 08/2021: 
            elif 2459430<dd<2459460: 
                if c==0:  
                    rv_list[ll]+=4.74 
                    rv_list[ll]+= 0.142
                elif c==1: 
                    rv_list[ll]+=3.59 
                    rv_list[ll]+= 0.111
            # 10/2021: 
            elif 2459510<dd<2459550: 
                if c==0:  
                    rv_list[ll]+=3.49 
                    rv_list[ll]+= 0.176
                elif c==1: 
                    rv_list[ll]+=1.44 
                    rv_list[ll]-= 0.217
            # 03/2022: 
            elif 2459655<dd<2459700:
                if c==0:  
                    rv_list[ll]-=0.67 
                    rv_list[ll]+= 0.305
                elif c==1: 
                    rv_list[ll]-=1.48 
                    rv_list[ll]+= 1.561
            # 05/2022: 
            elif 2459720<dd<2459750:
                if c==0:  
                    rv_list[ll]-= 3.04 
                    rv_list[ll]-= 0.436
                elif c==1: 
                    rv_list[ll]-= 1.51 
                    rv_list[ll]-= 1.284
            # 07/2022: 
            elif 2459765<dd<2459810:
                if c==0:  
                    rv_list[ll]-=3.72 
                    rv_list[ll]+= 0.462
                elif c==1: 
                    rv_list[ll]-=2.54 
                    rv_list[ll]+= 0.568
            # 06/2023: 
            elif 2460115<dd<2460165: 
                if c==0:  
                    rv_list[ll]-=4.0 
                    rv_list[ll]+= 0.386
                elif c==1: 
                    rv_list[ll]-=2.53 
                    rv_list[ll]-= 0.483
            # 10/2023: 
            elif 2460218<dd<2460248: 
                if c==0:  
                    rv_list[ll]-=11.88
                elif c==1: 
                    rv_list[ll]-=10.82
            # 11/2023: 
            elif 2460250<dd<2460279: 
                if c==0:  
                    rv_list[ll]-=11.26
                elif c==1: 
                    rv_list[ll]-=11.52
            # 12/2023: 
            elif 2460280<dd<2460320: 
                if c==0:  
                    rv_list[ll]-=12.79
                elif c==1: 
                    rv_list[ll]-=11.8
            else:
                print('error: date %s out of range'%dd) 
    return rv_list    

def loadparams(TOI = '', others={}):
    '''
    Load in planet parameters.

    Args:
        TOI (float): TOI (target of interest) ID as named by TESS
        others (dict): input for additional planet parameters, or alternate planet parameters. Requires P (period), t0 (epoch), r (planet radius), and m_s (stellar mass), with planet specified as '_pn'.
            ex: others = {'P_p1':10., 't0_p1':5., 'r_p1':1., 'm_s':15.}
        
    Returns:
        params (dict): the final set of parameters that will be used 
    '''

    params = {}
    necessary_params = ['P','t0','r','m_s']
    
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
                params['r_p'+str(n)] = exofop_p['Planet Radius (R_Earth)']
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
            params[param_others] = others[param_others]
            
    if (len(params.keys())-1)%3 != 0:
        print("Missing parameter in dictionary 'params'. Please review:",params)
        sys.exit()
            
    return params

# Create signal subtracted plots
# 07/17: this section needs to be updated to be compatible with exoFOP

def phasefold(TOI = '', others = {},
              instruments = ['MX_R','MX_B'], paths = [],
              order = [], sigsub = False):
    '''
    Plotting phase-folded and signal-subtracted data.

    Args:
        name: TOI number
        params: additional planets and their parameters outside of ExoFOP data; leave blank if all planets of interest are already listed in ExoFOP
        instruments: list of instruments
        paths: paths to user-uploaded RV data; must include RV, err_RV, and BJD
        order: order for plotting and/or signal-subtracting planet signals
        sigsub: whether or not to plot with signal-subtraction 
    
    Returns:
        plotted figures of signal-subtracted data
    '''
    
    # Import measured RV's, BJD's, and errors: --- can i turn this into an ssh login???
    pathlist = [Path('/Users/ninabrown/Documents/bean_team/local_data/rv_csv/TOI-2142').glob('**/*_r.csv'), Path('/Users/ninabrown/Documents/bean_team/local_data/rv_csv/TOI-2142').glob('**/*_b.csv')]
    pathcopy = [Path('/Users/ninabrown/Documents/bean_team/local_data/rv_csv/TOI-2142').glob('**/*_r.csv'), Path('/Users/ninabrown/Documents/bean_team/local_data/rv_csv/TOI-2142').glob('**/*_b.csv')]

    plt_ct = 0
    residuals = []
    
    # Load ExoFOP data:
    params = loadparams(TOI, others)
    n_p = int((len(params)-1)/3) # number of planets
    if order == []:
        order = range(1,n_p+1)

    fig, axs = plt.subplots(n_p*2,1,figsize=(12,7*n_p))

    for c in range(len(instruments)):
        if list(pathcopy[c]):
            rv_list,bjd,erv,rv,K=[],[],[],[],[]

            # Import data:
            for path in pathlist[c]:
                p = str(path)   
                if c<2:
                    rv_csv=pd.read_csv(p)
                else:
                    rv_csv=pd.read_csv(p,sep=' ',header=None, names=['bjd','rv','e_rv'], usecols=[0,1,2])
                    
                print(type(rv_list), type(rv_csv['rv'].values[:]))
                    
                rv_list=np.concatenate([rv_list,rv_csv['rv'].values[:]])
                bjd=np.concatenate([bjd,rv_csv['bjd'].values[:]])
                erv=np.concatenate([erv,rv_csv['e_rv'].values[:]])   
            rv_list = MX_offsets(rv_list,bjd,c) # add offsets for MAROON-X

            #rv_list-=c_off[0][c]
            #erv+=c_off[1][c]

            run4 = np.where((2459305<np.array(bjd))&(np.array(bjd)<2459335))[0]
            run5 = np.where((2459350<np.array(bjd))&(np.array(bjd)<2459380))[0]
            run6 = np.where((2459430<np.array(bjd))&(np.array(bjd)<2459460))[0]
            run7 = np.where((2459510<np.array(bjd))&(np.array(bjd)<2459550))[0]
            run8 = np.where((2459655<np.array(bjd))&(np.array(bjd)<2459700))[0]
            run9 = np.where((2459720<np.array(bjd))&(np.array(bjd)<2459750))[0]
            run10 = np.where((2459765<np.array(bjd))&(np.array(bjd)<2459810))[0]
            run11 = np.where((2460115<np.array(bjd))&(np.array(bjd)<2460165))[0]

            if sigsub == True:
                rv_sub = rv_list
                rv_exp_sum = np.zeros(len(rv_list))

            ct = []
            for ii in order:
                rv_exp = []
                print('\nPlanet %s (%sd)(%s):'%(ii,params['P_p'+str(ii)],instruments[c]))
                mrv,merv,mbjd=[],[],[]
                
                # Estimate mass using Forecaster by Chen & Kipping (2016):
                m, m_plus, m_min = mr.Rstat2M(mean=params['r_p'+str(ii)], std=0.01, unit='Earth', sample_size=1000, grid_size=1e3, classify='Yes')
                print(f'Planet mass estimated at {m:.2f} +{m_plus:.2f} -{m_min:.2f} Earth mass')


                # Calculate RV amplitude:
                K.append(28.4 * (m/317.83) * ((365.25/params['P_p'+str(ii)])**(1/3)) * ((params['m_s'])**(-2/3))) # 28.4 [m/s] = 2piG^1/3; 317.83 = MJ/ME; 365.25 = Yr/day
                print(f'Calculated RV semi-amplitude of {K[-1]:.2f} m s-1')

                # Get RV curve (let e = 0 and w = pi/2):
                tpi = 2 * np.pi
                phase = 1
                Tp = params['t0_p'+str(ii)] - (phase * params['P_p'+str(ii)])  # time of periastron
                dates = np.linspace(min(bjd)-1000,max(bjd)+1000,10000) # set desired date range
                ma = (tpi/params['P_p'+str(ii)]) * (dates - Tp) # mean anomaly
                ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
                rv.append(K[-1]*np.cos(np.pi/2 + ta))

                # Plot data per planet as phase-fold:
                rx = ((params['t0_p'+str(ii)]-bjd)/params['P_p'+str(ii)])%1 
                
                if sigsub == True:
                    rv_plot = rv_sub
                    Ps = [params['P_p'+str(cc)] for cc in ct]
                    title = '+ subtracted %sd signal(s)'%(Ps)
                    for cc in ct : print(params['P_p'+str(cc)]) 
                else:
                    rv_plot = rv_list
                    title = ''

                axs[plt_ct].errorbar(rx[run4], rv_plot[run4], yerr=erv[run4], fmt="o", color='red', label='04/2021')
                axs[plt_ct].errorbar(rx[run5], rv_plot[run5], yerr=erv[run5], fmt="o", color='orange', label='05/2021')
                axs[plt_ct].errorbar(rx[run6], rv_plot[run6], yerr=erv[run6], fmt="o", color='gold', label='08/2021')
                axs[plt_ct].errorbar(rx[run7], rv_plot[run7], yerr=erv[run7], fmt="o", color='yellowgreen', label='10/2021')
                axs[plt_ct].errorbar(rx[run8], rv_plot[run8], yerr=erv[run8], fmt="o", color='cyan', label='03/2022')
                axs[plt_ct].errorbar(rx[run9], rv_plot[run9], yerr=erv[run9], fmt="o", color='tab:blue', label='05/2022')
                axs[plt_ct].errorbar(rx[run10], rv_plot[run10], yerr=erv[run10], fmt="o", color='tab:purple', label='07/2022')
                axs[plt_ct].errorbar(rx[run11], rv_plot[run11], yerr=erv[run11], fmt="o", color='m', label='06/2023')
                axs[plt_ct].plot(np.linspace(0,1,10000),K[-1]*np.sin(np.linspace(0,2*np.pi,10000)), label = 'Est. curve', color='k', linewidth=0.7)
                axs[plt_ct].set_title('Estimated sine curve vs. real data\nPhase folded %s\n%s %s arm'%(title, TOI, instruments[c]))
                axs[plt_ct].set_xlabel('Period = %s days' %(params['P_p'+str(ii)]))
                axs[plt_ct].set_ylabel('RV [m/s]')
                axs[plt_ct].legend(bbox_to_anchor=(1., 1.05))
                plt_ct+=1

                ct.append(ii)

                if sigsub == True:
                    # Signal subtract: 
                    for jj in range(len(rv_list)):
                        ma = (tpi/params['P_p'+str(ii)]) * (bjd[jj] - Tp) # mean anomaly ---- double and triple check this
                        ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
                        rv_exp.append(K[-1]*np.cos(np.pi/2 + ta))

                    rv_sub = rv_list - rv_exp

    fig.tight_layout()
    plt.show()

phasefold(TOI = 2142, others = {'P_p2':38.5, 'r_p2': 2.9, 't0_p2': 2459024.970202545},  order = [2,1],sigsub = True)
                    
