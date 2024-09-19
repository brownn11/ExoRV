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

import load as l
colors = ['tab:blue','tab:red']

def curve_only(TOI = '', others = {}):
    '''
    Plotting expected RV sine curve. Does not plot any RV data points.

    Args:
        TOI (float): TOI number
        others (dict): input for additional planet parameters, or alternate planet parameters. Requires P (period [d]), t0 (epoch [BJD]), r (planet radius [earth rad]) OR m (planet mass [earth mass]), and m_s (stellar mass [sol mass]), with planet specified as '_pn'.
                    ex: others = {'P_p1':10., 't0_p1':5., 'r_p1':1., 'm_s':15.}    
    Returns:
        plot: plotted figure of expected RV sine curve
        K (float): expected semi-major amplitude 
    '''
    
    # Load ExoFOP data:
    params = l.loadparams(TOI, others)
    n_p = int((len(params)-1)/3) # number of planets

    # Build plot and initiate rv, K lists:
    fig, axs = plt.subplots(n_p+1,1,figsize=(6,3*n_p))
    plt_ct = 0
    rv,K=[],[]

    for ii in range(1,n_p+1):
        print('\nPlanet %s (%sd):'%(ii,params['P_p'+str(ii)]))
        
        try:
            m = params['m_p'+str(ii)]
        except:
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
        dates = np.linspace(2460200,2460500,10000) # set random date range
        ma = (tpi/params['P_p'+str(ii)]) * (dates - Tp) # mean anomaly
        ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
        rv.append(K[-1]*np.cos(np.pi/2 + ta)) 

        axs[plt_ct].plot(np.linspace(0,1,10000),K[-1]*np.sin(np.linspace(0,2*np.pi,10000)), label = 'Est. curve', color='k', linewidth=0.7)
        axs[plt_ct].set_title('Estimated sine curve vs. real data\nPhase folded %s'%(TOI))
        axs[plt_ct].set_xlabel('Period = %s days' %(params['P_p'+str(ii)]))
        axs[plt_ct].set_ylabel('RV [m/s]')
        axs[plt_ct].legend(bbox_to_anchor=(1., 1.05))
        plt_ct+=1

    if n_p>1:
        axs[-1].plot(dates, np.sum(rv,axis=0), label = 'Est. curve', color='k', linewidth=0.7)
    else: 
        axs[-1].plot(dates, rv[0], label = 'Est. curve', color='k', linewidth=0.7) 

    fig.tight_layout()
    plt.show()

    return K

def points_only(TOI = '', tn = '',
              path = '', scp_MX = False):
    '''
    Plotting RV data points. Does not plot RV expected curve.

    Args:
        TOI (float): TOI (target of interest) ID as named by TESS      
        tn (str): target name, if not a TESS target
        instruments (list, str): list of instrument names; MUST be in same order as local data when sorted alphabetically 
        path (str): path to user-uploaded RV data, default to local folder; must include RV, err_RV, and BJD
        scp_download (bool): scp download RV data points from remote host
        scp_MX (bool): scp download RV data points from remote host; tailored to MAROON-X

    
    Returns:
        Plotted figures of phase-folded data, as well as data over all time. Plots RV data points over an expected RV sine curve. 
    '''

    instruments = ['MX_B','MX_R']
    
    # Initialize a plot counter
    plt_ct = 0

    # set tn if TOI target is used -- simplifies things a bit:
    if tn == '' and TOI != '':
        tn = 'TOI-'+str(TOI)

    # Find matching files in specified path:
    paths = glob.glob(path+'*'+str(tn)+'*.csv')
    if len(paths)==0:
        print('Error: path empty, check target name or file location\n---> Attempted path:',path,'*',str(tn),'*.csv')

    paths.sort()
    print('Using files:',paths)

    # Initialize plot:
    fig, axs = plt.subplots(len(instruments),1,figsize=(8,3*len(instruments)))

    for c in range(len(instruments)): # Iterate over instruments:
        if list(paths[c]): # Confirm path exists:
            rv_list,bjd,erv=[],[],[]
            p = str(paths[c])  

            # Read in data: 
            if c<2:
                rv_csv=pd.read_csv(p)
            else:
                rv_csv=pd.read_csv(p,sep=' ',header=None, names=['bjd','rv','e_rv'], usecols=[0,1,2])
            rv_list=np.array(rv_csv['rv'].values[:])
            bjd=np.array(rv_csv['bjd'].values[:])
            erv=np.array(rv_csv['e_rv'].values[:])  

            # Add MAROON-X offsets:
            if ('MX' in instruments[c]) and (min(bjd) < 2460313.):
                rv_list = l.MX_offsets(rv_list,bjd,c) 
                
            # Plot all data and expected RV sine curve per instrument:
            axs[plt_ct].errorbar(bjd, rv_list, yerr=erv, fmt='.', color=colors[c], label = 'all data')
            axs[plt_ct].set_xlim(min(bjd)-10,max(bjd)+10)
            axs[plt_ct].set_title('All data -- %s -- %s arm'%(tn, instruments[c]))
            plt_ct+=1
    axs[0].legend(bbox_to_anchor=(1., 1.05))
    fig.tight_layout()
    plt.show()

def RV_plotter(TOI = '', others = {}, tn = '',
              path = '',
              order = [], sigsub = False, 
              scp_download = False, scp_MX = False):
    '''
    Plots phase-folded and signal-subtracted data. Uses both RV data points and expected curve.

    Args:
        TOI (float): TOI (target of interest) ID as named by TESS
        others (dict): input for additional planet parameters, or alternate planet parameters. Requires P (period [d]), t0 (epoch [BJD]), r (planet radius [earth rad]) OR m (planet mass [earth mass]), and m_s (stellar mass [sol mass]), with planet specified as '_pn'.
            ex: others = {'P_p1':10., 't0_p1':5., 'r_p1':1., 'm_s':15.}       
        tn (str): target name, if not a TESS target
        instruments (list, str): list of instrument names; MUST be in same order as local data when sorted alphabetically 
        path (str): path to user-uploaded RV data, default to local folder; must include RV, err_RV, and BJD
        order (list, int): order for plotting and/or signal-subtracting planet signals
        sigsub (bool): whether or not to plot with signal-subtraction 
        scp_download (bool): scp download RV data points from remote host
        scp_MX (bool): scp download RV data points from remote host; tailored to MAROON-X
    
    Returns:
        Plotted figures of phase-folded data, as well as data over all time. Plots RV data points over an expected RV sine curve. 
    '''
    
    instruments = ['MX_B','MX_R']

    # Initialize a plot counter
    plt_ct = 0
    residuals = []

    # set tn if TOI target is used -- simplifies things a bit:
    if tn == '' and TOI != '':
        tn = 'TOI-'+str(TOI)
    
    # Find matching files in specified path:
    paths = glob.glob(path+'*'+str(tn)+'*.csv')
    if len(paths)==0:
        print('Error: path empty, check target name or file location\n---> Attempted path:',path,'*',str(tn),'*.csv')
        exit
    
    paths.sort()
    print('Using files:',paths)
    
    # Load ExoFOP data:
    params = l.loadparams(TOI, others)
    n_p = int((len(params)-1)/3) # number of planets

    # Automate order if unspecified:
    if order == []:
        order = range(1,n_p+1) 

    # Initialize plot:
    fig, axs = plt.subplots(n_p+2,len(instruments),figsize=(10,6*n_p))

    for c in range(len(instruments)): # Iterate over instruments:
        if list(paths[c]): # Confirm path exists:
            rv_list,bjd,erv,rv,K,ct=[],[],[],[],[],[]
            p = str(paths[c])  
            plt_ct = 0 # reset plot counter

            # Read in data: 
            if c<2:
                rv_csv=pd.read_csv(p)
            else:
                rv_csv=pd.read_csv(p,sep=' ',header=None, names=['bjd','rv','e_rv'], usecols=[0,1,2])
            rv_list=np.array(rv_csv['rv'].values[:])
            bjd=np.array(rv_csv['bjd'].values[:])
            erv=np.array(rv_csv['e_rv'].values[:])  

            # Add MAROON-X offsets:
            if ('MX' in instruments[c]) and (min(bjd) < 2460313.):
                rv_list = l.MX_offsets(rv_list,bjd,c) 

            # Initialize dummy arrays for signal subtraction:
            if sigsub == True:
                rv_sub = rv_list
                rv_exp_sum = np.zeros(len(rv_list))

            rv_exp_multisig = 0 # expected RV, does not reset per signal
            for ii in order: # Iterate over expected planets:
                rv_exp_onesig = [] # expected RV, resets per signal
                print('\nPlanet %s (%sd)(%s):'%(ii,params['P_p'+str(ii)],instruments[c]))
                
                try:
                    m = params['m_p'+str(ii)]
                except:
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
                dates = np.linspace(min(bjd)-100,max(bjd)+100,10000) # set desired date range
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

                print(plt_ct,c)
                axs[plt_ct,c].errorbar(rx, rv_plot, yerr=erv, fmt=".", color=colors[c], label='all data')
                axs[plt_ct,c].plot(np.linspace(0,1,10000),K[-1]*np.sin(np.linspace(0,2*np.pi,10000)), label = 'Est. curve', color='k', linewidth=0.7)
                axs[plt_ct,c].set_title('Phase folded %s; %s %s arm'%(title, tn, instruments[c]))
                axs[plt_ct,c].set_xlabel('Period = %s days' %(params['P_p'+str(ii)]))
                axs[plt_ct,c].set_ylabel('RV [m/s]')

                plt_ct+=1
                ct.append(ii) # track which planets have been plotted

                if sigsub == True:
                    # Signal subtract: 
                    #for jj in range(len(rv_list)):
                    ma = (tpi/params['P_p'+str(ii)]) * (bjd - Tp) 
                    ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
                    rv_exp_onesig = (K[-1]*np.cos(np.pi/2 + ta)) #.append(K[-1]*np.cos(np.pi/2 + ta))
                    rv_exp_multisig += rv_exp_onesig
                    print(rv_exp_multisig)

                    rv_sub = rv_list - rv_exp_onesig
                else:
                    # Still get residuals:
                    ma = (tpi/params['P_p'+str(ii)]) * (bjd - Tp) 
                    ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
                    rv_exp_multisig += (K[-1]*np.cos(np.pi/2 + ta))
                
            # Plot all data and expected RV sine curve per instrument:
            axs[plt_ct,c].errorbar(bjd, rv_list, yerr=erv, fmt='.', color=colors[c], label = 'all data')
            if n_p>1:
                axs[plt_ct,c].plot(dates,np.sum(rv,axis=0), label = 'Est. curve', color='k', linewidth=0.7)
            else: 
                axs[plt_ct,c].plot(dates,rv[0], label = 'Est. curve', color='k', linewidth=0.7) 
            axs[plt_ct,c].set_xlim(min(bjd)-10,max(bjd)+10)
            axs[plt_ct,c].set_title('All data -- %s -- %s arm'%(tn, instruments[c]))
            plt_ct+=1

            # Plot residuals per instrument: 
            rv_rsd = rv_list - rv_exp_multisig
            axs[plt_ct,c].errorbar(bjd, rv_rsd, yerr=erv, fmt='.', color=colors[c], label = 'all data')
            axs[plt_ct,c].axhline(y=0,c='k')
            axs[plt_ct,c].set_xlim(min(bjd)-10,max(bjd)+10)
            axs[plt_ct,c].set_title('Residuals -- %s -- %s arm'%(tn, instruments[c]))
            plt_ct+=1

    axs[0,c].legend(bbox_to_anchor=(1., 1.05))
    fig.tight_layout()
    plt.show()