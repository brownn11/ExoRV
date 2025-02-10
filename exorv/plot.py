#!/usr/bin/python3
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from forecaster import forecaster3 as mr
import glob
plt.rcParams['lines.linewidth']=0.4
plt.rcParams.update({'font.size': 8})
import os

import load as l
colors = ['tab:blue','tab:red']
cmaps = ['winter','autumn']
colors2 = ['deepskyblue','coral']


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
              path = '', name = '', offsets = False, singleplot = False):
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
    
    # set tn if TOI target is used -- simplifies things a bit:
    if tn == '' and TOI != '':
        tn = 'TOI-'+str(TOI)

    # Create list of all file basenames:
    filenames = []

    if ',' in tn: # Looks for multiple additional datasets
        tn = tn.split(',')
        for _ in tn: 
            filenames.append(_)
            if filenames[-1][0] == ' ':
                filenames[-1] = filenames[-1][1:]
    else:
        filenames = [tn]

    # Create list of exact filenames:
    paths = []
    for nn in filenames:
        if len(name)==0:
            paths.append(glob.glob(path+'*'+str(nn)+'_r.csv')[0])
            paths.append(glob.glob(path+'*'+str(nn)+'_b.csv')[0])
        else:
            paths.append(list(name.split(', ')))
            for __ in range(len(paths)):
                paths[__] = path + paths[__]
        if len(paths)==0:
            print('Error: path empty, check target name or file location\n---> Attempted path:',path,'*',str(tn),'*.csv')
    print('Using files:',paths)

    # Initialize plot:
    if singleplot:
        fig, axs = plt.subplots(1,1,figsize=(8,2*len(instruments)))
    else:
        fig, axs = plt.subplots(len(instruments),1,figsize=(8,3*len(instruments)))

    bjd_all = []
    for ff in range(len(paths)): # Loop over each dataset
        if list(paths[ff]): # Confirm path exists
            rv_list,bjd,erv=[],[],[]
            p = str(paths[ff])  

            #color 
            if 'b' in p:
                c = 0
                if not singleplot:
                    ax = axs[0]
            else:
                c = 1
                if not singleplot:
                    ax = axs[1]
            if singleplot:
                ax = axs

            # Read in data: 
            if c<2:
                rv_csv=pd.read_csv(p)
            else:
                rv_csv=pd.read_csv(p,sep=' ',header=None, names=['bjd','rv','e_rv'], usecols=[0,1,2])
            rv_list=np.array(rv_csv['rv'].values[:])
            bjd=np.array(rv_csv['bjd'].values[:])
            bjd_all = np.concatenate([bjd_all, bjd])
            erv=np.array(rv_csv['e_rv'].values[:])  

            # Add MAROON-X offsets:
            if offsets:
                if ('MX' in instruments[c]) and (min(bjd) < 2460313.):
                    rv_list = l.MX_offsets(rv_list,bjd,c) 

            # Create cmap, normalized to number of datasets plotted:
            if c==0:
                cmap = cm.Blues
            else:
                cmap = cm.Reds
            norm = Normalize(vmin=-3, vmax=len(paths))
                    
            #Plot all data per instrument:
            ax.errorbar(bjd, rv_list, yerr=erv, fmt='.', label = paths[ff], c = cmap(norm(ff)), ecolor=cmap(norm(ff))) 
            if singleplot:
                ax.set_title('%s'%(tn))
            else:
                ax.set_title('%s -- %s arm'%(tn, instruments[c]))
            ax.set_ylabel('RV [m/s]')
            ax.set_xlabel('BJD')
    bjd_range = max(bjd_all)-min(bjd_all)
    if singleplot:
        ax.set_xlim(min(bjd_all)-0.1*bjd_range,max(bjd_all)+0.1*bjd_range)  
        ax.legend(loc = 'upper right', fontsize = 6)
    else:
        for cc in [0,1]:
            axs[cc].set_xlim(min(bjd_all)-0.1*bjd_range,max(bjd_all)+0.1*bjd_range)  
            axs[cc].legend(loc = 'upper right', fontsize = 6) 

    fig.tight_layout()
    plt.show()

def RV_plotter(TOI = '', others = {}, tn = '', 
              path = '', name = '',
              order = [], sigsub = True, offsets = False, singleplot = False):
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

    # set tn if TOI target is used -- simplifies things a bit:
    if tn == '' and TOI != '':
        tn = 'TOI-'+str(TOI)
    
    # Create list of all file basenames:
    filenames = []

    if ',' in tn: # Looks for multiple additional datasets
        tn = tn.split(',')
        for _ in tn: 
            filenames.append(_)
            if filenames[-1][0] == ' ':
                filenames[-1] = filenames[-1][1:]
    else:
        filenames = [tn]

    # Create list of exact filenames:
    paths = []
    for nn in filenames:
        if len(name)==0:
            paths.append(glob.glob(path+'*'+str(nn)+'_r.csv')[0])
            paths.append(glob.glob(path+'*'+str(nn)+'_b.csv')[0])
        else:
            paths.append(list(name.split(', ')))
            for __ in range(len(paths)):
                paths[__] = path + paths[__]
        if len(paths)==0:
            print('Error: path empty, check target name or file location\n---> Attempted path:',path,'*',str(tn),'*.csv')
    print('Using files:',paths)
    
    # Load ExoFOP data:
    params = l.loadparams(TOI, others)
    n_p = int((len(params)-1)/3) # number of planets

    # Automate order if unspecified:
    if order == []:
        order = range(1,n_p+1) 

    # Initialize plot:
    if singleplot:
        fig, axs = plt.subplots(n_p+2,1,figsize=(8,5*n_p))
    else:
        fig, axs = plt.subplots(n_p+2,len(instruments),figsize=(8,5*n_p))

    K = np.zeros(len(order))
    for ii in order: # Get K and expected planet mass for each planet -- 
        print('\nPlanet %s (%sd):'%(ii,params['P_p'+str(ii)]))
        
        try:
            m = params['m_p'+str(ii)]
        except:
            # Estimate mass using Forecaster by Chen & Kipping (2016):
            m, m_plus, m_min = mr.Rstat2M(mean=params['r_p'+str(ii)], std=0.01, unit='Earth', sample_size=1000, grid_size=1e3, classify='Yes')
            print(f'Planet mass estimated at {m:.2f} +{m_plus:.2f} -{m_min:.2f} Earth mass')

        # Calculate RV amplitude:
        K[ii-1] = (28.4 * (m/317.83) * ((365.25/params['P_p'+str(ii)])**(1/3)) * ((params['m_s'])**(-2/3))) # 28.4 [m/s] = 2piG^1/3; 317.83 = MJ/ME; 365.25 = Yr/day
        print(f'Calculated RV semi-amplitude of {K[-1]:.2f} m s-1')

    bjd_all = [] # compile ALL bjd dates to find absolute minimum and maximum of all datasets

    for ff in range(len(paths)): # Loop over each dataset
        if list(paths[ff]): # Confirm path exists

            rv_list,bjd,erv,rv,ct=[],[],[],[],[]
            p = str(paths[ff])  
            plt_ct = 0 # reset plot counter

            inst_name = ''

            #color 
            if 'b' in p:
                c = 0
                if not singleplot:
                    ax = axs[:,0]
                    inst_name = 'MXB arm'
            else:
                c = 1
                if not singleplot:
                    ax = axs[:,1]
                    inst_name = 'MXR arm'
            if singleplot:
                ax = axs[:]

            # Read in data: 
            if c<2:
                rv_csv=pd.read_csv(p)
            else:
                rv_csv=pd.read_csv(p,sep=' ',header=None, names=['bjd','rv','e_rv'], usecols=[0,1,2])
            rv_list=np.array(rv_csv['rv'].values[:])
            bjd=np.array(rv_csv['bjd'].values[:])
            bjd_all = np.concatenate([bjd_all,bjd])
            erv=np.array(rv_csv['e_rv'].values[:])  
            
            # Add MAROON-X offsets:
            if offsets:
                if ('MX' in instruments[c]) and (min(bjd) < 2460313.):
                    rv_list = l.MX_offsets(rv_list,bjd,c) 

            # Initialize dummy arrays for signal subtraction:
            if sigsub == True:
                rv_sub = rv_list

            rv_exp_multisig = 0 # expected RV, does not reset per signal
            for ii in order: # Iterate over expected planets:
                rv_exp_onesig = [] # expected RV, resets per signal

                # Get RV curve (let e = 0 and w = pi/2):
                tpi = 2 * np.pi
                phase = 1
                Tp = params['t0_p'+str(ii)] - (phase * params['P_p'+str(ii)])  # time of periastron
                dates = np.linspace(min(bjd)-100,max(bjd)+100,10000) # set desired date range
                ma = (tpi/params['P_p'+str(ii)]) * (dates - Tp) # mean anomaly
                ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
                rv.append(K[ii-1]*np.cos(np.pi/2 + ta)) # expected rv signal

                # Plot data per planet as phase-fold:
                rx = ((params['t0_p'+str(ii)]-bjd)/params['P_p'+str(ii)])%1 
                
                if sigsub == True:
                    rv_plot = rv_sub
                    Ps = [round(params['P_p'+str(cc)],2) for cc in ct]
                    if plt_ct >= 1:
                        title = '+ subtracted %sd signal(s)'%Ps
                    else:
                        title = ''
                else:
                    rv_plot = rv_list
                    title = ''

                # Create cmap, normalized to number of datasets plotted:
                if c==0:
                    cmap = cm.Blues
                else:
                    cmap = cm.Reds
                norm = Normalize(vmin=-3, vmax=len(paths))

                ax[plt_ct].errorbar(rx, rv_plot, yerr=erv, fmt=".", c = cmap(norm(ff)), ecolor=cmap(norm(ff)), label = paths[ff])
                ax[plt_ct].plot(np.linspace(0,1,10000),K[ii-1]*np.sin(np.linspace(0,2*np.pi,10000)), color='k', linewidth=0.7)
                ax[plt_ct].set_title('Phase folded %s %s'%(title, inst_name))
                ax[plt_ct].set_xlabel('Period = %s days' %round(params['P_p'+str(ii)],3))
                ax[plt_ct].set_ylabel('RV [m/s]')

                plt_ct+=1
                ct.append(ii) # track which planets have been plotted

                if sigsub == True:
                    # Signal subtract: 
                    ma = (tpi/params['P_p'+str(ii)]) * (bjd - Tp) 
                    ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
                    rv_exp_onesig = (K[ii-1]*np.cos(np.pi/2 + ta)) 
                    rv_exp_multisig += rv_exp_onesig

                    rv_sub = rv_list - rv_exp_onesig

                else:
                    # Still get residuals:
                    ma = (tpi/params['P_p'+str(ii)]) * (bjd - Tp) 
                    ta = np.arctan(np.tan(ma/2)) * 2 # true anomaly
                    rv_exp_multisig += (K[ii-1]*np.cos(np.pi/2 + ta))
                
            # Plot all data and expected RV sine curve per instrument:
            
            ax[plt_ct].errorbar(bjd, rv_list, yerr=erv, fmt='.',c = cmap(norm(ff)), ecolor=cmap(norm(ff)))
            if n_p>1:
                ax[plt_ct].plot(dates,np.sum(rv,axis=0), color='k', linewidth=0.7)
            else: 
                ax[plt_ct].plot(dates,rv[0], color='k', linewidth=0.7) 
            ax[plt_ct].set_title('All data %s'%(inst_name))
            ax[plt_ct].set_ylabel('RV [m/s]')
            ax[plt_ct].set_xlabel('BJD')
            plt_ct+=1

            # Plot residuals per instrument: 
            rv_rsd = rv_list - rv_exp_multisig
            sq_sum = np.sum([ii*ii for ii in rv_rsd])
            rms = np.sqrt(sq_sum/len(rv_rsd))
            ax[plt_ct].errorbar(bjd, rv_rsd, yerr=erv, fmt='.', c = cmap(norm(ff)), ecolor=cmap(norm(ff)), label = 'RMS = %s'%round(rms,2))
            ax[plt_ct].axhline(y=0,c='k')
            ax[plt_ct].set_title('Residuals %s'%(inst_name))
            ax[plt_ct].set_ylabel('[m/s]')
            ax[plt_ct].set_xlabel('BJD')

            plt_ct+=1

    bjd_range = max(bjd_all)-min(bjd_all)
    if singleplot:
        ax[-2].set_xlim(min(bjd_all)-0.1*bjd_range,max(bjd_all)+0.1*bjd_range)  
        ax[0].legend(loc = 'upper right', fontsize = 6)
        ax[-1].legend(loc = 'upper right', fontsize = 6)
    else:
        for cc in [0,1]:
            axs[-2,cc].set_xlim(min(bjd_all)-0.1*bjd_range,max(bjd_all)+0.1*bjd_range)  
            axs[0,cc].legend(loc = 'upper right', fontsize = 6)
            axs[-1,cc].legend(loc = 'upper right', fontsize = 6)
    fig.tight_layout()
    plt.show()