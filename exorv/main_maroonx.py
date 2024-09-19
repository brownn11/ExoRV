#!/usr/bin/python3
import numpy as np
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from forecaster import forecaster3 as mr
import glob
import json
plt.rcParams['lines.linewidth']=0.4
plt.rcParams.update({'font.size': 8})
import os
import argparse

import plot as p
import load as l

# Parse arguments:
parser = argparse.ArgumentParser(description="Plot RVs")
parser.add_argument('-TOI', '--TOI', type = int, required = False, default = 0, help = "Target TOI number.")
parser.add_argument('-tn', '--target_name', type = str, required = False, default = "", help = "Target name (SIMBAD-recognizeable).")
parser.add_argument('-f', '--folder_name', type = str, required = False, default = "", help = "SERVAL folder name.")
parser.add_argument('-scp', '--scp_download', action = "store_true", help = "Download new files or use available.")
parser.add_argument('-o', '--others', type = json.loads, help = "Additional or alternate planet parameters. Requires P (period [d]), t0 (epoch [BJD]), r (planet radius [earth rad]) OR m (planet mass [earth mass]), and m_s (stellar mass [sol mass]), with planet specified as '_pn'. \n ex: -o '{'P_p1':10, 't0_p1':5, 'r_p1':1, 'm_s':15}'")
parser.add_argument('--points_only', action = "store_true", help = "Plot only RV points.")
parser.add_argument('--curve_only', action = "store_true", help = "Plot only expected RV curve.")
parser.add_argument('-p', '--local_path', type = str, required = False, default = '', help = 'Path to local RV data.')
args = parser.parse_args()

if args.scp_download:
    if not args.folder_name:
        print("SCP download requires a specified folder name. Please enter folder name:")
        folder_name = input()
    else:
        folder_name = args.folder_name
    l.loaddata(TOI = args.TOI, tn = args.target_name, servalfolder = folder_name)

if args.points_only:
    p.points_only(tn =  args.target_name, TOI = args.TOI, path = args.local_path)
elif args.curve_only:
    p.curve_only(TOI = args.TOI, others = args.others)
else:
    p.RV_plotter(tn =  args.target_name, TOI = args.TOI, others = args.others, path = args.local_path)