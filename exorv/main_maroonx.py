#!/usr/bin/python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from forecaster import forecaster3 as mr
import json
plt.rcParams['lines.linewidth']=0.4
plt.rcParams.update({'font.size': 8})
import argparse

import plot as p
import load as l

# Parse arguments:
parser = argparse.ArgumentParser(description="Plot RVs")
parser.add_argument('-TOI', '--TOI', type = int, required = False, default = 0, help = "Target TOI number.")
parser.add_argument('-tn', '--target_name', type = str, required = False, default = "", help = "Target name as listed in filename (i.e. 'TOI-2142' for 'TOI-2142_r.csv').\nList multiple target names to overplot.")
parser.add_argument('-scp_mx', '--scp_download_mx', type = str, required = False, default = "", help = "Input SERVAL folder name to indicate scp download.")
parser.add_argument('-scp', '--scp_download', type = str, required = False, default = "", help = "Input full path + filename to data on remote server to indicate scp download.")
parser.add_argument('-o', '--others', type = json.loads, help = """Additional or alternate planet parameters. Requires P (period [d]), t0 (epoch [BJD]), r (planet radius [earth rad]) OR m (planet mass [earth mass]), and m_s (stellar mass [sol mass]), with planet specified as '_pn'.\nex: -o '{"P_p1":10, "t0_p1":5, "r_p1":1, "m_s":15}' """)
parser.add_argument('--pointsonly', action = "store_true", help = "Plot only RV points.")
parser.add_argument('--curveonly', action = "store_true", help = "Plot only expected RV curve.")
parser.add_argument('-p', '--local_path', type = str, required = False, default = '', help = 'Manually enter path to local RV data.')
parser.add_argument('-n', '--file_name', type = str, required = False, default = '', help = 'Manually enter local file name of RV data.\nex: "Gl 15 A_cd_r.csv, Gl 15 A_cd_b.csv"')
parser.add_argument('--offsets', action = "store_true", help = "Apply MAROON-X offsets")
parser.add_argument('--singleplot', action = "store_true", help = "Plot red and blue channels together.")
parser.add_argument('--MXdrift', action = "store_true", help = "Consider MX etalon drift of 2 m/s/day.")
parser.add_argument('-subfmt', type = str, required = False, default = "date", help = "Astropy Time ISO subformat: date, date_hm, or date_hms")

args = parser.parse_args()

if args.scp_download_mx:
    folder_name = args.scp_download_mx
    l.loaddata_mx(TOI = args.TOI, tn = args.target_name, servalfolder = folder_name)
elif args.scp_download:
    path = args.scp_download
    l.loaddata(path = path)

if args.pointsonly:
    p.points_only(tn =  args.target_name, TOI = args.TOI, path = args.local_path, name = args.file_name, offsets = args.offsets, singleplot = args.singleplot, subfmt = args.subfmt)
elif args.curveonly:
    p.curve_only(TOI = args.TOI, others = args.others)
else:
    p.RV_plotter(tn =  args.target_name, TOI = args.TOI, others = args.others, path = args.local_path, name = args.file_name, offsets = args.offsets, singleplot = args.singleplot, MXdrift = args.MXdrift, subfmt = args.subfmt)