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
parser.add_argument('-scp_mx', '--scp_download_mx', type = str, required = False, default = "", help = "Input SERVAL folder name to indicate scp download.")
parser.add_argument('-scp', '--scp_download', type = str, required = False, default = "", help = "Input full path + filename to data on remote server to indicate scp download.")
parser.add_argument('-o', '--others', type = json.loads, help = """Additional or alternate planet parameters. Requires P (period [d]), t0 (epoch [BJD]), r (planet radius [earth rad]) OR m (planet mass [earth mass]), and m_s (stellar mass [sol mass]), with planet specified as '_pn'.\nex: -o '{"P_p1":10, "t0_p1":5, "r_p1":1, "m_s":15}' """)
parser.add_argument('--points_only', action = "store_true", help = "Plot only RV points.")
parser.add_argument('--curve_only', action = "store_true", help = "Plot only expected RV curve.")
parser.add_argument('--compare', type = str, default = "", help = 'Enter another target name to compare.\nex: -tn "Gl 12_octd" --compare "Gl 12_novd"\Only works with points-only plots.')
parser.add_argument('-p', '--local_path', type = str, required = False, default = '', help = 'Manually enter path to local RV data.')
parser.add_argument('-n', '--file_name', type = str, required = False, default = '', help = 'Manually enter local file name of RV data.\nex: "Gl 15 A_cd_r.csv, Gl 15 A_cd_b.csv"')
parser.add_argument('--offsets', action = "store_true", help = "Apply MAROON-X offsets")
args = parser.parse_args()

if args.scp_download_mx:
    folder_name = args.scp_download_mx
    l.loaddata_mx(TOI = args.TOI, tn = args.target_name, servalfolder = folder_name)
elif args.scp_download:
    path = args.scp_download
    l.loaddata(path = path)

if args.points_only:
    p.points_only(tn =  args.target_name, tn_adtl = args.compare, TOI = args.TOI, path = args.local_path, name = args.file_name, offsets = args.offsets)
elif args.curve_only:
    p.curve_only(TOI = args.TOI, others = args.others)
else:
    p.RV_plotter(tn =  args.target_name, tn_adtl = args.compare, TOI = args.TOI, others = args.others, path = args.local_path, name = args.file_name, offsets = args.offsets)