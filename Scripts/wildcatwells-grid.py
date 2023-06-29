# ################
# Plot # in paper - How objective function landscape changes with its parameters
# Corresponding issue on git - #

# Uncomment all code below if running the script for the first time
# ################

import sys
sys.path.insert(1, '../src/optimizers')
sys.path.insert(1, '../src')

def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname

from objectives import objectives
import data_gen
import numpy as np
import pickle
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from utils import plot2d
from plotly.subplots import make_subplots

from ipyparallel import Client
import ipyparallel as ipp
import os
import subprocess
import time
import itertools
from matplotlib import cm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__=='__main__':
    import sys
    import argparse
    import asyncio
    from utils import bootstrap,wrapText
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--smoothness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--ruggedness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    args = parser.parse_args()
    
    
    smoothness_mat=np.arange(*args.smoothness_range).round(2)
    ruggedness_mat=np.arange(*args.ruggedness_range).round(2)
    
    with plt.style.context(['science','no-latex']):
        fig = plt.figure(figsize=(30,21))

        for i,smoothness in enumerate(smoothness_mat):
            for j,rug_amp in enumerate(ruggedness_mat):
                ax = fig.add_subplot(len(smoothness_mat), len(ruggedness_mat), i*len(smoothness_mat)+j+1, projection='3d')
                synth_func=objectives()
                synth_func.bounds=[(0,100),(0,100)]
                synth_func.args={'N':1,'Smoothness':smoothness,'rug_freq':1,'rug_amp':rug_amp}
                synth_func.seed=0
                X,surf=synth_func.wildcatwells()
                plot = ax.plot_surface(X.T[0],X.T[1] , surf, rstride=1, cstride=1, cmap=cm.get_cmap('gnuplot'),
                           linewidth=0, antialiased=False)
                ax.axes.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                ax.axes.zaxis.set_ticklabels([])    
                if j==0:
                    y_label=smoothness
                    ax.text2D(-0.12,-0.01,str(round(y_label,2)),fontsize='30')
                if i==len(smoothness_mat)-1:
                    x_label=rug_amp
                    ax.set_xlabel(str(round(x_label,2)),fontsize='30')
                if i==int(len(smoothness_mat)/2)-1 and j==0:
                    ax.text2D(-0.17,-0.20, "Smoothness",fontsize='40',rotation=90)
                if i==len(smoothness_mat)-1 and j==int(len(smoothness_mat)/2):
                    ax.text2D(-0.27,-0.15, "Ruggedness Amplitude",fontsize='40')
        plt.subplots_adjust(
                left  = 0.1,  # the left side of the subplots of the figure
                right = 0.8,    # the right side of the subplots of the figure
                bottom = 0.1,   # the bottom of the subplots of the figure
                top = 0.9,      # the top of the subplots of the figure
                wspace = 0.01,   # the amount of width reserved for blank space between subplots
                hspace = 0.01)   # the amount of height reserved for white space between subplot

        right,bottom,width,height=0.83,0.1,0.03,0.8
        cbar_ax = fig.add_axes([right,bottom,width,height])
        cbar=fig.colorbar(plot,ax=fig.get_axes(),cax=cbar_ax)
        cbar.ax.tick_params(labelsize='20')
        fig.suptitle('Wildcatwells grid',fontsize='50')
        filename,ext='../results/wildcatwells-grid','.eps'

        try:
            savename=unique_file(filename, ext)
            plt.savefig(os.path.abspath(savename),bbox_inches = 'tight',dpi=300)#, pad_inches = 0.3)
            plt.close()
        except:
            savename=unique_file(filename, ext)
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(os.path.abspath(savename) ,bbox_inches = 'tight',dpi=300) #, pad_inches = 0.3)
            plt.close()



