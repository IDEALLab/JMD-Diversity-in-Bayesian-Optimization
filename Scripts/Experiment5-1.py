# ################
# Experiment 5 - Validating diversity generation by the matrices.
# Corresponding issue on git - #

# Uncomment all code below if running the script for the first time
# ################


import sys
sys.path.insert(1, '../src/optimizers')
sys.path.insert(1, '../src')

import data_gen
import numpy as np
import pickle
from matplotlib import pyplot as plt
import itertools
import os



def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname

if __name__=="__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--trials", type=int, default=2,
                        help="Seed used for data generator.")
    
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used for data generator.")
    
    parser.add_argument("--training_size", type=int, default=10,
                        help="training size for the set.")
    
    args = parser.parse_args()
    
    #Set-up the data generator
    synth_data=data_gen.diverse_data_generator()
    synth_data.options['seed']=np.random.choice(range(100))
    #Change the bound here.
    synth_data.options['bounds']=[(0,100),(0,100)]
    synth_data.training_size=args.training_size
    synth_data.gamma=1e-5
    
    with plt.style.context(['science','no-latex']):
        fig, axs = plt.subplots(args.trials, 2,figsize=(10,4.5*args.trials),sharex='col', sharey='row')
        for trial in range(args.trials):
            percentilemat=[5,95]

            #Generate data for H-DPP
            synth_data.generate(method="subDPP")
            x_subdpp_samp1,y_subdpp_samp1=synth_data.sample(5,method="subDPP").T
            x_subdpp_samp2,y_subdpp_samp2=synth_data.sample(95,method="subDPP").T

            #Score_calculations
            _,subDPP_score_samp1=synth_data.compare_subDPP(synth_data.key,np.array([x_subdpp_samp1,y_subdpp_samp1]).T)
            _,subDPP_score_samp2=synth_data.compare_subDPP(synth_data.key,np.array([x_subdpp_samp2,y_subdpp_samp2]).T)

            axs[trial, 0].scatter(x_subdpp_samp1, y_subdpp_samp1)
            axs[trial, 0].set_title('5th percentile, score: '+ str(round(subDPP_score_samp1,2)),fontsize=21)
            axs[trial, 1].scatter(x_subdpp_samp2, y_subdpp_samp2)
            axs[trial, 1].set_title('95th percentile, score: ' + str(round(subDPP_score_samp2,2)),fontsize=21)
            
            
            axs[trial,0].tick_params(axis='y', labelsize=20)
            
            if trial==args.trials-1:
                axs[trial,0].tick_params(axis='x', labelsize=20)
                axs[trial,1].tick_params(axis='x', labelsize=20)
            
        fig.tight_layout(pad=3.0)
        [ax.set_xlim([0, 100]) for ax in axs.flatten()]
        [ax.set_ylim([0, 100]) for ax in axs.flatten()]
        plt.suptitle('Comparison of samples from 5th vs 95th percentile',fontsize='35', x=0.65, y=1,)
        
            
        plt.subplots_adjust(
            left  = 0.1,  # the left side of the subplots of the figure
            right = 1.2,    # the right side of the subplots of the figure
            bottom = 0.1,   # the bottom of the subplots of the figure
            top = 0.9,      # the top of the subplots of the figure
            wspace = 0.2,   # the amount of width reserved for blank space between subplots
            hspace = 0.2)   # the amount of height reserved for white space between subplots
    
    
        filename,ext= '../results/Experiment5-1/k=' + str(int(args.training_size)), '.eps'
        savename=unique_file(filename, ext)
        os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
        fig.savefig(os.path.abspath(savename),dpi=1000)

