# ################
# Experiment 4 - Validation process for choosing gamma.
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
    
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used for data generator.")
    
    parser.add_argument("--training_size", type=int, default=10,
                        help="training size for the set.")
    
    parser.add_argument("--g_range", default="-7, -1, 1" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for gammas with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    args = parser.parse_args()
    
    synth_data=data_gen.diverse_data_generator()
    synth_data.options['seed']=args.seed*2
    #Change the bound here.
    synth_data.options['bounds']=[(0,100),(0,100)]
    synth_data.training_size=args.training_size
    
    fig,goodrange=synth_data.gammacalc(plot=True,grange=args.g_range,min_samples=2000)
    
    filename,ext= '../results/Experiment4/k=' + str(int(args.training_size)), '.eps'
    savename=unique_file(filename, ext)
    os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
    fig.savefig(os.path.abspath(savename),dpi=300)

    print("Correlation mat has been saved to the results directory, for k ={}. The suitable range of gamma estimated from the plot is {}".format(args.training_size,goodrange))
    