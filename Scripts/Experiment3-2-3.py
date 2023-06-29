# ################
# Experiment 3-2-3 - Effects of diversity on the optimization process when hyper-parameters are fixed to the optimal values for each function
# Corresponding issue on git - #

# Generates data for BO optimizing with fixed optimal parameters for wildcatwells with args {'N':1,'Smoothness':0.2,'rug_freq':1,'rug_amp':0.7} and seed in the range [0,numfuncs] and saves the figures generated to Experiment3-2-3 directory.

# Needed arguments:
# #Trials- Number of trials to generate the distribution of data
# #Numfuncs-Number of functions to generate the data for i.e from seed 0 to numfuncs-1 seed.
# #intent- choices -> {gen-data: To generate and save data to suitable directory, plot-data: To load data and plot data from the suitable directory}
# ################

import sys
sys.path.insert(1, '../src/optimizers')
sys.path.insert(1, '../src')

from objectives import objectives

import data_gen
import numpy as np
import random
import string
import pickle
import dill
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import kde

from ipyparallel import Client
import ipyparallel as ipp
import os
import subprocess
import time
import itertools
from tqdm import tqdm
import itertools
import pandas as pd
import results
from os.path import exists
import fileinput
if sys.platform.startswith('win'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def savefile(filename,variable):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        dill.dump(variable, f)
    return variable

def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s (%d).%s" % (basename, next(c), ext)
    return actualname

def load_variable(filename):
    with open(filename, 'rb') as f:
        variable = dill.load(f)
    return variable

def singleopt(opt,f):
    counter=0
    while True:
        try:
            result= opt.optimize(f)
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Keyboard interrupt')
        except:
            print('Error occured restarting.')
            pass
            if counter>=5:
                raise Exception("Too many restarts.")
    return result

def parallel(trials,fun,args,parallelcomp):
    """
    fun: function to be parallized
    args: RxN list with args for each run.
    runs: number of runs of the function.
    """
    from tqdm import tqdm
    rc,bview,_=parallelcomp
    async_results=[]
    for _ in range(trials):
        args[0].train=args[0].datagenmodule.sample(args[2])
        async_results.append(bview.apply_async(fun, args[0],args[1]))
    results=[]
    for ar in tqdm(async_results):
        try:
            current_res=ar.get(timeout=10)
            results.append(current_res)
        except KeyboardInterrupt:
            raise Exception('Keyboard interrupt')
        except:
            pass
    return results

async def startengines():
    import multiprocessing as mp
    ncpus=mp.cpu_count()
    clust=ipp.Cluster()
    await clust.start_controller()
    clear_output()
    await clust.start_engines(n=ncpus);
    rc= await clust.connect_client()
    rc.wait_for_engines(ncpus)
    clear_output()
    return rc,clust

async def stopengines(clust):
    await clust.stop_engines(engine_set_id=list(clust.engines.keys())[0]);
    await clust.stop_cluster()
    clear_output()
    print('Engines stopped')

def setupengines(rc,clust):
    bview=rc.load_balanced_view()
    dview=rc.direct_view()
    parallelcomp=[rc,bview,dview]
    dview.map(os.chdir, [os.path.abspath('..\src')]*len(rc))
    with dview.sync_imports():
        import data_gen
        import objectives
    dview.map(os.chdir, [os.path.abspath('..\src\optimizers')]*len(rc))
    with dview.sync_imports():
        import Optimizers
    print("Engine Set-up complete")
    return parallelcomp


import math
import results
from IPython.display import clear_output

#need to add log arg
async def main(trials,ind_trials,max_iter,sm_range,rug_range,sampling_method=None,reverse=False):
    
    #Set-up variables
    completedata={}    
    if sampling_method is None or sampling_method=="diverse":
        percentilemat=[5,95]
    else:
        percentilemat=[None]
    smoothness_mat=np.arange(*sm_range).round(2)

    #Change this to configure the Fixed parameters of the optimizer.
    available_locs=[0]
    
    while len(available_locs)>0:
        from os.path import exists
        available_locs=[]
        for smoothness in np.arange(*sm_range).round(2):
            for rug_amp in np.arange(*rug_range).round(2):
                for dataseed in range(trials):
                    filename='../Data/experiment3-2-3/'+sampling_method+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
                    if exists(filename):
                        pass
                    else:
                        available_locs.append((smoothness,rug_amp,dataseed))
                        
        if not reverse:
            smoothness,rug_amp,dataseed=available_locs[0]
        else:
            smoothness,rug_amp,dataseed=available_locs[-1]
            
        filename='../Data/experiment3-2-1/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        try:
            hpseries=load_variable(filename)
        except:
            raise Exception('Optimal Hyperparameter data has not been generated yet for these set of wildcatwells functions, run script Exp 3-2-1.')
        print('Working on Trial {} of {}, for smoothness= {}, rug_amp ={}'.format(dataseed+1,trials,smoothness,rug_amp))
        hpseries.extract_hyperparam()
        
        
        #Create a result object for the trial data
        result=results.result_diversity_trial()
        result.percentiles=percentilemat        
        
        filename='../Data/experiment3-2-3/'+sampling_method+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        savefile(filename,result)
        
        # Set-up the function generator
        synth_func=objectives()
        synth_func.bounds=[(0,100),(0,100)]
        synth_func.seed=dataseed
        synth_func.args={'N':1,'Smoothness':smoothness,'rug_freq':1,'rug_amp':rug_amp}
        f=synth_func.generate_cont()


        if sampling_method is None or sampling_method=="diverse":
            synth_data=data_gen.diverse_data_generator()
            synth_data.options['bounds']=synth_func.bounds
            synth_data.options['N_samples']=10000
            synth_data.gamma=1e-5
            synth_data.options['seed']=0
            synth_data.training_size=10
        else:
            synth_data=data_gen.random_data_generator()
            synth_data.options['bounds']=synth_func.bounds
            synth_data.options['seed']=0
            synth_data.training_size=10

        #Start and define the engines
        rc,clust = await startengines()
        parallelcomp=setupengines(rc,clust)

        funcname='smoothness-{}-ruggedness_amplitude-{}'.format(smoothness,rug_amp)
        #For diversity trial we need to iterate over different percentiles of diversity.
        for percentile in percentilemat:
            print('{}th percentile of diversity being evaluated'.format(percentile))
            #Set-up the optimizer
            from Optimizers import optimizer
            BO=optimizer()
            BO.max_iter=max_iter
            BO.opt="BO-fixparam"
            BO.paramset=hpseries.optdict
            BO.tol=0.1
            BO.minimize=synth_func.minstate
            BO.bounds=synth_func.bounds
            #Add appropriate training data to BO.
            BO.datagenmodule=synth_data

            #Run the parallel process/Trial
            current_result=parallel(ind_trials,singleopt,[BO,f,percentile],parallelcomp)
            [result.addresult(percentile,ind_result) for ind_result in current_result]

        #Stop the engines
        await stopengines(clust)
        #Update log file
        # log.logger.info('complete: smoothness:{}, rug_amp:{}'.format(smoothness,rug_amp))
        #Add a theoretical maximum or minima dependent on the problem.
        result.opt=100
        #Save Data to the Result trial object and append to an outer list collecting data for each seed in the trial.
        filename='../Data/experiment3-2-3/'+sampling_method+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        savefile(filename,result)
    return

if __name__=='__main__':
    
    import argparse
    import asyncio
    import os
    from utils import bootstrap
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of seeds for each iteration of ruggedness and smoothness that need to be evaluated.")
    parser.add_argument("--ind_trials", type=int, default=10, help="Number of evaluation for each each seed")
    
    parser.add_argument("--smoothness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--ruggedness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    
    parser.add_argument("--intent", choices={"gen-data","optimal_init_iter" ,"plot-data","save_compile_data"}, default="gen-data")

    parser.add_argument("--sampling_type", choices={"diverse", "random"}, default="diverse")
    
    parser.add_argument("--grid_type",choices={"difference","comparison"},default="comparison")
    
    parser.add_argument("--max_iter", type=int, default=100, help="Number of iterations for BO during each trial.")
    
    parser.add_argument("--reverse_gen_direction",action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    success=False
    if args.intent=='gen-data' or args.intent=='both':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        completedata=asyncio.run(main(args.trials,args.ind_trials,
                                      args.max_iter,
                                      args.smoothness_range,
                                      args.ruggedness_range,
                                      sampling_method=args.sampling_type,
                                      reverse=args.reverse_gen_direction))
        success=True
        
    def plot_data(percentile,smoothness,rug_amp,dataseed,plot_type,sample_type):
        filename='../Data/experiment3-2-3/'+sample_type+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        result=load_variable(filename)
        if plot_type=='comparison':
            return result.bootstrap(percentile)
        else:
            return result.difference_bootstrap_plotdata()

    def get_all_seed_data(smoothness,rug_amp,plot_type,sample_type):
        import ray
        @ray.remote
        def map_(obj, f):
            import sys
            sys.path.insert(1, '../src/optimizers')
            sys.path.insert(1, '../src')
            import results
            return f(*obj)

        if plot_type=='comparison' and sample_type=='diverse':
            return ray.get([map_.remote([5,smoothness,rug_amp,seed,plot_type,sample_type], plot_data) for seed in range(100)]),ray.get([map_.remote([95,smoothness,rug_amp,seed,plot_type], plot_data) for seed in range(100)])
        elif plot_type=='differnce':
            return ray.get([map_.remote(['None',smoothness,rug_amp,seed,plot_type,sample_type], plot_data) for seed in range(100)])
        else:
            print('executed right if statment')
            return ray.get([map_.remote([None,smoothness,rug_amp,seed,plot_type,sample_type], plot_data) for seed in range(100)])
        
    if args.intent=='save_compile_data': 
        completedata={}
        smoothness_mat=np.arange(*args.smoothness_range).round(2)
        ruggedness_mat=np.arange(*args.ruggedness_range).round(2)
        for smoothness in smoothness_mat:
            completedata[smoothness]={}
            for rug_amp in ruggedness_mat:
                completedata[smoothness][rug_amp]={}
                if args.sampling_type=='diverse':
                    grids=['comparison','difference']
                else:
                    grids=['comparison']
                for grid_type in grids:
                    plotdata=get_all_seed_data(smoothness,rug_amp,grid_type,args.sampling_type)
                    if grid_type=='comparison':
                        completedata[smoothness][rug_amp][grid_type]=[bootstrap(single_percentile_data) for single_percentile_data in np.array(plotdata)[:,:,0,:]]
                    else:
                        completedata[smoothness][rug_amp][grid_type]=bootstrap(np.array(plotdata)[:,0,:])
            
        filename='../Data/experiment3-2-3/'+args.sampling_type+'-plot-data.pkl'
        savefile(filename,completedata)
        
    if args.intent=='plot-data':
        filename='../Data/experiment3-2-3/plot-data.pkl'
        
        if exists(filename):
            completedata=load_variable(filename)
        else:
            raise Exception('Save plot data using --intent "save_compile_data"')
        
        def average_cumoptgap(percentilemat,smoothness,rug_amp):
            import numpy as np
            cumoptgap_data=[]
            for dataseed in range(100):
                filename='../Data/experiment3-2-3/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
                result=load_variable(filename)
                cumoptgap_data.append(result.cum_opt_gap_improvement_overall(percentilemat)[0])
            return np.average(cumoptgap_data)
        
        smoothness_mat=np.arange(*args.smoothness_range).round(2)
        ruggedness_mat=np.arange(*args.ruggedness_range).round(2)      
        #Generate plot and save it to the appropriate directory.
        with plt.style.context(['science','no-latex']):
            fig, ax = plt.subplots(len(smoothness_mat), len(ruggedness_mat), sharex='col', sharey='row',figsize=(30,21))
            
            for i,smoothness in enumerate(smoothness_mat):
                for j,rug_amp in enumerate(ruggedness_mat):
                    percentilemat=[5,95]
                    cumoptgap=average_cumoptgap(percentilemat,smoothness,rug_amp)
                    cumoptgaptext1='Diversity helped imrove'
                    cumoptgaptext2='performance by '+ str(round(cumoptgap*100,2)) + ' percent'
                    if not exists(filename):
                        plotdata=get_all_seed_data(smoothness,rug_amp,args.grid_type)
                    else:
                        plotdata=completedata[smoothness][rug_amp]
                    if args.grid_type=="comparison":
                        #bootstraps the data for the line plot for each percentile in the diversity trial result object.
                        for k,percentile in enumerate(percentilemat):
                            if not exists(filename):
                                bootstrap_on_trials=bootstrap(np.array(plotdata)[:,:,0,:][k])
                            else:
                                bootstrap_on_trials=plotdata[args.grid_type][k]
                            label=str(percentile)+'th percentile'
                            ax[i,j].plot(np.arange(args.max_iter+2), bootstrap_on_trials[0], '-',label=label) #Plotting the mean data
                            ax[i,j].fill_between(np.arange(args.max_iter+2),bootstrap_on_trials[1], 
                                            bootstrap_on_trials[2], alpha=0.3) #Plotting the 90% confidence intervals.
                            ax[i,j].text(15,20, cumoptgaptext1,fontsize='15')
                            ax[i,j].text(10,17, cumoptgaptext2,fontsize='15')
                            
                        ax[i,j].legend(fontsize='x-small')
                        ax[i,j].set_ylim([0, 40])
                        ax[i,j].set_xlim([0, int(args.max_iter/2)])
                    else:
                        if not exists(filename):
                            bootstrap_on_trials=bootstrap(np.array(plotdata)[:,0,:])
                        else:
                            bootstrap_on_trials=plotdata[args.grid_type]
                        def create_colormap_and_ls(x,y):
                            # select how to color
                            cmap = (mpl.colors.ListedColormap(['tomato','blue','yellowgreen']).with_extremes(over='yellowgreen', under='tomato'))
                            bounds = [-1e-10,0.0,1e-10]
                            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                            
                            # get segments
                            xy = np.array([x, y]).T.reshape(-1, 1, 2)
                            segments = np.hstack([xy[:-1], xy[1:]])
                            
                            # make line collection
                            lc = LineCollection(segments, cmap = cmap, norm = norm, linewidths=2)
                            lc.set_array(y)
                            return cmap,norm,lc
                        
                        cmap,norm,lc=create_colormap_and_ls(np.arange(args.max_iter+2),bootstrap_on_trials[0])

                        ax[i,j].add_collection(lc) #Plotting the mean data as a line segment

                        ax[i,j].fill_between(np.arange(args.max_iter+2),bootstrap_on_trials[1], 
                                        bootstrap_on_trials[2], alpha=0.1, hatch='\\\\\\\\',facecolor='azure') 
                        ax[i,j].axhline(y=0,label='Insignificant difference in perfromance',color='blue')

                        if cumoptgap<0: 
                            c='r'
                        else:
                            c='g'
                        ax[i,j].text(20,5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                        handles, labels = ax[i,j].get_legend_handles_labels()
                        ax[i,j].set_ylim([-10, 10])
                        ax[i,j].tick_params(axis='x', labelsize=25)
                        ax[i,j].tick_params(axis='y', labelsize=25)
                        ax[i,j].set_xlim([0, int(args.max_iter/2)])
                        
                    if j==0:
                        y_label=smoothness
                        ax[i,j].set_ylabel(str(round(y_label,2)),fontsize='35')
                    if i==len(smoothness_mat)-1:
                        x_label=rug_amp
                        ax[i,j].set_xlabel(str(round(x_label,2)),fontsize='35')
            if args.grid_type!="comparison":
                right,bottom,width,height=0.83,0.1,0.03,0.8
                cbar_ax = fig.add_axes([right,bottom,width,height])
                cbar=fig.colorbar(
                    mpl.cm.ScalarMappable(cmap=cmap, norm=norm), extend='both',
                    extendfrac='auto',ax=ax.ravel().tolist(),cax=cbar_ax)
                cbar.set_label( label='Effect of diversity on performance of optimizer',fontsize='30')
                cbar.set_ticks([])
            ax[int(len(smoothness_mat)/2)-1,0].text(-24,-22, "Smoothness",fontsize='45',rotation=90)
            ax[len(smoothness_mat)-1,int(len(smoothness_mat)/2)].text(-45 ,-19 , "Ruggedness Amplitude",fontsize='45')
            
            plt.subplots_adjust(
                left  = 0.1,  # the left side of the subplots of the figure
                right = 0.8,    # the right side of the subplots of the figure
                bottom = 0.1,   # the bottom of the subplots of the figure
                top = 0.9,      # the top of the subplots of the figure
                wspace = 0.2,   # the amount of width reserved for blank space between subplots
                hspace = 0.2)   # the amount of height reserved for white space between subplots

            if args.grid_type=="comparison":
                fig.suptitle('Comparison in optimality gap when hyperparameters are fixed for the optimizer',fontsize='48', x=0.91, horizontalalignment='right')
            else:
                fig.suptitle('Difference in optimality gap when hyperparameters are fixed for the optimizer',fontsize='48', x=0.91, horizontalalignment='right')
                filename,ext= '../results/Experiment3-2-3/variable-ww-difference-sm-'+str(args.smoothness_range)+'-rug-'+str(args.ruggedness_range), '.eps'

            try:
                savename=unique_file(filename, ext)
                plt.savefig(os.path.abspath(savename),bbox_inches = 'tight',dpi=1000)
                plt.close()
            except:
                savename=unique_file(filename, ext)
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                plt.savefig(os.path.abspath(savename),bbox_inches = 'tight',dpi=1000)
                plt.close()