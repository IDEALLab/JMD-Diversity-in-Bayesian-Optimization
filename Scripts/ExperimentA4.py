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
import ray
from os.path import exists
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

@ray.remote(num_gpus=0.1,num_cpus=0.5)
def map_(opt,objective,trial,percentile):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    objective.seed=trial
    if objective.fun_name!='wildcatwells':
        wildcatwells=objective.generate_cont()
    else:
        wildcatwells=objective.generate_cont(from_saved=True,local_dir='C:\\temp\\wildcatwells')
    counter=0
    opt.datagenmodule.options['seed']=trial*7
    opt.datagenmodule.generate()
    while counter<5:
        try:
            opt.train=opt.datagenmodule.sample(percentile)
            result= opt.optimize(wildcatwells)
            result.seed=trial
            return result
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Keyboard interrupt')
        except:
            counter+=1
            print('Error occured restarting.')

import math
import results
from IPython.display import clear_output

#need to add log arg
def main(trials,ind_trials,dim_mat,objective_func,reverse=False):
    
    #Set-up variables
    completedata={}    
    percentilemat=[5,95]
    
    available_locs=[0]
    batch_size=5
    
    if objective_func=='wildcatwells':
        ruggedness_mat=['low','medium','high']
    else:
        ruggedness_mat=['None']
        
    while len(available_locs)>0:
        from os.path import exists
        available_locs=[]
        for dim in np.arange(*dim_mat):
            for level_of_ruggedness in ruggedness_mat:
                for seed in range(trials):
                    filename='../Data/ExperimentA4/'+objective_func+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
                    if exists(filename):
                        print(dim,level_of_ruggedness,seed)
                        pass
                    else:
                        available_locs.append((dim,level_of_ruggedness,seed))
                    
        if not reverse:
            dim,level_of_ruggedness,seed=available_locs[0]
        else:
            dim,level_of_ruggedness,seed=available_locs[-1]

#         dim,level_of_ruggedness,seed=3.0,'None',4
        filename='../Data/ExperimentA2/'+objective_func+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
        hpseries=load_variable(filename)
        print('Working on Trial {} of {}, for dim= {}, rug ={}'.format(seed+1,trials,dim,level_of_ruggedness))
        hpseries.extract_hyperparam()
        
#         Create a result object for the trial data
        result=results.result_diversity_trial()
        result.percentiles=percentilemat        
        
        filename='../Data/experimentA4/'+objective_func+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
        savefile(filename,result)
        
        print("Computing data for {} dimensions".format(str(dim)))
        
        # Set-up the function generator
        synth_func=objectives()
        if objective_func=='wildcatwells':
            synth_func.bounds=[(0,100)]*int(dim)
        elif objective_func=='Rastrigin':
            synth_func.bounds=[(-5.12,5.12)]*int(dim)
        elif objective_func=='Rosenbrock':
            synth_func.bounds=[(-10,10)]*int(dim)
        else:
            synth_func.bounds=[(-10,10)]*int(dim)
        synth_func._dim=int(dim)
        synth_func.fun_name=objective_func
        synth_func.budget=50**int(dim)

        #Set-up the diversity training data generator
        synth_data=data_gen.diverse_data_generator()
        synth_data.options['bounds']=synth_func.bounds
        synth_data.options['N_samples']=1000
        synth_data.gamma=1e-4
        synth_data.options['seed']=seed*7
        if objective_func=='wildcatwells':
            if int(dim)==3:
                synth_data.training_size=40
            else:
                synth_data.training_size=10
        elif objective_func=='Sphere':
            if int(dim)==2:
                synth_data.training_size=8
            elif dim==3:
                synth_data.training_size=12
            elif dim==4:
                synth_data.training_size=38
            else:
                synth_data.training_size=75
        elif objective_func=='Rastrigin':
            if int(dim)==2:
                synth_data.training_size=5
            elif dim==3:
                synth_data.training_size=7
            elif dim==4:
                synth_data.training_size=30
            else:
                synth_data.training_size=60
        elif objective_func=='Rosenbrock':
            if int(dim)==2:
                synth_data.training_size=4
            elif dim==3:
                synth_data.training_size=5
            elif dim==4:
                synth_data.training_size=8
            else:
                synth_data.training_size=20
        
        print('Computation started for {} ruggedness'.format(level_of_ruggedness))

        #Vary the synthetic black-box function for the sensitivity analysis
        if level_of_ruggedness=='low':
            synth_func.args={'N':1,'Smoothness':0.8,'rug_freq':1,'rug_amp':0.2,'A':10}
        elif level_of_ruggedness=='medium':
            synth_func.args={'N':1,'Smoothness':0.4,'rug_freq':1,'rug_amp':0.4,'A':10}
        else:
            synth_func.args={'N':1,'Smoothness':0.2,'rug_freq':1,'rug_amp':0.8,'A':10}

        
        #For diversity trial we need to iterate over different percentiles of diversity.
        for percentile in percentilemat:
            print('{}th percentile of diversity being evaluated'.format(percentile))
            
            #Set-up the optimizer
            from Optimizers import optimizer
            BO=optimizer()
            if dim==2:
                BO.max_iter=100
            elif dim>=3:
                BO.max_iter=400
                
            BO.opt="BO-fixparam"
            BO.optima=synth_func.optimal_y
            BO.paramset=hpseries.optdict
            BO.tol=0.1
            BO.minimize=synth_func.minstate
            BO.bounds=synth_func.bounds
            
            #Add appropriate training data to BO.
            BO.datagenmodule=synth_data
            
            #Run the parallel process/Trial
            ray.shutdown()
            time.sleep(10)
            ray.init(runtime_env={"working_dir": "../src"}, num_cpus=10,num_gpus=2,log_to_driver=False)
            from tqdm.autonotebook import tqdm
            
            init_trial=0
            with tqdm(total=ind_trials-init_trial) as pbar:
                for batch_num in range(init_trial//batch_size,ind_trials//batch_size):
                    current_result=ray.get([map_.remote(BO,synth_func,trial_num,percentile) for trial_num in range(batch_num*batch_size,(batch_num+1)*batch_size)])
                    [[result.addresult(percentile,ind_result) for ind_result in current_result if ind_result is not None]]
                    pbar.update(batch_size)
                    #Save Data to the Result trial object and append to an outer list collecting data for each seed in the trial.
                    filename='../Data/experimentA4/'+objective_func+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
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
    10,
    parser.add_argument("--ind_trials", type=int, default=20, help="Number of evaluation for each each seed")
    
    parser.add_argument("--fun_name",choices={"wildcatwells", "Sphere","Rastrigin","Rosenbrock"},default="wildcatwells")
    
    parser.add_argument("--dimension_mat", default="2, 6, 1" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--intent", choices={"gen-data","optimal_init_iter" ,"plot-data","save_compile_data"}, default="gen-data")
    
    parser.add_argument("--grid_type",choices={"difference","comparison"},default="comparison")
    
    args = parser.parse_args()


    success=False
    if args.intent=='gen-data' or args.intent=='both':
        main(args.trials,args.ind_trials,args.dimension_mat,args.fun_name)
    
        
    def plot_data(percentile,dim,level_of_ruggedness,seed,args):
        if args.fun_name=='wildcatwells':
            filename='../Data/experimentA4/wildcatwells/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
            result=load_variable(filename)
            result.opt=100
        else:
            filename='../Data/experimentA4/'+level_of_ruggedness+'/'+str(dim)+'/None/data'+str(seed)+'.pkl'
            result=load_variable(filename)
        if args.grid_type=='comparison':
            try:
                return result.bootstrap(percentile)
            except:
                print(dim,level_of_ruggedness,seed)
                raise Exception('Error')
        else:
            try:
                return result.difference_bootstrap_plotdata()
            except:
                print(dim,level_of_ruggedness,seed)
                raise Exception('Error')

    def get_all_seed_data(dim,level_of_ruggednss,grid_type,args):
        import ray
        @ray.remote
        def map_(obj, f):
            import sys
            sys.path.insert(1, '../src/optimizers')
            sys.path.insert(1, '../src')
            import results
            return f(*obj)
        ray.shutdown()
        ray.init()
        if grid_type=='comparison':
            return ray.get([map_.remote([5,dim,level_of_ruggedness,seed,args], plot_data) for seed in range(10)]),ray.get([map_.remote([95,dim,level_of_ruggedness,seed,args], plot_data) for seed in range(10)])
        elif grid_type=='difference':
            return ray.get([map_.remote(['None',dim,level_of_ruggedness,seed,args], plot_data) for seed in range(10)])
        
        
    if args.intent=='save_compile_data': 
        completedata={}
        dim_mat=np.arange(*args.dimension_mat)
        if args.fun_name=='wildcatwells':
            ruggedness_mat=['low','medium','high']
        else:
            ruggedness_mat=['Sphere','Rosenbrock','Rastrigin']
        for dim in dim_mat:
            completedata[dim]={}
            for level_of_ruggedness in ruggedness_mat:
                completedata[dim][level_of_ruggedness]={}
                grids=['comparison','difference']
                for grid_type in grids:
                    plotdata=get_all_seed_data(dim,level_of_ruggedness,grid_type,args)
                    if grid_type=='comparison':
                        completedata[dim][level_of_ruggedness][grid_type]=[bootstrap(single_percentile_data) for single_percentile_data in np.array(plotdata)[:,:,0,:]]
                    else:
                        completedata[dim][level_of_ruggedness][grid_type]=bootstrap(np.array(plotdata)[:,0,:])
        
        if args.fun_name=='wildcatwells':
            filename='../Data/experimentA4/wildcatwells-plot-data.pkl'
        else:
            filename='../Data/experimentA4/test-plot-data.pkl'
        savefile(filename,completedata)
        
    if args.intent=='plot-data':
        
        if args.fun_name=='wildcatwells':
            filename='../Data/experimentA4/wildcatwells-plot-data.pkl'
        else:
            filename='../Data/experimentA4/test-plot-data.pkl'
        
        if exists(filename):
            completedata=load_variable(filename)
        else:
            raise Exception('Save plot data using --intent "save_compile_data"')
        
        def average_cumoptgap(percentilemat,dim,level_of_ruggedness):
            import numpy as np
            cumoptgap_data=[]
            for seed in range(10):
                if args.fun_name=='wildcatwells':
                    filename='../Data/experimentA4/wildcatwells/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
                    result=load_variable(filename)
                    result.opt=100
                    cumoptgap_data.append(result.percentage_imporvement_cum_opt_gap(percentilemat)[0])
                else:
                    filename='../Data/experimentA4/'+level_of_ruggedness+'/'+str(dim)+'/None/data'+str(seed)+'.pkl'
                    result=load_variable(filename)
                    result.opt=0
                    cumoptgap_data.append(result.percentage_imporvement_cum_opt_gap(percentilemat)[0])
            return np.average(cumoptgap_data)
        
        dim_mat=np.arange(*args.dimension_mat)
        if args.fun_name=='wildcatwells':
            ruggedness_mat=['low','medium','high']
        else:
            ruggedness_mat=['Sphere','Rosenbrock','Rastrigin']
            
        #Generate plot and save it to the appropriate directory.
        with plt.style.context(['science','no-latex']):
            fig, ax = plt.subplots(len(dim_mat), len(ruggedness_mat), sharex='row',figsize=(30,21))
            
            for i,dim in enumerate(dim_mat):
                for j,level_of_ruggedness in enumerate(ruggedness_mat):
                    percentilemat=[5,95]
                    cumoptgap=average_cumoptgap(percentilemat,dim,level_of_ruggedness)
                    cumoptgaptext1='Diversity helped imrove'
                    cumoptgaptext2='performance by '+ str(round(cumoptgap*100,2)) + ' percent'
                    if not exists(filename):
                        plotdata=get_all_seed_data(smoothness,rug_amp,args.grid_type)
                    else:
                        plotdata=completedata[dim][level_of_ruggedness]
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
                        
                        cmap,norm,lc=create_colormap_and_ls(np.arange(len(bootstrap_on_trials[0])),bootstrap_on_trials[0])

                        ax[i,j].add_collection(lc) #Plotting the mean data as a line segment

                        ax[i,j].fill_between(np.arange(len(bootstrap_on_trials[0])),bootstrap_on_trials[1], 
                                        bootstrap_on_trials[2], alpha=0.1, hatch='\\\\\\\\',facecolor='azure') 
                        ax[i,j].axhline(y=0,label='Insignificant difference in perfromance',color='blue')

                        if cumoptgap<0: 
                            c='r'
                        else:
                            c='g'
#                         ax[i,j].text(20,5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                        handles, labels = ax[i,j].get_legend_handles_labels()
                        if level_of_ruggedness!='Rosenbrock':
                            ax[i,j].set_ylim([-20, 10])
                            ax[i,j].text(20,3.5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                        else:
                            ax[i,j].set_ylim([-20000, 10000])
                            ax[i,j].text(20,3500.5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                        ax[i,j].tick_params(axis='x', labelsize=25)
                        ax[i,j].tick_params(axis='y', labelsize=25)
                        
                        if dim==2:
                            ax[i,j].set_xlim([0, int(len(bootstrap_on_trials[0]))/2])
                        else:
                            ax[i,j].set_xlim([0, 100])
                    if j==0:
                        y_label=dim
                        ax[i,j].set_ylabel(str(round(y_label,2)),fontsize='30')
                    if i==len(dim_mat)-1:
                        x_label=level_of_ruggedness
                        ax[i,j].set_xlabel(x_label,fontsize='30')
                        
            if args.grid_type!="comparison":
                right,bottom,width,height=0.83,0.1,0.03,0.8
                cbar_ax = fig.add_axes([right,bottom,width,height])
                cbar=fig.colorbar(
                    mpl.cm.ScalarMappable(cmap=cmap, norm=norm), extend='both',
                    extendfrac='auto',ax=ax.ravel().tolist(),cax=cbar_ax)
                cbar.set_label( label='Effect of diversity on performance of optimizer',fontsize='30')
                cbar.set_ticks([])
            ax[int(len(dim_mat)/2)-1,0].text(-38,-60 , "Number of Dimensions",fontsize='45',rotation=90)
            if args.fun_name=='wildcatwells':
                ax[len(dim_mat)-1,int(len(ruggedness_mat)/2)].text(0,-17 ,"Level of Ruggedness",fontsize='45')
            else:
                ax[len(dim_mat)-1,int(len(ruggedness_mat)/2)].text(20,-33500 ,"Test functions",fontsize='45')
            
            plt.subplots_adjust(
                left  = 0.1,  # the left side of the subplots of the figure
                right = 0.8,    # the right side of the subplots of the figure
                bottom = 0.1,   # the bottom of the subplots of the figure
                top = 0.9,      # the top of the subplots of the figure
                wspace = 0.2,   # the amount of width reserved for blank space between subplots
                hspace = 0.2)   # the amount of height reserved for white space between subplots

            if args.grid_type=="comparison":
                fig.suptitle('Comparison in optimality gap when hyperparameters are fixed for the optimizer',fontsize='48', x=0.4, horizontalalignment='center')
                filename,ext= '../results/ExperimentA4/comparison-plot','.png'
            else:
                fig.suptitle('Absolute difference in optimality gap (y-axis) vs iterations (x-axis) \n when hyperparameters are fixed',fontsize='48', x=0.45, horizontalalignment='center')
                filename,ext= '../results/ExperimentA4/difference-plot','.png'

            try:
                savename=unique_file(filename, ext)
                plt.savefig(os.path.abspath(savename),bbox_inches = 'tight')
                plt.close()
            except:
                savename=unique_file(filename, ext)
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                plt.savefig(os.path.abspath(savename),bbox_inches = 'tight')
         