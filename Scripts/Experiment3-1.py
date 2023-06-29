################
# Experiment 3-1 - Effects of ν on diversity for wildcatwells function with args {'N':1,'Smoothness':0.2,'rug_freq':1,'rug_amp':0.7} and seed as specified.
# Corresponding issue on git - #
# ################

import sys
sys.path.insert(1, '../src/optimizers')
sys.path.insert(1, '../src')

from objectives import objectives
import data_gen
import numpy as np
import pickle
import dill
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from ipyparallel import Client
import ipyparallel as ipp
import os
import subprocess
import time
import itertools
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

def singleopt(opt,f,percentile,seed):
    counter=0
    
    while True:
        try:
            opt.train=opt.datagenmodule.sample(percentile)
            result= opt.optimize(f)
            result.seed=seed
            break
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Keyboard interrupt')
        except:
            counter+=1
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
    async_results=list()
    for trial in tqdm(range(trials),desc="Tasks scheduled:"):
        args[1].seed=trial
        wildcatwells=args[1].generate_cont()
        args[0].datagenmodule.generate()
        async_results.append(bview.apply_async(fun, args[0],wildcatwells,args[2],trial))
    rc.wait_interactive(async_results)
    results=[ar.get() for ar in list(itertools.compress(async_results,[result.successful() for result in async_results]))]
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

async def main(trials,seed,max_iter):
    
    percentilemat=[5,95]
    nu_mat=[0.5,1.5,2.5,10]
    completedata=[]
    available_locs=[0]
    
    # Set-up the function generator
    synth_func=objectives()
    synth_func.bounds=[(0,100),(0,100)]
    synth_func.seed=seed
    synth_func.args={'N':1,'Smoothness':0.4,'rug_freq':1,'rug_amp':0.4}
    
    #Create the diversity training data generator
    synth_data=data_gen.diverse_data_generator()
    
    
    while len(available_locs)>0:
        from os.path import exists
        available_locs=[]
        for nu in nu_mat:
            filename='../Data/experiment3-1/nu/'+str(nu)+'/data.pkl'
            if exists(filename):
                pass
            else:
                available_locs.append((nu))
        nu=available_locs[0]
        print('Diversity trial started for nu : {}'.format(nu))
        #Create a result object for the trial data
        result=results.result_diversity_trial()
        result.percentiles=percentilemat

        #Start and define the engines
        rc,clust = await startengines()
        parallelcomp=setupengines(rc,clust)
        
        funcname='wildcatwells'+str(seed)
        #For diversity trial we need to iterate over different percentiles of diversity.
        for percentile in percentilemat:
            print('Running Trials with training data from {}th percentile of diversity for {}'.format(percentile,funcname))
            
            #Prepare synthetic data (function and training points) for the parallel process.
            #Set-up the optimizer
            from Optimizers import optimizer
            BO=optimizer()
            BO.max_iter=max_iter
            BO.opt="BO"
            BO.options['tol']=0.1
            BO.options['nu']=nu
            BO.minimize=synth_func.minstate
            BO.bounds=synth_func.bounds
            
            #Add the data gen sampling module to BO after generating appropriate data it.
            
            synth_data.options['seed']=seed*2
            synth_data.options['bounds']=synth_func.bounds
            synth_data.training_size=10
            synth_data.options['N_samples']=10000
            synth_data.gamma=1e-5
            synth_data.generate()
            BO.datagenmodule=synth_data
            
            #Run the parallel process/Trial
            current_result=parallel(trials,singleopt,[BO,synth_func,percentile],parallelcomp)
            [result.addresult(percentile,ind_result) for ind_result in current_result]
        filename='../Data/experiment3-1/nu/'+str(nu)+'/data.pkl'
        savefile(filename,result)
            
        #Stop the engines
        await stopengines(clust)
        completedata.append(result)
    
    return completedata

if __name__=='__main__':
    import sys
    import argparse
    import asyncio
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of Trials of BO optimizer")
    
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed used for objective function generation.")
    
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Max number of runs for BO.")
    
    parser.add_argument("--intent", choices={"gen-data", "plot-data", "save_compile_data"}, default="gen-data")
    parser.add_argument("--grid_type", choices={"difference", "comparison"}, default="difference")

    args = parser.parse_args()
    
    if args.intent=='gen-data' or args.intent=='both':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        completedata=asyncio.run(main(args.trials,args.seed,args.max_iter))
        filename='../Data/experiment3-1/completedata.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(completedata, f)
            
    def compile_existing_data(objective,args):
        """
        objective = dict{savedata, create_var}
        """
        import numpy as np
        
    
        nu_mat=[0.5,1.5,2.5,10]
        objectives = {'savedata':0, 'create_var':0}
        objectives[objective]+=1
        
        completedata=list()
        breakcond=False
        for nu in nu_mat:
            filename='../Data/experiment3-1/nu/'+str(nu)+'/data.pkl'
            if os.path.exists(filename):
                result = load_variable(filename)
            else:
                breakcond=True
            if breakcond:
                break
            completedata.append(result)
        if objective=='savedata':
            filename='../Data/experiment3-1/completedata.pkl'
            savefile(filename,completedata)
        else:
            return completedata
    if args.intent=='save_compile_data':
        compile_existing_data('savedata',args)
    
    nu_mat=[0.5,1.5,2.5,'inf']
    if args.intent=='plot-data' or args.intent=='both':
        with plt.style.context(['science','no-latex']):
            completedata=compile_existing_data('create_var',args)
            fig, ax = plt.subplots(2,2,figsize=(20,20))
            
            for i,nu in enumerate(nu_mat):
                result=completedata[i]
                result.opt=100 #Set the theoretical optimal value of the function here.
                percentilemat=[5,95]
                funcname='wildcatwells'+str(args.seed)
                title='BO with nu =' + str(nu)
                ax.flatten()[i].set_title(title,fontsize='25')
                cumoptgap=result.cum_opt_gap_improvement_overall(percentilemat)[0]
                cumoptgaptext=round(cumoptgap,2)
                if args.grid_type=='comparison':
                    for percentile in percentilemat:
                        label=str(percentile)+'th percentile'
                        ax.flatten()[i].plot(np.arange(args.max_iter+2), result.bootstrap(percentile)[0], '-',label=label) #Plotting the mean data
                        ax.flatten()[i].fill_between(np.arange(args.max_iter+2),result.bootstrap(percentile)[1], 
                                        result.bootstrap(percentile)[2], alpha=0.3) #Plotting the 90% confidence intervals.
                        ax.flatten()[i].text(35,30, cumoptgaptext,fontsize='35')
                    ax.flatten()[i].set_ylim([0, 35])
                    ax.flatten()[i].set_xlim([0, 0.5*args.max_iter])
                    ax.flatten()[i].legend()
                else:
                    bootstrap_on_trials=result.difference_bootstrap_plotdata_by_seed()
                    def create_colormap_and_ls(x,y):
                        # select how to color
                        cmap = (mpl.colors.ListedColormap(['tomato','blue','lime']).with_extremes(over='lime', under='tomato'))
                        bounds = [-1,0.0,1]
                        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                        # get segments
                        xy = np.array([x, y]).T.reshape(-1, 1, 2)
                        segments = np.hstack([xy[:-1], xy[1:]])

                        # make line collection
                        lc = LineCollection(segments, cmap = cmap, norm = norm)
                        lc.set_array(y)
                        return cmap,norm,lc

                    cmap,norm,lc=create_colormap_and_ls(np.arange(args.max_iter+2),bootstrap_on_trials[0])

                    ax.flatten()[i].add_collection(lc) #Plotting the mean data as a line segment

                    ax.flatten()[i].fill_between(np.arange(args.max_iter+2),bootstrap_on_trials[1], 
                                    bootstrap_on_trials[2], alpha=0.1, hatch='\\\\\\\\',facecolor='azure') #Plotting the 95% confidence intervals.
                    ax.flatten()[i].axhline(y=0,label='Insignificant difference in performance',color='blue')

                    if cumoptgap<0: 
                        c='r'
                    else:
                        c='g'
                    ax.flatten()[i].text(20,3.5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                    handles, labels = ax.flatten()[i].get_legend_handles_labels()
                    ax.flatten()[i].set_ylim([-10, 10])
                    ax.flatten()[i].tick_params(axis='x', labelsize=25)
                    ax.flatten()[i].tick_params(axis='y', labelsize=25)
                    ax.flatten()[i].set_xlim([0, int(args.max_iter/2)])
                    
                if args.grid_type!="comparison":
                    right,bottom,width,height=0.87,0.1,0.04,0.75
                    cbar_ax = fig.add_axes([right,bottom,width,height])
                    cbar=fig.colorbar(
                        mpl.cm.ScalarMappable(cmap=cmap, norm=norm), extend='both',
                        extendfrac='auto',ax=ax.ravel().tolist(),cax=cbar_ax, shrink=0.5)
                    cbar.set_label( label='Effect of diversity on performance of optimizer',fontsize='40')
                    cbar.set_ticks([])
                fig.legend(handles,labels,loc='lower center',fontsize=30)

            plt.subplots_adjust(
                left  = 0.1,  # the left side of the subplots of the figure
                right = 0.85,    # the right side of the subplots of the figure
                bottom = 0.1,   # the bottom of the subplots of the figure
                top = 0.85,      # the top of the subplots of the figure
                wspace = 0.2,   # the amount of width reserved for blank space between subplots
                hspace = 0.2)   # the amount of height reserved for white space between subplots
            plt.suptitle('Optimality gap comparing performance less diverse and highly \n diverse examples with different nu',fontsize='44')
            filename,ext= '../results/Experiment3-1/wildcatwells' + str(args.seed), '.eps'
            savename=unique_file(filename, ext)
            os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
            plt.savefig(os.path.abspath(savename),dpi=300)

