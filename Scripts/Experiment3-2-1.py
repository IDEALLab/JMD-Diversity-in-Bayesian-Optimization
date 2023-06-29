# ################
# Experiment 3-2-1 - Determining the optimal hyper-parameters for a set of 10 wildcat-wells functions.
# Corresponding issue on git - #

# This script generates optimal hyperparameter data for wildcatwells with args {'N':1,'Smoothness':0.2,'rug_freq':1,'rug_amp':0.7} and seed in range [0,numfuncs], and saves the figures generated to Experiment3-2-1 directory.

# Needed arguments:
# #Numfuncs-Number of functions to generate the data for i.e from seed 0 to numfuncs-1 seed.
# #intent- {gen-data: To generate and save data to suitable directory, plot-data: To load data and plot data from the suitable directory}
# ################

import sys
sys.path.insert(1, '../src/Optimizers')
sys.path.insert(1, '../src')

from objectives import objectives
import data_gen
import numpy as np
import pickle
import dill
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kde

from ipyparallel import Client
import ipyparallel as ipp
import os
import subprocess
import time
from tqdm import tqdm
import itertools
import pandas as pd
import results
from os.path import exists
if sys.platform.startswith('win'):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    

def reject_outliers(data, m=4):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
    
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


def HPextract(X,f):
    import numpy as np
    import results
    def fitmodel(X,Y):
        from botorch.fit import fit_gpytorch_model
        import BO
        train_x_ei, train_obj_ei, _ = BO.generate_initial_data(X,Y,False,n=len(Y))
        mll, model = BO.initialize_model(train_x_ei, train_obj_ei,nu=1.5,covar="matern")
        fit_gpytorch_model(mll);
        return model
    try:
        res=results.hp_result()
        Y=np.array([f(X[a])[0] for a in range(len(X))])
        model=fitmodel(X,Y)
        res.addhp(model)
    except:
        res=results.hp_result()
        X=X[:len(X)-1]
        Y=np.array([f(X[a])[0] for a in range(len(X))])
        try:
            model=fitmodel(X,Y)
            res.addhp(model)
        except:
            res.hyperparameters= None
    return res

def parallel(fun,args,parallelcomp):
    """
    fun: function to be parallized
    args: RxN list with args for each run.
    runs: number of runs of the function.
    """
    rc,bview,_=parallelcomp
    async_result= bview.map_async(fun, args[0],args[1])
    async_result.wait_interactive()
    result=async_result.get()
    return result

def parallel_miniter(fun,init_trainset,randomlist,f,parallelcomp):
    """
    fun: function to be parallized
    args: RxN list with args for each run.
    runs: number of runs of the function.
    """
    def update_pos_mat(pos_mat,bisection_search_results):
        binary_result=[]
        for result in bisection_search_results:
            if result.hyperparameters is not None:
                check=result.hyperparameters['likelihood.noise_covar.raw_noise'][0]<0.1
            else:
                check=False
            binary_result.append(check)
        try:
            arg_min=min(np.where(np.array(binary_result)==True)[0])
            max_width=pos_mat[1]-pos_mat[0]
            gap=max_width/len(pos_mat)
            updated_pos_range=(pos_mat[arg_min-1],pos_mat[arg_min],gap)
        except ValueError:
            gap=100
            updated_pos_range=(pos_mat[len(pos_mat)-1],pos_mat[len(pos_mat)-1]+1000,gap)
        max_width=pos_mat[len(pos_mat)-1]-pos_mat[len(pos_mat)-2]
        return max_width,np.arange(*updated_pos_range,dtype=int)
    
    rc,bview,_=parallelcomp
    n_runs=min(len(rc),10)
    init_gap=int(len(randomlist)/n_runs)
    found=False
    counter=0
    while not found:
        async_results=[]
        counter+=1
        if counter==1:
            pos_mat=np.arange(0,len(randomlist),init_gap)
        X=[np.concatenate([init_trainset,randomlist[:n]]) for n in pos_mat]
        print('Bi-Section search, count : {}, current_range: {}'.format(counter,[pos_mat[0],pos_mat[-1]]))
        for x in tqdm(X,desc="Bisection scheduled:"):
            async_results.append(bview.apply_async(fun,x,f))
        
        rc.wait_interactive(async_results)
        result=[ar.get() for ar in list(itertools.compress(async_results,[result.successful() for result in async_results]))]
        filename='../Data/experiment3-2-1/result_obj.pkl'
        savefile(filename,result)
        max_width,pos_mat=update_pos_mat(pos_mat,result)
        found=max_width<10
    optimal_range=[pos_mat[0],pos_mat[n_runs-1]]
    return optimal_range

def setupengines(rc,clust):
    bview=rc.load_balanced_view()
    dview=rc.direct_view()
    parallelcomp=[rc,bview,dview]
    if sys.platform.startswith('win'):
        dview.map(os.chdir, [os.path.abspath('..\src')]*len(rc))
    else:
        dview.map(os.chdir, [os.path.abspath('../src')]*len(rc))
    with dview.sync_imports():
        import data_gen
        import objectives
    
    if sys.platform.startswith('win'):
        dview.map(os.chdir, [os.path.abspath('..\src\optimizers')]*len(rc))
    else:
        dview.map(os.chdir, [os.path.abspath('../src/Optimizers')]*len(rc))
    with dview.sync_imports():
        import BO
    print("Engine Set-up complete")
    return parallelcomp

import math
import results
from IPython.display import clear_output

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


async def search_inititer(sm_range,rug_range,alt_rug=None):
    
    #Set-up variables
    smoothness_mat=np.arange(*sm_range).round(2)
    iterations=2500
    for smoothness in smoothness_mat:
        miniterdata=[]
        rug_data_trial_data=list()
        ruggedness_mat=np.arange(*rug_range).round(2)
        
        #Defines the conditions for the loop if partial data exists for the function
        if  smoothness==smoothness_mat[0] and alt_rug is not None:
            ruggedness_mat=np.arange(*(alt_rug,*rug_range[1:])).round(2)
            
        for rug_amp in ruggedness_mat:

            # Set-up the function generator
            synth_func=objectives()
            synth_func.bounds=[(0,100),(0,100)]
            synth_func.seed=0
            synth_func.args={'N':1,'Smoothness':smoothness,'rug_freq':1,'rug_amp':rug_amp}
            f=synth_func.generate_cont()

            #Set-up the diversity training data generator
            synth_data=data_gen.diverse_data_generator()
            synth_data.options['bounds']=synth_func.bounds
            synth_data.options['N_samples']=10000
            synth_data.options['seed']=0
            synth_data.gamma=1e-5
            synth_data.training_size=10
            synth_data.generate()
            init_trainset=synth_data.sample(95)
            randomlist=synth_data.optgammarand(init_trainset,iterations)

            #Create a result object for the series data
            resultdata=[]
            
            try:
                #Start and define the engines
                rc,clust = await startengines()

                parallelcomp=setupengines(rc,clust)

                #Run the parallel process
                result_data=parallel_miniter(HPextract,init_trainset,randomlist,f,parallelcomp)
            except KeyboardInterrupt:
                await stopengines(clust)
                sys.exit(0)
                

            #Restart the engines
            await stopengines(clust)

            filename='../Data/experiment3-2-1/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/optimal_range.pkl'
            savefile(filename,result_data)
            miniterdata.append(result_data)
    return miniterdata


import signal
async def main(trials,sm_range,rug_range,reverse=False):
    #Set-up variables
    smoothness_mat=np.arange(*sm_range).round(2)
    
    init_iter=1000
    iterations=1200
    completedata=[]
    available_locs=[0]
    
    while len(available_locs)>0:
        from os.path import exists
        available_locs=[]
        for smoothness in np.arange(*sm_range).round(2):
            for rug_amp in np.arange(*rug_range).round(2):
                for dataseed in range(trials):
                    filename='../Data/experiment3-2-1/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
                    if exists(filename):
                        pass
                    else:
                        available_locs.append((smoothness,rug_amp,dataseed))
                    
        if not reverse:
            smoothness,rug_amp,dataseed=available_locs[0]
        else:
            smoothness,rug_amp,dataseed=available_locs[-1]
        print('Working on Trial {} of {}, for smoothness= {}, rug_amp ={}'.format(dataseed+1,trials,smoothness,rug_amp))
        
        # Set-up the function generator
        synth_func=objectives()
        synth_func.bounds=[(0,100),(0,100)]
        synth_func.seed=0
        synth_func.args={'N':1,'Smoothness':smoothness,'rug_freq':1,'rug_amp':rug_amp}
        f=synth_func.generate_cont()

        #Set-up the diversity training data generator
        synth_data=data_gen.diverse_data_generator()
        synth_data.options['bounds']=synth_func.bounds
        synth_data.options['N_samples']=10000
        synth_data.options['seed']=dataseed*2
        synth_data.gamma=1e-5
        synth_data.training_size=10
        synth_data.generate()
        init_trainset=synth_data.sample(95)
        randomlist=synth_data.optgammarand(init_trainset,iterations)

        #Prepare synthetic data (function and training points) for the parallel process.
        synthlist,flist=[],[]
        for n in range(iterations):
            training_set=np.concatenate([init_trainset,randomlist[:n]])
            synthlist.append(training_set)
            flist.append(f)

        #Create a result object for the series data
        result=results.hp_series()
        resultdata=[]
        
        # Placeholder file so that other workers do not schedule the same work.
        filename='../Data/experiment3-2-1/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        savefile(filename,result)
        
        #Start and define the engines
        rc,clust = await startengines()

        parallelcomp=setupengines(rc,clust)

        #Run the parallel process
        current_resultdata=parallel(HPextract,[synthlist[init_iter:iterations],flist[init_iter:iterations]],parallelcomp)
        [resultdata.append(current_result) for current_result in current_resultdata]
        rc.purge_everything()
                

        #Restart the engines

        await stopengines(clust)

        #Save Data to for each trial.
        result.data=resultdata
        result.updateres()
        result.wwargs=synth_func.args
        # result.extract_hyperparam()
        filename='../Data/experiment3-2-1/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        savefile(filename,result)
    return completedata


if __name__=='__main__':
    import argparse
    import asyncio
    import os
    import gc
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of functions need to be evaluated.")
    
    parser.add_argument("--smoothness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--ruggedness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--intent", choices={"gen-data","optimal_init_iter" ,"plot-data","save_compile_data"}, default="gen-data")
    
    parser.add_argument("--data_save_type", choices={"plot_data", "bootstrap_plot_data", "optdata", "bootstrap_optdata", "object"},default="object")
    
    parser.add_argument("--reverse_gen_direction",action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    if args.intent=='gen-data':
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        completedata=asyncio.run(main(args.trials,
                                      args.smoothness_range,
                                      args.ruggedness_range,
                                      reverse=args.reverse_gen_direction))

    elif args.intent=='optimal_init_iter':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        miniterdata=asyncio.run(search_inititer(args.smoothness_range,args.ruggedness_range))
        #Save Data to a pkl file
        filename='../Data/experiment3-2-1/init-iterdata','.pkl'
        savename=unique_file(filename, ext)
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        with open(savename, 'wb') as f:
            pickle.dump(miniterdata, f)
    
    #Create optimality hyperparameter dictionary to be used in the Experiment 3-2-3 and 3-2-2.
    
        
    def plot_data(percentile,smoothness,rug_amp,dataseed,plot_type):
        filename='../Data/experiment3-2-3/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        result=load_variable(filename)
        if plot_type=='comparison':
            return result.bootstrap(percentile)
        else:
            return result.difference_bootstrap_plotdata()
            
        def get_all_seed_data(smoothness,rug_amp,plot_type):
            import ray
            @ray.remote
            def map_(obj, f):
                import sys
                sys.path.insert(1, '../src/optimizers')
                sys.path.insert(1, '../src')
                import results
                return f(*obj)
    
            if plot_type=='comparison':
                return ray.get([map_.remote([5,smoothness,rug_amp,seed,plot_type], plot_data) for seed in range(100)]),ray.get([map_.remote([95,smoothness,rug_amp,seed,plot_type], plot_data) for seed in range(100)])
            else:
                return ray.get([map_.remote(['None',smoothness,rug_amp,seed,plot_type], plot_data) for seed in range(100)])
    
    def compile_existing_data(data_type,objective,args):
        """
        data_type = dict{plot_data, bootstrap_plot_data, optdata, bootstrap_optdata, object}
        objective = dict{savedata, create_var}
        """
        import numpy as np
        data_types={'plot_data':0, 'bootstrap_plot_data':0, 'optdata':0, 'bootstrap_optdata':0,'object':0}
        data_types[data_type]+=1
        
        objectives = {'savedata':0, 'create_var':0}
        objectives[objective]+=1
        
        completedata=dict()
        breakcond=False
        for smoothness in np.arange(*args.smoothness_range).round(2):
            completedata[smoothness]=dict()
            for rug_amp in np.arange(*args.ruggedness_range).round(2):
                result_trial=results.hp_trial()
                for dataseed in range(args.trials):
                    filename='../Data/experiment3-2-1/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
                    if os.path.exists(filename):
                        result = load_variable(filename)
                        result_trial.add_hpseries(result)
                    else:
                        breakcond=True
                    if breakcond:
                        break
                if breakcond:
                    break
                else:
                    if data_type=='object':
                        completedata[smoothness][rug_amp]=result_trial
                    else:
                        completedata[smoothness][rug_amp]=pd.DataFrame.from_dict(result_trial.extract_data(data_type))
            if breakcond:
                break
        if objective=='savedata':
            filename='../Data/experiment3-2-1/'+data_type+'.pkl'
            savefile(filename,completedata)
        else:
            return completedata
     
    if args.intent=='save_compile_data':
        compile_existing_data(args.data_save_type,'savedata',args)
            
    if args.intent=='plot-data' or args.intent=='both':
        
        # Creating a more readable version of completedata using pandas
        complete_data=compile_existing_data('plot_data','create_var',args)
        complete_df= pd.DataFrame.from_dict(complete_data)
        
        # print('progress: smoothness:  {} rug_amp: {}'.format(smoothness_key,ruggedness_key))
        init_iter=1000
        iterations=1200
        #Plot the data and save image
        seed=0
        for smoothness_key in complete_df.columns:
            for ruggedness_key in complete_df.T.columns:
                    with plt.style.context(['science','no-latex']):
                        fig = plt.figure(figsize=(18, 13))
                        outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
                        for count,hyperparameter in enumerate(complete_df[smoothness_key][ruggedness_key].columns):
                            funcname='smoothness-{}-ruggedness_amplitude-{}'.format(smoothness_key,ruggedness_key)
                            inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[count], wspace=0.0, hspace=0.0,width_ratios=[2, 1])
                            ax_plot = plt.Subplot(fig, inner[0])
                            ax_hist = plt.Subplot(fig, inner[1])
                            title= hyperparameter 

                            ax_plot.set_title(str(title))
                            ax_plot.set_xlabel('iterations')
                            ax_plot.set_ylabel('hyperparameter value')
                            ax_plot.plot(np.arange(init_iter,iterations),complete_df[smoothness_key][ruggedness_key][hyperparameter][0],label='Hyperparameter values observed',color='b')

                            try:
                                density = kde.gaussian_kde(reject_outliers(complete_df[smoothness_key][ruggedness_key][hyperparameter][seed]))
                            except:
                                noise = np.random.normal(0,2e-2,len(complete_df[smoothness_key][ruggedness_key][hyperparameter][0]))
                                density = kde.gaussian_kde(reject_outliers(complete_df[smoothness_key][ruggedness_key][hyperparameter][seed]+noise))

                            y_range=np.round(ax_plot.get_ylim())
                            try:
                                y=np.arange(*y_range,(y_range[1]-y_range[0])/100)
                            except:
                                y=np.arange(0,1e-1,100)
                            
                            ax_hist.plot(density(y),y,color='r', label='Kernel Density Plot')
                            ax_hist.set_ylim(ax_plot.get_ylim())
                            fig.add_subplot(ax_plot)
                            fig.add_subplot(ax_hist)
                            ax_hist.axes.xaxis.set_ticklabels([])
                            ax_hist.axes.yaxis.set_ticklabels([])
                            lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
                            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
                            fig.legend(lines[0:2], labels[0:2],fontsize='15',loc='lower center')
                        fig.suptitle('Finding optimal hyperparameters for smoothness ={} and ruggedness amplitude = {} '.format(smoothness_key,ruggedness_key),fontsize='25')
                        savename='../results/Experiment3-2-1/'+funcname+'.png'
                        os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
                        plt.savefig(os.path.abspath(savename))
    gc.collect(generation=2)
