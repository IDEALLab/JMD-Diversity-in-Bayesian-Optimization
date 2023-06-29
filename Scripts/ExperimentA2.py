# ################
# Experiment A2 - Determining the optimal hyper-parameters for a set of 10 wildcat-wells functions.
# Corresponding issue on git - #

# This script generates optimal hyperparameter data for wildcatwells over 'low', 'medium' and high ruggedness and also varying in dimensions.

# Needed arguments:

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

import itertools
import gc
import ray

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


@ray.remote(num_gpus=0.1,num_cpus=1)
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
        Y=f(X)
        model=fitmodel(X,Y)
        res.addhp(model)
    except:
        res=results.hp_result()
        X=X[:len(X)-1]
        Y=f(X)
        try:
            model=fitmodel(X,Y)
            res.addhp(model)
        except:
            res.hyperparameters= None
    return res


import signal
def main(trials,dim_mat,objective_func,reverse=False):
    #Set-up variables
    
    init_iter=1000
    iterations=1100
    completedata=[]
        
    percentilemat=[5,95]
    
    available_locs=[0]
    batch_size=10
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
                    filename='../Data/ExperimentA2/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
                    if exists(filename):
                        print(dim,level_of_ruggedness,seed)
                        pass
                    else:
                        available_locs.append((dim,level_of_ruggedness,seed))
                    
        if not reverse:
            dim,level_of_ruggedness,seed=available_locs[0]
        else:
            dim,level_of_ruggedness,seed=available_locs[-1]
        print('Working on Trial {} of {}, for dim = {}, level_of_ruggeness ={}'.format(seed+1,trials,dim,level_of_ruggedness))
        
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

        #Vary the synthetic black-box function for the sensitivity analysis
        if level_of_ruggedness=='low':
            synth_func.args={'N':1,'Smoothness':0.8,'rug_freq':1,'rug_amp':0.2,'A':10}
        elif level_of_ruggedness=='medium':
            synth_func.args={'N':1,'Smoothness':0.4,'rug_freq':1,'rug_amp':0.4,'A':10}
        else:
            synth_func.args={'N':1,'Smoothness':0.2,'rug_freq':1,'rug_amp':0.8,'A':10}

        if objective_func!='wildcatwells':
            f=synth_func.generate_cont()
        else:
            f=synth_func.generate_cont(from_saved=True,local_dir='C:\\temp\\wildcatwells')

        #Set-up the diversity training data generator
        synth_data=data_gen.diverse_data_generator()
        synth_data.options['bounds']=synth_func.bounds
        synth_data.options['N_samples']=10000
        synth_data.options['seed']=seed*2
        synth_data.gamma=1e-5
        synth_data.training_size=10
        synth_data.generate()
        init_trainset=synth_data.sample(5)
        randomlist=synth_data.optgammarand(init_trainset,iterations)

        #Prepare synthetic data (function and training points) for the parallel process.
        synthlist=[]
        for n in range(iterations):
            training_set=np.concatenate([init_trainset,randomlist[:n]])
            synthlist.append(training_set)

        #Create a result object for the series data
        result=results.hp_series()
        resultdata=[]
        
        # Placeholder file so that other workers do not schedule the same work.
        filename='../Data/ExperimentA2/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
        savefile(filename,result)

        #Run the parallel process
        ray.shutdown()
        time.sleep(10)
        current_resultdata=[]
        ray.init(runtime_env={"working_dir": "../src"}, num_cpus=10,num_gpus=2,log_to_driver=False)
        from tqdm.autonotebook import tqdm
        with tqdm(total=iterations-init_iter) as pbar:
            for batch_num in range(iterations-init_iter//batch_size):
                current_data=synthlist[init_iter+batch_num*batch_size:init_iter+(batch_num+1)*batch_size]
                current_batchresult=ray.get([HPextract.remote(X,f) for X in current_data])
                current_resultdata=current_resultdata+current_batchresult
                pbar.update(batch_size)
        [resultdata.append(current_result) for current_result in current_resultdata if current_result is not None]

        #Save Data to for each trial.
        result.data=resultdata
        result.updateres()
        result.wwargs=synth_func.args
        # result.extract_hyperparam()
        filename='../Data/ExperimentA2/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
        savefile(filename,result)


if __name__=='__main__':
    import argparse
    import asyncio
    import os
    import gc
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of functions need to be evaluated.")
    
    parser.add_argument("--dimension_mat", default="2, 6, 1" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--fun_name",choices={"wildcatwells", "Sphere","Rastrigin","Rosenbrock"},default="wildcatwells")
    
    
    parser.add_argument("--intent", choices={"gen-data","optimal_init_iter" ,"plot-data","save_compile_data"}, default="gen-data")
    
    parser.add_argument("--data_save_type", choices={"plot_data", "bootstrap_plot_data", "optdata", "bootstrap_optdata", "object"},default="object")
    
    parser.add_argument("--reverse_gen_direction",action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    if args.intent=='gen-data':
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        main(args.trials,args.dimension_mat,args.fun_name,reverse=args.reverse_gen_direction)

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
        filename='../Data/experiment3-2-3/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(seed)+'.pkl'
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
        for dim in np.arange(*args.dimension_mat):
            completedata[dim]=dict()
            if args.fun_name=='wildcatwells':
                ruggedness_mat=['low','medium','high']
            else:
                ruggedness_mat=['None']
            for level_of_ruggedness in ruggedness_mat:
                result_trial=results.hp_trial()
                for seed in range(args.trials):
                    filename='../Data/ExperimentA2/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
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
                        completedata[dim][level_of_ruggedness]=result_trial
                    else:
                        completedata[dim][level_of_ruggedness]=pd.DataFrame.from_dict(result_trial.extract_data(data_type))
            if breakcond:
                break
        if objective=='savedata':
            filename='../Data/experimentA2/'+args.fun_name+'/'+data_type+'.pkl'
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
