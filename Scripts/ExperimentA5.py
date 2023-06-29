# ################
# Experiment 3-2-4 - Effects of training size on the average value of hyper-parameters when the GP is fit to the initial training set.
# Corresponding issue on git - #

# This script generates average hyperparameter data for wildcatwells function with args {'N':1,'Smoothness':0.2,'rug_freq':1,'rug_amp':0.7} and seed in range [0,numfuncs] and and saves the figures generated to Experiment3-2-2 directory.

# Needed arguments:
# #Trials- Number of trials to generate the distribution of data
# #Numfuncs-Number of functions to generate the data for i.e from seed 0 to numfuncs-1 seed.
# #intent- {gen-data: To generate and save data to suitable directory, plot-data: To load data and plot data from the suitable directory}
# ################

import sys
sys.path.insert(1, '../src/optimizers')
sys.path.insert(1, '../src')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from objectives import objectives
import data_gen
import numpy as np
from utils import bootstrap,div_data_gen,closest_N
import random
import pickle
import dill
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kde

from HPparallel import HP_parallel
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
import scipy.stats
import ray


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

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

def reject_outliers(data, m=4):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def compile_all_optdata(dim,level_of_ruggedness,num_trials):
    import results
    res_trial=results.hp_trial()
    for seed in range(num_trials):
        filename='../Data/ExperimentA2/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
        optdict=load_variable(filename)
        res_trial.add_hpseries(optdict)
    return res_trial

@ray.remote(num_gpus=0.1,num_cpus=1)
def HPextract(synth_data,f):
    import numpy as np
    import results
    def fitmodel(X,Y):
        from botorch.fit import fit_gpytorch_model
        import BO
        train_x_ei, train_obj_ei, _ = BO.generate_initial_data(X,Y,False,n=synth_data.training_size)
        mll, model = BO.initialize_model(train_x_ei, train_obj_ei,nu=1.5,covar="matern")
        fit_gpytorch_model(mll);
        return model
    try:
        res=results.hp_result()
        X1=synth_data.sample(95)
        Y1=f(X1).astype(np.float32)
        model1=fitmodel(X1,Y1)
        res.adddiversehp(model1,synth_data)
        
        X2=synth_data.sample(5)
        Y2=f(X2).astype(np.float32)
        model2=fitmodel(X2,Y2)
        res.adddiversehp(model2,synth_data)
    except:
        res=results.hp_result()
        X1=synth_data.sample(95)
        Y1=f(X1).astype(np.float32)
        model1=fitmodel(X1,Y1)
        res.adddiversehp(model1,synth_data)
        
        X2=synth_data.sample(5)
        Y2=f(X2).astype(np.float32)
        model2=fitmodel(X2,Y2)
        res.adddiversehp(model2,synth_data)
    return res

from IPython.display import clear_output

def main(trials,ind_trials,dim_mat,objective_func,reverse=False):
    #Set-up variables
    import results
    available_locs=[0]
    level_of_ruggedness='None'
    while len(available_locs)>0:
        from os.path import exists
        available_locs=[]
        for dim in np.arange(*dim_mat):
            if dim==2:
                training_mat=(2.0,15.0,1)
            elif dim==3:
                training_mat=(2.0,25.0,2)
            elif dim==4:
                training_mat=(2.0,65.0,4)
            else:
                training_mat=(5.0,95.0,5)
            for training_size in np.arange(*training_mat):
                for seed in range(trials):
                    filename='../Data/ExperimentA5/'+objective_func+'/'+str(dim)+'/'+str(training_size)+'/data'+str(seed)+'.pkl'
                    if exists(filename):
                        print(dim,training_size,seed)
                        pass
                    else:
                        available_locs.append((dim,training_size,seed))
                    
        if not reverse:
            dim,training_size,seed=available_locs[0]
        else:
            dim,training_size,seed=available_locs[-1]
            
        print('Working on trial {} of {} for dimension= {}, training_size ={}'.format(seed,trials,dim,training_size))
        
        
        result=results.hp_series()
        filename='../Data/ExperimentA5/'+objective_func+'/'+str(dim)+'/'+str(training_size)+'/data'+str(seed)+'.pkl'
        savefile(filename,result)
        
        
        # Set-up the function generator
        synth_func=objectives()
        synth_func.bounds=[(0,100)]*int(dim)
        synth_func.seed=seed
        if level_of_ruggedness=='low':
            synth_func.args={'N':1,'Smoothness':0.8,'rug_freq':1,'rug_amp':0.2}
        elif level_of_ruggedness=='medium':
            synth_func.args={'N':1,'Smoothness':0.4,'rug_freq':1,'rug_amp':0.4}
        elif level_of_ruggedness=='high':
            synth_func.args={'N':1,'Smoothness':0.2,'rug_freq':1,'rug_amp':0.8}
        else:
            synth_func.args={'A':10}
        synth_func.fun_name=objective_func
        synth_func.budget=50**int(dim)
        if objective_func!='wildcatwells':
            f=synth_func.generate_cont()
        else:
            f=synth_func.generate_cont(from_saved=True,local_dir='C:\\temp\\wildcatwells')
        
        #Set-up the diversity training data generator
        synth_data=data_gen.diverse_data_generator()
        synth_data.options['bounds']=synth_func.bounds
        synth_data.options['N_samples']=1000
        synth_data.options['seed']=seed*2
        synth_data.gamma=1e-4
        synth_data.training_size=training_size
        synth_data.generate()
        
        ##Run the main experiment
        ray.shutdown()
        time.sleep(10)
        resultdata=[]
        ray.init(runtime_env={"working_dir": "../src"}, num_cpus=10,num_gpus=2,log_to_driver=False)
        current_resultdata=ray.get([HPextract.remote(synth_data,f) for _ in range(ind_trials)])
        [resultdata.append(current_result) for current_result in current_resultdata if current_result is not None]

        #Save Data to for each trial.
        result.data=resultdata
        result.updatediverseres()
        result.wwargs=synth_func.args
        result.wwargs['seed']=seed
        savefile(filename,result)
        gc.collect(generation=2)
            

if __name__=='__main__':
    
    import argparse
    import asyncio
    import os
    import gc
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of seeds for each iteration of ruggedness and smoothness that need to be evaluated.")
    parser.add_argument("--ind_trials", type=int, default=50, help="Number of evaluation for each each seed")
    
    
    parser.add_argument("--dimension_mat", default="2, 10, 1" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--fun_name",choices={"wildcatwells", "Sphere","Rastrigin","Rosenbrock"},default="wildcatwells")
    
    
    parser.add_argument("--intent", choices={"gen-data","optimal_init_iter" ,"plot-data","save_compile_data"}, default="gen-data")
    
    parser.add_argument("--data_save_type", choices={"plot_data", "bootstrap_plot_data", "optdata", "bootstrap_optdata", "object"},default="object")
    
    args = parser.parse_args()
 
        
    if args.intent=='gen-data':
        main(args.trials,args.ind_trials,args.dimension_mat,args.fun_name)
        
            
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
            for dim in np.arange(*dim_mat):
                for training_size in training_mat:
                    filename='../Data/ExperimentA5/'+str(dim)+'/'+str(training_size)+'/data.pkl'
                    if exists(filename):
                        result = load_variable(filename)
                        result_trial.add_hpseries(result)
                    else:
                        break
                else:
                    if data_type=='object':
                        completedata[dim]=result_trial
                    else:
                        completedata[dim]=pd.DataFrame.from_dict(result_trial.extract_data(data_type))
            if breakcond:
                break
        if objective=='savedata':
            filename='../Data/experimentA5/'+data_type+'.pkl'
            savefile(filename,completedata)
        else:
            return completedata
     
    if args.intent=='save_compile_data':
        compile_existing_data(args.data_save_type,'savedata',args)
        
    if args.intent=='plot-data':
        pass