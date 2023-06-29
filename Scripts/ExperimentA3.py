# ################
# Experiment 3-2-2 - Effects of diversity on the average value of hyper-parameters when the GP is fit to the initial training set.
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

def reject_outliers(data, m=2):
    inlier_idx=np.abs(data - np.mean(data)) < m * np.std(data)
    return [inlier for idx,inlier in enumerate(data) if inlier_idx[idx]]

def compile_all_optdata(dim,level_of_ruggedness,args):
    import results
    res_trial=results.hp_trial()
    for seed in range(args.trials):
        if args.fun_name=='wildcatwells':
            filename='../Data/ExperimentA2/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
        else:
            filename='../Data/ExperimentA2/'+level_of_ruggedness+'/'+str(dim)+'/None/data'+str(seed)+'.pkl'
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
                    filename='../Data/ExperimentA3/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
                    if exists(filename):
                        print(dim,level_of_ruggedness,seed)
                        pass
                    else:
                        available_locs.append((dim,level_of_ruggedness,seed))
                    
        if not reverse:
            dim,level_of_ruggedness,seed=available_locs[0]
        else:
            dim,level_of_ruggedness,seed=available_locs[-1]
            
        print('Working on Trial {} of {}, for dimension= {}, ruggedness ={}'.format(seed+1,trials,dim,level_of_ruggedness))
        
        
        result=results.hp_series()
        filename='../Data/ExperimentA3/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
        savefile(filename,result)
        
        
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
        synth_data.gamma=1e-4
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
    
    
    parser.add_argument("--dimension_mat", default="2, 6, 1" , type=lambda s: tuple(float(item) for item in s.split(',')),
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
            if args.fun_name=='wildcatwells':
                ruggedness_mat=['low','medium','high']
            else:
                ruggedness_mat=['None']
            for level_of_ruggedness in ruggedness_mat:
                result_trial=results.hp_trial()
                for seed in range(args.trials):
                    filename='../Data/ExperimentA3/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data'+str(seed)+'.pkl'
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
            filename='../Data/experimentA3/'+data_type+'.pkl'
            savefile(filename,completedata)
        else:
            return completedata
     
    if args.intent=='save_compile_data':
        compile_existing_data(args.data_save_type,'savedata',args)
        
    if args.intent=='plot-data':
        #Adds optimaldata extracted in experiment3-2-1 a local variable.
        if args.fun_name=='wildcatwells':
            filename='../Data/experimentA2/wildcatwells/object.pkl'
            try:
                with open(filename, 'rb') as f:
                    optdata = pickle.load(f)
            except:
                raise Exception('Optimal Hyperparameter data has not been saved yet for these set of wildcatwells functions, run script Exp A2 with intent to save object data.')
            
        #Creating a more readable version of completedata using pandas
        import pandas as pd
        
        
        dim_mat=np.arange(*args.dimension_mat)
        
        if args.fun_name=='wildcatwells':
            ruggedness_mat=['low','medium','high']
            filename='../Data/experimentA3/'+args.objective_func+'/object.pkl'
            if os.path.exists(filename):
                completedata=load_variable(filename)
            else:
                completedata=compile_existing_data('object','create_var',args)
            for i,dim in enumerate(dim_mat):
                for j,level_of_ruggedness in enumerate(ruggedness_mat):
                    completedata[dim][level_of_ruggedness].opt=100
        else:
            ruggedness_mat=['Sphere','Rosenbrock','Rastrigin']
            completedata={2.0:{},3.0:{}}
            for func_name in ruggedness_mat:
                filename='../Data/experimentA1/'+func_name+'/object.pkl'
                if os.path.exists(filename):
                    current_func_data=load_variable(filename)
                    for dim in current_func_data:
                        try:
                            completedata[dim][func_name]=current_func_data[dim]['None']
                        except KeyError:
                            completedata[dim]={}
                            completedata[dim][func_name]=current_func_data[dim]['None']
                else:
                    args.fun_name=func_name
                    current_func_data=compile_existing_data('object','create_var',args)
                    for dim in current_func_data:
                        try:
                            completedata[dim][func_name]=current_func_data[dim]['None']
                        except KeyError:
                            completedata[dim]={}
                            completedata[dim][func_name]=current_func_data[dim]['None']
                            
        complete_df= pd.DataFrame.from_dict(completedata)
        
        
        percentage_absolute_error= lambda percentile,hyperparameter,res,box_data: [np.abs((box_data[percentile][hyperparameter][seed]-res.optdata[hyperparameter][seed])/res.optdata[hyperparameter][seed])*100 for seed in range(10)]
        
        #Generate plot and save it to the appropriate directory.
        with plt.style.context(['science','no-latex']):
            fig, ax = plt.subplots(len(dim_mat), len(ruggedness_mat),figsize=(30,25))

            for i,dim in enumerate(dim_mat):
                for j,level_of_ruggedness in enumerate(ruggedness_mat):

                    funcname='dim-{} level of ruggedness-{}'.format(dim,level_of_ruggedness)
                    res_trial=compile_all_optdata(dim,level_of_ruggedness,args)
                    res_trial.extract_plotdata()
                    res_trial.extract_optdata()
                    box_data=completedata[dim][level_of_ruggedness].bootstrap_plotdata()
                    figtitle='Hyperparameter distributuion for '+funcname
                    hyperparameter='covar_module.base_kernel.raw_lengthscale'
                    

                    superdict={'data5':reject_outliers(box_data[5][hyperparameter]),
                               'data95':reject_outliers(box_data[95][hyperparameter])}

                    box_5=ax[i,j].boxplot(superdict['data5'], notch=True, positions=[1],patch_artist=True, boxprops=dict(facecolor="C0"))

                    box_95=ax[i,j].boxplot(superdict['data95'], notch=True,positions=[2],patch_artist=True, boxprops=dict(facecolor="C2"))
                    # ax[i,j].legend([box_5["boxes"][0], box_95["boxes"][0]], ["Data 5", "Data 95"], loc='upper right',fontsize='15')
                    for line in res_trial.optdata[hyperparameter]:
                        ax[i,j].axhline(line)
                    ax[i,j].axhline(line,label='Optimal hyperparameters')
                    handles, labels = ax[i,j].get_legend_handles_labels()

                    mean_absolute_error_5=mean_confidence_interval(percentage_absolute_error(5,hyperparameter,res_trial,box_data))
                    mean_absolute_error_95=mean_confidence_interval(percentage_absolute_error(95,hyperparameter,res_trial,box_data))

                    text1='MAE 5 : {:.1e} - {:.1e}'.format(np.round(mean_absolute_error_5[1],2),np.round(mean_absolute_error_5[2],2))
                    text2='MAE 95 : {:.1e} - {:.1e}'.format(np.round(mean_absolute_error_95[1],2),np.round(mean_absolute_error_95[2],2))
                    
                    bottom,top=ax[i,j].get_ylim()
                    ax[i,j].text(0.55,top*0.75, text1 ,fontsize='28')
                    ax[i,j].text(0.55,top*0.55, text2 ,fontsize='28')

#                     ax[i,j].set_ylim([-5, 100])
                    # ax[i,j].tick_params(axis='x', labelsize=16)
                    ax[i,j].tick_params(axis='y', labelsize=25)


                    if j==0:
                        y_label=dim
                        ax[i,j].set_ylabel(str(round(y_label,2)),fontsize='40')
                    if i==len(dim_mat)-1:
                        x_label=level_of_ruggedness
                        ax[i,j].set_xlabel(x_label,fontsize='40')
                        ax[i,j].set_xticklabels([])

            ax[int(len(dim_mat)/2)-1,0].text(-0.2,-355, "Number of dimensions",fontsize='45',rotation=90)
            if args.fun_name=='wildcatwells':
                ax[len(dim_mat)-1,int(len(dim_mat)/2)].text(0.4,50, "Level of ruggedness",fontsize='45')
            else:
                ax[len(dim_mat)-1,int(len(ruggedness_mat)/2)].text(0.7,-2.7*10**6 ,"Test functions",fontsize='45')

            handles, labels = ax[i,j].get_legend_handles_labels()

            fig.legend(handles,labels,loc='lower right',fontsize='35',bbox_to_anchor=(0.43,-0.032))
            fig.legend([box_5["boxes"][0], box_95["boxes"][0]], 
                       ["Less-Diverse Initial Samples", "Diverse Initial Samples"],
                       loc='lower center', fontsize='35',bbox_to_anchor=(0.6,-0.045))

            plt.subplots_adjust(
                left  = 0.1,  # the left side of the subplots of the figure
                right = 0.85,    # the right side of the subplots of the figure
                bottom = 0.12,   # the bottom of the subplots of the figure
                top = 0.9,      # the top of the subplots of the figure
                wspace = 0.2,   # the amount of width reserved for blank space between subplots
                hspace = 0.2)   # the amount of height reserved for white space between subplots

            fig.suptitle('Distribution of lengthscale learned by BO on initial samples',fontsize='55', x=0.87, horizontalalignment='right')
            filename,ext= '../results/ExperimentA3/'+args.fun_name+'_grid_plot', '.png'

            try:
                savename=unique_file(filename, ext)
                plt.savefig(os.path.abspath(savename),bbox_inches = 'tight')#, pad_inches = 0.3)
                plt.close()
            except:
                savename=unique_file(filename, ext)
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                plt.savefig(os.path.abspath(savename) ,bbox_inches = 'tight') #, pad_inches = 0.3)
                plt.close()