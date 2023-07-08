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

def compile_all_optdata(smoothness,rug_amp):
    import results
    res_trial=results.hp_trial()
    for dataseed in range(100):
        filename='../Data/experiment3-2-1/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
        optdict=load_variable(filename)
        res_trial.add_hpseries(optdict)
    return res_trial

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
        Y1=np.array([f(X1[a])[0] for a in range(len(X1))])
        model1=fitmodel(X1,Y1)
        res.adddiversehp(model1,synth_data)
        
        X2=synth_data.sample(5)
        Y2=np.array([f(X2[a])[0] for a in range(len(X2))])
        model2=fitmodel(X2,Y2)
        res.adddiversehp(model2,synth_data)
    except:
        res=results.hp_result()
        X1=synth_data.sample(95)
        Y1=np.array([f(X1[a])[0] for a in range(len(X1))])
        model1=fitmodel(X1,Y1)
        res.adddiversehp(model1,synth_data)
        
        X2=synth_data.sample(5)
        Y2=np.array([f(X2[a])[0] for a in range(len(X2))])
        model2=fitmodel(X2,Y2)
        res.adddiversehp(model2,synth_data)
    return res

def parallel(fun,trials,args):
    """
    fun: function to be parallized
    args: RxN list with args for each run.
    runs: number of runs of the function.
    """
    synth_data,f,parallelcomp=args
    import results
    result=results.hp_series()
    rc,bview,_=parallelcomp
    async_results=[]
    dir(synth_data)
    for _ in tqdm(range(trials),desc="Scheduling the Individual Trial for current function and seed:"):
        synth_data.generate()
        async_results.append(bview.apply_async(fun,synth_data,f))
    rc.wait_interactive(async_results)
    print('len_results:',len(async_results))
    result.data=[ar.get() for ar in list(itertools.compress(async_results,[result.successful() for result in async_results]))]
    print('len_results_after:',len(result.data))
    result.updatediverseres()
    return result

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
        import BO
    print("Engine Set-up complete")
    return parallelcomp

from IPython.display import clear_output

async def main(trials,ind_trials,sm_range,rug_range,alt_rug=None,alt_seed=None):
    #Set-up variables
    completedata={}
    smoothness_mat=np.arange(*sm_range).round(2)
    
    for smoothness in smoothness_mat:
        
        rug_data_trial_data=list()
        ruggedness_mat=np.arange(*rug_range).round(2)
        
        #Defines the conditions for the loop if partial data exists for the function
        if  smoothness==smoothness_mat[0] and alt_rug is not None:
            ruggedness_mat=np.arange(*(alt_rug,*rug_range[1:])).round(2)
        completedata[smoothness]={}
        for rug_amp in ruggedness_mat:
            seed_mat=range(trials)
            completedata[smoothness][rug_amp]={}
            #Defines the conditions for the loop if partial data exists for the function
            if  smoothness==smoothness_mat[0] and rug_amp==alt_rug and alt_seed is not None:
                seed_mat=range(*(alt_seed,trials))
            
            rc,clust = await startengines()
            parallelcomp=setupengines(rc,clust)
            for dataseed in seed_mat:
                print('Working on Trial {} of {}, for smoothness= {}, rug_amp ={}'.format(dataseed+1,trials,smoothness,rug_amp))
                
                # Set-up the function generator
                synth_func=objectives()
                synth_func.bounds=[(0,100),(0,100)]
                synth_func.seed=dataseed
                synth_func.args={'N':1,'Smoothness':smoothness,'rug_freq':1,'rug_amp':rug_amp}
                f=synth_func.generate_cont()

                #Set-up the diversity training data generator
                synth_data=data_gen.diverse_data_generator()
                synth_data.options['bounds']=synth_func.bounds
                synth_data.options['N_samples']=10000
                synth_data.options['seed']=dataseed*2
                synth_data.gamma=1e-5
                synth_data.training_size=10
                
                args=[synth_data,f,parallelcomp]
                result=parallel(HPextract,ind_trials,args)
            
                filename='../Data/experiment3-2-2/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
                savefile(filename,result)
                
                completedata[smoothness][rug_amp][dataseed]=result
                
            #Stop the engines
            await stopengines(clust)
    return completedata

if __name__=='__main__':
    
    import argparse
    import asyncio
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of seeds for each iteration of ruggedness and smoothness that need to be evaluated.")
    parser.add_argument("--ind_trials", type=int, default=100, help="Number of evaluation for each each seed")
    
    parser.add_argument("--smoothness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--ruggedness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    
    parser.add_argument("--intent", choices={"gen-data","optimal_init_iter" ,"plot-data","save_compile_data"}, default="gen-data")
    
    parser.add_argument("--data_save_type", choices={"plot_data", "bootstrap_plot_data", "optdata", "bootstrap_optdata", "object"},default="object")
    
    parser.add_argument("--plot_type", choices={ "grid", "individual"},default="grid")
    
    parser.add_argument("--check_previous_data",action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    
    if args.check_previous_data:
        # Checks if there is data that already exists for the current trials and skips over those.
        from os.path import exists
        for smoothness in np.arange(*args.smoothness_range).round(2):
            for rug_amp in np.arange(*args.ruggedness_range).round(2):
                for dataseed in range(args.trials):
                    filename='../Data/experiment3-2-2/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
                    if exists(filename):
                        breakcond=False
                    else:
                        current_smoothness=smoothness
                        current_rug_amp=rug_amp
                        current_seed=dataseed
                        breakcond=True
                    if breakcond:
                        break
                if breakcond:
                    break
            if breakcond:
                break
                
    if args.intent=='gen-data' and not args.check_previous_data:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        completedata=asyncio.run(main(args.trials,args.ind_trials,args.smoothness_range,args.ruggedness_range))
        #Save Data to a pkl file
        filename='../Data/experiment3-2-2/overall-data.pkl'
        savefile(filename,completedata)
        
    elif  args.intent=='gen-data' and args.check_previous_data:
        current_smoothness_range=(current_smoothness,*args.smoothness_range[1:])
        
        
        completedata=asyncio.run(main(args.trials,
                                      args.ind_trials,
                                      current_smoothness_range,
                                      args.ruggedness_range,
                                      alt_rug=current_rug_amp,
                                      alt_seed=current_seed))
    else:
        filename='../Data/experiment3-2-2/object.pkl'
        with open(filename, 'rb') as f:
            completedata = pickle.load(f)    #Create optimality hyperparameter dictionary to be used in the Experiment 3-2-3 and 3-2-2.
            
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
                    filename='../Data/experiment3-2-2/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data'+str(dataseed)+'.pkl'
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
            filename='../Data/experiment3-2-2/'+data_type+'.pkl'
            savefile(filename,completedata)
        else:
            return completedata
     
    if args.intent=='save_compile_data':
        compile_existing_data(args.data_save_type,'savedata',args)
        
    if args.intent=='plot-data':
        #Adds optimaldata extracted in experiment3-2-1 a local variable.
        filename='../Data/experiment3-2-1/object.pkl'
        try:
            with open(filename, 'rb') as f:
                optdata = pickle.load(f)
        except:
            raise Exception('Optimal Hyperparameter data has not been saved yet for these set of wildcatwells functions, run script Exp 3-2-1 with intent to save object data.')
            
        #Creating a more readable version of completedata using pandas
        import pandas as pd
        complete_df= pd.DataFrame.from_dict(completedata)
        
        smoothness_mat=np.arange(*args.smoothness_range).round(2)
        ruggedness_mat=np.arange(*args.ruggedness_range).round(2)
        
        percentage_absolute_error= lambda percentile,hyperparameter,res,box_data: [np.abs((box_data[percentile][hyperparameter][seed]-res.optdata[hyperparameter][seed])/res.optdata[hyperparameter][seed])*100 for seed in range(100)]
        
        if args.plot_type=='grid':
            #Generate plot and save it to the appropriate directory.
            with plt.style.context(['science','no-latex']):
                fig, ax = plt.subplots(len(smoothness_mat), len(ruggedness_mat), sharex='col', sharey='row',figsize=(30,21))

                for i,smoothness in enumerate(smoothness_mat):
                    for j,rug_amp in enumerate(ruggedness_mat):
                        
                        funcname='smoothness-{}-ruggedness_amplitude-{}'.format(smoothness,rug_amp)
                        res_trial=compile_all_optdata(smoothness,rug_amp)
                        res_trial.extract_plotdata()
                        res_trial.extract_optdata()
                        box_data=completedata[smoothness][rug_amp].bootstrap_plotdata()
                        figtitle='Hyperparameter distributuion for '+funcname
                        hyperparameter='covar_module.base_kernel.raw_lengthscale'
                        
                        superdict={'data5':box_data[5][hyperparameter],
                                   'data95':box_data[95][hyperparameter]}
                        
                        box_5=ax[i,j].boxplot(superdict['data5'], notch=True, positions=[1],patch_artist=True, boxprops=dict(facecolor="C0"))
    
                        box_95=ax[i,j].boxplot(superdict['data95'], notch=True,positions=[2],patch_artist=True, boxprops=dict(facecolor="C2"))
                        # ax[i,j].legend([box_5["boxes"][0], box_95["boxes"][0]], ["Data 5", "Data 95"], loc='upper right',fontsize='15')
                        for line in res_trial.optdata[hyperparameter]:
                            ax[i,j].axhline(line)
                        ax[i,j].axhline(line,label='Optimal hyperparameters')
                        handles, labels = ax[i,j].get_legend_handles_labels()
                        
                        mean_absolute_error_5=mean_confidence_interval(percentage_absolute_error(5,hyperparameter,res_trial,box_data))
                        mean_absolute_error_95=mean_confidence_interval(percentage_absolute_error(95,hyperparameter,res_trial,box_data))
                        
                        text1='MAE 5 : {} - {}'.format(*np.round(mean_absolute_error_5[1:],2))
                        text2='MAE 95 : {} - {}'.format(*np.round(mean_absolute_error_95[1:],2))
                        
                        ax[i,j].text(0.75,48, text1 ,fontsize='28')
                        ax[i,j].text(0.75,35, text2 ,fontsize='28')
                        
                        ax[i,j].set_ylim([-5, 60])
                        # ax[i,j].tick_params(axis='x', labelsize=16)
                        ax[i,j].tick_params(axis='y', labelsize=25)


                        if j==0:
                            y_label=smoothness
                            ax[i,j].set_ylabel(str(round(y_label,2)),fontsize='40')
                        if i==len(smoothness_mat)-1:
                            x_label=rug_amp
                            ax[i,j].set_xlabel(str(round(x_label,2)),fontsize='40')
                            ax[i,j].set_xticklabels([])
                            
                ax[int(len(smoothness_mat)/2)-1,0].text(-0.28,-35, "Smoothness",fontsize='45',rotation=90)
                ax[len(smoothness_mat)-1,int(len(smoothness_mat)/2)].text(-0.94,-32, "Ruggedness Amplitude",fontsize='45')
                
                handles, labels = ax[i,j].get_legend_handles_labels()
                
                fig.legend(handles,labels,loc='lower right',fontsize='35',bbox_to_anchor=(0.53,-0.035))
                fig.legend([box_5["boxes"][0], box_95["boxes"][0]], 
                           ["Less-Diverse Initial Samples", "Diverse Initial Samples"],
                           loc='lower center', fontsize='35',bbox_to_anchor=(0.7,-0.045))
            
                plt.subplots_adjust(
                    left  = 0.1,  # the left side of the subplots of the figure
                    right = 0.9,    # the right side of the subplots of the figure
                    bottom = 0.12,   # the bottom of the subplots of the figure
                    top = 0.9,      # the top of the subplots of the figure
                    wspace = 0.2,   # the amount of width reserved for blank space between subplots
                    hspace = 0.2)   # the amount of height reserved for white space between subplots
            
                fig.suptitle('Distribution of lengthscale learned by BO on initial samples',fontsize='55', x=0.88, horizontalalignment='right')
                filename,ext= '../results/Experiment3-2-2/grid_plot', '.eps'

                try:
                    savename=unique_file(filename, ext)
                    plt.savefig(os.path.abspath(savename),bbox_inches = 'tight',dpi=1000)#, pad_inches = 0.3)
                    plt.close()
                except:
                    savename=unique_file(filename, ext)
                    os.makedirs(os.path.dirname(savename), exist_ok=True)
                    plt.savefig(os.path.abspath(savename) ,bbox_inches = 'tight',dpi=1000) #, pad_inches = 0.3)
                    plt.close()
            
        else:
            seedmat=np.arange(0,args.trials)
            for smoothness in completedata.keys():
                for rug_amp in completedata[smoothness].keys():
                    with plt.style.context(['science','no-latex']):
                        funcname='smoothness-{}-ruggedness_amplitude-{}'.format(smoothness,rug_amp)
                        # print(optdict)
                        res_trial=compile_all_optdata(smoothness,rug_amp)
                        res_trial.extract_plotdata()
                        res_trial.extract_optdata()
                        box_data=completedata[smoothness][rug_amp].bootstrap_plotdata()
                        fig = plt.figure(figsize=(18, 13))
                        outer = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2)
                        i,j=0,0
                        figtitle='Hyperparameter distributuion for '+funcname
                        for count,hyperparameter in enumerate(completedata[smoothness][rug_amp].plotdata[5][0].keys()):
                            inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[count], wspace=0.0, hspace=0.0)
                            ax_box = plt.Subplot(fig, inner[0])
                            ax_hist = plt.Subplot(fig, inner[1])
                            superdict={'data5':box_data[5][hyperparameter],
                                       'data95':box_data[95][hyperparameter]}
                            
                            box_5=ax_box.boxplot(superdict['data5'], notch=True, positions=[1],patch_artist=True, boxprops=dict(facecolor="C0"))
                            box_95=ax_box.boxplot(superdict['data95'], notch=True,positions=[2],patch_artist=True, boxprops=dict(facecolor="C2"))
                            ax_box.set_xticklabels([])
                            ax_box.set_title('\n\n'+hyperparameter,fontsize=20)
                            
                            ax_box.axhline(res_trial.optdata[hyperparameter][0],label='Optimal hyperparameters')
                            handles, labels = ax_box.get_legend_handles_labels()
                            for line in res_trial.optdata[hyperparameter]:
                                ax_box.axhline(line)

                            #add a density plot next to the distribution.
                            optdata[smoothness][rug_amp].extract_plotdata()

                            for seed in range(100):
                                try:
                                    density = kde.gaussian_kde(reject_outliers(res_trial.plotdata[hyperparameter][seed]))
                                except:
                                    noise = np.random.normal(0,2e-2,len(list(res_trial.data.values())[0].data[hyperparameter]))
                                    density = kde.gaussian_kde(reject_outliers(res_trial.plotdata[hyperparameter][seed]+noise))

                                y_range=np.round(ax_box.get_ylim())
                                y=np.arange(*y_range,(y_range[1]-y_range[0])/100)
                                ax_hist.plot(density(y),y)
                            ax_hist.set_xticklabels([])
                            ax_hist.set_ylim(ax_box.get_ylim())
                            fig.add_subplot(ax_box)
                            fig.add_subplot(ax_hist)
                        line_leg=fig.legend(handles,labels,loc='lower center',fontsize='18',bbox_to_anchor=(0.4,0.0))
                        for line in line_leg.get_lines():
                            line.set_linewidth(2.0)
                        box_leg=fig.legend([box_5["boxes"][0], box_95["boxes"][0]], ["Data 5", "Data 95"],loc='lower center',fontsize='18',bbox_to_anchor=(0.6,-0.01))
                        fig.suptitle(figtitle,fontsize=25)
                        savename='../results/Experiment3-2-2/'+funcname + '.png'
                        os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
                        plt.savefig(os.path.abspath(savename))