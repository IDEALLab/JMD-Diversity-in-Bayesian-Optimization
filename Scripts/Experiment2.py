
# ################
# Experiment 2 - Sensitivity analysis for the objective function (wildcat-wells)
# Corresponding issue on git - #

# Uncomment all code below if running the script for the first time
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
from scipy.stats import kde

from ipyparallel import Client
import ipyparallel as ipp
import os
import subprocess
import time
import itertools
import gc
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

def reject_outliers(data, m=4):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

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

def parallel_fun_modifier(trials,fun,args,parallelcomp):
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
        args[0].datagenmodule.options['seed']=trial
        args[0].datagenmodule.generate()
        async_results.append(bview.apply_async(fun, args[0],wildcatwells,args[2],trial))
    rc.wait_interactive(async_results)
    results=[ar.get() for ar in list(itertools.compress(async_results,[result.successful() for result in async_results]))]
    return results

def series_fun_modifier(trials,fun,args):
    """
    fun: function to be parallized
    args: RxN list with args for each run.
    runs: number of runs of the function.
    """
    
    results=list()
    for trial in range(trials):
        args[1].seed=trial
        wildcatwells=args[1].generate_cont()
        args[0].datagenmodule.generate()
        results.append(fun(args[0],wildcatwells,args[2]))
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
    time.sleep(10)
    dview.map(os.chdir, [os.path.abspath('..\src\optimizers')]*len(rc))
    with dview.sync_imports():
        import Optimizers
    print("Engine Set-up complete")
    return parallelcomp


import results
import math
import results
from IPython.display import clear_output

async def main(trials,max_iter,sm_range,rug_range,sampling_method=None,alt_rug=None):
    
    if sampling_method is None or sampling_method=="diverse":
        percentilemat=[5,95]
    else:
        percentilemat=[None]
    
    smoothness_mat=np.arange(*sm_range).round(2)
    ruggedness_mat=np.arange(*rug_range).round(2)
    
    # Set-up the function generator
    synth_func=objectives()
    synth_func.bounds=[(0,100),(0,100)]
    
    #Set-up the diversity training data generator
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
        
    
    available_locs=[0]
    
    while len(available_locs)>0:
        from os.path import exists
        available_locs=[]
        for smoothness in np.arange(*sm_range).round(2):
            for rug_amp in np.arange(*rug_range).round(2):
                filename='../Data/experiment2/'+sampling_method+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data.pkl'
                if exists(filename):
                    pass
                else:
                    available_locs.append((smoothness,rug_amp))
        smoothness,rug_amp=available_locs[0]
    
        print('Computation started for smoothness = {}, rug_amp= {}'.format(smoothness,rug_amp))

        #Vary the synthetic black-box function for the sensitivity analysis
        synth_func.args={'N':1,'Smoothness':smoothness,'rug_freq':1,'rug_amp':rug_amp}

        #Create a result object for the trial data
        result=results.result_diversity_trial()
        result.percentiles=percentilemat
        
        
        filename='../Data/experiment2/'+sampling_method+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data.pkl'
        savefile(filename,result)
        
        #For diversity trial we need to iterate over different percentiles of diversity.
        for percentile in percentilemat:
            #Start and define the engines
            rc,clust = await startengines()
            time.sleep(10)
            parallelcomp=setupengines(rc,clust)

            print('{}th percentile of diversity being evaluated'.format(percentile))
            #Set-up the optimizer
            from Optimizers import optimizer
            BO=optimizer()
            BO.max_iter=max_iter
            BO.opt="BO"
            BO.tol=0.1
            BO.minimize=synth_func.minstate
            BO.bounds=synth_func.bounds
            #Add appropriate training data to BO.
            BO.datagenmodule=synth_data

            #Run the parallel process/Trial
            current_result=parallel_fun_modifier(trials,singleopt,[BO,synth_func,percentile],parallelcomp)
            [[result.addresult(percentile,ind_result) for ind_result in current_result]]
            rc.purge_results('all')
            #Stop the engines
            await stopengines(clust)
        time.sleep(15)
        filename='../Data/experiment2/'+sampling_method+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data.pkl'
        savefile(filename,result)
        gc.collect(generation=2)
    return 

if __name__=='__main__':
    import sys
    import argparse
    import asyncio
    from utils import bootstrap,wrapText
    import gc
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of Trials of Each setting of the smoothness and rug_amp")
    
    parser.add_argument("--max_iter", type=int, default=100,
                        help="Max number of runs for BO.")
    
    parser.add_argument("--smoothness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--ruggedness_range", default="0.2, 0.85, 0.2" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--intent", choices={"gen-data", "plot-data", "both","save_compile_data"}, default="gen-data")

    parser.add_argument("--sampling_type", choices={"diverse", "random"}, default="diverse")
    
    parser.add_argument("--plot_type",choices={"grid", "correlation_mat","hyperparameter_convergence"})
    
    parser.add_argument("--grid_type",choices={"difference","comparison"},default="comparison")
    
    parser.add_argument("--check_previous_data",action=argparse.BooleanOptionalAction)
    
    args = parser.parse_args()
    
    if args.intent=='gen-data' or args.intent=='both':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        completedata=asyncio.run(main(args.trials,args.max_iter,args.smoothness_range,args.ruggedness_range,
                                      sampling_method=args.sampling_type))
            
    def compile_existing_data(objective,args):
        """
        objective = dict{savedata, create_var}
        """
        import numpy as np
        
        objectives = {'savedata':0, 'create_var':0}
        objectives[objective]+=1
        
        completedata=dict()
        breakcond=False
        for smoothness in np.arange(*args.smoothness_range).round(2):
            completedata[smoothness]=dict()
            for rug_amp in np.arange(*args.ruggedness_range).round(2):
                filename='../Data/experiment2/'+args.sampling_type+'/smoothness/'+str(smoothness)+'/ruggedness/'+str(rug_amp)+'/data.pkl'
                if os.path.exists(filename):
                    result = load_variable(filename)
                    completedata[smoothness][rug_amp]=result
                else:
                    breakcond=True
                if breakcond:
                    break
            if breakcond:
                break
        if objective=='savedata':
            filename='../Data/experiment2/complete-'+args.sampling_type+'-data.pkl'
            savefile(filename,completedata)
        else:
            return completedata
        
    if args.intent=='save_compile_data':
        compile_existing_data('savedata',args)
        
    if args.intent=='plot-data' or args.intent=='both':
        
        filename='../Data/experiment2/completedata.pkl'
        with open(filename, 'rb') as f:
            completedata = pickle.load(f)
        
        smoothness_mat=np.arange(*args.smoothness_range).round(2)
        ruggedness_mat=np.arange(*args.ruggedness_range).round(2)
        data_dict=dict()
        for i,smoothness in enumerate(smoothness_mat):
            for j,rug_amp in enumerate(ruggedness_mat):
                completedata[smoothness][rug_amp].opt=100
        data_dict=completedata
        if args.plot_type!='hyperparameter_convergence':        
            #Generate plot and save it to the appropriate directory.
            with plt.style.context(['science','no-latex']):
                fig, ax = plt.subplots(len(smoothness_mat), len(ruggedness_mat), sharex='col', sharey='row',figsize=(30,21))

                for i,smoothness in enumerate(smoothness_mat):
                    for j,rug_amp in enumerate(ruggedness_mat):
                        percentilemat=[5,95]
                        cumoptgap=data_dict[smoothness][rug_amp].cum_opt_gap_improvement_overall(percentilemat)[0]
                        if args.grid_type=="comparison":
                            #bootstraps the data for the line plot for each percentile in the diversity trial result object.
                            for percentile in percentilemat:
                                bootstrap_on_trials=data_dict[smoothness][rug_amp].bootstrap(percentile)
                                label=str(percentile)+'th percentile'
                                ax[i,j].plot(np.arange(args.max_iter+2), bootstrap_on_trials[0], '-',label=label) #Plotting the mean data
                                ax[i,j].fill_between(np.arange(args.max_iter+2),bootstrap_on_trials[1], 
                                                bootstrap_on_trials[2], alpha=0.1) #Plotting the 90% confidence intervals.
                                ax[i,j].text(15,20, cumoptgaptext1,fontsize='15')
                                ax[i,j].text(10,17, cumoptgaptext2,fontsize='15')

                            ax[i,j].legend(fontsize='x-small')
                            ax[i,j].set_ylim([0, 40])
                            ax[i,j].set_xlim([0, int(args.max_iter/2)])
                        else:
                            bootstrap_on_trials=data_dict[smoothness][rug_amp].difference_bootstrap_plotdata_by_seed()
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

                            ax[i,j].add_collection(lc) #Plotting the mean data as a line segment

                            ax[i,j].fill_between(np.arange(args.max_iter+2),bootstrap_on_trials[1], 
                                            bootstrap_on_trials[2], alpha=0.1, hatch='\\\\\\\\',facecolor='azure') #Plotting the 95% confidence intervals.
                            ax[i,j].axhline(y=0,label='Insignificant difference in perfromance',color='blue')

                            if cumoptgap<0: 
                                c='r'
                            else:
                                c='g'
                            ax[i,j].text(20,3.5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                            handles, labels = ax[i,j].get_legend_handles_labels()
                            ax[i,j].set_ylim([-10, 10])
                            ax[i,j].tick_params(axis='x', labelsize=25)
                            ax[i,j].tick_params(axis='y', labelsize=25)
                            ax[i,j].set_xlim([0, int(args.max_iter/2)])


                        if j==0:
                            y_label=smoothness
                            ax[i,j].set_ylabel(str(round(y_label,2)),fontsize='30')
                        if i==len(smoothness_mat)-1:
                            x_label=rug_amp
                            ax[i,j].set_xlabel(str(round(x_label,2)),fontsize='30')
                if args.grid_type!="comparison":
                    right,bottom,width,height=0.83,0.1,0.03,0.8
                    cbar_ax = fig.add_axes([right,bottom,width,height])
                    cbar=fig.colorbar(
                        mpl.cm.ScalarMappable(cmap=cmap, norm=norm), extend='both',
                        extendfrac='auto',ax=ax.ravel().tolist(),cax=cbar_ax, shrink=0.5)
                    cbar.set_label( label='Effect of diversity on performance of optimizer',fontsize='40')
                    cbar.set_ticks([])
                ax[int(len(smoothness_mat)/2)-1,0].text(-24,-20, "Smoothness",fontsize='45',rotation=90)
                ax[len(smoothness_mat)-1,int(len(smoothness_mat)/2)].text(-43,-19, "Ruggedness Amplitude",fontsize='45')
            # fig.legend(handles,labels,loc='lower center',fontsize=20)
            
            plt.subplots_adjust(
                left  = 0.1,  # the left side of the subplots of the figure
                right = 0.8,    # the right side of the subplots of the figure
                bottom = 0.1,   # the bottom of the subplots of the figure
                top = 0.9,      # the top of the subplots of the figure
                wspace = 0.2,   # the amount of width reserved for blank space between subplots
                hspace = 0.2)   # the amount of height reserved for white space between subplots
            
            if args.grid_type=="comparison":
                fig.suptitle('Comparison in optimality gap when optimizer is initiated with 95th vs 5th percentile of diversity',fontsize='38')
                filename,ext= '../results/Experiment2/variable-ww-sm-'+str(args.smoothness_range)+'-rug-'+str(args.ruggedness_range), '.png'
            else:
                fig.suptitle('Difference in optimality gap when optimizer is fitting hyperparameters at each iteration',fontsize='42', x=0.88, horizontalalignment='right')
                filename,ext= '../results/Experiment2/variable-ww-difference-sm-'+str(args.smoothness_range)+'-rug-'+str(args.ruggedness_range), '.eps'

            try:
                savename=unique_file(filename, ext)
                plt.savefig(os.path.abspath(savename),bbox_inches = 'tight',dpi=1200)#, pad_inches = 0.3)
                plt.close()
            except:
                savename=unique_file(filename, ext)
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                plt.savefig(os.path.abspath(savename) ,bbox_inches = 'tight',dpi=1200) #, pad_inches = 0.3)
                plt.close()
        elif args.plot_type=='hyperparameter_convergence':
            
            filename='../Data/experiment3-2-1/optdata.pkl'
            with open(filename, 'rb') as f:
                optdict = dill.load(f)
            
            filename='../Data/experiment3-2-1/object.pkl'
            with open(filename, 'rb') as f:
                optdata = pickle.load(f)
            seed=0
            for smoothness in completedata.keys():
                for rug_amp in completedata[smoothness].keys():
                    with plt.style.context(['science','no-latex']):
                        funcname='smoothness-{}-ruggedness_amplitude-{}'.format(smoothness,rug_amp)
                        completedata[smoothness][rug_amp].extract_hpdata()
                        fixparams=optdict[smoothness][rug_amp]
                        fig = plt.figure(figsize=(18, 13))
                        outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
                        i,j=0,0
                        figtitle='Hyperparameter convergence for '+funcname
                        plot_data={5:completedata[smoothness][rug_amp].hpdata[5].bootstrap_plotdata(),95:completedata[smoothness][rug_amp].hpdata[95].bootstrap_plotdata()}
                        for count,hyperparameter in enumerate(list(optdict[0.2][0.2].keys())[0:2]):
                            inner = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=outer[count], wspace=0.0, hspace=0.0)
                            ax_series = plt.Subplot(fig, inner[0])
                            ax_hist = plt.Subplot(fig, inner[1])
                            ax_series.plot(np.arange(len(plot_data[5][hyperparameter][0])),plot_data[5][hyperparameter][0],label='Non diverse examples')
                            ax_series.fill_between(np.arange(len(plot_data[5][hyperparameter][0])),plot_data[5][hyperparameter][1],plot_data[5][hyperparameter][2], alpha=0.3)
                            ax_series.plot(np.arange(len(plot_data[95][hyperparameter][0])),plot_data[95][hyperparameter][0],label='Diverse Examples')
                            ax_series.fill_between(np.arange(len(plot_data[95][hyperparameter][0])),plot_data[95][hyperparameter][1],plot_data[95][hyperparameter][2], alpha=0.3)
                            ax_series.set_title('\n\n'+hyperparameter,fontsize=20)
                            ax_series.axhline(fixparams[hyperparameter][0], c='r',label='Optimal Value of the hyperparameter')
                            
                            ax_series.set_xlim([0, 10])
                            handles, labels = ax_series.get_legend_handles_labels()

                            #add a density plot next to the distribution.
                            optdata[smoothness][rug_amp].extract_plotdata()
                            try:
                                density = kde.gaussian_kde(reject_outliers(list(optdata[smoothness][rug_amp].data.values())[0].data[hyperparameter]))
                            except:
                                noise = np.random.normal(0,2e-2,len(list(optdata[smoothness][rug_amp].data.values())[0].data[hyperparameter]))
                                density = kde.gaussian_kde(reject_outliers(list(optdata[smoothness][rug_amp].data.values())[0].data[hyperparameter]+noise))
                                

                            y_range=np.round(ax_series.get_ylim())
                            y=np.arange(*y_range,(y_range[1]-y_range[0])/100)
                            ax_hist.plot(density(y),y)
                            ax_hist.set_ylim(ax_series.get_ylim())
                            fig.add_subplot(ax_series)
                            fig.add_subplot(ax_hist)
                            fig.legend(handles,labels,loc='lower center',fontsize=18)
                            fig.suptitle(figtitle,fontsize=25)
                            savename='../results/Experiment2/hyperparameter-convergence/'+funcname + '.eps'
                            plt.savefig(os.path.abspath(savename),bbox_inches = 'tight', pad_inches = 0, dpi=1000)
