
# ################
# Experiment A1 - Sensitivity analysis for the objective function (wildcat-wells)
# Corresponding issue on git - #

# Uncomment all code below if running the script for the first time
# ################

import sys
# sys.path.insert(1, '../src/optimizers')
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

import os
import subprocess
import time
import itertools
import gc
import ray

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
    opt.datagenmodule.options['seed']=trial*9
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

import results
import math
import results
from IPython.display import clear_output

def main(trials,dim_mat,objective_func):
    
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
                filename='../Data/experimentA1/'+objective_func+'/'+str(dim)+'/'+level_of_ruggedness+'/data.pkl'
                if exists(filename):
                    print(dim,level_of_ruggedness)
                    pass
                else:
                    available_locs.append((dim,level_of_ruggedness))
        dim,level_of_ruggedness=available_locs[0]
        
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
        synth_data.gamma=1e-5
        synth_data.options['seed']=0
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

        #Create a result object for the trial data
        result=results.result_diversity_trial()
        result.percentiles=percentilemat


        filename='../Data/experimentA1/'+objective_func+'/'+str(dim)+'/'+level_of_ruggedness+'/data.pkl'
        savefile(filename,result)

        #For diversity trial we need to iterate over different percentiles of diversity.
        for percentile in percentilemat:
            #Start and define the engines

            print('{}th percentile of diversity being evaluated'.format(percentile))
            #Set-up the optimizer
            from Optimizers import optimizer
            BO=optimizer()
            if int(dim)==2:
                max_iter=100
            else:
                max_iter=300
            BO.max_iter=max_iter
            BO.opt="BO"
            BO.optima=synth_func.optimal_y
            BO.tol=0.1
            BO.minimize=synth_func.minstate
            if objective_func=='Sphere':
                BO.options['verbose']=True
                BO.tol=0.2
            BO.bounds=synth_func.bounds
            #Add appropriate training data to BO.
            BO.datagenmodule=synth_data

            #Run the parallel process/Trial
            ray.shutdown()
            time.sleep(10)
            ray.init(runtime_env={"working_dir": "../src"}, num_cpus=10,num_gpus=1,log_to_driver=False)
            from tqdm.autonotebook import tqdm
            with tqdm(total=trials) as pbar:
                for batch_num in range(trials//batch_size):
                    current_result=ray.get([map_.remote(BO,synth_func,trial_num,percentile) for trial_num in range(batch_num*batch_size,(batch_num+1)*batch_size)])
                    [[result.addresult(percentile,ind_result) for ind_result in current_result if ind_result is not None]]
                    pbar.update(batch_size)
                    filename='../Data/experimentA1/'+objective_func+'/'+str(dim)+'/'+level_of_ruggedness+'/data.pkl'
                    savefile(filename,result)
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
    
    parser.add_argument("--dimension_mat", default="2, 6, 1" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for smoothness with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--fun_name",choices={"wildcatwells", "Sphere","Rastrigin","Rosenbrock"},default="wildcatwells")
    
    parser.add_argument("--intent", choices={"gen-data", "plot-data", "both","save_compile_data"}, default="gen-data")
    
    parser.add_argument("--plot_type",choices={"grid", "correlation_mat","hyperparameter_convergence"})
    
    parser.add_argument("--grid_type",choices={"difference","comparison"},default="comparison")
    
    args = parser.parse_args()
    
    if args.intent=='gen-data' or args.intent=='both':
        main(args.trials,args.dimension_mat,args.fun_name)
            
    def compile_existing_data(objective,args):
        """
        objective = dict{savedata, create_var}
        """
        import numpy as np

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
                filename='../Data/experimentA1/'+args.fun_name+'/'+str(dim)+'/'+level_of_ruggedness+'/data.pkl'
                if os.path.exists(filename):
                    result = load_variable(filename)
                    completedata[dim][level_of_ruggedness]=result
                else:
                    breakcond=True
                if breakcond:
                    break
            if breakcond:
                break
        if objective=='savedata':
            filename='../Data/experimentA1/'+args.objective_func+'/completedata.pkl'
            savefile(filename,completedata)
        else:
            return completedata
        
    if args.intent=='save_compile_data':
        compile_existing_data('savedata',args)
        
    if args.intent=='plot-data' or args.intent=='both':
        
        dim_mat=np.arange(*args.dimension_mat)
        if args.fun_name=='wildcatwells':
            ruggedness_mat=['low','medium','high']
            filename='../Data/experimentA1/'+args.objective_func+'/completedata.pkl'
            if os.path.exists(filename):
                completedata=load_variable(filename)
            else:
                completedata=compile_existing_data('create_var',args)
            for i,dim in enumerate(dim_mat):
                for j,level_of_ruggedness in enumerate(ruggedness_mat):
                    completedata[dim][level_of_ruggedness].opt=100
        else:
            ruggedness_mat=['Sphere','Rosenbrock','Rastrigin']
            completedata={2.0:{},3.0:{}}
            for func_name in ruggedness_mat:
                filename='../Data/experimentA1/'+func_name+'/completedata.pkl'
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
                    current_func_data=compile_existing_data('create_var',args)
                    for dim in current_func_data:
                        try:
                            completedata[dim][func_name]=current_func_data[dim]['None']
                        except KeyError:
                            completedata[dim]={}
                            completedata[dim][func_name]=current_func_data[dim]['None']
         
        data_dict=completedata
        
        if args.plot_type!='hyperparameter_convergence':        
            #Generate plot and save it to the appropriate directory.
            with plt.style.context(['science','no-latex']):
                fig, ax = plt.subplots(len(dim_mat), len(ruggedness_mat),figsize=(30,21))

                for i,dim in enumerate(dim_mat):
                    for j,level_of_ruggedness in enumerate(ruggedness_mat):
                        percentilemat=[5,95]
                        cumoptgap=data_dict[dim][level_of_ruggedness].percentage_imporvement_cum_opt_gap(percentilemat)[0]
                        if args.grid_type=="comparison":
                            #bootstraps the data for the line plot for each percentile in the diversity trial result object.
                            for percentile in percentilemat:
                                bootstrap_on_trials=data_dict[dim][level_of_ruggedness].bootstrap(percentile)
                                label=str(percentile)+'th percentile'
                                ax[i,j].plot(np.arange(len(bootstrap_on_trials[0])), bootstrap_on_trials[0], '-',label=label) #Plotting the mean data
                                ax[i,j].fill_between(np.arange(len(bootstrap_on_trials[0])),bootstrap_on_trials[1], 
                                                bootstrap_on_trials[2], alpha=0.1) #Plotting the 90% confidence intervals.
                                cumoptgaptext1=str(cumoptgap)
                                ax[i,j].text(15,20, cumoptgaptext1,fontsize='15')
#                                 ax[i,j].text(10,17, cumoptgaptext2,fontsize='15')

                            ax[i,j].legend(fontsize='x-small')
                            ax[i,j].set_ylim([0, 40])
                            ax[i,j].set_xlim([0, int(len(bootstrap_on_trials[0]))])
                        else:
                            bootstrap_on_trials=data_dict[dim][level_of_ruggedness].difference_bootstrap_plotdata()
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

                            cmap,norm,lc=create_colormap_and_ls(np.arange(len(bootstrap_on_trials[0])),bootstrap_on_trials[0])

                            ax[i,j].add_collection(lc) #Plotting the mean data as a line segment

                            ax[i,j].fill_between(np.arange(len(bootstrap_on_trials[0])),bootstrap_on_trials[1], 
                                            bootstrap_on_trials[2], alpha=0.1, hatch='\\\\\\\\',facecolor='azure') #Plotting the 95% confidence intervals.
                            ax[i,j].axhline(y=0,label='Insignificant difference in perfromance',color='blue')

                            if cumoptgap<0: 
                                c='r'
                            else:
                                c='g'
                            handles, labels = ax[i,j].get_legend_handles_labels()
                            if level_of_ruggedness!='Rosenbrock':
                                ax[i,j].set_ylim([-20, 10])
                                ax[i,j].text(20,3.5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                            else:
                                ax[i,j].set_ylim([-20000, 10000])
                                ax[i,j].text(20,3500.5, str(round(cumoptgap,2)) ,fontsize='50',color=c)
                            ax[i,j].tick_params(axis='x', labelsize=25)
                            ax[i,j].tick_params(axis='y', labelsize=25)
                            ax[i,j].set_xlim([0, len(bootstrap_on_trials[0])/2])


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
                        extendfrac='auto',ax=ax.ravel().tolist(),cax=cbar_ax, shrink=0.5)
                    cbar.set_label( label='Effect of diversity on performance of optimizer',fontsize='40')
                    cbar.set_ticks([])
                ax[int(len(dim_mat)/2)-1,0].text(-45,-60, "Number of Dimensions",fontsize='45',rotation=90)
                if args.fun_name=='wildcatwells':
                    ax[len(dim_mat)-1,int(len(ruggedness_mat)/2)].text(-10,-20 ,"Level of Ruggedness",fontsize='45')
                else:
                    ax[len(dim_mat)-1,int(len(ruggedness_mat)/2)].text(30,-31500 ,"Test functions",fontsize='45')
#             fig.legend(handles,labels,loc='lower center',fontsize=20)
            
            plt.subplots_adjust(
                left  = 0.1,  # the left side of the subplots of the figure
                right = 0.8,    # the right side of the subplots of the figure
                bottom = 0.1,   # the bottom of the subplots of the figure
                top = 0.9,      # the top of the subplots of the figure
                wspace = 0.2,   # the amount of width reserved for blank space between subplots
                hspace = 0.2)   # the amount of height reserved for white space between subplots
            
            if args.grid_type=="comparison":
                fig.suptitle('Comparison in optimality gap when optimizer is initiated with 95th vs 5th percentile of diversity',fontsize='38')
                filename,ext= '../results/ExperimentA1/'+args.fun_name+'/comparison', '.png'
            else:
                fig.suptitle('Absolute difference in optimality gap (y-axis) vs iterations (x-axis) \n when hyperparameters are fit at each iteration.',fontsize='48', x=0.45, horizontalalignment='center')
                filename,ext= '../results/ExperimentA1/'+args.fun_name+'/variable-ww-difference', '.png'

            try:
                savename=unique_file(filename, ext)
                plt.savefig(os.path.abspath(savename),bbox_inches = 'tight')#, pad_inches = 0.3)
                plt.close()
            except:
                savename=unique_file(filename, ext)
                os.makedirs(os.path.dirname(savename), exist_ok=True)
                plt.savefig(os.path.abspath(savename) ,bbox_inches = 'tight') #, pad_inches = 0.3)
                plt.close()
        elif args.plot_type=='hyperparameter_convergence':
            
            filename='../Data/experimentA2/optdata.pkl'
            with open(filename, 'rb') as f:
                optdict = dill.load(f)
            
            filename='../Data/experimentA2/object.pkl'
            with open(filename, 'rb') as f:
                optdata = pickle.load(f)
            seed=0
            
            
            if objective_func=='wildcatwells':
                ruggedness_mat=['low','medium','high']
            else:
                ruggedness_mat=['None']
                
            for dim in completedata.keys():
                for level_of_ruggendess in completedata[dim].keys():
                    with plt.style.context(['science','no-latex']):
                        funcname='dim-{}-level_of_ruggendess-{}'.format(dim,level_of_ruggedness)
                        completedata[dim][level_of_ruggedness].extract_hpdata()
                        fixparams=optdict[dim][level_of_ruggedness]
                        fig = plt.figure(figsize=(18, 13))
                        outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.2)
                        i,j=0,0
                        figtitle='Hyperparameter convergence for '+funcname
                        plot_data={5:completedata[dim][level_of_ruggedness].hpdata[5].bootstrap_plotdata(),95:completedata[dim][level_of_ruggedness].hpdata[95].bootstrap_plotdata()}
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
                            optdata[dim][level_of_ruggedness].extract_plotdata()
                            try:
                                density = kde.gaussian_kde(reject_outliers(list(optdata[dim][level_of_ruggedness].data.values())[0].data[hyperparameter]))
                            except:
                                noise = np.random.normal(0,2e-2,len(list(optdata[dim][level_of_ruggedness].data.values())[0].data[hyperparameter]))
                                density = kde.gaussian_kde(reject_outliers(list(optdata[dim][level_of_ruggedness].data.values())[0].data[hyperparameter]+noise))
                                

                            y_range=np.round(ax_series.get_ylim())
                            y=np.arange(*y_range,(y_range[1]-y_range[0])/100)
                            ax_hist.plot(density(y),y)
                            ax_hist.set_ylim(ax_series.get_ylim())
                            fig.add_subplot(ax_series)
                            fig.add_subplot(ax_hist)
                            fig.legend(handles,labels,loc='lower center',fontsize=18)
                            fig.suptitle(figtitle,fontsize=25)
                            savename='../results/ExperimentA2/hyperparameter-convergence/'+funcname + '.png'
                            plt.savefig(os.path.abspath(savename),bbox_inches = 'tight', pad_inches = 0)
