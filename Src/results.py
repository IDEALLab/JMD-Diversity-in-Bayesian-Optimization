
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
from random import choices
import string
from scipy.signal import find_peaks
from scipy.stats import kde
from matplotlib.ticker import FormatStrFormatter


### UTILITY Functions ###
def intersection(listoflists):
    """
    Returns an intersection of the lists.
    """
    from functools import reduce
    return list(reduce(set.intersection, [set(item) for item in listoflists])) 

def replacetool(lst,replacee,replacer=None,method='nearest-element'):
    import numpy as np
    waslist=False
    if isinstance(lst,list):
        waslist=True
        lst=np.array(lst)
    
    #Checks if the list even has elements to be replaced.    
    initcheck=len(np.where(lst==replacee)[0])<1
    if initcheck:
        if waslist:
            return list(lst)
        return lst
        
    if method=='nearest-element' and replacer==None:
        wherenotreplacee=np.where(lst!=replacee)[0]
        nearestsub=np.array([np.argmin(np.abs(wherenotreplacee-wherereplacee)) for wherereplacee in np.where(lst==replacee)[0]])
        lst[np.where(lst==replacee)[0]]=lst[wherenotreplacee[nearestsub]]
        
    elif method=='replace-term' and replacer!=None:
        for wherereplacee in np.where(lst==replacee)[0]:
            lst[wherereplacee]=replacer
            
    elif replacer!=None:
        for wherereplacee in np.where(lst==replacee)[0]:
            lst[wherereplacee]=replacer
            
    else:
        print('Function Incorrectly defined, please check again.')
        
    if waslist:
        return list(lst)
    return lst


def reject_outliers(data, m=4):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def calc_zero_grad(peak_loc,x,density_function,peak_locs,biggest_peak=False):
    import numpy as np
    if biggest_peak:
        significant_digits=-int(np.log(num_der(x,density_function(x))[peak_loc]**2)/np.log(10))
    else:
        significant_digits=-int(np.log(num_der(x,density_function(x))[peak_loc]**2)/np.log(10)+1)
    zero_grads=np.round(num_der(x,density_function(x))**2,significant_digits)==0
    zero_grads=np.append(zero_grads,True)
    zero_grads[peak_locs]=False
    return zero_grads


def peak_extremes(peak_arg_loc,peak_num,x,density_function,peak_locs):
    if peak_num==0:
        zero_grad_args=calc_zero_grad(peak_arg_loc,x,density_function,peak_locs,biggest_peak=True)
    else:
        zero_grad_args=calc_zero_grad(peak_arg_loc,x,density_function,peak_locs)
    try:
        peak_end=np.arange(peak_arg_loc+2,len(x))[zero_grad_args[peak_arg_loc+2:]][0]
    except:
        peak_end=len(x)
    try:
        peak_start=np.arange(0,peak_arg_loc-2)[zero_grad_args[:peak_arg_loc-2]][-1]
    except:
        peak_start=0
    return (peak_start,peak_end)


def num_der(x,y):
    return np.array([(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range(len(x)-1)])

####-x-####

class result_opt():
    """
    Create an instance to store results from an optimizer, particularly BOTorch.
    
    Attributes:
    success (boolean) - Succesful run of the optimizer
    nit (int) - Number of iteration of the optimizer
    xopt (float)- optimal argument in X.
    yopt (float) - optimal point in Y.
    minstate (boolean) - Is the optima a minima or maxima.
    _opt (float) - Theoretical optima.
    xall (list) - All the Xs explored by otpimizers
    yall (list) - Best Y at each iteration
    traindata (list) - Training data used to initialize the optimizer.
    hpdata (dictionary) - Hyperparameters at each iteration
    key (alphanumeric) - Key to identify result instance.
    
    Methods:
    Optimality Gap [opt_gap()] - Caclulates the optimality gap based on theoretical optima (_opt) and Best Y at each iteration (y_all)
    Cumulative optimality Gap [cum_opt_gap()] - Calculates Cumulative optimality gap by taking the area under the curve of optimality gap.
    """
    def __init__(self):
        self.success=None
        self.nit=None
        self.xopt=None
        self.yopt=None
        self.minstate=None
        self._opt=None
        self.xall=[]
        self.yall=[]
        self.traindata=[]
        self.hpdata=None
        self.key= ''.join(random.choice(string.ascii_letters) for i in range(8))
        
    def __repr__(self):
        return str(self.__dict__)
    
    @property
    def opt_gap(self):
        if self._opt==None:
            raise Exception('Enter a thoeretical min/max for this result in self._opt')
        if self.minstate:
            return self.yall-self._opt
        else:
            return self._opt-self.yall
    
    def tracepath(self):
        pass
    
    def cum_opt_gap(self):
        from utils import cumoptgap
        return cumoptgap(self.opt_gap[1:])
    
    def __str__(self):
        return vars(self)
        
class result_trial(result_opt):
    def __init__(self, *args, **kwargs):
        super(hp_trial, self).__init__(*args, **kwargs)
        self.data={}
        
        
    def addresult(self,current_result):
        """
        Add a result_opt instance to the data dictionary.
        """
        self.data[current_result.key]=current_result
    
    def update_async(self):
        """
        Get's the actual result from the async result object.
        """
        for keys in self.data:
            self.data[keys]=self.data[keys].get()
        
    def getresult(self,key):
        return self.data[key]
    
    def bootstrap(self):
        """
        Returns bootstrap plot data
        """
        from utils import bootstrap
        return bootstrap(self.data.values().yall)
    
    def bootstrap_cumoptgap(self):
        """
        Get boostraped value of cumuluative optimality gap over all trials.
        """
        from utils import bootstrap
        return bootstrap(self.data.values().cum_opt_gap())
        
class scipy_result(result_opt):
    def __init__(self, *args, **kwargs):
        self.xall=[]
        self.yall=[]
        self.funcseed=None
        super(scipy_result, self).__init__(*args, **kwargs)
        
class hp_result():
    """
    Store hyperparameters from a botorch model.
    
    Attribute:
    hyperparameters - Dictionary of saved parameters
    
    Methods:
    Add Diverse Hyperparameters [adddiversehp()] - If its a diversity related expreiment and data from different diversity needs to be saved in the result object, use this method.
    Add hyperparamters [addhp()] - 
    
    """
    def __init__(self, *args, **kwargs):
        self.hyperparameters={}
    
    def adddiversehp(self,model,synth_data):
        """
        Given a botorch model, and a synth_data object the method finds the appropriate percentile of diversity the data is from and the hyyperprameters of the botorch model.
        """
        import botorch
        if isinstance(model,botorch.models.gp_regression.SingleTaskGP):
            if synth_data!='smart':
                ind=np.argmin([np.abs(np.array([synth_data.acquiredsets[-1]])/synth_data.options['N_samples']*100-95),np.abs(np.array([synth_data.acquiredsets[-1]])/synth_data.options['N_samples']*100-5)])
                percentile=[95,5][ind]
            else:
                percentile='smart'
            try:
                isinstance(self.hyperparameters[percentile],dict)
            except KeyError:
                self.hyperparameters[percentile]=dict()
                
            for param_name, param in model.named_parameters():
                try:
                    self.hyperparameters[percentile][param_name].append(param.item())
                except KeyError:
                    self.hyperparameters[percentile][param_name]=[param.item()]
        else:
            print('Unexpected input! Input botorch model to extract hyperparamers and data_gen object to get the percentile.')
            
    def addhp(self,model):
        """
        Given a botorch model finds the hyperprameter for its gaussian process.
        """
        import botorch
        if isinstance(model,botorch.models.gp_regression.SingleTaskGP):
            for param_name, param in model.named_parameters():
                try:
                    self.hyperparameters[param_name].append(param.item())
                except KeyError:
                    self.hyperparameters[param_name]=[param.item()] 
        else:
            print('Unexpected input! Input botorch model to extract hyperparamers and data_gen object to get the percentile.')
        
          
class hp_series(hp_result):
    """
    Contains a series of hyperparameter results from a single run of the experiment.
    
    Attributes:
    data- Can store data as multiple entries of hp_result instances or a dictionary of the extracted data from each instance of hp_result.
    key- Assigns a random key to each instance for its identifcation.
    optdict- Can be popualted with optimal hyperparameter results observed on the 'data' stored within the series.
    seed- If the series corresponds to a particular seed in a larger experiment, that can be referenced here.
    wwargs- This attribute is used to store the configuration of the wildcatwells function that the hp_series object data corresponds to.
    
    Methods:
    Update Diverse Result [updatediverseres()] - Given a diverse list of hp_results as data update the data attribute to generate a dictionary from these results.
    Update Result [updateres()] - Given a list of hp_results that have non-diverse results, update the data attribure to genrate a dictionary from the results.
    Extract hyperparameters [extract_hyperparam()] - If the data observed is believed to have modes of optimal hyperparameters, find these modes and save them to optdict attribute.
    Visualise [visualise()] - Creates a visual representation of how the hyperparameters were extracted.
    """
    def __init__(self, *args, **kwargs):
        super(hp_series, self).__init__(*args, **kwargs)
        self.data=None
        self.key= ''.join(random.choice(string.ascii_letters) for i in range(8))
        self.optdict=None
        self.seed=None
        self.wwargs=None
        
    def updatediverseres(self):
        """
        Create a dictionary from hpresult objects.
        """
        if self.data==None:
            raise Exception('Data not collected yet')
            
        super_dict={}
        if not hasattr(self,'additional_data'):
            tempresult=[singleresult.hyperparameters for singleresult in self.data]
            tempresult=replacetool(tempresult,None,method='nearest-element') #manages the exceptions that occur during bo
        else:
            tempresult=[singleresult.hyperparameters for singleresult in self.additional_data]
            tempresult=replacetool(tempresult,None,method='nearest-element') #manages the exceptions that occur during bo
            
        for percentile in intersection([list(singleresult.keys()) for singleresult in tempresult]):
            for param in intersection([singleresult[percentile].keys() for singleresult in tempresult]):
                try:
                    super_dict[percentile][param] = np.array([singleresult[percentile][param] for singleresult in tempresult]).flatten()
                except:
                    super_dict[percentile]={}
                    super_dict[percentile][param] = np.array([singleresult[percentile][param] for singleresult in tempresult]).flatten()
        
        if not hasattr(self,'additional_data'):
            self.data=super_dict
        else:
            self.data.update(super_dict)
    
    @property
    def hyperparameters(self):
        if hasattr(self,'_hyperparameters'):
            if self._hyperparameters is None:
                if self.data is not None:
                    self._hyperparameters=self.getkeys()
                    if isinstance(self._hyperparameters[0],(int,float)):
                        self.div_trial=True
                        self.percentiles=self._hyperparameters
                        self._hyperparameters=list(self.data[self.getkeys()[0]].keys())
                    return self._hyperparameters
                else:
                    print('No data found')
            else:
                return self._hyperparameters
        else:
            self._hyperparameters=None
            return self.hyperparameters
        
    def num_trials_for_percentile(self,percentile):
        if percentile in self.data:
            return len(self.data[percentile].keys())
        else:
            return 0
        
    def updateres(self):
        """
        Update the existing asyncmap data to generate a dictionary from the data.
        """
        if self.data==None:
            raise Exception('Data not collected yet')

        tempresult=np.array([singleresult.hyperparameters for singleresult in self.data])
        tempresult=replacetool(tempresult,None,method='nearest-element') #manages the exceptions that occur during fiting of the GPy model.
        for param in intersection([list(singleresult.keys()) for singleresult in tempresult]):
            try:
                super_dict[param] = np.array([singleresult[param] for singleresult in tempresult]).flatten()
            except:
                super_dict={}
                super_dict[param] = np.array([singleresult[param] for singleresult in tempresult]).flatten()
        self.data=super_dict
        
    def extract_hyperparam(self):
        """
        Create optimality hyperparameter dictionary to be used in the Experiment 3-2-3 and 3-2-2.
        """
        from utils import bootstrap

        npoints=1000 #Determines the mesh fineness, when selecting two optimal hyperparameter values.
        rel_area_factor=0.333 #Determines how much smaller than the biggest peak can nmodes of hyperparameters be slected at.

        self.optdict={}
        self.opthp_alldata={}
        for col in self.data.keys():
            
            self.opthp_alldata[col]={}
            
            noise = np.random.normal(0,2e-3,len(self.data[col]))
            new_dist=reject_outliers(self.data[col]+noise)
            density_function=kde.gaussian_kde(new_dist)
                
                
            stdev,mean=np.std(new_dist),np.mean(new_dist)
            x=np.arange(mean-4*stdev,mean+4*stdev,8*stdev/npoints) #Range of hyperparameter to be explored, mean +/- 4*stdev.
            
            #Finds peaks in the density function over the range X and arange them by their height
            peak_locs=find_peaks(density_function(x))[0]
            arg_sort_peaks=np.argsort(density_function(x)[peak_locs])
            peak_locs=peak_locs[arg_sort_peaks][::-1]
            
            self.opthp_alldata[col]['all_peaks']=peak_locs
            
            #Finds the locations of Xs where the density function has a 0 slope.

            all_peaks_info=[peak_extremes(peak_loc,i,x,density_function,peak_locs) for i,peak_loc in enumerate(peak_locs)] #defines the starting and ending point of all peak

            area_of_peaks=[np.trapz(density_function(x)[list(range(*peak_info))]) for peak_info in all_peaks_info]

            biggest_peak_arg=np.argmax(area_of_peaks)
            biggest_peak_loc=peak_locs[biggest_peak_arg]

            accepted_peaks_args=peak_locs[area_of_peaks>rel_area_factor*area_of_peaks[biggest_peak_arg]]
            
            
            self.opthp_alldata[col]['peak_extremes']=all_peaks_info
            if col=='mean_module.raw_constant':
                self.optdict[col]=x[accepted_peaks_args]
                self.optdict['mean_module.constant']=x[accepted_peaks_args]
            elif col=='mean_module.constant':
                self.optdict[col]=x[accepted_peaks_args]
                self.optdict['mean_module.raw_constant']=x[accepted_peaks_args]
            else:
                self.optdict[col]=x[accepted_peaks_args]
                
    
    def getkeys(self):
        """
        Returns the result keys corresponding to different results.
        """
        return list(self.data.keys())
                
                
    def visualize(self):
        """
        Visualise the optimal hyperparameters observed in the series and a kernel density funciton estimating the distribution of each hyperparameter.
        """
        try:
            exists=self.opthp_alldata
        except AttributeError:
            raise Exception('Optimal hyperparameters need to be extracted to use the visualisation tool, please use extract_hyperparam method first.')
        with plt.style.context(['science','no-latex']):
            fig, ax = plt.subplots(2,2,figsize=(18,10))
            for i,col in enumerate(self.data.keys()):

                noise = np.random.normal(0,2e-5,len(self.data[col]))
                new_dist=reject_outliers(self.data[col]+noise)
                density_function=kde.gaussian_kde(new_dist)
                stdev,mean=np.std(new_dist),np.mean(new_dist)
                npoints=1000
                x=np.arange(mean-4*stdev,mean+4*stdev,8*stdev/npoints) #Range of hyperparameter to be explored, mean +/- 4*stdev.
                ax.flatten()[i].plot(density_function(x),x)
                ax.flatten()[i].set_title(col,fontsize=18)
                for peak_num,peak_info in enumerate(self.opthp_alldata[col]['peak_extremes']):
                    ax.flatten()[i].scatter(density_function(x)[list(range(*peak_info))],x[list(range(*peak_info))],label='peak '+str(peak_num+1))
                    # plot_text='Area of peak '+str(peak_num+1)+' :' + str(round(np.trapz(density_function(x)[list(range(*peak_info))]),3))
                    # ax.flatten()[i].text(max(density_function(x))*1.1,max(x)*(0.9-0.1*peak_num),plot_text)
                peaks=self.opthp_alldata[col]['all_peaks']
                ax.flatten()[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                ax.flatten()[i].get_yaxis().get_offset_text().set_visible(False)
                ax_max = max(ax.flatten()[i].get_yticks())
                exponent_axis = np.floor(np.log10(ax_max)).astype(int)
                ax.flatten()[i].annotate(r'$\times$10$^{%i}$'%(exponent_axis),
                             xy=(.015, .94), xycoords='axes fraction',fontsize=12)
                ax.flatten()[i].scatter(density_function(x)[peaks],x[peaks],c='r',label='All Evaluated Peaks')
                this_plots_legend=ax.flatten()[i].legend(fontsize=13)
                ax.flatten()[i].tick_params(axis='x', labelsize=15)
                ax.flatten()[i].tick_params(axis='y', labelsize=15)
            fig.suptitle('Finding optimal hyperparameters for smoothness ={}, \n and ruggedness amplitude = {} '.format(self.wwargs['Smoothness'],self.wwargs['rug_amp']),fontsize='25')
        
        plt.subplots_adjust(
            left  = 0.1,  # the left side of the subplots of the figure
            right = 0.9,    # the right side of the subplots of the figure
            bottom = 0.1,   # the bottom of the subplots of the figure
            top = 0.88,      # the top of the subplots of the figure
            wspace = 0.1,   # the amount of width reserved for blank space between subplots
            hspace = 0.2)   # the amount of height reserved for white space between subplots
        return fig
            

class hp_histogram():
    def __init__(self):
        self.data=dict()
        pass
    
class hp_trial(hp_series):
    def __init__(self):
        self.data=dict()
        self.optdata=None
        self.plotdata=None
        self.modes=None
        self.n_trials=0
        self._hyperparameters=None
        self.div_trial=False
        self.percentiles=None
            
    @property
    def hyperparameters(self):
        if hasattr(self,'_hyperparameters'):
            if self._hyperparameters is None:
                if self.data is not None:
                    self._hyperparameters=list(self.data[self.getkeys()[0]].data.keys())
                    if isinstance(self._hyperparameters[0],(int,float)):
                        self.div_trial=True
                        self.percentiles=self._hyperparameters
                        self._hyperparameters=list(self.data[self.getkeys()[0]].data[self._hyperparameters[0]].keys())
                    else:
                        self.div_trial=False
                    return self._hyperparameters
                else:
                    print('No data found')
            else:
                return self._hyperparameters
        else:
            self._hyperparameters=None
            return self.hyperparameters
        
        
    def __len__(self):
        """
        The number of trials in the object.
        """
        return len(self.data.keys())
    
    def getkeys(self):
        """
        Returns the result keys corresponding to different results.
        """
        return list(self.data.keys())
    
    
    def extract_data(self,objective):
        """
        objectives = dict{ plot_data, bootstrap_plot_data, optdata, bootstrap_optdata}
        """
        if self.plotdata is None:
            self.extract_plotdata()
        if self.optdata is None:
            self.extract_optdata()
        objectives={'plot_data':0, 'bootstrap_plot_data':0, 'optdata':0, 'bootstrap_optdata':0}
        objectives[objective]+=1
        if objectives['plot_data']:
            needed_data=self.plotdata
        elif objectives['bootstrap_plot_data']:
            needed_data=self.bootstrap_plotdata()
        elif objectives['optdata']:
            needed_data=self.optdata
        else:
            needed_data=self.bootstrap_opthyperparam()
        return needed_data
    
    def get_key_for_seed(self,seed):
        """
        Find the hpseries object's key corresponding to a particular seed, if the trial is seeded and the seed information is saved to hpseries object.
        """
        try:
            seed_arg_mat=[hpseries.seed for hpseries in self.data.values()]
            correct_arg=np.where(np.array(seed_arg_mat)==seed)[0][0]
            return list(self.data.keys())[correct_arg]
        except:
            print('series for seed not found')
        
    def add_hpseries(self,hpseries,seed=None):
        """
        Add an hpseries object as a trial.
        """
        self.n_trials+=1
        if seed is None:
            self.data[hpseries.key]=hpseries
    
    def extract_optdata(self):
        """
        Extract the optimal hyperparameter data from each hpseries object and save to optdata attribute.
        """
        self.optdata=dict()
        self.optdata_second_mode=dict()
        for key in self.data.keys():
            self.data[key].extract_hyperparam()
            for hyperparameter in self.hyperparameters:
                try:
                    self.optdata[hyperparameter].append(self.data[key].optdict[hyperparameter][0])
                except KeyError:
                    self.optdata[hyperparameter]=[]
                    self.optdata[hyperparameter].append(self.data[key].optdict[hyperparameter][0])
                ## Check for a second mode
                if len(self.data[key].optdict[hyperparameter])>1:
                    try:
                        self.optdata_second_mode[hyperparameter].append(self.data[key].optdict[hyperparameter][1])
                    except KeyError:
                        self.optdata_second_mode[hyperparameter]=[]
                        self.optdata_second_mode[hyperparameter].append(self.data[key].optdict[hyperparameter][1])
            
    def extract_plotdata(self):
        """
        Extract and save the plotdata from each hpseries object to the plotdata attribute.
        """
        self.plotdata=dict()
        self.hyperparameters
        if self.div_trial:
            for percentile in self.percentiles:
                self.plotdata[percentile]=dict()
        for key in self.data.keys():
            for hyperparameter in self.hyperparameters:
                    
                if not self.div_trial:
                
                    if hyperparameter=='mean_module.raw_constant' and hyperparameter not in self.data[key].data:
                        variable_hyperparameter='mean_module.constant'
                    elif hyperparameter=='mean_module.constant' and hyperparameter not in self.data[key].data:
                        variable_hyperparameter='mean_module.raw_constant'
                    else:
                        variable_hyperparameter=hyperparameter
                        
                    if hyperparameter in self.plotdata:
                        self.plotdata[hyperparameter].append(self.data[key].data[variable_hyperparameter])
                    else:
                        self.plotdata[hyperparameter]=[self.data[key].data[variable_hyperparameter]]
                else:
                    
                    if hyperparameter=='mean_module.raw_constant' and hyperparameter not in self.data[key].data[self.percentiles[0]]:
                        variable_hyperparameter='mean_module.constant'
                    elif hyperparameter=='mean_module.constant' and hyperparameter not in self.data[key].data[self.percentiles[0]]:
                        variable_hyperparameter='mean_module.raw_constant'
                    else:
                        variable_hyperparameter=hyperparameter
                        
                    for percentile in self.percentiles:
                        if hyperparameter in self.plotdata[percentile]:
                            self.plotdata[percentile][hyperparameter].append(self.data[key].data[percentile][variable_hyperparameter])
                        else:
                            self.plotdata[percentile][hyperparameter]=[self.data[key].data[percentile][variable_hyperparameter]]
                

    def bootstrap_opthyperparam(self):
        """
        bootstrap the optimal hyperparameters observed in each trial.
        """
        if self.optdata==None:
            self.extract_optdata()
        bootstrap_data=dict()
        for hyperparameter in self.optdata.keys():
            from utils import bootstrap
            bootstrap_data[hyperparameter]=bootstrap(self.optdata[hyperparameter]) 
        return bootstrap_data
    
    def bootstrap_plotdata(self):
        """
        Bootstrap the hyperparameter data from different trials.
        """
        if 5 in self.data[self.getkeys()[0]].data.keys():
            self.div_trial=True
        else:
            self.div_trial=False
        
        if self.plotdata is None:
            self.extract_plotdata()
        bootstrap_data=dict()
        if not self.div_trial:
            for hyperparameter in self.plotdata.keys():
                from utils import bootstrap
                bootstrap_data[hyperparameter]=bootstrap(self.plotdata[hyperparameter])
        else:
            for percentile in self.plotdata.keys():
                bootstrap_data[percentile]=dict()
                for hyperparameter in self.hyperparameters:
                    from utils import bootstrap
                    bootstrap_data[percentile][hyperparameter]=[bootstrap(self.plotdata[percentile][hyperparameter][seed])[0] for seed in range(len(self.plotdata[percentile][hyperparameter]))]
            
        return bootstrap_data
    
    def plot_avg(self):
        """
        Plots the average of the hyperparameters at each iteration observed over n trials within the result_trial object.
        """
        #These are the parameters that have been fixed in experiment 3-2-1~(line 120-13), if changed then will need adjustment here.
        init_iter=1000
        iterations=1200
        #Plot the data
        bootstrap_data=self.bootstrap_plotdata()
        with plt.style.context(['science','no-latex']):
            fig, ax = plt.subplots(2,2,figsize=(20,20))
            for i,hyperparameter in enumerate(bootstrap_data.keys()):
                title='Hyperparameter: [' + hyperparameter  +']'
                ax.flatten()[i].set_title(str(title))
                ax.flatten()[i].set_xlabel('iterations')
                ax.flatten()[i].set_ylabel('hyperparameter value at xth iteration')
                ax.flatten()[i].fill_between(np.arange(init_iter,iterations),bootstrap_data[hyperparameter][2],bootstrap_data[hyperparameter][1],alpha=0.3)
                ax.flatten()[i].plot(np.arange(init_iter,iterations),bootstrap_data[hyperparameter][0],label='mean value of the hyperparameter at each iteration')
            plt.legend()
            plt.show()
            
    def plot_ind(self):
        """
        Plots the average of the hyperparameters at each iteration observed over n trials within the result_trial object.
        """
        #These are the parameters that have been fixed in experiment 3-2-1~(line 120-13), if changed then will need adjustment here.
        init_iter=1000
        iterations=1200
        #Plot the data
        bootstrap_data=self.bootstrap_plotdata()
        with plt.style.context(['science','no-latex']):
            fig, ax = plt.subplots(2,2,figsize=(20,20))
            for i,hyperparameter in enumerate(bootstrap_data.keys()):
                title='Hyperparameter: [' + hyperparameter  +']'
                ax.flatten()[i].set_title(str(title))
                ax.flatten()[i].set_xlabel('iterations')
                ax.flatten()[i].set_ylabel('hyperparameter value at xth iteration')
                for key in self.data.keys():
                    ax.flatten()[i].plot(np.arange(init_iter,iterations),self.data[key].data[hyperparameter],label=key, alpha=0.5)
            plt.legend()
            plt.show()

class result_diversity_trial(result_opt):
    def __init__(self, *args, **kwargs):
        self.percentiles=None
        self.data={}
        self._opt={}
        self.key_seed=None
        super(result_diversity_trial, self).__init__(*args, **kwargs)
        
    def __repr__(self):
        return '<Diversity_trial>.object with data from percentiles: {}. Each percentile has data from {} trials.'.format(self.percentiles,
                                                                                                                        len(self))
    def __len__(self):
        return len(self.data[self.percentiles[0]].keys())
    
    def num_res_percentile(self,percentile):
        if percentile in self.data:
            return len(self.data[percentile].keys())
        else:
            return 0
    
    def addresult(self,percentile,current_result):
        """
        Add a result of a particular percentile to this instance.
        percentile (int) - percentile for which the operation needs to be exected.
        current_result (object) - Result object that needs to be adde to the instance.
        """
        if percentile not in self.percentiles:
            raise Exception("Percentile not found")
        try:
            self.data[percentile][current_result.key]=current_result
        except KeyError:
            self.data[percentile]={}
            self.data[percentile][current_result.key]=current_result
    
    def update_async(self):
        """
        Get's the actual result from the async result object.
        """
        for percentile in self.percentiles:
             for keys in self.data[percentile]:
                self.data[percentile][keys]=self.data[percentile][keys].get()
    @property
    def opt(self):
        return self._opt
    @opt.setter
    def opt(self,val):
        if math.isnan(val):
            raise ValueError("The input you have provided is not recognised "
                             "as a valid number")
        self._opt=np.float32(val)
        for percentile in self.percentiles:
            for keys in self.data[percentile]:
                self.data[percentile][keys]._opt=np.float32(val)
    def create_key_seed_dict(self):
        """
        Creates a dictionary with keys to common seeds between the results.
        """
        key_dict={}
        for seed in range(100):
            key_dict[seed]={}
            break_cond=False
            for percentile in self.percentiles:
                try:
                    arg_pos=np.where(np.array([res.seed for res in self.data[percentile].values()])==seed)[0][0]
                    key_dict[seed][percentile]=list(self.data[percentile].keys())[arg_pos]
                except:
                    break_cond=True
            if break_cond:
                del(key_dict[seed])
        self.key_seed=key_dict
                    
        
    def extract_hpdata(self):
        """
        Creates the hyperparameter dictionary in hpdata attribute of the result object.
        """
        self.hpdata=dict()
        for percentile in self.percentiles:
            hp_trial_inst=hp_trial()
            for key in self.data[percentile].keys():
                self.data[percentile][key].hpdata.seed=self.data[percentile][key].seed
                hp_trial_inst.add_hpseries(self.data[percentile][key].hpdata)
            self.hpdata[percentile]=hp_trial_inst
        
    def getkeys(self,percentile):
        """
        Returns the result keys corresponding to different results for the specified percentile.
        """
        return list(self.data[percentile].keys())
    
    def get_traindata(self,percentile):
        """
        Returns training data used in trials for a particular percentile.
        """
        return [self.data[percentile][key].traindata for key in self.getkeys(percentile)]
    
    def getresult(self,percentile,key):
        """
        Returns the result corresponding to the key and percentile.
        percentile (int) - percentile for which the operation needs to be exected.
        key (alphanumeric) - key of the needed result.
        """
        if percentile not in self.percentiles:
            raise Exception("Percentile not found")
        return self.data[percentile][key]
    
    def bootstrap(self,percentile):
        """
        Returns the mean, 5th percentile and 95th percentile variation in the optimality gap at each iteration for the whole trial.
        percentile (int) - percentile for which the operation needs to be exected.
        """
        from utils import bootstrap
        
        return bootstrap([obj.opt_gap for obj in self.data[percentile].values()])
    
    def bootstrap_cumoptgap(self,percentile):
        """
        Returns the mean, 5th percentile and 95th percentile variation in cumulative optimality gap values for the whole trial.
        percentile (int) - percentile for which the operation needs to be exected.
        """
        from utils import bootstrap
        return bootstrap([obj.cum_opt_gap() for obj in self.data[percentile].values()])
    
    
    def difference_bootstrap_plotdata_by_seed(self):
        """
        Returns the difference of bootstrap value when subtracting the mean, and the confidence intervals of the 5th from the 95th, 
        so shows how much better the 95th does on each iteration on an avergae.
        """
        from utils import bootstrap
        self.create_key_seed_dict()
        
        return bootstrap(+np.array([self.data[5][key_dict[5]].opt_gap for key_dict in self.key_seed.values()]) - 
                         np.array([self.data[95][key_dict[95]].opt_gap for key_dict in self.key_seed.values()])
                        )
    def difference_bootstrap_plotdata(self):
        """
        Returns the difference of bootstrap value when subtracting the mean, and the confidence intervals of the 5th from the 95th, 
        so shows how much better the 95th does on each iteration on an avergae.
        """
        from utils import bootstrap
        if len(self.percentiles)==2:
            return bootstrap(self.bootstrap(5) - self.bootstrap(95))
        else:
            return bootstrap(self.bootstrap(5) - self.bootstrap(95)), bootstrap(self.bootstrap(5) - self.bootstrap('smart'))
    
    def percentage_difference_bootstrap_plotdata(self):
        """
        Returns the difference of bootstrap value when subtracting the mean, and the confidence intervals of the 5th from the 95th, 
        so shows how much better the 95th does on each iteration on an avergae.
        """
        from utils import bootstrap
        
        return (self.bootstrap(5) - self.bootstrap(95))/self.bootstrap(95)*100
    
    def cum_opt_gap_improvement_overall(self,percentiles):
        """
        Returns the mean improvement of cumulative optimality gap when going from percentile 1 to percentile 2.
        percentiles (list) - [percentile1, percentile2] The two percentiles which are being compared.
        """
        return -self.bootstrap_cumoptgap(percentiles[1])+self.bootstrap_cumoptgap(percentiles[0])
    
    def percentage_imporvement_cum_opt_gap(self,percentiles):
        """
        Returns the mean improvement of cumulative optimality gap when going from percentile 1 to percentile 2.
        percentiles (list) - [percentile1, percentile2] The two percentiles which are being compared.
        """
        return (-self.bootstrap_cumoptgap(percentiles[1])+self.bootstrap_cumoptgap(percentiles[0]))/self.bootstrap_cumoptgap(percentiles[1])*100
    
    def cum_opt_gap_improvement_overall_by_seed(self):
        
        from utils import bootstrap
        self.create_key_seed_dict()
        
        ## Filter out zero from the array.
        return bootstrap(
            np.array(
                [
            -self.data[95][key_dict[95]].cum_opt_gap()+self.data[5][key_dict[5]].cum_opt_gap() for key_dict in self.key_seed.values()
                ]
            ))