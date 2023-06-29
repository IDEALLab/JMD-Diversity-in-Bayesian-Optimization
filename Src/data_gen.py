
import sys
# sys.path.insert(1, '../../Delta-design-python/src')
# from delta_design_game import graph_representation

import dill
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import sqrtm, inv
import random
import string, os, csv
from random import choices
import pandas as pd
import time
import scipy.stats
import ray
from itertools import compress
from IPython.display import clear_output
from Combination import combination_finder
import decimal,ast
from tqdm.autonotebook import tqdm
import scienceplots


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

def dist_mat(subset,distance_measure):
    """
    Generates dist_mat for delta design game
    """
    if isinstance(subset[0],graph_representation) or torch.is_tensor(subset[0]):
        mat=[]
        for i,game_i in enumerate(subset):
            for j,game_j in enumerate(subset):
                if j<i:
                    mat.append(distance_measure(game_i,game_j))
        return np.array(mat)
    


def set_diff2d(A, B):
    """
    Description : ADD to utils file.: Gives the difference between two sets A and B, such that it returns elements in the larger set that are not in the smaller one.
    """
    if len(A)<len(B):
        A,B=B,A
    res = (A[:, None] == B).all(-1).any(-1)
    return A[~res]

def rows_uniq_elems(a):
    return np.array([v for v in a if len(set(v)) == len(v)])


class diverse_data_generator:
    """
    Description : Object to generate diverse samples from a discrete set on a 2D metric space. This object maybe in the future generalized to N-D metric spaces.
    Parameters:
        X (type: np.ndarray) - This is a Nx2 dimensional array on a metric 2D space where both dimensions are the feature vectors for N samples.
        acquiredsets (type:list) - For the subdpp they define the points that have already been sampled from the discrete set of points that define our sample set.
        data (type: np.ndarray) - Dx2 dimensional array (only populated once generator has been used with subdpp method) with the first dimension as a list of the diversity scores corresponding to the sampled sets in the 2nd dimension.
        gamma (type: float) - Gamma is the hyperparameter for the RBF kernel that is used as a similarity measure for our subDPP.
        options (type:dict) - 
            GPU (type: boolean) - Not compatible yet.
            paralllel-process (type: boolean) - Use parallel process during the subDPP sampling process.
            seed (type: int) - Can be used to generate consistent sets of samples.
            Training-size (type: int) - This is the k in k-dpp(s) it means how many points do you want to sample.
            N_samples (type: int) - This is the D dimension in the data, or the dimension of the distribution of the subDPP.
            bounds (type: list) - [(x1-min,x1-max),(x2-min,x2-max)] Creates a 2D eucliedean-discrete-integer sample space X using the bounds on the space.
            
    """
    def __init__(self):
        self.acquiredsets=[]
        self.data={}
        self.gamma=None
        self.X=None
        self._training_size=None
        self.rng=None
        self._key=None
        self._dim=None
        self.mean=None #Mean for current scores
        self.sd=None #Standard deviation of current scores
        self.tracker={'N_results':0, 'time-tracking':{}}
        self.options={'GPU':False, 'parallel-process': True, 'seed': None, 'N_samples': 10000, 'bounds': None,'time-debug':False,
                      'local_dir':None}
        self.gamma_correlation_dict={}
        self.combination_sampler=combination_finder()
        
    def __repr__(self):
        self.update_data_dict()
        return str(self.data)
    
    @property
    def key(self):
        return self._key
    
    @key.setter
    def key(self,seed):
        if isinstance(seed,int):
            raise Exception('Seed for keygen can only be an int, not {}'.format(type(seed)))
        random.seed(seed)
        self._key= ''.join(random.choice(string.ascii_letters) for i in range(8))
        self.create_data_dict()
        
    @property
    def training_size(self):
        return self._training_size
    
    @training_size.setter
    def training_size(self,value):
        #Clears out the gamma correlation dictionary
        self.gamma_correlation_dict=None
        self._training_size=int(value)
        
    @property
    def dim(self):
        if self.options['bounds'] is None:
            raise Exception('Specify bounds for each dimension')
        else:
            if self._dim is None:
                self.dim=len(self.options['bounds'])
        return self._dim
    
    @dim.setter
    def dim(self,value):
        if not isinstance(value,int):
            raise Exception(f'Dimension can be only an integer not of type {type(value)}')
        if value!=len(self.options['bounds']):
            print('The bounds and dimension set are diffeernt dimensions.')
        #Clears out the gamma correlation dictionary
        self.gamma_correlation_dict=None
        self._dim=value
    
    def create_data_dict(self):
        if not self.options['local_dir']:
            self.data[self.key]={'scores':None, 
                                 'dist': None,
                                 'training_size':self.training_size,
                                 'bounds':self.options['bounds'],
                                 'seed':self.options['seed'],
                                 'N_samples':self.options['N_samples'],
                                 'gamma':self.gamma,
                                 'mean':None,
                                 'standard-deviation':None}
        else:
            self.data[self.key]={'scores':None, 
                                 'dist': None,
                                 'training_size':self.training_size,
                                 'bounds':self.options['bounds'],
                                 'seed':self.options['seed'],
                                 'N_samples':self.options['N_samples'],
                                 'gamma':self.gamma,
                                 'mean':None,
                                 'standard-deviation':None,
                                 'data_loc':self.options['local_dir']+'\\temp_'+self.key+'.csv'}
            
    def update_data_dict(self,scored=False):
        """
        Description:
        """
        if self.options['local_dir']:
            if not scored:
                self.data[self.key].update({'scores':'in temp data file', 
                                         'dist': 'in temp data file',
                                         'gamma':self.gamma,
                                         'mean':self.mean,
                                         'standard-deviation':self.sd})
            else:
                self.data[self.key].update({'scores':'in temp data file', 
                                         'dist': 'in temp data file',
                                         'gamma':self.gamma,
                                         'mean':self.mean,
                                         'standard-deviation':self.sd,
                                         'data_loc':self.options['local_dir']+'\\temp_'+self.key+'_updated.csv'})
       
        else:
            self.data[self.key].update({'mean':self.mean,
                                     'standard-deviation':self.sd})
            
        
    def save_data_dict(self):
        """
        Description:
        """
        import csv,os
        
        data_df=pd.DataFrame.from_dict(self.data)
        file_exists=False
        column_labels_exist=False
        key_exists=False
        
        if os.path.isfile(self.options['local_dir']+'\\data-info.csv'):
            file_exists=True
            current_data_df=pd.read_csv(self.options['local_dir']+'\\data-info.csv')
            if current_data_df.columns[0]=='key':
                column_labels_exist=True
                if self.key in current_data_df['key'].tolist():
                    key_exists=True
                    
        if file_exists:
            current_data_df=current_data_df.set_index('key')
            if column_labels_exist:
                if key_exists:
                    current_data_df.update(data_df.T)
                    current_data_df.to_csv(self.options['local_dir']+'\\data-info.csv',mode='w+',index_label='key')
                else:
                    current_data_df=pd.concat([current_data_df,pd.DataFrame(data_df[self.key]).T])
                    current_data_df.to_csv(self.options['local_dir']+'\\data-info.csv',mode='w+',index_label='key')
            else:
                data_df.T.to_csv(self.options['local_dir']+'\\data-info.csv',mode='w+',index_label='key')
        else:
            data_df.T.to_csv(self.options['local_dir']+'\\data-info.csv',mode='w+',index_label='key')
            
    def find_existing_keys(self):
        if os.path.exists(self.options['local_dir']+'\\data-info.csv'):
            return pd.read_csv(self.options['local_dir']+'\\data-info.csv')['key'].tolist()
        else:
            print('No keys found')

    def load_data_dict(self,key):
        self._key=key
        data_file=pd.read_csv(self.options['local_dir']+'\\data-info.csv',index_col='key').T
        self.data[self.key]=data_file[key].to_dict()
        
    def load_score_stats(self,key):
        self.load_data_dict(key)
        if self.data[key]['mean'] is not None and not np.isnan(self.data[key]['mean']):
            print('stats loaded')
        else:
            all_data=pd.read_csv(self.data[self.key]['data_loc'].split('.')[0]+'_updated'+'.csv',on_bad_lines='skip',header=None)
            scores=all_data[7].to_numpy()
            scores=scores[~np.isnan(scores)]
            scores=scores[scores>-1e10]
            self.data[key]['mean']=scores.mean()
            self.data[key]['standard-deviation']=np.std(scores)
            print('stats loaded')
            self.save_data_dict()

    def update_genereator_from_data_dict(self):
        if not self.key:
            raise Exception('Define generate a key by defining a seed for the keygen, obj.key=seed. Usee seed as None for random key')
            
        self.training_size=self.data[self.key]['training_size']
        self.options['seed']=self.data[self.key]['seed']
        self.options['bounds']=ast.literal_eval(self.data[self.key]['bounds'])
        self.gamma=self.data[self.key]['gamma']
        
        if self.data[self.key]['mean']:
            self.mean=self.data[self.key]['mean']
            
        if self.data[self.key]['standard-deviation']:
            self.sd=self.data[self.key]['standard-deviation']
    
    def update_gamma_correlation_dict(self,g1,g2,correlation):
        if self.gamma_correlation_dict is None:
            self.gamma_correlation_dict={}
        if g1 not in self.gamma_correlation_dict:
            self.gamma_correlation_dict[g1]={}
            self.gamma_correlation_dict[g1][g2]=correlation
            
        else:
            self.gamma_correlation_dict[g1][g2]=correlation
            
        if g2 not in self.gamma_correlation_dict:
            self.gamma_correlation_dict[g2]={}
            self.gamma_correlation_dict[g2][g1]=correlation
            
        else:
            self.gamma_correlation_dict[g2][g1]=correlation
    
    def constructrng(self):
        if self.options['seed'] is None:
            print(self.options['seed'])
            print('There is no seed specified, so the generated results will be inconsistent')
        self.rng={'sampler':np.random.default_rng(self.options['seed'])}
        
    def constructX(self,budget=10000):
        from functools import reduce
        if self.options['bounds'] is None:
            raise Exception('Neither the sample space was given and can not be constructed because bounds have not been specified.')
  
        dim_budget=((budget**(1/self.dim))//2+1)*2
        round_resolution=max([abs(int(np.log(np.ptp(bound)/dim_budget))) for bound in self.options['bounds']]+[1])
        resolution_of_bounds=[np.round(np.ptp(bound)/dim_budget,round_resolution) for bound in self.options['bounds']]
        meshgrid_input=[np.arange(bound[0],bound[1],resolution_of_bounds[i]) for i,bound in enumerate(self.options['bounds'])]
        self.X=np.stack(np.meshgrid(*meshgrid_input), axis=-1).reshape(-1,self.dim)
    
    def generate(self,method="subDPP"):
        
        if self.options['time-debug']:
            self.start=time.perf_counter()
            
        self.acquiredsets=[]
        
        #Generates a key to save the data.
        self.key=None
        
        #Generate a random number generator if there is none.
        if self.rng is None:
            self.constructrng()

        #Generate sample space if there is none.
        if self.X is None:
            self.constructX()
        
        #Save data information to local directory if available.
        if self.options['local_dir']:
            self.save_data_dict()
        
        if method=="subDPP" and self.training_size is None or self.gamma is None:
            raise Exception('Training size or gamma has not been specified, so data can not be generated')
        try:
            options = {"k-dpp-eig" : self.k_dpp_eig,"subDPP": self.sub_dpp}
        except KeyError:
            raise Exception('Method ',method,' is not available at this point in time, the current options are:',list(options.keys()))
            
        return options[method]()
    
    def sample(self,args=None,method="subDPP"):
        try:
            options = {"k-dpp-eig" : self.k_dpp_sample_eig, "subDPP": self.sub_dpp_sample}
        except KeyError:
            raise Exception('Method {} is not available at this point in time, the current options are: {}.'.format(method,list(options.keys())))
        return options[method](args)
        
    def sub_dpp(self):
        """
        Description : Generates subset of DPP distribution of size N_samples. This distribution is then sorted and placed in the data attribute of the object.
        gamma - Controls how the DPP scores each training set
        """
        @ray.remote
        def map_(obj, f):
            return f(*obj)
        
        def twoD_score(subset,gamma):
            """ 
            [Added here temprorarily to fix an import issue with ray]
            Description : Generates a diversity score for a given subset of data given a certain gamma. The diversity score is generated by taking log-determinant of a squared distance kernel.
            """
            if isinstance(subset,(list,np.ndarray)) and isinstance(subset[0][0],(np.int32,np.float64,np.float32,np.int64)):
                D = squareform(pdist(subset, 'euclidean').astype('float64')**2)
            elif not isinstance(subset[0][0],(np.int32,np.float64,np.float32,np.int64)):
                raise ValueError('Elements of subset need to be a numpy int, but are {}'.format(type(subset[0][0])))
            else:
                raise TypeError('A list or np.array was expected but a {} was entered as the subset'.format(type(subset)))
            S = np.exp(-gamma*D)
            (sign, logdet) = np.linalg.slogdet(S)
            return logdet
            
        #Select Gamma
        if self.gamma==None and self.training_size==10:
            self.gamma=1e-6
        elif self.gamma==None:
            raise Exception("self.gamma has not been fixed, look for a range of functional gammas using self.gammacalc")
            
        if self.options['time-debug']:
            t_setup=time.perf_counter()
            
        #Generate the subDPP distribution

        if self.options['local_dir']:
            
            if not ray.is_initialized():
                ray.init(runtime_env={"working_dir": "../src"},num_cpus=6,num_gpus=1)
            self.gen_pos_mat()
            if self.options['time-debug']:
                t_datagen=time.perf_counter()
            
            if not ray.is_initialized():
                ray.init(runtime_env={"working_dir": "../src"},num_cpus=6,num_gpus=1)
            self.score_parallel_data()
            self.load_score_stats(self.key)
        else:
            elements=self.iid_comb_locs()
            pos_mat=[self.combination_sampler.find_elm(self.X.shape[0],self.training_size,element_index) for element_index in elements]
            
            superset=[self.X[pos_array] for pos_array in pos_mat]
            if self.options['time-debug']:
                t_datagen=time.perf_counter()

            #Evaluate the DPP score for each sample.
            if self.training_size>1:
                if self.training_size>50 and self.options['parallel-process']:
                    logdet= np.array(ray.get([map_.remote([subset,self.gamma], twoD_score) for subset in superset]))
                else:
                    logdet=np.array([self.twoD_score(subset,self.gamma) for subset in superset])

                #Normalise the score such that higher score means more diverse.
                self.mean=logdet.mean()
                self.sd=np.std(logdet)
                logdet=(logdet-self.mean)/self.sd
                ind = np.argsort(logdet)
                logdet=logdet[ind]
                superset=np.array(superset)[ind]


                self.data[self.key].update({'scores':logdet, 
                                     'dist': superset})
                self.update_data_dict()
            else:
                self.data[self.key].update({'scores':'Can not be calculated for just a sample_set of size 1.', 
                                     'dist':superset})
        self.tracker['N_results']+=1
        if self.options['time-debug']:
            t_scoregen=time.perf_counter()
            total=t_scoregen-self.start
            self.tracker['time-tracking']={'Total':total,'setting-up': (t_setup-self.start)/total, 'sampling': (t_datagen-t_setup)/total, 'scoring': (t_scoregen-t_datagen)/total}
            delattr(self,'start')
        
    def sub_dpp_sample(self,percentiles):
        """
        Description: Sample points from 'percentile' percentile of the 10,000 subset distribution that is ranked in order of diversity.
        Input: percentile for randomly sampling from the 'percentile'th percentile of the distribution.
        Output: 
        """
        try:
            self.data[self.key]
        except:      
            raise Exception("Data needs to be generated first using the generator method")
            
        current_loc=[]
        
        #Checks if there is a list of percentiles and accordingly returns a list of samples
        if isinstance(percentiles,(list,np.ndarray)):
            for percentile in percentiles:
                if percentile<97.5 and percentile>2.5:
                    possible_locs_percentile=list(range(int((percentile-2.5)/100*self.options['N_samples']),int((percentile)/100*self.options['N_samples'])))
                elif percentile>97.5:
                    possible_locs_percentile=list(range(int(95/100*self.options['N_samples']),self.options['N_samples']))
                else:
                    possible_locs_percentile=list(range(0,int(5/100*self.options['N_samples'])))
                    
                possible_locs = np.setdiff1d(possible_locs_percentile,self.acquiredsets)
                try:
                    loc_temp=self.rng['sampler'].choice(list(possible_locs),1)[0]
                except ValueError:
                    raise Exception('Constructed batch-DPP space exhausted above its {}th percentile. Re-generate the distribution or sample from a smaller percentile.'.format(percentile,self.options['N_samples']))
                current_loc.append(loc_temp)
                self.acquiredsets.append(loc_temp)
        else:
            if percentiles<97.5 and percentiles>2.5:
                possible_locs_percentile=list(range(int((percentiles-2.5)/100*self.options['N_samples']),int((percentiles)/100*self.options['N_samples'])))
            elif percentiles>97.5:
                possible_locs_percentile=list(range(int(95/100*self.options['N_samples']),self.options['N_samples']))
            else:
                possible_locs_percentile=list(range(0,int(5/100*self.options['N_samples'])))
            possible_locs = np.setdiff1d(possible_locs_percentile,self.acquiredsets)
            try:
                current_loc=self.rng['sampler'].choice(list(possible_locs),1)[0]
            except ValueError:
                raise Exception('Constructed batch-DPP space exhausted above its {}th percentile. Re-generate the distribution or sample from a smaller percentile.'.format(percentiles,self.options['N_samples']))
            self.acquiredsets.append(current_loc)
        return self.data[self.key]['dist'][current_loc]
    
    def compare_subDPP(self,resultkey,subset):
        """
        Returns the rank of given subset relative to the result referenced by the key.
        """
        
        current_score=self.twoD_score(subset,self.data[resultkey]['gamma'])
        normal_current_score=(current_score-self.data[resultkey]['mean'])/self.data[resultkey]['standard-deviation']
        rank=np.sum(self.data[resultkey]['scores']<normal_current_score)
        # print('The subset lies in the {} percentile, and its rank is {}'.format(int(rank/self.data[key]['N_samples']),rank))
        
        return rank,normal_current_score
              
    def twoD_score(self,subset,gamma):
        """ 
        Description : Generates a diversity score for a given subset of data given a certain gamma. The diversity score is generated by taking log-determinant of a squared distance kernel.
        """
        if isinstance(subset,(list,np.ndarray)) and isinstance(subset[0][0],(np.int32,np.float64,np.float32,np.int64)):
            D = squareform(pdist(subset, 'euclidean').astype('float64')**2)
        elif isinstance(subset[0][0],(list,np.ndarray)):
            raise ValueError('Elements of subset need to be a (numpy int,numpy int) but are ({},{})'.format(type(subset[0][0]),type(subset[0][0])))
        else:
            print(subset)
            raise TypeError('A list or np.array was expected but a {} was entered as the subset'.format(type(subset)))
        S = np.exp(-gamma*D)
        (sign, logdet) = np.linalg.slogdet(S)
        return logdet
    
    def k_dpp_eig(self):
        """
        Adapted from : https://github.com/ChengtaoLi/dpp
        Description : Constructs the L-ensemble and decomposes it into eigen-values for faster computation. 
        """
        import scipy.linalg
        
        if self.X is None:
            self.constructX()
        pairwise_dists = squareform(pdist(self.X, 'euclidean'))
        L = np.exp(-pairwise_dists ** 2 / 0.5 ** 2)
        # Get eigendecomposition of kernel matrix
        self.D, self.V = scipy.linalg.eigh(L)
        
    def E(self,D, k):
        N = D.shape[0]
        E = np.zeros((k+1, N+1))

        E[0] = 1.
        for l in list(range(1,k+1)):
            E[l,1:] = np.copy(np.multiply(D, E[l-1,:N]))
            E[l] = np.cumsum(E[l], axis=0)
        return E
    
    def k_dpp_sample_eig(self,k=None):
        """
        Adapted from : https://github.com/ChengtaoLi/dpp
        Description: Constructs a discrete DPP using the decomposed eigenvalues of the L-ensemble matrix. Samples K-diverse points from this distribution.
        """
        
        if self.X is None:
            self.constructX()
        if k is None:
            k=self.training_size
        def sample_k(D, E, k):
            i = D.shape[0]
            remaining = k
            rst = list()

            while remaining > 0:
                if i == remaining:
                    marg = 1.
                else:
                    marg = D[i-1] * E[remaining-1, i-1] / E[remaining, i]

                if np.random.rand() < marg:
                    rst.append(i-1)
                    remaining -= 1
                i -= 1
            return np.array(rst)
        
        def sym(X):
            return X.dot(inv(np.real(sqrtm(X.T.dot(X)))))
        
        N = self.D.shape[0]
        v_idx = sample_k(self.D, self.E(self.D,k), k)
        V = self.V[:,v_idx]
        rst = list()
        for i in list(range(k-1,-1,-1)):
            # choose indices
            
            P = np.sum(V**2, axis=1)

            row_idx = np.random.choice(range(N), p=P/np.sum(P))
            col_idx = np.nonzero(V[row_idx])[0][0]

            rst.append(row_idx)

            # update V
            V_j = np.copy(V[:,col_idx])
            V = V - np.outer(V_j, V[row_idx]/V_j[row_idx])
            V[:,col_idx] = V[:,i]
            V = V[:,:i]

            # reorthogonalize
            if i > 0:
                V = sym(V)

        rst = np.sort(rst)
        return self.X[rst]
    
    def gen_pos_mat(self):
        """
        Description:
        """

        @ray.remote(num_gpus=0.01,num_cpus=0.1)
        def map_(element_nums, total_elements,training_size, comb_finder,data_loc):
            import csv
            elements={}
            for element_num in element_nums:
                elements[element_num]=comb_finder.find_elm(total_elements,training_size,element_num)
            with open(data_loc,'a') as f:
                writer = csv.writer(f, dialect='excel')
                for element_num in element_nums:
                    writer.writerow([element_num, *elements[element_num]])
            del elements

        def parallel_pos_mat(self,elements_for_ray):
            print('Starting_parallel_process')
            def create_batch_data(data,batch_size=1000):
                return [data[init:min(init+batch_size,len(data)+1)] for init in range(0,len(data),batch_size)]
            batch_element_data=create_batch_data(elements_for_ray)
            subset=ray.get([map_.remote(element_nums, self.X.shape[0],self.training_size, self.combination_sampler,
                                        self.data[self.key]['data_loc']) for element_nums in batch_element_data])
            ray.shutdown()

        def post_process(self,elements,comb,skip_rows=None,all_data=None):
            print('Post parallel processing')

            if all_data is not None:
                all_data.append(pd.read_csv(self.data[self.key]['data_loc'],
                                            header=None,on_bad_lines='skip',skiprows=skip_rows).dropna())
            else:
                all_data=pd.read_csv(self.data[self.key]['data_loc'],
                                     header=None,on_bad_lines='skip',skiprows=skip_rows).dropna()
            all_data=all_data.set_index(0)
            correct_elms=list(set(all_data.index).intersection(elements))
            all_data=all_data[all_data.index.isin(correct_elms)]
            print('Completed reading data as a pandas dataframe')
            pos_mat=list(all_data.to_numpy().astype(np.int32))
            missing_elm_indices=list(set(elements)-set(all_data.index))
            
            #Check if the parallel process missed too many elements that it warrants a re-run
            if len(missing_elm_indices)>0.1*len(elements):
                with open(self.data[self.key]['data_loc']) as f:
                    skip_rows=sum(1 for line in f)
                parallel_post_mat(self,missing_elm_indices,local_dir,comb)
                post_process(local_dir,elements,comb,skip_rows=skip_rows,all_data=all_data.reset_index(level=0))

            missing_elms=[np.array(self.combination_sampler.find_elm(self.X.shape[0],self.training_size,element_index)) for element_index in missing_elm_indices]
            pos_mat=pos_mat+missing_elms
            missing_elms_df=pd.DataFrame(missing_elms)
            missing_elms_df=missing_elms_df.rename(columns={0:1,1:2,2:3,3:4,4:5})
            missing_elms_df.insert(0,0,list(missing_elm_indices))
            missing_elms_df=missing_elms_df.set_index(0)
            all_data=all_data.append(missing_elms_df)
            self.save_data_dict()
            
            print('Found the missing elements, and compiled full pos_mat')
            if not self.options['local_dir']:
                self.pos_mat_complete=True
                del all_data
                return pos_mat
            else:
                print('Compiled data frame with all elements. \n Updating the temp csv file...')
                self.pos_mat_complete=True
                all_data.to_csv(self.data[self.key]['data_loc'], mode='w', index=True, header=False)
                del all_data
                
            
        elements=self.iid_comb_locs()
        if not hasattr(self,'pos_mat_complete'):
            self.pos_mat_complete=False
        if not self.pos_mat_complete:
            if os.path.exists(self.data[self.key]['data_loc']):
                with open(self.data[self.key]['data_loc'],'r') as f:
                    reader = csv.reader(f, dialect='excel', delimiter=',')
                    current_elements=[]
                    for row in reader:
                        if row:
                            current_elements.append(int(row[0]))
                elements_for_ray=list(set(elements)-set(current_elements))
                del current_elements,reader
            else:
                elements_for_ray=elements
                with open(self.data[self.key]['data_loc'],'w+') as f:
                    pass
            parallel_pos_mat(self,elements_for_ray)
        else:
            delattr(self,'pos_mat_complete')
            
        post_process(self,elements,self.combination_sampler,skip_rows=None,all_data=None)
            
    def score_parallel_data(self,batch_size=10000):
        import os
        from itertools import repeat
        if os.path.isfile(self.options['local_dir']+'\\score_tracker.csv'):
            existing_intervals=pd.read_csv(self.options['local_dir']+'\\score_tracker_'+self.key+'.csv',header=None)[0].tolist()
        else:
            with open(self.options['local_dir']+'\\score_tracker_'+self.key+'.csv','w+') as f:
                writer = csv.writer(f, dialect='excel')
                writer.writerow([])
            existing_intervals=[]

        @ray.remote(num_gpus=0.01,num_cpus=0.1)
        def map_(self,interval_num,interval_size):
            import pandas as pd
            from itertools import repeat
            if not os.path.isfile(self.data[self.key]['data_loc']):
                with open(self.data[self.key]['data_loc'],'w+'):
                    pass
            pos_mat_df=pd.read_csv(self.data[self.key]['data_loc'], skiprows=(interval_num)*interval_size, nrows=interval_size,header=None)
            pos_mat=pos_mat_df.to_numpy()[:,1:].astype(np.int32)
            superset=[self.X[pos_array] for pos_array in pos_mat]
            logdet= np.array(list(map(self.twoD_score,superset,repeat(self.gamma))))
            pos_mat_df['emperical_scores']=logdet
            pos_mat_df.set_index(0)
            new_file=self.data[self.key]['data_loc'].split('.')[0]+'_updated'+'.csv'
            if not os.path.isfile(new_file):
                with open(new_file,'w+'):
                    pass
            pos_mat_df.to_csv(new_file,mode='a',header=False)
            with open(self.options['local_dir']+'\\score_tracker_'+self.key+'.csv','a') as f: 
                writer = csv.writer(f, dialect='excel')
                writer.writerow([interval_num])
            del pos_mat
            return logdet
        if self.options['N_samples']%batch_size>0:
            all_intervals=list(range(int(self.options['N_samples']/batch_size)+1))
        else:
            all_intervals=list(range(int(self.options['N_samples']/batch_size)))
        needed_intervals=set(all_intervals)-set(existing_intervals)
        
        logdet=ray.get([map_.remote(self,interval_num,batch_size) for interval_num in needed_intervals])
        
        #Final check
        old_file=pd.read_csv(self.options['local_dir']+'\\temp_'+self.key+'.csv',header=None).set_index(0)
        updated_file=pd.read_csv(self.data[self.key]['data_loc'],header=None)
        
        elements_in_updated_file=set(updated_file[0].tolist())
        elements_in_old_file=set(old_file.index.tolist())
        
        missing_elms = list(elements_in_updated_file-elements_in_old_file)
        if len(missing_elms)>0:
            missing_pos_mat = [old_file.T[elm].to_numpy() for elm in missing_elms]
            missing_superset=[self.X[pos_array] for pos_array in missing_pos_mat]
            missing_logdet= np.array(list(map(self.twoD_score,missing_superset,repeat(self.gamma))))

            updated_file.append({0:missing_elms,
                                 1:np.array(missing_pos_mat)[:,0],
                                 2:np.array(missing_pos_mat)[:,1],
                                 3:np.array(missing_pos_mat)[:,2],
                                 4:np.array(missing_pos_mat)[:,3],
                                 5:missing_logdet},ignore_index=True)
            logdet=np.concatenate([logdet,missing_logdet])
        
        logdet=np.array(logdet).flatten()
        
        self.mean=logdet.mean()
        self.sd=np.std(logdet)
        logdet=(logdet-self.mean)/self.sd
        self.update_data_dict(scored=True)
        self.save_data_dict()
        
    def load_score_stats(self,key):
        self.load_data_dict(key)
        if self.data[key]['mean'] is not None and not np.isnan(self.data[key]['mean']):
            print('stats loaded')
        else:
            all_data=pd.read_csv(self.data[self.key]['data_loc'].split('.')[0]+'_updated'+'.csv',on_bad_lines='skip',header=None)
            scores=all_data[int(self.training_size+2)].to_numpy()
            scores=scores[~np.isnan(scores)]
            scores=scores[scores>-1e10]
            self.data[key]['mean']=scores.mean()
            self.data[key]['standard-deviation']=np.std(scores)
            print('stats loaded')
            self.save_data_dict()
    
    def gammacorrelation(self,g1,g2,min_samples):
        """
        Description: Calculates the correlation between the relative ordering of subsets in the N_sampled distribution.
        """
        self.options['N_samples']=min_samples
        self.options['parallel-process']=False
        self.gamma=10**(g1)
        self.generate()
        score1=self.data[self.key]['scores']
        self.gamma=10**(g2)
        score2=np.array([self.twoD_score(subset,self.gamma) for subset in self.data[self.key]['dist']])
        score2_mean=score2.mean()
        score2_sd=np.std(score2)
        score2=(score2-score2_mean)/score2_sd
        x= np.argsort(score1)
        y= np.argsort(score2)
        if np.isnan(score2).any():
            np.random.shuffle(y)
        return scipy.stats.linregress(x,y).rvalue
    
    def gammacalc(self,grange=None,plot=False,annotations=False,min_samples=200):
        """
        Description : Returns the biggest range of gammas that will work for a specific training size.
        Input: 
        grange: (optional), Default-value: (-12,4), type : (tuple,list,numpy.ndarray) : Specify the range of gammas to be checked.
        Output:
        correlatedgammas: type: tuple : Gives the log range of gamma(s) that work with the current training_size of the 
        """
        
        from multiprocessing import Pool
        from itertools import repeat
         
        # Assigns a log range of gamma's to compute the compatibility of the sampler over.
        if grange is None:
            gamma_range=range(-12,4)
        elif isinstance(grange,(tuple,list,np.ndarray)):
            gamma_range=np.arange(*grange)
        else:
            raise Exception('gamma\'s range should be a tuple, list or numpy.ndarray ')
        
        # Calculate the correlation of two gamma's in the range
        
        if self.gamma_correlation_dict is None:
            self.gamma_correlation_dict=dict()
            
        correlation_mat=np.zeros(shape=(len(gamma_range),len(gamma_range)))
        for i,g1 in enumerate(gamma_range):
            for j,g2 in enumerate(gamma_range):
                try:
                    correlation_mat[i,j]=self.gamma_correlation_dict[g1][g2]
                except KeyError:
                    correlation_mat[i,j]=self.gammacorrelation(g1,g2,min_samples)
                    self.update_gamma_correlation_dict(g1,g2,correlation_mat[i,j])
        
        #Plot the correlation matrix (optional)
        if plot:
            with plt.style.context(['science','no-latex']):
                fig, ax = plt.subplots(figsize=(12,12))
                im = ax.imshow(correlation_mat,cmap='viridis')
                ax.set_xticks(np.arange(len(gamma_range)), labels=gamma_range,fontsize='18')
                ax.set_yticks(np.arange(len(gamma_range)), labels=gamma_range,fontsize='18')
                cbar=fig.colorbar(im)
                cbar.ax.tick_params(labelsize='15')
                if annotations:
                    for i in range(len(gamma_range)):
                        for j in range(len(gamma_range)):
                            text = ax.text(j, i, correlation_mat[i, j])
                
        #Fill the diagonals with 0 because we do not care about picking a highly correlated single gamma which always happen when gammas are the same.
        np.fill_diagonal(correlation_mat,0)
        #Appends gamma to a list that satisfy a certain level of minimum correlation threshold.
        correlated_gammas=[]
        min_cor=0.95
        while len(correlated_gammas)<2:
            correlated_gammas=np.stack([[gamma_range[i] for i in np.where([correlation_mat>min_cor][0])[0]],[gamma_range[i] for i in np.where([correlation_mat>min_cor][0])[1]]]).T
            min_cor+=-0.05
            
            # If none of the correlations meet the threshold return Nonetype object.
            if min_cor<0.5:
                print("Only Poorly coorelated gammas could be found for [{},{}] range of gamma(s). Try to change the range of gammas by adding a custom range.".format(min(grange),max(grange)))
                return
        if plot:
            return fig,correlated_gammas[np.abs([np.abs(i)[0]-np.abs(i)[1] for i in correlated_gammas]).argmax()]
        # The return statements looks at the list of correlated gamma(s) and generates a tuple of the widest range of these gamma values.
        return correlated_gammas[np.abs([np.abs(i)[0]-np.abs(i)[1] for i in correlated_gammas]).argmax()]
    
    def optgammarand(self,current_set,set_size):
        """
        Description : Generates a list with length set_size of random of 2D points that are not present in current_set, but lie in the sample space for X.
        """
        random.seed(self.options['seed'])
        if self.X is None:
            self.constructX()
        return np.array(random.sample(list(set_diff2d(current_set,self.X)),set_size))
    
    def iid_comb_locs(self,combination=None,sample_size=None):
        """
        Description: Used to randomly choose 'sample_size' # of element locations from possible locations in range(0,math.comb(*combinations)).
        Input (optional_arguments):
            combination [tuple] (default: (# of elements in X, training_size))  
            sample_size [int] (default: N_samples to construct the sub-DPP) 
        """
        if combination is None:
            combination=self.X.shape[0],self.training_size
        if sample_size is None:
            sample_size=self.options['N_samples']
        total_combinations=math.comb(*combination)
        if total_combinations>np.iinfo(np.int64).max:
            np.random.seed(self.options['seed'])
            seed_list=np.random.randint(low=np.iinfo(np.int32).min,high=np.iinfo(np.int32).max,size=sample_size)
            elements=[]
            for int_seed in seed_list:
                random.seed(int_seed)
                elements.append(random.randint(0,total_combinations))
        else:
            if self.options['N_samples']/total_combinations<0.6:
                elements=random.sample(range(0,total_combinations),sample_size)
            else:
                inverse_sample=set(random.sample(range(0,total_combinations),total_combinations-sample_size))
                full_sample=set(range(0,total_combinations))
                our_sample=full_sample-inverse_sample
                elements=list(our_sample)
        return elements