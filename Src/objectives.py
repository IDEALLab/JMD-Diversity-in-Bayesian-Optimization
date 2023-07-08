
import math
import numpy as np
from scipy.stats import multivariate_normal
from opensimplex import OpenSimplex
from scipy.spatial.distance import pdist, squareform
import random
from scipy import interpolate
# from utils import closest_N
from SimpleNoise import SimplexNoise        
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
import os
import dill
import pandas as pd
import ray
import torch

def load_variable(filename):
    with open(filename, 'rb') as f:
        variable = dill.load(f)
    return variable

def savefile(filename,variable):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        dill.dump(variable, f)
    return variable


def cont_func(x,f):
    try:
        res=f(*x)
    except:
        x=np.stack([b for b in x])
        if x.shape[1]!=len(X):
            x=x.T
        res=np.array([f(*a)[0] for a in x ],dtype=np.float32)
    return res.ravel()

## Utitlity lambda functions


construct_slice= lambda dims,dim_num: (slice(None),)*dims+(dim_num,)

class objectives:
    def __init__(self):
        self.args=None
        self.bounds=None
        self.seed=None
        self._dim=None
        self._fun_name='wildcatwells'
        self.noise_based_funcs=['wildcatwells']
        self.minstate=False
        self.budget=None
        self.parallel=False
        pass
    
    @property
    def fun_name(self):
        return self._fun_name
    
    @fun_name.setter
    def fun_name(self,func_name):
        if func_name in ['Sphere','Rastrigin','Rosenbrock']:
            self.minstate=True
            self._fun_name=func_name
            self.optimal_y=0
        elif func_name=='wildcatwells':
            self.minstate=False
            self._fun_name=func_name
            self.optimal_y=100
        else:
            print(f'Choose a different function_name {func_name} is not available. \n Available functions are: wildcatwells, Sphere, Rastrigin and Rosenbrock') 
    
    @property
    def dim(self):
        if self.bounds is None:
            raise Exception('Specify bounds for each dimension')
        else:
            if self._dim is None:
                self._dim=len(self.bounds)
        return self._dim

    def construct_X(self,with_noise_coords=False):
        """
        Constructs X based on given bounds for each dimension.
        """
        from functools import reduce
        
        if self.budget is None:
            budget=10000
        else:
            budget=self.budget

        dim_budget=((budget**(1/len(self.bounds)))//2+1)*2
        round_resolution=max([abs(int(np.log(np.ptp(bound)/dim_budget))) for bound in self.bounds]+[1])
        resolution_of_bounds=[np.round(np.ptp(bound)/dim_budget,round_resolution) for bound in self.bounds]
        meshgrid_input=[np.arange(bound[0], bound[1]+resolution_of_bounds[i], resolution_of_bounds[i]).astype(np.float32) for i,bound in enumerate(self.bounds)]
        if with_noise_coords:
            noise_input=[np.arange(0,100,100/len(meshgrid_input[i])) for i,bound in enumerate(self.bounds)]
            return np.stack(np.meshgrid(*meshgrid_input), axis=-1),np.stack(np.meshgrid(*noise_input), axis=-1)
        else:
            return np.stack(np.meshgrid(*meshgrid_input), axis=-1)
                
    
    def wildcatwells(self):
        """
        args={
        N: Number of peaks [Integer value]
        Smoothness: Input is a decimal value in range [0,1]
        rug_freq: Input is a decimal value in range [0,1]
        rug_amp: Input is a decimal value in range [0,1]
        }
        """
        
        N=self.args['N']
        smoothness=self.args['Smoothness']
        rug_freq=self.args['rug_freq']
        rug_amp=self.args['rug_amp']
        
        def gen_loc_list(possible_loc,num_points,min_dist):
            """
            Given a list of locations, returns a list of locations that are at least a min_dist away from each other.
            Use: for ruggedness frequency.
            """
            loc_list=[possible_loc[0]]
            def pick_point(loc_list,point,min_dist):
                loc_list.append(point)
                if squareform(pdist(loc_list, 'euclidean'))[len(loc_list)-1].min()>min_dist:
                    return loc_list
                return loc_list[:-1]
            for points in possible_loc[1:]:
                if len(loc_list)==num_points:
                    break
                pick_point(loc_list,points,min_dist)
            return loc_list

        def gen_ND_rand(self):
            """
            Get a random location on the input space of the features.
            """
            possible_loc=list()
            np.random.seed(self.seed)
            for bound in self.bounds:
                randsamp=(bound[1]-bound[0])*np.random.rand(100,1)+bound[0]
                possible_loc.append(randsamp)
            possible_loc=np.concatenate(possible_loc,axis=1)
            return possible_loc

        def get_ND_loc(flattened_loc,X):
            """
            Converts a flattened location on a numpy array to the orignal ND array location given the orignal array.
            """
            ND_loc=[]
            current_remainder=flattened_loc
            for elms in np.flip(X.shape)[1:]:
                ND_loc.append(current_remainder%elms)
                current_remainder=current_remainder//elms
            ND_loc.reverse()
            return tuple(ND_loc)
        
        @ray.remote(num_gpus=0.1,num_cpus=1)
        def get_noise_data_parallel(noise_gen,noise_coord,feature_size):
            return noise_gen.simplexNoise(noise_coord/feature_size)+noise_gen.simplexNoise(noise_coord/(feature_size/4)) / 4 + noise_gen.simplexNoise(noise_coord/(feature_size/8)) / 8
        
        def get_noise_data(noise_gen,noise_coord,feature_size):
            return noise_gen.simplexNoise(noise_coord/feature_size)+noise_gen.simplexNoise(noise_coord/(feature_size/4)) / 4 + noise_gen.simplexNoise(noise_coord/(feature_size/8)) / 8
            
        if self.budget is not None:
            X,noise_feature=self.construct_X(self.budget,with_noise_coords=True)
        else:
            X,noise_feature=self.construct_X(with_noise_coords=True)
            
        gridsize=np.average(self.bounds,axis=1)
        new_loc=tuple([gridsize]*int(self.dim))

        loc_list=gen_loc_list(gen_ND_rand(self),N,min((1-rug_freq)*100/N+np.average(gridsize)/5-N,np.max(self.bounds)-1))
        gausMat=0
        for new_loc in loc_list:
            if N>1:
                mv_norm_mean=np.identity(len(self.bounds))*((1.1-rug_freq)/N)*(0.3*gridsize)**len(self.bounds)
                rv = multivariate_normal(new_loc, mv_norm_mean)
            else:
                mv_norm_mean=np.identity(len(self.bounds))*(0.2*gridsize)**len(self.bounds)
                rv = multivariate_normal(new_loc, mv_norm_mean)
            gausMat+= rv.pdf(X)

        A=np.zeros(X.shape[:-1])
        noise_gen=SimplexNoise(self.seed,self.dim)
        feature_size = 100*(1-(1-smoothness)**2)
        
        from tqdm.autonotebook import tqdm
        if self.parallel:
            ray.init(runtime_env={"working_dir": "../src"}, num_cpus=10,num_gpus=2,log_to_driver=False)
            async_values=[]
            with tqdm(total=noise_feature.reshape(-1,len(self.bounds)).shape[0]) as pbar:
                for oneD_loc,noise_coord in enumerate(noise_feature.reshape(-1,len(self.bounds))):
                    async_values.append(get_noise_data_parallel.remote(noise_gen,noise_coord,feature_size))
                    pbar.update(1)
            values=ray.get(async_values)
        else:
            values=[]
            with tqdm(total=noise_feature.reshape(-1,len(self.bounds)).shape[0]) as pbar:
                for oneD_loc,noise_coord in enumerate(noise_feature.reshape(-1,len(self.bounds))):
                    values.append(get_noise_data(noise_gen,noise_coord,feature_size))
                    pbar.update(1)
        
        for oneD_loc,noise_coord in enumerate(noise_feature.reshape(-1,len(self.bounds))):
            color = (values[oneD_loc] + 1)*N
            A[get_ND_loc(oneD_loc,X)]=color

        power=-math.log(A.max()/gausMat.max())/math.log(10)
        surf = gausMat+A*10**(power-(1-rug_amp)**5)
        surf = (surf / surf.flatten().max())*100
        surf=surf.astype(np.float32)
        self.minstate=False
        return X,surf
            
    def Rastrigin(self,X):
        """
        (ND function)
        Recommended bounds x_i \in [(-5.12,5.12)]*(dims)
        args- {
        A - measure of distance between height and peaks}
        """
      
        A=self.args['A']
        if isinstance(X,(np.ndarray)):
            if len(X.shape)!=self.dim or len(X.shape)==2:
                surf=np.sum(X.T**2+A*np.cos(self.dim*np.pi*X.T),axis=0)+A*self.dim
            else:
                surf=np.sum(X**2+A*np.cos(self.dim*np.pi*X),axis=self.dim)+A*self.dim
        elif torch.is_tensor(X):
            if len(X.shape)!=self.dim or len(X.shape)==2:
                surf=torch.sum(X.T**2+A*np.cos(self.dim*np.pi*X.T),axis=0)+A*self.dim
            else:
                surf=torch.sum(X**2+A*np.cos(self.dim*np.pi*X),axis=self.dim)+A*self.dim
            
        self.minstate=True
        return surf
    
    def Sphere(self,X):
        """
        (ND function)
        Recommended bounds x_i \in [(-10,10)]*(dims)
        """
        if isinstance(X,(np.ndarray)) and len(X.shape)>1:
            surf=np.sum(X**2,axis=-1)
        elif (isinstance(X,(np.ndarray)) and len(X.shape)==1) or isinstance(X,(list)):
            surf=np.array([sum(x_i**2) for x_i in X])
        elif torch.is_tensor(X) and len(X.shape)>1:
            surf=torch.sum(X**2,axis=-1)
        self.minstate=True
        return surf
        
    def Rosenbrock(self,X):
        """
        (ND function)
        Recommended bounds x_i \in [(-10,10)]*(dims)
        args- {
        A - measure of distance between height and peaks}
        """
        if (isinstance(X,np.ndarray) or torch.is_tensor(X)) and (len(X.shape)==2 and X.shape[-1]==self.dim) :
            if len(X.shape)==2 and X.shape[-1]==self.dim:
                if torch.is_tensor(X):
                    surf=torch.zeros_like(X[:,0])
                else:
                    surf=np.zeros_like(X[:,0])
                for dim in range(self.dim-1):
                    surf+=100*(X[:,dim+1]-X[:,dim]**2)**2 + (1-X[:,dim])**2
        else:
            surf=[]
            for x_i in X:
                surf_ind=0
                for dim in range(self.dim-1):
                    surf_ind+=100*(x_i[dim+1].T-x_i[dim].T**2)**2 + (1-x_i[dim].T)**2
                surf.append(surf_ind)
            surf=type(X)(surf)
        self.minstate=False
        return surf
    
    def xinsheyang3(self,X):
        """
        (2D function)
        Recommended bounds x_i \in [(-20,20)]
        args - {
        b - spread
        m - steepness}
        """
        surf= np.exp(-np.add(*(X/b)**(2*m))) - 2*np.exp(-np.add(*X**2))*np.multiply(np.cos(X)**2)
        self.minstate=True
        return surf
    
    def xinsheyang4(self,X):
        """
        (2D function)
        Recommended bounds x_i \in [(-20,20)]
        """
        surf= (np.add(*np.sin(X)**2) - np.exp(-np.add(*X**2)))*np.exp(-np.add(*np.sin(np.sqrt(np.abs(X)))**2))
        self.minstate=True
        return surf

    def damavandi(self,X):
        """
        (2D function)
        Recommended bounds x_i \in [(0,14)]
        """
        surf= (1-np.abs(np.multiply(*np.sinc((X-2)/np.pi)**5)))*(2+(X[0]-7)**2+2*(X[1]-7)**2)
        self.minstate=True
        return surf
    
    def forrester(x):
        """
        (1D function)
        Recomended bounds x_0 \in [(-5,5)]
        """
        y=np.sin(12*x-4)*(6*x-2)**2
        self.minstate=False
        return y
    
    def generate_cont(self,from_saved=True,local_dir=None,budget=10000):
        """
        Interpolates to generate a continuous function
        """
        if self.fun_name in self.noise_based_funcs: 
            return self.noise_based_cont(from_saved=from_saved,local_dir=local_dir,budget=budget)
        else:
            return getattr(self,self.fun_name)
        
    def noise_based_cont(self,from_saved=False,local_dir=None,budget=10000):
        """
        Interpolates to generate a continuous function
        """
        
        if from_saved and local_dir:
            X,surf=self.load_saved_func(local_dir)
        else:
            test_func=self.fun_name
            X,surf=getattr(self,test_func)()
        if self.dim>2:
            budget=np.prod(X.shape[:-1])
            dim_budget=((budget**(1/len(self.bounds)))//2)*2
            round_resolution=max([abs(int(np.log(np.ptp(bound)/dim_budget))) for bound in self.bounds]+[1])
            resolution_of_bounds=[np.round(np.ptp(bound)/dim_budget,round_resolution) for bound in self.bounds]
            meshgrid_input=[np.arange(bound[0], bound[1]+resolution_of_bounds[i], resolution_of_bounds[i]).astype(np.float32) for i,bound in enumerate(self.bounds)]
            continous_func = RegularGridInterpolator(meshgrid_input, surf, method='linear', bounds_error=True)
        else:
            x_shape=list(X.shape)
            continous_func = LinearNDInterpolator(X.reshape(np.prod(x_shape[:-1]),self.dim), surf.flatten())
        return continous_func
    
    def save_objective_func(self,local_dir):
        """
        Save the current defined objective function.
        """
        data_df=pd.DataFrame(self.args,index=[0])
        data_df['fun_name']=self.fun_name
        data_df['dims']=self.dim
        data_df['seed']=self.seed
        
        file_exists=False
        column_labels_exist=False
        key_exists=False
        def check_if_larger_df_has_smaller_df(small_df,larger_df,args):
            """
            Checks if the `small_df' is a row inside the `larger_df.'
            """
            element_wise_comparison=lambda small_df,current_df,args: [(current_df[arg]==small_df[arg][0]).tolist() for arg in args]
            row_comparison=list(map(all, zip(*element_wise_comparison(small_df,larger_df,args))))
            try:
                return [row_comparison.index(True)]
            except ValueError:
                return False
            
        ##Create a pandas df for the current objective function instance.
        if os.path.isfile(local_dir+'\\objective-data.csv'):
            file_exists=True
            current_data_df=pd.read_csv(local_dir+'\\objective-data.csv')
            current_data_df=current_data_df.set_index('key')
            if current_data_df.columns[0]=='N':
                column_labels_exist=True
                if check_if_larger_df_has_smaller_df(data_df,current_data_df,list(self.args.keys())+['dims','seed']):
                    key_exists=True
                    data_df.index=check_if_larger_df_has_smaller_df(data_df,current_data_df,list(self.args.keys())+['dims','seed'])
                    data_df['loc']=local_dir+'\\'+self.fun_name+str(data_df.index.tolist()[0])+'.pkl'
                else:
                    data_df.index=[len(current_data_df.index)]
                    data_df['loc']=local_dir+'\\'+self.fun_name+str(len(current_data_df.index))+'.pkl'
            else:
                data_df.index=[len(current_data_df.index)]
                data_df['loc']=local_dir+'\\'+self.fun_name+str(len(current_data_df.index))+'.pkl'
        else:
            data_df['loc']=local_dir+'\\'+self.fun_name+str(data_df.index.tolist()[0])+'.pkl'

        ##Save the objective function.
        savefile(data_df['loc'].tolist()[0],getattr(self,self.fun_name)())

        ##Update the local directory objective functions log.
        if file_exists:
            if column_labels_exist:
                if key_exists:
                    print('key_exists')
                    current_data_df.update(data_df)
                    current_data_df.to_csv(local_dir+'\\objective-data.csv',mode='w+',index_label='key')
                else:
                    print('columns exist but key doesn\'t')
                    current_data_df=pd.concat([current_data_df,data_df])
                    current_data_df.to_csv(local_dir+'\\objective-data.csv',mode='w+',index_label='key')
            else:
                print('File exists but columns don\'t')
                data_df.to_csv(local_dir+'\\objective-data.csv',mode='w+',index_label='key')
        else:
            data_df.to_csv(local_dir+'\\objective-data.csv',mode='w+',index_label='key')
            
    def list_saved_funcs(self,local_dir):
        """
        Lists the objective functions saved in the `local_dir.'
        """
        if os.path.isfile(local_dir+'\\objective-data.csv'):
            current_data_df=pd.read_csv(local_dir+'\\objective-data.csv')
            current_data_df=current_data_df.set_index('key')
            print(current_data_df)
        else:
            return

    def load_saved_func(self,local_dir):
        """
        Load the saved function from given `local_dir.' 
        """
        data_df=pd.DataFrame(self.args,index=[0])
        data_df['dims']=self.dim
        data_df['fun_name']=self.fun_name
        data_df['seed']=self.seed

        def check_if_larger_df_has_smaller_df(small_df,larger_df,args):
            element_wise_comparison=lambda small_df,current_df,args: [(current_df[arg]==small_df[arg][0]).tolist() for arg in args]
            row_comparison=list(map(all, zip(*element_wise_comparison(small_df,larger_df,args))))
            try:
                return [row_comparison.index(True)]
            except ValueError:
                return False

        if os.path.isfile(local_dir+'\\objective-data.csv'):
            current_data_df=pd.read_csv(local_dir+'\\objective-data.csv')
            current_data_df=current_data_df.set_index('key')
            if check_if_larger_df_has_smaller_df(data_df,current_data_df,list(self.args.keys())+['dims','seed']):
                return load_variable(current_data_df['loc'][check_if_larger_df_has_smaller_df(data_df,current_data_df,
                                                                                              list(self.args.keys())+['dims','seed'])[0]])
            else:
                print('The current objective function settings do not have a saved object. \nHere is a list of saved objective functions:')
                self.list_saved_funcs(local_dir)
        else:
            print('The current objective function settings do not have a saved object. \nHere is a list of saved objective functions:')
            self.list_saved_funcs(local_dir)