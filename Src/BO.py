import sys
sys.path.insert(1, '../')

import numpy as np
import random
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import time
import gpytorch
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from utils import div_data_gen, closest_N
import results
import warnings
from IPython.display import clear_output

def generate_initial_data(X,Y,minstate,n=10):
    # generate training data
    if range(len(X)<n):
        n=len(X)
    picks=np.array([random.sample(range(len(X)), n)])
    train_x = torch.from_numpy(X[picks])
    train_x = train_x.type(torch.FloatTensor)[0]
    train_obj = torch.from_numpy(Y[picks])[0]
    if minstate==True:
        best_observed_value = train_obj.min()
    else:
        best_observed_value = train_obj.max()
    return train_x, train_obj.reshape(-1,1), best_observed_value

def initialize_model(train_x, train_obj,nu=1.5,covar="matern"):
    """
    Inputs
    train_X: nD tensor with n Training points
    train_obj: nD tensor with n evaluations at train_x
    nu: hyperparameter for the matern kernel
    covar: type of covariance module used for kernel
    """    
    from gpytorch.priors.torch_priors import GammaPrior
    from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood)
    from gpytorch.constraints.constraints import GreaterThan
    
    noise_prior = GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    MIN_INFERRED_NOISE_LEVEL = 1e-3
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=GreaterThan(
            MIN_INFERRED_NOISE_LEVEL,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )
    
    # define models for objective and constraint
    if covar=="matern":
        covar_x=MaternKernel(nu=nu)
    elif covar=="rbf":
        covar_x=gpytorch.kernels.RBFKernel()
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj, covar_module = ScaleKernel(
            base_kernel=covar_x))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
def optimize_acqf_and_get_observation(acq_func,train_x,obj_func,raw,bound,restarts=1):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    try:
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.tensor([[float(format(bound[0][0], '.1f'))] * train_x.shape[1], [float(format(bound[0][1]-1, '.1f'))] * train_x.shape[1]]),
            q=1,
            num_restarts=restarts,
            raw_samples=raw,
            options={},
        )
    except:
        restarts+=1
        optimize_acqf_and_get_observation(acq_func,train_x,obj_func,raw,bound,restarts)
    # observe new values 
    exact_x= torch.from_numpy(np.array([candidates.detach()[0][a] for a in range(train_x.shape[1])]))
    #Adding jitter to prevent problem during cholesky decomposition.
    n=3
    while torch.round(exact_x * 10**n) / (10**n) in torch.round(train_x * 10**n) / (10**n):
        exact_x = torch.round(exact_x * 10**n) / (10**n) - torch.randn_like(exact_x)
    new_x= exact_x
    new_x= new_x.view(1,len(new_x))
    less_than_min=[idx for idx,ind_bound in enumerate(bound) if new_x[0,idx]<ind_bound[0]]
    greater_than_max=[idx for idx,ind_bound in enumerate(bound) if new_x[0,idx]>ind_bound[1]]
    
    for incorrect_idx in less_than_min:
        new_x[0,incorrect_idx]=bound[incorrect_idx][0]+1
        
    for incorrect_idx in greater_than_max:
        new_x[0,incorrect_idx]=bound[incorrect_idx][1]-1
    
    new_obj = torch.tensor(obj_func(new_x),dtype=torch.float32).reshape(1).view(1,1)
    return new_x, new_obj


def rejectpoint(train_x_ei,train_obj_ei,result,result_hpdata,nt,new_iter,iteration,exceptioncounter,nu,covar):
    train_x_ei=train_x_ei[:nt+new_iter]
    train_obj_ei=train_obj_ei[:nt+new_iter]
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei, nu=nu, covar=covar)
    result.yall=result.yall[:new_iter]
    result_hpdata=result_hpdata[:new_iter]
    iteration=new_iter-1 #This moves back the iteration counter to the iteration that the error occured on.
    exceptioncounter.append(iteration)
    return train_x_ei,train_obj_ei,result,iteration,exceptioncounter,mll_ei,model_ei

################################################################################
# #BO with hyperparameter optimization.
# ###############################################################################

from botorch.optim.stopping import ExpMAStoppingCriterion as OptimizerConstraints
def BOloop(obj_func,max_iter,X,Y,bounds=[(0,100),(0,100)],verbose=False,tol=1,nu=2.5,covar="matern",minstate=False,optimal_y=100):
    result=results.result_opt()
    result_hpseries=results.hp_series()
    exceptioncounter=[]
    warningstor=[]
    result_hpdata=[]
    debug=True
    overall_warning_limit=False
    
    #Changes the kernel type to mattern.
    if nu>2.5:
        covar="rbf"
        
    # call helper functions to generate initial training data and initialize model
    if len(X)>10:
        nt=10
    else:
        nt=len(X)
    train_x_ei, train_obj_ei, result.yopt = generate_initial_data(X,Y,minstate,n=nt)
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei,nu=nu,covar=covar)
    #Initial fit
    fit_gpytorch_model(mll_ei)
    
    result.yall.append(result.yopt)
    error=100
    iteration=0
    print('processing')

    # run N_iters rounds of BayesOpt after the initial random batch
    while error>tol and iteration<max_iter:
        
        res_hp=results.hp_result()
        iteration+=1

        # for best_f, we use the best observed noisy values as an approximation
        EI = ExpectedImprovement(model=model_ei, best_f=result.yopt)

        # optimize and get new observation
        new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(EI,train_x_ei,obj_func,len(train_x_ei),bounds)

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
        
        # update progress
        result.yopt = train_obj_ei.max()
        result.yall.append(result.yopt)
        
        result.traindata=X

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        try:
            mll_ei, model_ei = initialize_model(
                train_x_ei, 
                train_obj_ei,
                nu=nu,
                covar=covar
            )
        except:
            print(train_x_ei,train_obj_ei)
            raise Exception('error')
        
        # fit the model
        try:
            with warnings.catch_warnings(record=True) as w:
                fit_gpytorch_model(mll_ei)
                warningstor.append(w)
                if len(w)>=1 and debug:
                    print([warnings.category for warnings in w])
                    raise Exception("Typical behavior before a crash from BOTorch")
                    
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Keyboard interrupt')
        except:
            if len(exceptioncounter)>0:
                recent_warnings=np.where([len(element)>=1 for element in warningstor])[0][-1]
                overall_warning_limit= len(recent_warnings)>=3 or (iteration-recent_warnings[-1])<1
            
            if iteration>10 and len(exceptioncounter)<1 and not overall_warning_limit:
                new_iter=iteration-5
                if debug:
                    print("First exception has occured, solver will quit at the 3rd.".format(len(exceptioncounter)))
                    print("Exception occured at iteration # {}".format(iteration))
                train_x_ei,train_obj_ei,result,iteration,exceptioncounter,mll_ei,model_ei=rejectpoint(train_x_ei,
                                                                                                      train_obj_ei,
                                                                                                      result,result_hpdata,nt,
                                                                                                      new_iter,
                                                                                                      iteration,
                                                                                                      exceptioncounter,
                                                                                                      nu,covar)
                if verbose:
                    print("Updated iteration # {}".format(iteration))
                    continue
            elif iteration>25 and iteration not in exceptioncounter and not overall_warning_limit:
                new_iter=iteration-10
                if debug:
                    print("Exception found at iteration # {}".format(iteration))
                train_x_ei,train_obj_ei,result,iteration,exceptioncounter,mll_ei,model_ei=rejectpoint(train_x_ei,
                                                                                                      train_obj_ei,
                                                                                                      result,result_hpdata,nt,new_iter
                                                                                                      ,iteration,
                                                                                                      exceptioncounter,
                                                                                                      nu,covar)
            elif iteration>50 and not overall_warning_limit:
                new_iter=np.where([len(element)>=1 for element in warningstor])[0][-1]-15
                train_x_ei,train_obj_ei,result,iteration,exceptioncounter,mll_ei,model_ei=rejectpoint(train_x_ei,
                                                                                                      train_obj_ei,
                                                                                                      result,result_hpdata,nt,
                                                                                                      new_iter,
                                                                                                      iteration,
                                                                                                      exceptioncounter,
                                                                                                      nu,covar)
            else:
                if debug:
                    print("The exception couldn't be handled or recurring error found at current_iteration : {}; Exceptionstore: {} or Exceptions were recorded at more than N iterations or iteration count was too low for the exception to be dealt with.".format(iteration,exceptioncounter))
                raise Exception('Recurring Error, result couldn''t be generated')
        
        error=optimal_y-result.yopt
        
        if verbose:
            print(
                f"\nIteration {iteration:>2}: best_value (EI) = "
                f"({result.yopt:>4.2f}), "
                f"error = {error:>2}", end=""
            )
        else:
            print('.',end="")
    
        res_hp.addhp(model_ei)
        result_hpdata.append(res_hp)
    result_hpseries.data=result_hpdata
    result_hpseries.updateres()
    result.hpdata=result_hpseries
    result.xopt=train_x_ei[np.where(train_obj_ei.numpy()==result.yopt.numpy())[0]]
    if minstate:
        result.yall.append(result.yopt)
        result.yall=-np.array([result.yall[a].numpy() for a in range(len(result.yall))])
    else:
        result.yall.append(result.yopt)
        result.yall=np.array([result.yall[a].numpy() for a in range(len(result.yall))])
        
    result.xall=np.array([train_x_ei[nt:][a].numpy() for a in range(len(train_x_ei[nt:]))])
    
    if iteration<max_iter:
        if minstate:
            result.yall=-np.concatenate([-result.yall,result.yall[-1]*np.ones(max_iter+2-len(result.yall))])
        else:
            result.yall=np.concatenate([result.yall,result.yall[-1]*np.ones(max_iter+2-len(result.yall))])
    result.nit=iteration
    result.success=True
    result._opt=optimal_y
    result.minstate=minstate
    if not verbose:
        clear_output()
    return result

################################################################################
# # Single-hyperparameter optimization initially to set the hyperparameters.
# ###############################################################################

def BOloop_singleopt(obj_func,max_iter,X,Y,bounds=[(0,100),(0,100)],verbose=False,tol=1,nu=2.5,covar="matern",minstate=False,optimal_y=100):
    result=results.result_opt()
    exceptioncounter=[]
    if nu>2.5:
        covar="rbf"
        
    # call helper functions to generate initial training data and initialize model
    if len(X)>10:
        nt=10
    else:
        nt=len(X)
    train_x_ei, train_obj_ei, result.yopt = generate_initial_data(X,Y,minstate,n=nt)
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei,nu=nu,covar=covar)
    passfilter=0
    while passfilter==0 and nt>2:
        try:
            fit_gpytorch_model(mll_ei)
            passfilter=1
        except KeyboardInterrupt:
            raise KeyboardInterrupt('Keyboard interrupt')   
        except:
            X[~(X==closest_N(X,2)[0]).all(axis=1),:]
            nt+=-1
            train_x_ei, train_obj_ei, result.yopt  = generate_initial_data(X,Y,n=nt)
            mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei,nu=nu,covar=covar)
            pass
    
    result.yall.append(result.yopt)
    result.traindata=X
    error=100
    iteration=0
    dotholder='.'

    # run N_iters rounds of BayesOpt after the initial random batch
    while error>tol and iteration<max_iter:
        iteration+=1

        # for best_f, we use the best observed noisy values as an approximation
        EI = ExpectedImprovement(model=model_ei, best_f=result.yopt)

        # optimize and get new observation
        new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(EI,train_x_ei,obj_func,len(train_x_ei),bounds)

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])

        # update progress
        
        result.yopt = train_obj_ei.max()
        result.yall.append(result.yopt)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei, 
            train_obj_ei,
            nu=nu,
            covar=covar
        )
        error=optimal_y-result.yopt
        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (EI) = "
                f"({result.yopt:>4.2f}), "
                f"error = {error:>2}", end=""
            )
        else:
            clear_output()
            print('processing',dotholder)
            dotholder+='.'
            
    result.xopt=train_x_ei[np.where(train_obj_ei.numpy()==result.yopt.numpy())[0]]
    result.xall=np.array([train_x_ei[nt:][a].numpy() for a in range(len(train_x_ei[nt:]))])
    if minstate:
        result.yall.append(-result.yopt)
        result.yall=-np.array([result.yall[a].numpy() for a in range(len(result.yall))])
    else:
        result.yall.append(result.yopt)
        result.yall=np.array([result.yall[a].numpy() for a in range(len(result.yall))])
    
    if iteration<max_iter:
        if minstate:
            result.yall=-np.concatenate([-result.yall,result.yall[-1]*np.ones(max_iter+2-len(result.yall))])
        else:
            result.yall=np.concatenate([result.yall,result.yall[-1]*np.ones(max_iter+2-len(result.yall))])
    result.nit=iteration
    result.success=True
    result._opt=optimal_y
    result.minstate=minstate
    return result

################################################################################
# # No hyperparameter optimization with this loop
# ###############################################################################

def BOloop_fixparam(obj_func,max_iter,X,Y,fixparam,bounds=[(0,100),(0,100)],verbose=False,tol=1,nu=2.5,covar="matern",minstate=False, optimal_y=100):
    result=results.result_opt()
    exceptioncounter=[]
    if nu>2.5:
        covar="rbf"
        
    # call helper functions to generate initial training data and initialize model
    if len(X)>10:
        nt=10
    else:
        nt=len(X)
    train_x_ei, train_obj_ei, result.yopt = generate_initial_data(X,Y,minstate,n=nt)
    mll_ei, model_ei = initialize_model(train_x_ei, train_obj_ei,nu=nu,covar=covar)
    with torch.no_grad():
        for param_name, param in model_ei.named_parameters():
            param.copy_(torch.tensor(fixparam[param_name][0]))
        
    result.yall.append(result.yopt)
    result.traindata=X
    error=100
    iteration=0
    dotholder='.'

    # run N_iters rounds of BayesOpt after the initial random batch
    while error>tol and iteration<max_iter:
        iteration+=1

        # for best_f, we use the best observed noisy values as an approximation
        EI = ExpectedImprovement(model=model_ei, best_f=result.yopt)

        # optimize and get new observation
        new_x_ei, new_obj_ei = optimize_acqf_and_get_observation(EI,train_x_ei,obj_func,len(train_x_ei),bounds)
        
        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
        
        # update progress
        
        result.yopt = train_obj_ei.max()
        result.yall.append(result.yopt)
        
        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei, 
            train_obj_ei,
            nu=nu,
            covar=covar
        )
        
        error=optimal_y-result.yopt
        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (EI) = "
                f"({result.yopt:>4.2f}), "
                f"error = {error:>2}", end=""
            )
        else:
            clear_output()
            print('processing',dotholder)
            dotholder+='.'
            
    result.xopt=train_x_ei[np.where(train_obj_ei.numpy()==result.yopt.numpy())[0]]
    result.xall=np.array([train_x_ei[nt:][a].numpy() for a in range(len(train_x_ei[nt:]))])
    if minstate:
        result.yall.append(result.yopt)
        result.yall=-np.array([result.yall[a].numpy() for a in range(len(result.yall))])
    else:
        result.yall.append(result.yopt)
        result.yall=np.array([result.yall[a].numpy() for a in range(len(result.yall))])
    
    if iteration<max_iter:
        if minstate:
            result.yall=np.concatenate([result.yall,result.yall[-1]*np.ones(max_iter+2-len(result.yall))])
        else:
            result.yall=np.concatenate([result.yall,result.yall[-1]*np.ones(max_iter+2-len(result.yall))])
    result.nit=iteration
    result.success=True
    result._opt=optimal_y
    result.minstate=minstate
    return result
