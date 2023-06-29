import sys
# sys.path.insert(1, '../')

import numpy as np
from BO import BOloop,BOloop_fixparam
import results

class optimizer:
    def __init__(self):
        """
        Used to define an instance that can be then used as an optimizer with the optimize method.
        max_iter= Maximum iterations for the optimizer.
        opt= The type of optimizer
        tol= Tolerated percent error from the Maxima
        bounds=bounds of the input function
        """
        self.max_iter=100
        self.obj_func=None
        self.train=None
        self.opt="BO"
        self.options={'tol':0.1, 'nu':2.5,'verbose':False}
        self.optima=100
        self.minimize=False
        self.paramset=None
        self.datagenmodule=None
        self.funcseed=None
    
    def optimize(self,f,method=None):
        """
        
        """
        if method is None:
            method=self.opt
        try:
            options = {"BO" : self.BO,"BO-fixparam": self.BO_fixparam, "CG": self.CG, "NM": self.NM, "SA":self.SA}
        except KeyError:
            raise Exception('Method ',method,' is not available at this point in time, the current options are:',list(options.keys()))    
        
        return options[method](f)
    
    def BO_fixparam(self,f):
        """
        
        """
        self.nu=2.5
        self.covar="matern"
        X=self.train
        if self.minimize:
            f_c=lambda X: -f(X)
        else:
            f_c=f
        Y=f_c(X).astype(np.float32)
        return BOloop_fixparam(f_c,self.max_iter,X,Y,self.paramset,bounds=self.bounds,tol=self.options['tol'],nu=self.options['nu'],minstate=self.minimize,optimal_y=self.optima)
        
    def BO(self,f):
        """
        
        """
        self.nu=2.5
        self.covar="matern"
        X=self.train
        if self.minimize:
            f_c=lambda X: -f(X)
        else:
            f_c=f
        Y=f_c(X).astype(np.float32)
        return BOloop(f_c,self.max_iter,X,Y,bounds=self.bounds,tol=self.options['tol'],verbose=self.options['verbose'],minstate=self.minimize,optimal_y=self.optima)
    
    def CG(self,f):
        """
        
        """
        from scipy.optimize import minimize
        
        if len(self.train)>1:
            training_samples=self.train[np.argmax([f(X_train) for X_train in self.train])]
        else:
            training_samples=self.train
            

        f_c = f_corrected(f,self.minimize)
        
        res= minimize(f_c, training_samples , method='CG',options={'maxiter':self.max_iter})
        
        result=results.result_opt()
        result.traindata=self.train
        result.success=res.message
        result.nit=res.nfev
        result.xopt=res.x
        if not self.minimize:
            result.yopt=-res.fun
        result.minstate=self.minimize
        result.xall=np.array(f_c.xall)
        result.yall=np.array(f_c.yall).flatten()
        if result.nit<=self.max_iter:
            result.yall=np.concatenate([result.yall,result.yall[-1]*np.ones(self.max_iter+2-len(result.yall))])
        elif result.nit>self.max_iter:
            result.yall=result.yall[:self.max_iter+2]
            
        
        return result
    
    def NM(self,f):
        """
        
        """
        from scipy.optimize import minimize
        
        if len(self.train)>1:
            training_samples=self.train[np.argmax([f(X_train) for X_train in self.train])]
        else:
            training_samples=self.train
        
        f_c = f_corrected(f,self.minimize)
        res= minimize(f_c, training_samples, method='Nelder-Mead', options={'maxiter':self.max_iter})
        
        result=results.result_opt()
        
        result.traindata=self.train
        result.success=res.message
        result.nit=res.nfev
        result.xopt=res.x
        if not self.minimize:
            result.yopt=-res.fun
        result.minstate=self.minimize
        result.xall=np.array(f_c.xall)
        result.yall=np.array(f_c.yall).flatten()
        if result.nit<=self.max_iter:
            result.yall=np.concatenate([result.yall,result.yall[-1]*np.ones(self.max_iter+2-len(result.yall))])
        elif result.nit>self.max_iter:
            result.yall=result.yall[:self.max_iter+2]
        
        return result
        
    def SA(self,f):
        """
        
        """
        from scipy.optimize import dual_annealing
        
        if len(self.train)>1:
            training_samples=self.train[np.argmax([f(X_train) for X_train in self.train])]
        else:
            training_samples=self.train

        f_c = f_corrected(f,self.minimize)
            
        res= dual_annealing(f_c, x0=training_samples,bounds=self.bounds,maxfun=self.max_iter)
        
        result=results.result_opt()
        
        result.traindata=self.train
        result.success=res.message
        result.nit=res.nfev
        result.xopt=res.x
        if not self.minimize:
            result.yopt=-res.fun
        result.minstate=self.minimize
        result.xall=np.array(f_c.xall)
        result.yall=np.array(f_c.yall).flatten()
        if result.nit<=self.max_iter:
            result.yall=np.concatenate([result.yall,result.yall[-1]*np.ones(self.max_iter+2-len(result.yall))])
        elif result.nit>self.max_iter:
            result.yall=result.yall[:self.max_iter+2]
        
        return result


class f_corrected():
    def __init__(self,func,minimize):
        self.xall=[]
        self.yall=[]
        self.fun=func
        self.minimize=minimize

    def calltracker(self,args):
        self.xall.append(args)
        if len(self.yall)>1 and max(self.yall)<self.fun(args):
            append_y=self.fun(args)
        elif len(self.yall)<1:
            append_y=self.fun(args)
        else:
            append_y=self.yall[-1]
        self.yall.append(append_y)

    def __call__(self,args):
        self.calltracker(args)
        if not self.minimize:
            return -self.fun(args)
        else:
            return self.fun(args)