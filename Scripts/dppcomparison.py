import sys
sys.path.insert(1, '../src/optimizers')
sys.path.insert(1, '../src')

import data_gen
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from IPython.display import clear_output
from utils import bootstrap
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def create_synth_data(seed=None,bounds=[(0,100),(0,100)]):
    #Set-up the diversity training data generator
    synth_data=data_gen.diverse_data_generator()
    if seed is None:
        seed=np.random.randint(100)
    synth_data.options['bounds']=bounds
    synth_data.gamma=1e-6
    synth_data.options['N_samples']=10000
    synth_data.options['seed']=seed*2
    return synth_data

def gammagen(maxtrainsize):
    gammavals={}
    timegammacalc={}
    synth_data=create_synth_data()
    for training_size in range(5,maxtrainsize,5):
        synth_data.training_size=training_size
        start=time.time()
        gammavals[training_size]=synth_data.gammacalc(min_samples=200)
        timegammacalc[training_size]=time.time()-start
        clear_output()
        print('Progress: Gamma Value Calc. Completed {} of {}'.format(training_size,maxtrainsize))
    return gammavals,timegammacalc

def distgentime(training_mat,N_size_mat):
    #Does time comparisons on distribution generation of k-dpp, H-DPP,H-DPP parallel
    synth_data=create_synth_data()
    datagencomplexity=dict()
    datagencomplexity['OurApproach-parallel']=dict()
    datagencomplexity['OurApproach-not-parallel']=dict()
    
    
    filename='pkldata/dppcomparison/gammavals5-'+str(args.maxtrainsize)+'.pkl'
    with open(filename, 'rb') as f:
        gammavals,timegammacalc = pickle.load(f)

    for N_samples in np.arange(*N_size_mat):
        print("N_samples = {} started".format(N_samples))
        
        for training_size in np.arange(*training_mat):
            print('Training for size= {}'.format(training_size))
            synth_data.training_size=training_size
            try:
                synth_data.gamma=10**np.median(np.arange(*np.sort(gammavals[training_size])))
            except:
                pass
            synth_data.options['parallel-process']=True
            start=time.time()
            synth_data.generate()
            try:
                datagencomplexity['OurApproach-parallel'][N_samples].append(time.time()-start)
            except KeyError:
                datagencomplexity['OurApproach-parallel'][N_samples]=list()
                datagencomplexity['OurApproach-parallel'][N_samples].append(time.time()-start)
            
            synth_data.options['parallel-process']=False
            start=time.time()
            synth_data.generate()
            try:
                datagencomplexity['OurApproach-not-parallel'][N_samples].append(time.time()-start)
            except KeyError:
                datagencomplexity['OurApproach-not-parallel'][N_samples]=list()
                datagencomplexity['OurApproach-not-parallel'][N_samples].append(time.time()-start)
        print("N_samples = {} completed".format(N_samples))
        clear_output()
    return datagencomplexity

#Does time comparisons on sampling from HDPP and coreDPP
def samplingtime(trials,maxtrainsize):
    synth_data=create_synth_data()
    timestor=dict()
    timestor['KDPP']=dict()
    timestor['OurApproach']=dict()
    synth_data.generate(method='k-dpp-eig')
    filename='pkldata/dppcomparison/gammavals5-'+str(args.maxtrainsize)+'.pkl'
    with open(filename, 'rb') as f:
        gammavals,timegammacalc = pickle.load(f)
    for training_size in range(5,maxtrainsize,5):
        print("Generating data for training size={}".format(training_size))
        synth_data.training_size=training_size
        synth_data.gamma=10**np.median(np.arange(*np.sort(gammavals[training_size])))
        print(synth_data.gamma,synth_data.training_size)
        synth_data.generate()
        for trial in range(trials):
            start=time.time()
            synth_data.k_dpp_sample_eig(training_size)
            try:
                timestor['KDPP'][training_size].append(time.time()-start)
            except KeyError:
                timestor['KDPP'][training_size]=list()
                timestor['KDPP'][training_size].append(time.time()-start)
        for _ in range(trials):
            start=time.time()
            synth_data.sub_dpp_sample(5)
            try:
                timestor['OurApproach'][training_size].append(time.time()-start)
            except KeyError:
                timestor['OurApproach'][training_size]=list()
                timestor['OurApproach'][training_size].append(time.time()-start)
        print('Progress : Sampling Time recorded {} of {}'.format(training_size,maxtrainsize))
    filename='pkldata/sampling-time-comparison.pkl'
    with open(os.path.abspath(filename), 'wb') as f:
        pickle.dump(timestor, f)
    return timestor


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25,
                        help="Number of Trials")
    
    parser.add_argument("--maxtrainsize", type=int, default=100,
                        help="Max training size for samplegen and ")
    parser.add_argument("--training_mat", default="5, 100, 5" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for training set size with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    parser.add_argument("--N_size_mat", default="10000, 50001, 10000" , type=lambda s: tuple(float(item) for item in s.split(',')),
                       help="Define the range generator for N_samples with a delimited string seperated by commas of the form :- ''start,stop,step'' ")
    
    parser.add_argument("--intent", choices={"gammacalc", "sampletime","distgen","all"}, default="all")
    parser.add_argument("--reason", choices={"data-gen","plot-data","both"}, default="data-gen")
    
    args = parser.parse_args()
    
    if args.reason=="data-gen" or args.reason=="both":
        if args.intent=="all" or args.intent=="gammacalc":
            gammavals,timegammacalc=gammagen(args.maxtrainsize)
            filename='pkldata/dppcomparison/gammavals5-'+str(args.maxtrainsize)+'.pkl'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump([gammavals,timegammacalc], f)
                
        if args.intent=="all" or args.intent=="sampletime":
            timestor=samplingtime(args.trials,args.maxtrainsize)
            filename='pkldata/dppcomparison/sampling-time-comparison'+str(args.maxtrainsize)+'.pkl'
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            with open(os.path.abspath(filename), 'wb') as f:
                pickle.dump(timestor, f)

        if args.intent=="all" or args.intent=="distgen":
            datagencomplexity=distgentime(args.training_mat,args.N_size_mat)
            filename='pkldata/dppcomparison/datagen-speed-comparison'+str(args.maxtrainsize)+'.pkl'
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(datagencomplexity, f)

    if args.reason=="plot-data" or args.reason=="both":
        #Load the files
        
        filename='pkldata/dppcomparison/sampling-time-comparison'+str(args.maxtrainsize)+'.pkl'
        with open(filename, 'rb') as f:
            timestor = pickle.load(f)
            
        filename='pkldata/dppcomparison/datagen-speed-comparison'+str(args.maxtrainsize)+'.pkl'
        with open(filename, 'rb') as f:
            datagencomplexity = pickle.load(f)
            
        filename='pkldata/dppcomparison/gammavals5-'+str(args.maxtrainsize)+'.pkl'
        with open(filename, 'rb') as f:
            gammavals,timegammacalc = pickle.load(f)
        
        #Define tbe neccesary x variables
        training_mat=np.arange(*args.training_mat)
        N_size_mat=np.arange(*args.N_size_mat)
        training_set=range(5,args.maxtrainsize,50)
        
        if args.intent=="gammacalc" or args.intent=="all":
            #plot 1
            with plt.style.context(['science','no-latex']):
                fig=plt.figure()
                plt.plot(training_set,[np.median(np.arange(*np.sort(gammavals[training_size]))) for training_size in training_set])
                plt.xlabel('Training size')
                plt.ylabel('Median gamma value for training size ''x''')
                plt.ylabel('Median sigma for training size ''x''')
                savename='../results/dppcomparison/gammavals-5-' + str(args.maxtrainsize) + '.png'
                os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
                plt.savefig(os.path.abspath(savename))
                plt.close()
            with plt.style.context(['science','no-latex']):
                fig=plt.plot()
                plt.plot(training_set,timegammacalc)
                plt.ylabel('Time taken to generate gamma range for training size ''x''')
                plt.xlabel('Training size')
                savename='../results/dppcomparison/gammatimegen-5-' + str(args.maxtrainsize) + '.png'
                os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
                plt.savefig(os.path.abspath(savename))
                plt.close()
        
         
        
        #plot2 distgen
        
        if args.intent=="distgen" or args.intent=="all":
            with plt.style.context(['science','no-latex']):
                fig=plt.figure(figsize=(20,20))
                
                plt.plot(training_mat,302.3*np.ones(len(training_mat)) , label='vanilla k-dpp') 
                for i,key in enumerate(datagencomplexity.keys()):
                    for N_samples in N_size_mat:
                        if N_samples==50000:
                            if key == 'OurApproach-parallel':
                                # also add what the colors mean on a seperate scale
                                label='our approach with parallel computing'
                            else:
                                label='our approach without parallel computing'
                        else:
                            label=''
                        y_data=np.array(datagencomplexity[key][N_samples])
                        plt.plot(training_mat,y_data, label=label,alpha=N_samples/max(N_size_mat))
                    plt.ylabel('Time taken to generate a distribution of training size x',fontsize=22)
                    plt.xlabel('training size',fontsize=22)
                    plt.legend(loc='center left',fontsize=22)
                savename='../results/dppcomparison/datagencomparison.eps'
                os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
                plt.savefig(os.path.abspath(savename),dpi=300)
                plt.close()
            
            
        #plot3 sampletime
        if args.intent=="sampletime" or args.intent=="all":
            with plt.style.context(['science','no-latex']):
                fig=plt.figure(figsize=(15,15))
                for key in timestor.keys():
                    y_data=np.array([bootstrap(timestor[key][training_size]) for training_size in training_set])
                    y_mean=y_data[:,0]
                    y_5=y_data[:,1]
                    y_95=y_data[:,2]
                    if key == 'KDPP':
                        plt.plot(training_set,y_mean,label='vanilla k-dpp')
                    else:
                        plt.plot(training_set,y_mean,label='our tool')
#                     plt.fill_between(training_set,y_5, y_95, alpha=0.3)
                    plt.ylabel('Time taken to sample from the generated distribution',fontsize=22)
                    plt.xlabel('training size',fontsize=22)
                plt.legend(fontsize=22)
                savename='../results/dppcomparison/sampletimecomparison-5-' + str(args.maxtrainsize) + '.eps'
                os.makedirs(os.path.dirname(os.path.abspath(savename)), exist_ok=True)
                plt.savefig(os.path.abspath(savename),dpi=300)
                plt.close()