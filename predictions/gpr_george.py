#!/usr/bin/env python3
from gpr_base import GPR_base
import numpy as np
from pathlib import Path
import joblib
import os
# import plot
import lal
import george
from george import kernels
import plot
import json
import scipy.optimize as opt
from scipy.optimize import minimize
import climin
os.sys.path.append("../data/waveforms/")
import compare_waveforms_base as compare_wf
import matplotlib.pyplot as plt

class GPR_george(GPR_base):
    def __init__(self, savedir):
        super().__init__()
        
        self.alpha = 0.0001
        self.floor = 0.01
        self.n_restarts_optimizer = 500
        self.variable_name = "phase"
        
        self.N_par = 4 # m1, m2, s1z, s2z, lambda1, lambda2
        self.path_to_train_files = "../data/LAL_data/LAL_data_delta"+self.variable_name+"_train/"
        self.path_to_test_files = "../data/LAL_data/LAL_data_delta"+self.variable_name+"_test/"

        self.savedir = savedir

        self.nodes = False
        self.freq_train = np.linspace(0.0004, 0.015, 10)
        self.freq_test = np.linspace(0.0004, 0.015, 40)

        self.gpr_save = []

        self.load_train_test_from_files()

        self.y_train = np.ravel(self.y_train)

        #self.scaler = StandardScaler()
        #self.x_train = self.scaler.fit_transform(self.x_train)
        #self.x_test = self.scaler.transform(self.x_test)
    
    def train(self, node_number=None, verbose=False):
        """
        Find the hyperparameter values that obtain Lmarg max,
        with Lmarg defined by the training points
        """

        if self.nodes==False:

            k0 = kernels.ConstantKernel(log_constant=np.log(1), ndim=4)
            k1 = kernels.Matern32Kernel(metric=[1,1,1,1], ndim=4)
            kernel = kernels.ConstantKernel(log_constant=np.log(1), ndim=5) \
                * kernels.ExpSquaredKernel([0.005,0.005],ndim=5,axes=[0,1]) \
                * kernels.ExpSquaredKernel([0.005, 0.005], ndim=5, axes=[2,3]) \
                * kernels.ExpSquaredKernel(100,ndim=5,axes=4)
            # k2 = kernels.ExpSquaredKernel(metric=1, ndim=4, axes=[3])
            # k2 = kernels.DotProductKernel(ndim=4, axes=[3])
            # k2 = kernels.ExpSine2Kernel(gamma=0.1, log_period=5.0, ndim=4, axes=3)
            # k2 = kernels.RationalQuadraticKernel(log_alpha=np.log(1), metric=1.2**2, ndim=4, axes=[3])
            # kernel = k0*k1

            print(kernel.get_parameter_names())
            print(np.exp(kernel.get_parameter_vector()))
            gpr_model = george.GP(kernel, solver=george.HODLRSolver, tol=1e-6, min_size=100, mean=0.0, seed=42)
            gpr_model.white_noise.set_parameter_vector(0.1)
            yerr = np.ones(len(self.x_train)) * 0
            gpr_model.compute(self.x_train)
            print(gpr_model.log_likelihood(self.y_train))

            def neg_ln_like(p):
                gpr_model.set_parameter_vector(p)
                return -gpr_model.log_likelihood(self.y_train)

            def grad_neg_ln_like(p):
                gpr_model.set_parameter_vector(p)
                return -gpr_model.grad_log_likelihood(self.y_train)

            opt = climin.Adam(gpr_model.get_parameter_vector(), grad_neg_ln_like)

            for info in opt:
                if info['n_iter']%10 == 0:
                    k = gpr_model.get_parameter_vector()
                    print("{} - {} - {}- {}".format(info['n_iter'], neg_ln_like(k), grad_neg_ln_like(k), np.exp(k)))
                    if info['n_iter'] > 100000:
                        break
            results = gpr_model.get_parameter_vector()
            
            # minimizer_kwargs = dict(method="L-BFGS-B", jac="True")
            # result = opt.basinhopping(neg_ln_like, gpr_model.get_parameter_vector(), niter=100, minimizer_kwargs=minimizer_kwargs)
            # result = minimize(neg_ln_like, gpr_model.get_parameter_vector(), jac=grad_neg_ln_like)
            gpr_model.set_parameter_vector(results)
            print(kernel.get_parameter_names())
            print(np.exp(kernel.get_parameter_vector()))
            print(gpr_model.log_likelihood(self.y_train))

            Path(self.savedir).mkdir(parents=True, exist_ok=True)
            joblib.dump(gpr_model, self.savedir+"onlyf.pkl")
            if verbose == True:
                gpr_verbose(gpr_model)
        
    def test(self, x_test=None, verbose=False):
        if self.nodes==False:
            gpr_model = joblib.load(self.savedir+"onlyf.pkl")
            if x_test is None:
                self.y_predict, self.y_predict_var = gpr_model.predict(self.y_train, self.x_test, return_var=True)
                self.y_predict_std = np.sqrt(self.y_predict_var)
            else:
                self.y_predict, self.y_predict_var = gpr_model.predict(self.y_train, x_test, return_var=True)
                self.y_predict_std = np.sqrt(self.y_predict_var)

def gpr_verbose(gpr_model):
    """
    Verbose mode with info on initial and train hyperparameters
    """

    print("Initial")
    params = gpr_model.kernel.get_params()
    for key in sorted(params):
        print("%s : %s" % (key, params[key]))
    print("Trained") 
    params = gpr_model.kernel_.get_params()
    for key in sorted(params):
        print("%s : %s" % (key, params[key]))
    print("") 

def gpr_one_out(train=True):
    """
    Train and test gpr models in which one hybrid waveform is left out
    """

    full_dict = json.load(open("../data/NR_data/BAM_SACRA_data/NRinfo.json"))
    for key1, value1 in full_dict.items():
        if train == True:
            test_dict = {}
            train_dict = {}
            for key2, value2 in full_dict.items():
                if key1 == key2:
                    test_dict[key2[:-3]] = value2 
                else:
                    train_dict[key2[:-3]] = value2
        
        savedir = "saved_nodes/"+key1+"/"
        #savedir = "saved_nodes_Mf_lt/"+key1+"/"
            
        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        if train == True:
            json.dump(test_dict, open(savedir+"dict_test.json", 'w'), sort_keys=True, indent=4)
            json.dump(train_dict, open(savedir+"dict_train.json", 'w'), sort_keys=True, indent=4)
            
        gpr = GPR_sklearn(savedir)
        if train == True:
            gpr.train()
        else:
            gpr.test()
            plot.plot_frequency_evolution(gpr)

if __name__ == "__main__":

    """ Test on the base set """
    # dict_train = json.load(open("../data/NR_data/BAM_SACRA_data/NRinfo_nospin.json"))
    dict_train = json.load(open("../data/LAL_data/fd_v4TSurr_IMR_train/train_points.json"))
    dict_test = json.load(open("../data/LAL_data/fd_v4TSurr_IMR_test/test_points.json"))

    savedir = "LAL_data/v4T_IMR_phase_george/"
    par_train = {}
    par_test = {}

    for key, value in dict_train.items():
        par_train[key] = value
    for key, value in dict_test.items():
        par_test[key] = value
    
    json.dump(par_train, open(savedir+"dict_train.json", 'w'), sort_keys=True, indent=4)
    json.dump(par_test, open(savedir+"dict_test.json", 'w'), sort_keys=True, indent=4)
    
    gpr = GPR_george(savedir)
    gpr.train()
    gpr.test()
    plot.plot_frequency_evolution(gpr)
    
    # gpr.add_training_points()
    #gpr_one_out(train=False)
    #plot.plot_distance()
    #gpr.add_training_points() 
    #plot.plot_uncertainty(gpr)
    #plot.plot_uncertainty(gpr)
