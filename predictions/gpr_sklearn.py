#!/usr/bin/env python3
from gpr_base import GPR_base
import numpy as np
from pathlib import Path
import joblib
import os
import plot
import lal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Product, Matern, DotProduct, ConstantKernel as C
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plot
import json
import scipy.optimize as opt
os.sys.path.append("../data/waveforms/")
import compare_waveforms_base as compare_wf

import matplotlib.pyplot as plt

class GPR_sklearn(GPR_base):
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
        # self.freq_train = np.logspace(np.log10(0.0003), np.log10(0.07), 20)
        # self.freq_train = np.logspace(-1.5, 0.3, 20)
        # self.freq_train = np.logspace(np.log10(20), np.log10(4096), 40)
        self.freq_train = np.linspace(0.0004, 0.015, 10)
        self.freq_test = np.linspace(0.0004, 0.015, 40)

        self.gpr_save = []

        self.load_train_test_from_files()
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        
    def train(self, node_number=None, verbose=True):
        """
        Find the hyperparameter values that obtain Lmarg max,
        with Lmarg defined by the training points
        """
        if self.nodes == True:
            for i, y_train in enumerate(self.y_train):
                if node_number!=None and node_number!=i:
                    continue
                else:
                    if os.path.exists(self.savedir+str(i)+".pkl"):
                        print("Remove model file")
                        os.remove(self.savedir+str(i)+".pkl")
                    
                    print("Node number: ", i)
                    kernel = C(1.0, (1e-3, 1e5))*RBF(np.array([1,1,1,1,1,1]), [(1e-3, 10), (1e-3, 10), (1e-3, 10), (1e-3, 10), (1e-3, 1e4), (1e-3, 1e4)])
                    # kernel = C(1.0, (1e-3, 1e5))*Matern(np.array([1,1,1,1,1,1]), (1e-3, 5e3))
                    gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer) 
                    
                    gpr_model.fit(self.x_train, y_train)
                    #self.gpr_save.append(gpr_model) 
                    Path(self.savedir).mkdir(parents=True, exist_ok=True)
                    joblib.dump(gpr_model, self.savedir+str(i)+".pkl")
                    
                    if verbose == True:
                        gpr_verbose(gpr_model) 
        
        elif self.nodes==False:
            # kernel = C(1.0, (1e-3, 1e1))*RBF(np.array([1,1,1,1]), [(1e-3, 1e1), (1e-3, 1e1), (1e-3, 1e1), (1e-3, 1e1)])
            # kernel = C(1.0, (1e-3, 1e1))*RBF(np.array([1,1,1,1]), [(1e-3, 1e1), (1e-3, 1e1), (1e-3, 1e1), (1e-3, 1e1)])
            kernel = C(1.0, (1e-3, 1e1))*RBF(np.array([1,1,1,1]), [(1e-3, 1e1), (1e-3, 1e1), (1e-3, 1e1), (1e-3, 1e1)])
            gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha, n_restarts_optimizer=self.n_restarts_optimizer) 
            gpr_model.fit(self.x_train, self.y_train)
            Path(self.savedir).mkdir(parents=True, exist_ok=True)
            joblib.dump(gpr_model, self.savedir+"onlyf.pkl")
            if verbose == True:
                gpr_verbose(gpr_model)
        
    def test(self, x_test=None, verbose=False):
        if self.nodes == True:
            if x_test is None: 
                for i, f in enumerate(self.freq_train):
                    gpr_model = joblib.load(self.savedir+str(i)+".pkl")
                    #gpr_model = self.gpr_save[i] 
                    y_predict, y_predict_std = gpr_model.predict(self.x_test, return_std=True)
                    self.y_predict = np.vstack([self.y_predict, y_predict]) if self.y_predict.size else y_predict 
                    self.y_predict_std = np.vstack([self.y_predict_std, y_predict_std]) if self.y_predict_std.size else y_predict_std 
           
                    if verbose == True:
                        gpr_verbose(gpr_model) 
            else:
                y_predict_otf = np.array([])
                y_predict_std_otf = np.array([])
                for i, f in enumerate(self.freq_train):
                    gpr_model = joblib.load(self.savedir+str(i)+".pkl")
                    y_predict, y_predict_std = gpr_model.predict(x_test, return_std=True)
                    #self.y_predict_otf.append(y_predict)
                    #self.y_predict_std_otf.append(y_predict_std)
                    y_predict_otf = np.vstack([y_predict_otf, y_predict]) if y_predict_otf.size else y_predict 
                    y_predict_std_otf = np.vstack([y_predict_std_otf, y_predict_std]) if y_predict_std_otf.size else y_predict_std
                return y_predict_otf, y_predict_std_otf
        
        else:
            gpr_model = joblib.load(self.savedir+"onlyf.pkl")
            if x_test is None:
                self.y_predict, self.y_predict_std = gpr_model.predict(self.x_test, return_std=True)
            else:
                self.y_predict, self.y_predict_std = gpr_model.predict(x_test, return_std=True)

    def add_training_points(self):
        """
        Add N_new training points, one at a time, in the places where they are
        more necessary
        """
        def err_func(new_point):
            print(new_point)
            m1, m2, s1z, s2z, lambda1, lambda2 = new_point
            par = np.array([[m1, m2, s1z, s2z, lambda1, lambda2]])
            pardict = {
                    'm1':m1, 
                    'm2':m2,
                    's1z':s1z,
                    's2z':s2z,
                    'lambda1':lambda1,
                    'lambda2':lambda2}
            x_test, y_test = self.load_new_point_otf(pardict)
            y_predict, y_predict_std = self.test(x_test=par)
            #rms_err = 1/len(gpr.freq_train)*np.sqrt(np.sum(np.array(y_predict_std)**2))
            rms_err = 1/len(gpr.freq_train)*np.sqrt(np.sum(np.array(y_predict-y_test)**2))
            return rms_err
        
        N_new = 1
        par_dict = {}
        bounds = []
        bounds.append((0.8, 2.4))
        bounds.append((0.8, 2.4))
        bounds.append((-0.5,0.5))
        bounds.append((-0.5,0.5))
        bounds.append((0, 3000))
        bounds.append((0, 3000))

        for n in range(N_new):
            rms_err_max=-np.inf
            print(n)
            minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
            worst_point = opt.basinhopping(err_func, [1.4, 1.4, 0, 0, 200, 200], niter=1, minimizer_kwargs=minimizer_kwargs)
            #worst_point = opt.minimize(err_func, [1.4, 1.4, 0, 0, 200, 200], bounds=bounds)
            m1_new, m2_new, s1z_new, s2z_new, lambda1_new, lambda2_new = worst_point.x

            print("Worst new point: m1={}, m2={}, s1z={}, s2z={}, lambda1={}, lambda2={}".format(m1_new, m2_new, s1z_new, s2z_new, lambda1_new, lambda2_new))
            print("Adding new point...")
            par_dict[n] = {
                    'm1': m1_new,
                    'm2': m2_new,
                    's1z': s1z_new,
                    's2z': s2z_new,
                    'lambda1': lambda1_new,
                    'lambda2': lambda2_new
            }
            x_new, y_new = self.load_new_point_otf(par_dict[n]) 
            x_new = x_new.reshape(1, -1)
            y_new = y_new.reshape(-1, 1) # a row value for each freq node
            
            self.x_train = np.vstack([self.x_train, x_new]) # add row
            self.y_train = np.hstack([self.y_train, y_new]) # add column

            self.savedir = self.savedir+"_"+str(n+1)+"_added_points/"
            Path(self.savedir).mkdir(parents=True, exist_ok=True)
            self.train(verbose=False)

        json.dump(par_dict, open("new_par.json", 'w'), sort_keys=True, indent=4)

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

    savedir = "LAL_data/v4T_IMR_phase/"
    par_train = {}
    par_test = {}

    for key, value in dict_train.items():
        par_train[key] = value
    for key, value in dict_test.items():
        par_test[key] = value
    
    json.dump(par_train, open(savedir+"dict_train.json", 'w'), sort_keys=True, indent=4)
    json.dump(par_test, open(savedir+"dict_test.json", 'w'), sort_keys=True, indent=4)
    
    gpr = GPR_sklearn(savedir)
    gpr.train()
    gpr.test()
    plot.plot_frequency_evolution(gpr)

    # gpr.add_training_points()
    #gpr_one_out(train=False)
    #plot.plot_distance()
    
    #gpr.add_training_points() 
    #plot.plot_uncertainty(gpr)
    #plot.plot_uncertainty(gpr)