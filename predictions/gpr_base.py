import glob
import numpy as np
import json
import os
from os.path import basename
from pathlib import Path
import scipy as scp
import copy
os.sys.path.append("../data/waveforms/")
import compare_waveforms_base as compare_wf
import lal

from waveform_base import waveform_base
from waveform_lal import waveform_lal
from waveform_nr import waveform_nr
from waveform_hybrid import waveform_hybrid
import waveform_lal as wf_lal_lib

class GPR_base(object):
    def __init__(self):
        self.x_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self.y_predict = np.array([])
        self.y_predict_std = np.array([])
        
        self.par_train = []
        self.par_test = []

    def hyperparameters_initialize(self, x_train, y_train):
        # Hyperparameter initial values
        hyp_0 = [] 
        hyp_lim = []
        
        dimension = len(x_train[0])
        
        theta_0 = np.average(np.abs(np.sort(y_train)))
        theta_lim = (1e-3, 1e6)  
        
        hyp_0.append(theta_0**2)
        hyp_lim.append(theta_lim)
         
        for i in range(dimension):
            par_i = [item[i] for item in x_train]
            l0 = np.average(np.abs(np.diff(np.sort(par_i))))
            l_lim = (1e-3, 5e3)
            hyp_0.append(l0)
            hyp_lim.append(l_lim)
        
        return hyp_0, hyp_lim

    def load_train_test_from_files(self):
        train_dict = json.load(open(self.savedir+"dict_train.json","r"))
        test_dict = json.load(open(self.savedir+"dict_test.json","r"))

        train_files = train_dict.keys()
        test_files = test_dict.keys()

        x_train = np.zeros([len(train_files), self.N_par])
        x_test = np.zeros([len(test_files), self.N_par])

        y_train = np.zeros([len(train_files), len(self.freq_train)])
        y_test = np.zeros([len(test_files), len(self.freq_test)])
        
        x_dict_train = load_x_from_file(train_dict)
        x_dict_test = load_x_from_file(test_dict)
        
        for i, train_file in enumerate(train_files):
            self.par_train.append(x_dict_train[train_file])
            x_train[i,:] = x_dict_train[train_file] 
            y_train[i,:] = self.load_y_from_file(self.path_to_train_files+train_file, x_dict_train[train_file], self.freq_train)
        for i, test_file in enumerate(test_files):
            self.par_test.append(x_dict_test[test_file])
            x_test[i,:] = x_dict_test[test_file]
            y_test[i,:] = self.load_y_from_file(self.path_to_test_files+test_file, x_dict_test[test_file], self.freq_test)
        
        self.reorganize_data(x_train, x_test, y_train, y_test)
    
    def load_y_from_file(self, path_to_file, par, freq_train):
        # M = par[0]+par[1]
        # s1z = par[2]
        freq, data_y = np.loadtxt(path_to_file, unpack=True) 
        # freq_norm = normalize_freq(freq, M, s1z)
        data_y_interp = scp.interpolate.interp1d(freq, data_y, fill_value='extrapolate') 
        data_y = data_y_interp(freq_train)
        
        if self.variable_name == "amp":
            data_y[np.isnan(data_y)] = self.floor
            data_y[data_y<self.floor] = self.floor
            data_y = np.log(data_y)
        
        return data_y
    
    def load_new_point_otf(self, par_dict): 
        print("Generating new point")
        wf_bns = waveform_lal(par_dict, approx_fd="SEOBNRv4T_Surrogate")
        wf_bns.get_h_fd()
        par_copy = copy.deepcopy(wf_bns.par)
        par_copy['lambda1'] = 0
        par_copy['lambda2'] = 0

        wf_bbh = waveform_lal(par_copy, approx_fd="SEOBNRv4T_Surrogate")
        wf_bbh.get_h_fd()
        m1 = wf_bns.par['m1']
        m2 = wf_bns.par['m2']
        s1z = wf_bns.par['s1z']
        s2z = wf_bns.par['s2z']
        lambda1 = wf_bns.par['lambda1']
        lambda2 = wf_bns.par['lambda2']
        M = m1+m2
        q = m1/m2
        Mc = (m1*m2)**(3./5)/M**(1./5)
        
        data_x = np.array([q, np.log10(1+lambda1/100), np.log10(1+lambda2/100)])
        # data_x = np.array([np.log(Mc), np.log(q), s1z, s2z, np.log(lambda1), np.log(lambda2)])
        print(data_x)

        if self.variable_name == "amp": 
            data_y = wf_bns.amp_fd/wf_bbh.amp_fd
        elif self.variable_name == "phase": 
            data_y = wf_bns.phase_fd-wf_bbh.phase_fd
        
        #data_y = compare_wf.phase_shifted_fd(wf_bns.freq, data_y, wf_bns.freq, np.zeros(len(data_y)), 15, 20)

        freq_norm = normalize_freq(wf_bns.freq, M, s1z)
        data_y_interp = scp.interpolate.interp1d(freq_norm, data_y, fill_value='extrapolate') 
        data_y = data_y_interp(self.freq_train)

        data_y[np.isnan(data_y)] = self.floor
        
        if self.variable_name == "amp":
            data_y = np.log(data_y)
            data_y[data_y<self.floor] = self.floor
        
        return data_x, data_y 

    def reorganize_data(self, x_train, x_test, y_train, y_test):
        """
        reorganize train and test data according to the frequency nodes  
        """
        if self.nodes == True:
            self.x_train = np.array(x_train)
            self.x_test = np.array(x_test)
            
            for i in range(len(self.freq_train)):
                self.y_train = np.vstack([self.y_train, y_train[:,i]]) if self.y_train.size else y_train[:,i] 
                self.y_test = np.vstack([self.y_test, y_test[:,i]]) if self.y_test.size else y_test[:,i] 
        
        if self.nodes == False:
            for i in range(x_train.shape[0]):
                for j, f in enumerate(self.freq_train):
                    self.x_train = np.vstack((self.x_train, np.append(x_train[i,:], f))) if self.x_train.size else np.append(x_train[i,:], f)
                    self.y_train = np.vstack((self.y_train, y_train[i,j])) if self.y_train.size else y_train[i,j]

            for i in range(x_test.shape[0]):
                for j, f in enumerate(self.freq_test):
                    self.x_test = np.vstack((self.x_test, np.append(x_test[i,:], f))) if self.x_test.size else np.append(x_test[i,:], f) 
                    self.y_test = np.vstack((self.y_test, y_test[i,j])) if self.y_test.size else y_test[i,j]

def load_x_from_file(full_dict):
    """
    The key correspond to the filename for such parameters
    """

    x_dict = {}
    for key, value in full_dict.items():
        m1=full_dict[key]['m1']
        m2=full_dict[key]['m2']
        s1z = full_dict[key]['s1z']
        s2z = full_dict[key]['s2z']
        lambda1=full_dict[key]['lambda1']
        lambda2=full_dict[key]['lambda2']
        q = m1/m2
        M = m1+m2
        Mc = (m1*m2)**(3./5)/M**(1./5)

        # x_dict[key] = [np.log(Mc), np.log(q), s1z, s2z, np.log(lambda1), np.log(lambda2)]
        # x_dict[key] = [np.log(Mc), np.log(q), s1z, s2z, np.log(lambda1), np.log(lambda2)]
        # x_dict[key] = [q, np.log10(1+lambda1/100), np.log10(1+lambda2/100)]
        x_dict[key] = [m1, m2, np.log10(1+lambda1/100), np.log10(1+lambda2/100)]

    return x_dict

def normalize_freq(freq, M, s1z):

    k1=1.5251
    k2=-1.1568
    k3=0.1292
    Mf_rd =  (1/(2*np.pi)*(k1+k2*(1-s1z)**k3))
    #freq = freq*M*lal.MTSUN_SI/Mf_rd
    #freq = freq*M*lal.MTSUN_SI
    return freq 


