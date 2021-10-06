#!/usr/bin/env python3

from waveform_base import waveform_base
from waveform_lal import waveform_lal
from waveform_nr import waveform_nr
from waveform_hybrid import waveform_hybrid
import matplotlib.pyplot as plt
import copy 
import numpy as np
import pathlib
import compare_waveforms_base as compare_wf
import json
import os
import glob 
import lal

def compare_wf_hybrid_v4Tsurrogate():
    full_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo.json"))
    
    for key, value in full_dict.items():
        if os.path.exists("../NR_data/BAM_SACRA_data_deltaamp/"+key[:-3]):
            continue

        wf_nr = waveform_nr() 
        wf_nr.set_parameters(key, value)
        
        wf_nr.get_h_td()
        wf_nr.get_h_fd_from_td()
        
        par_bns = copy.deepcopy(wf_nr.par)

        wf_bns = waveform_lal(par_bns, approx_fd="SEOBNRv4T_Surrogate")
        wf_bns.get_h_fd()
        wf_bns.get_h_td_from_fd()
        
        wf_hyb = waveform_hybrid()
        wf_hyb.t_align_start = value["t_align_start"] 
        wf_hyb.get_h_td(wf_bns, wf_nr)
        wf_hyb.get_h_fd_from_td(wf_bns)

        f = np.arange(20, 4096, 0.25) 

        delta_phase = compare_wf.interpolate_grid(wf_hyb.freq, wf_hyb.phase_fd, f) - compare_wf.interpolate_grid(wf_bns.freq, wf_bns.phase_fd, f)
        
        delta_amp = compare_wf.interpolate_grid(wf_hyb.freq, wf_hyb.amp_fd, f)/compare_wf.interpolate_grid(wf_bns.freq, wf_bns.amp_fd, f)
        np.savetxt("../NR_data/BAM_SACRA_data_deltaamp/"+str(key[:-3]), np.transpose([f, delta_amp])) 
        np.savetxt("../NR_data/BAM_SACRA_data_deltaphase/"+str(key[:-3]), np.transpose([f, delta_phase])) 


def compare_wf_lal_train():
    """
    Compute delta between v4Tsurr and IMRPhenomD_NRTidalv2
    Parameter set: NR simulation from SACRA and BAM
    """
    par_bns = {}
    sacra_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_sacra.json"))
    yes_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_yes.json"))
    maybe_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_maybe.json"))   
    doubts_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_doubts.json"))
    
    full_dict = {} 
    full_dict.update(sacra_dict)
    full_dict.update(yes_dict)
    full_dict.update(maybe_dict)

    for key, value in full_dict.items():
        if os.path.exists("../LAL_data/fd_v4TSurr_IMR/placeholder/"+str(key[:-3])):
            continue
        if not os.path.exists("../LAL_data/fd_v4TSurr_IMR/placeholder/"+str(key[:-3])):
            os.makedirs("../LAL_data/fd_v4TSurr_IMR/placeholder/"+str(key[:-3]))

        print(key)
        par_bns = value 

        wf_bns1 = waveform_lal(par_bns, approx_fd="SEOBNRv4T_Surrogate")
        wf_bns1.get_h_fd()
        
        wf_bns2 = waveform_lal(par_bns, approx_fd="IMRPhenomD_NRTidalv2")
        wf_bns2.get_h_fd()
        
        phase_1 = compare_wf.phaseshift_fd(wf_bns1.freq, wf_bns1.phase_fd, wf_bns2.freq, wf_bns2.phase_fd, 20, 25)
        
        # DELTA
        Mf = np.linspace(0.0004, 0.015, 1000)
        f = Mf/lal.MTSUN_SI/(par_bns['m1']+par_bns['m2'])
        delta_phase = compare_wf.interpolate_grid(wf_bns1.freq, phase_1, f) - compare_wf.interpolate_grid(wf_bns2.freq, wf_bns2.phase_fd, f)
        delta_amp = compare_wf.interpolate_grid(wf_bns1.freq, wf_bns1.amp_fd, f)/compare_wf.interpolate_grid(wf_bns2.freq, wf_bns2.amp_fd, f)
        
        # PLOT
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 1)
        #fig.suptitle(key.decode('ascii'), fontsize=16)
        ax1 = fig.add_subplot(gs[0, :]) 
        ax1.loglog(Mf, compare_wf.interpolate_grid(wf_bns1.freq, wf_bns1.amp_fd, f), label='v4TSurr')
        ax1.loglog(Mf,compare_wf.interpolate_grid(wf_bns2.freq, wf_bns2.amp_fd, f), label='PhenomD_NRT')
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[1, :]) 
        ax2.plot(Mf, delta_amp)
        ax2.set_xlabel('Mf')
        ax2.set_ylabel('delta amp')
        ax2.set_xscale('log')
        
        ax3 = fig.add_subplot(gs[2, :]) 
        ax3.plot(Mf, delta_phase)
        ax3.set_xlabel('Mf')
        ax3.set_ylabel('delta phase')
        ax3.set_xscale('log')

        plt.savefig('../LAL_data/fd_v4TSurr_IMR/'+key[:-3]+".png", dpi=500)

        np.savetxt("../LAL_data/LAL_data_deltaamp/"+str(key[:-3]), np.transpose([Mf, delta_amp])) 
        np.savetxt("../LAL_data/LAL_data_deltaphase/"+str(key[:-3]), np.transpose([Mf, delta_phase])) 
        

def compare_wf_lal_test():
    """
    Compute delta between v4Tsurr and IMRPhenomD_NRTidalv2
    Parameter set: NR simulation from SACRA and BAM
    """

    test_points = {}
    for i in range(50):
        test_point = {}
        m1_new = np.random.uniform(1.4,1.6) 
        m2_new = np.random.uniform(1.1,m1_new) 
        lambda1_new = np.random.uniform(1, 2000)
        lambda2_new = np.random.uniform(lambda1_new, 2000)
        s1z_new = 0
        s2z_new = 0
        test_point = {'m1':m1_new,
                'm2': m2_new,
                's1z': s1z_new,
                's2z': s2z_new,
                'lambda1': lambda1_new,
                'lambda2': lambda2_new}
        test_points[str(i)] = test_point
    json.dump(test_points, open("../LAL_data/fd_v4TSurr_IMR_train/train_points.json", 'w'), sort_keys = True, indent=4)
    print(test_points)
    dict_test = json.load(open("../LAL_data/fd_v4TSurr_IMR_train/train_points.json"))
    for key, value in dict_test.items():
        print(key)
        if os.path.exists("../LAL_data/fd_v4TSurr_IMR_train/placeholder/"+str(key)):
            continue
        if not os.path.exists("../LAL_data/fd_v4TSurr_IMR_train/placeholder/"+str(key)):
            os.makedirs("../LAL_data/fd_v4TSurr_IMR_train/placeholder/"+str(key))

        par_bns = value 

        wf_bns1 = waveform_lal(par_bns, approx_fd="SEOBNRv4T_Surrogate")
        wf_bns1.get_h_fd()
        
        wf_bns2 = waveform_lal(par_bns, approx_fd="IMRPhenomD_NRTidalv2")
        wf_bns2.get_h_fd()
        
        phase_1 = compare_wf.phaseshift_fd(wf_bns1.freq, wf_bns1.phase_fd, wf_bns2.freq, wf_bns2.phase_fd, 20, 25)
        
        # DELTA
        Mf = np.linspace(0.0004, 0.015, 1000)
        f = Mf/lal.MTSUN_SI/(par_bns['m1']+par_bns['m2'])
        delta_phase = compare_wf.interpolate_grid(wf_bns1.freq, phase_1, f) - compare_wf.interpolate_grid(wf_bns2.freq, wf_bns2.phase_fd, f)
        delta_amp = compare_wf.interpolate_grid(wf_bns1.freq, wf_bns1.amp_fd, f)/compare_wf.interpolate_grid(wf_bns2.freq, wf_bns2.amp_fd, f)
        
        # PLOT
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 1)
        #fig.suptitle(key.decode('ascii'), fontsize=16)
        ax1 = fig.add_subplot(gs[0, :]) 
        ax1.loglog(Mf, compare_wf.interpolate_grid(wf_bns1.freq, wf_bns1.amp_fd, f), label='v4TSurr')
        ax1.loglog(Mf,compare_wf.interpolate_grid(wf_bns2.freq, wf_bns2.amp_fd, f), label='PhenomD_NRT')
        ax1.legend()
        
        ax2 = fig.add_subplot(gs[1, :]) 
        ax2.plot(Mf, delta_amp)
        ax2.set_xlabel('Mf')
        ax2.set_ylabel('delta amp')
        ax2.set_xscale('log')
        
        ax3 = fig.add_subplot(gs[2, :]) 
        ax3.plot(Mf, delta_phase)
        ax3.set_xlabel('Mf')
        ax3.set_ylabel('delta phase')
        ax3.set_xscale('log')

        plt.savefig('../LAL_data/fd_v4TSurr_IMR_train/'+key+".png", dpi=500)

        np.savetxt("../LAL_data/LAL_data_deltaamp_train/"+str(key), np.transpose([Mf, delta_amp])) 
        np.savetxt("../LAL_data/LAL_data_deltaphase_train/"+str(key), np.transpose([Mf, delta_phase])) 
        


def plot_wf_deltaphase_td():
    wf_files = glob.glob("../NR_data/BAM_SACRA_data_deltaphase_td/*")
    i = 0
    for wf_file in wf_files:
        i = i+1
        #if i>1:
        #    continue
        f, d = np.loadtxt(wf_file, unpack=True)
        if any(data<-48 for data in d):
            print(wf_file)
        #d = compare_wf.phase_shifted_fd(f, d, f, np.zeros(len(d)), 20,25)
        plt.plot(f,d)
    plt.xlabel('freq')
    plt.ylabel('Deltaphase')
    plt.show()

def plot_wf_tidal_tidal0():
    wf_files = glob.glob("../LAL_data/LAL_data_deltaphase/*")
    i = 0
    for wf_file in wf_files:
        i = i+1
        #if i>1:
        #    continue
        f, d = np.loadtxt(wf_file, unpack=True)
        if any(data<-48 for data in d):
            print(wf_file)
        #d = compare_wf.phase_shifted_fd(f, d, f, np.zeros(len(d)), 20,25)
        plt.plot(f,d)
    plt.xlabel('freq')
    plt.ylabel('Deltaphase')
    plt.xscale('log')
    plt.show()

def plot_wf_hyb_v4tsurrogate():
    wf_files = glob.glob("../NR_data/BAM_SACRA_data_deltaamp/*")
    for wf_file in wf_files:

        f, d = np.loadtxt(wf_file, unpack=True)
        plt.loglog(f,d)
    print(wf_file)
    plt.xlabel('freq')
    plt.ylabel('Delta amp')
    plt.show()

    wf_files = glob.glob("../NR_data/BAM_SACRA_data_deltaphase/*")
    for wf_file in wf_files:
  
        f, d = np.loadtxt(wf_file, unpack=True)
        plt.plot(f,d)
        plt.xlabel('freq')
        plt.ylabel('Delta phase')
        plt.show()

def create_dict_from_samples():
    full_dict = {}
    for i in range(5):
        par = compare_wf.sample_parameters()
        full_dict[i] = par 
    return full_dict

def study_tc(full_dict):
    wf_bns = waveform_lal(par_bns[1], approx_fd="SEOBNRv4T_Surrogate")
    wf_bns.get_h_fd()
    wf_bns.h_fd_to_td()
    plt.plot(wf_bns.time, wf_bns.hp_td)
    plt.show()

if __name__=="__main__":
    """
    Compare tidal-notidal
    """
    # compare_wf_lal_train()
    compare_wf_lal_test()

    # plot_wf_tidal_tidal0()

    """
    Compare hybrid-v4tsurrogate
    """
    # compare_wf_hybrid_v4Tsurrogate()
    # plot_wf_hyb_v4tsurrogate()
    # plot_wf_deltaphase_td()

