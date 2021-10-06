from waveform_lal import waveform_lal
from waveform_base import waveform_base
import lal
import os
import glob
import h5py
import lalsimulation as lalsim
import numpy as np
import json
from os.path import basename
import matplotlib.pyplot as plt
import copy
import pylab
from pylab import arange,pi,sin,cos,sqrt

# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0068"
# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0054"
# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0055"
# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0056"
# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0073"
# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0091"
# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0080"
# if wf_file == "../NR_data/BAM_SACRA_data_deltaamp/BAM_0018"

no_wf_files = [
        "1.25_1.46_477_168_0.0155_182_.txt.h5",
        "BAM_0002.h5",
        "BAM_0003.h5",
        "BAM_0004.h5",
        "BAM_0009.h5",
        "BAM_0010.h5",
        "BAM_0022.h5",
        "BAM_0036.h5",
        "BAM_0038.h5",
        "BAM_0040.h5",
        "BAM_0041.h5",
        "BAM_0046.h5",
        "BAM_0058.h5",
        "BAM_0059.h5",
        "BAM_0060.h5",
        "BAM_0061.h5",
        "BAM_0063.h5",
        "BAM_0067.h5",
        "BAM_0069.h5",
        "BAM_0074.h5",
        "BAM_0075.h5",
        "BAM_0076.h5",
        "BAM_0077.h5",
        "BAM_0078.h5",
        "BAM_0079.h5",
        "BAM_0080.h5",
        "BAM_0081.h5", # lambda>5000
        "BAM_0082.h5", # lambda>5000
        "BAM_0083.h5", # lambda>5000
        "BAM_0084.h5", # lambda>5000
        "BAM_0085.h5", # lambda>5000
        "BAM_0086.h5", # lambda>5000
        "BAM_0087.h5", # lambda>5000
        "BAM_0098.h5",
        "BAM_0102.h5",
        "BAM_0105.h5",
        "BAM_0106.h5",
        "BAM_0110.h5",
        "BAM_0111.h5",
        "BAM_0112.h5",
        "BAM_0113.h5",
        "BAM_0114.h5",
        "BAM_0115.h5",
        "BAM_0116.h5",
        "BAM_0117.h5",
        "BAM_0118.h5",
        "BAM_0119.h5",
        "BAM_0121.h5",
        "BAM_0122.h5",
        "BAM_0123.h5",
        "BAM_0128.h5"]

short_wf_files = [
        "BAM_0023.h5",
        "BAM_0024.h5",
        "BAM_0025.h5",
        "BAM_0026.h5",
        "BAM_0027.h5",
        "BAM_0028.h5",
        "BAM_0029.h5",
        "BAM_0030.h5",
        "BAM_0031.h5",
        "BAM_0032.h5",
        "BAM_0033.h5",
        "BAM_0034.h5",
        "BAM_0099.h5",
        "BAM_0100.h5",
        "BAM_0103.h5",
        "BAM_0125.h5",
        "THC_0001.h5",
        "THC_0002.h5",
        "THC_0003.h5",
        "THC_0004.h5",
        "THC_0005.h5",
        "THC_0006.h5",
        "THC_0007.h5",
        "THC_0008.h5",
        "THC_0009.h5",
        "THC_0010.h5",
        "THC_0011.h5",
        "THC_0012.h5",
        "THC_0013.h5",
        "THC_0014.h5",
        "THC_0015.h5",
        "THC_0016.h5",
        "THC_0017.h5",
        "THC_0018.h5",
        "THC_0019.h5",
        "THC_0020.h5",
        "THC_0021.h5",
        "THC_0022.h5",
        "THC_0023.h5",
        "THC_0024.h5",
        "THC_0025.h5",
        "THC_0026.h5",
        "THC_0027.h5",
        "THC_0030.h5",
        "THC_0031.h5",
        "THC_0032.h5",
        "THC_0033.h5",
        "THC_0034.h5",
        "THC_0035.h5"]

yes_wf_files = [
        "BAM_0035.h5",
        "BAM_0037.h5",
        "BAM_0039.h5",
        "BAM_0047.h5",
        "BAM_0062.h5",
        "BAM_0064.h5",
        "BAM_0065.h5",
        "BAM_0068.h5",
        "BAM_0070.h5",
        "BAM_0088.h5",
        "BAM_0089.h5",
        "BAM_0091.h5",
        "BAM_0095.h5",
        "BAM_0101.h5",
        "BAM_0104.h5"]

thc_wf_files = [        
        "THC_0028.h5"]

maybe_wf_files = [
        "BAM_0005.h5",
        "BAM_0006.h5",
        "BAM_0007.h5",
        "BAM_0011.h5",
        "BAM_0012.h5",
        "BAM_0013.h5",
        "BAM_0014.h5",
        "BAM_0015.h5",
        "BAM_0016.h5",
        "BAM_0017.h5",
        "BAM_0018.h5",
        "BAM_0019.h5",
        "BAM_0020.h5",
        "BAM_0021.h5",
        "BAM_0042.h5",
        "BAM_0043.h5",
        "BAM_0044.h5",
        "BAM_0045.h5",
        "BAM_0048.h5",
        "BAM_0049.h5",
        "BAM_0050.h5",
        "BAM_0051.h5",
        "BAM_0052.h5",
        "BAM_0053.h5",
        "BAM_0054.h5",
        "BAM_0055.h5",
        "BAM_0056.h5",
        "BAM_0066.h5",
        "BAM_0071.h5",
        "BAM_0072.h5",
        "BAM_0073.h5",
        "BAM_0090.h5",
        "BAM_0092.h5",
        "BAM_0096.h5",
        "BAM_0107.h5",
        "BAM_0108.h5",
        "BAM_0124.h5",
        "BAM_0126.h5",
        "BAM_0127.h5"]

doubts_wf_files = [
        "BAM_0001.h5",
        "BAM_0008.h5",
        "BAM_0109.h5",
        "BAM_0120.h5",
        "THC_0029.h5",
        "THC_0036.h5"]

lambda5000_wf_files = [
        "BAM_0057.h5", # lambda>5000
        "BAM_0093.h5", # lambda>5000
        "BAM_0094.h5"] # lambda>5000

sacra_wf_files = [
        "1.07_1.46_1216_168_0.015_182_.txt.h5",
        "1.07_1.46_1695_252_0.015_182_.txt.h5",
        "1.07_1.46_2329_369_0.015_182_.txt.h5",
        "1.07_1.46_3196_535_0.015_182_.txt.h5",
        "1.07_1.46_4361_760_0.015_182_.txt.h5",
        "1.12_1.4_1304_333_0.015_182_.txt.h5",
        "1.12_1.4_1812_484_0.015_182_.txt.h5",
        "1.12_1.4_2490_693_0.015_182_.txt.h5",
        "1.12_1.4_3411_975_0.015_182_.txt.h5",
        "1.12_1.4_933_225_0.015_182_.txt.h5",
        "1.16_1.58_1079_142_0.0155_182_.txt.h5",
        "1.16_1.58_1506_215_0.0155_182_.txt.h5",
        "1.16_1.58_2085_319_0.0155_182_.txt.h5",
        "1.16_1.58_2863_465_0.0155_182_.txt.h5",
        "1.16_1.58_765_91_0.0155_182_.txt.h5",
        "1.17_1.56_1013_157_0.0155_182_.txt.h5",
        "1.17_1.56_1415_238_0.0155_182_.txt.h5",
        "1.17_1.56_1963_350_0.0155_182_.txt.h5",
        "1.17_1.56_2692_509_0.0155_182_.txt.h5",
        "1.17_1.56_719_101_0.0155_182_.txt.h5",
        "1.18_1.55_1354_249_0.0155_182_.txt.h5",
        "1.18_1.55_1875_366_0.0155_182_.txt.h5",
        "1.18_1.55_2575_530_0.0155_182_.txt.h5",
        "1.18_1.55_681_107_0.0155_182_.txt.h5",
        "1.18_1.55_966_165_0.0155_182_.txt.h5",
        "1.21_1.51_1163_298_0.0155_182_.txt.h5",
        "1.21_1.51_1621_435_0.0155_182_.txt.h5",
        "1.21_1.51_2238_625_0.0155_182_.txt.h5",
        "1.21_1.51_581_131_0.0155_182_.txt.h5",
        "1.21_1.51_827_200_0.0155_182_.txt.h5",
        "1.25_1.25_1352_1352_0.015_182_.txt.h5",
        "1.25_1.25_1875_1875_0.015_182_.txt.h5",
        "1.25_1.25_476_476_0.015_182_.txt.h5",
        "1.25_1.25_683_683_0.015_182_.txt.h5",
        "1.25_1.25_966_966_0.015_182_.txt.h5",
        "1.25_1.46_1351_535_0.0155_182_.txt.h5",
        "1.25_1.46_1871_760_0.0155_182_.txt.h5",
        "1.25_1.46_684_252_0.0155_182_.txt.h5",
        "1.25_1.46_966_369_0.0155_182_.txt.h5",
        "1.35_1.35_1211_1211_0.0155_182_.txt.h5",
        "1.35_1.35_289_289_0.0155_182_.txt.h5",
        "1.35_1.35_422_422_0.0155_182_.txt.h5",
        "1.35_1.35_460_460_0.0155_182_.txt.h5",
        "1.35_1.35_607_607_0.0155_182_.txt.h5",
        "1.35_1.35_863_863_0.0155_182_.txt.h5"]

class waveform_nr(waveform_base):
    def __init__(self):
        super().__init__()

        self.f_upper = 65536*2
        self.srate = self.f_upper*2
        self.dt = 1/self.srate
        self.distance_Mpc = 1
        self.distance = self.distance_Mpc*lal.PC_SI*1e6
        self.inclination = 0.
        self.coa_phase = 0.
        self.par = {}

    def set_parameters(self, key, value):
        """
        Extract parameters of the NR simulation from the LVC formatted h5 file
        """

        par_dict = value
        self.par['m1'] = par_dict['m1'] # grav mass
        self.par['m2'] = par_dict['m2'] # grav mass
        self.par['M'] = self.par['m1']+self.par['m2']
        self.par['lambda1'] = par_dict['lambda1']
        self.par['lambda2'] = par_dict['lambda2']
        self.par['s1x'] = par_dict['s1x']
        self.par['s1y'] = par_dict['s1y']
        self.par['s1z'] = par_dict['s1z']
        self.par['s2x'] = par_dict['s2x']
        self.par['s2y'] = par_dict['s2y']
        self.par['s2z'] = par_dict['s2z']
        
        self.ecc = par_dict['ecc']
        #self.par['kappa2t'] = par_dict['kappa2t']
        self.par['kappa2t'] = lalsim.SimNRTunedTidesComputeKappa2T(self.par['m1']*lal.MSUN_SI, self.par['m2']*lal.MSUN_SI, self.par['lambda1'], self.par['lambda2'])
        
        self.f_merg = lalsim.SimNRTunedTidesMergerFrequency(self.par['M'], self.par['kappa2t'], self.par['m1']/self.par['m2'])
        self.f_isco = lal.C_SI**3/(lal.MSUN_SI*self.par['M']*lal.G_SI*6**1.5*np.pi)
        self.f_lower = par_dict['f_lower'] 
        self.wf_file = "../NR_data/BAM_SACRA_data/"+key

    def get_h_td(self):
        """ 
        Create TD hp and hc from NR data in LVC format
        """
        
        params_lal = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda1(params_lal, self.par['lambda1'])
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(params_lal, self.par['lambda2'])
        ma = lalsim.SimInspiralCreateModeArray()
        lalsim.SimInspiralModeArrayActivateMode(ma, 2, 2)

        lalsim.SimInspiralWaveformParamsInsertModeArray(params_lal, ma)
        lalsim.SimInspiralWaveformParamsInsertNumRelData(params_lal, self.wf_file)
        approx_lal = lalsim.NR_hdf5

        hp_td, hc_td = lalsim.SimInspiralChooseTDWaveform(
                self.par['m1']*lal.MSUN_SI, 
                self.par['m2']*lal.MSUN_SI,
                self.par['s1x'],
                self.par['s1y'],
                self.par['s1z'],
                self.par['s2x'],
                self.par['s2y'],
                self.par['s2z'],
                self.distance,
                self.inclination,
                self.coa_phase,
                0.,
                self.ecc,
                0.,
                self.dt,
                self.f_lower, 
                self.f_lower,
                params_lal,
                lalsim.NR_hdf5)
        self.hp_td = hp_td.data.data
        self.hc_td = hc_td.data.data
        self.h_td_to_amp_phase_freq_td()
        #self.time = (np.arange(self.hp_td.data.length)-np.argmax(hp_td.data.data))*hp_td.deltaT
        self.time = (np.arange(int(len(self.amp_td)))-np.argmax(self.amp_td))*hp_td.deltaT
        self.dt = np.diff(self.time)[0]
    
    def get_h_fd_from_td(self):
        self.hp_fd, self.freq = self.h_td_fft(self.hp_td, self.time)
        self.hc_fd, self.freq = self.h_td_fft(self.hc_td, self.time)
       
        self.df = np.diff(self.freq)[0]
        self.h_fd_to_amp_phase_fd()
    
def create_dict_NR_BNS():
    """
    Create a dictionary of all NR simulations
    """
    par_NR_BNS = {}
    wf_files = glob.glob("../NR_data/BAM_SACRA_data/*.h5")

    for wf_file in wf_files:
        if basename(wf_file) not in sacra_wf_files:
            continue


        par = {} 
        f = h5py.File(wf_file, 'r')
        f_attr = f.attrs
        eos_name = f_attr.get('EOS-name', None)
        # if f_attr['mass2-msol']>1.3:
        #     print('skip')
        #     continue
        # if f_attr['mass2-msol']<1.15:
        #     continue

        par['m1'] = f_attr['mass1-msol'] # grav mass
        par['m2'] = f_attr['mass2-msol'] # grav mass
        par['lambda1'] = f_attr.get('tidal-lambda1', None)
        par['lambda2'] = f_attr.get('tidal-lambda2', None)
        par['s1x'] = f_attr['spin1x']
        par['s1y'] = f_attr['spin1y']
        par['s1z'] = f_attr['spin1z']
        par['s2x'] = f_attr['spin2x']
        par['s2y'] = f_attr['spin2y']
        par['s2z'] = f_attr['spin2z']
        
        par['ecc'] = f_attr['eccentricity']

        if basename(wf_file) in short_wf_files:
            par['t_align_start'] = 200
        elif basename(wf_file) in sacra_wf_files:
            par['t_align_start'] = 1000
        else:
            par['t_align_start'] = 700

        #par['kappa2t'] = f_attr['kappa2t']
        par['kappa2t'] = lalsim.SimNRTunedTidesComputeKappa2T(par['m1']*lal.MSUN_SI, par['m2']*lal.MSUN_SI, par['lambda1'], par['lambda2'])
#omega_orbit_msol = fa['Omega'] / mtotal
        par['f_lower'] = f_attr['f_lower_at_1MSUN'] 
        par_NR_BNS[basename(wf_file)] = par
    json.dump(par_NR_BNS, open("../NR_data/BAM_SACRA_data/NRinfo_sacra.json", 'w'), sort_keys=True, indent=4)

 
def create_dict_NR_BNS_nospin():
    """
    Create a dictionary of all non spinning NR simulations
    """
    par_NR_BNS = {}
    wf_files = glob.glob("../NR_data/BAM_SACRA_data/*.h5")

    for wf_file in wf_files:
        if basename(wf_file) not in sacra_wf_files:
            if basename(wf_file) not in yes_wf_files:
                if basename(wf_file) not in maybe_wf_files:
                    continue

        par = {} 
        f = h5py.File(wf_file, 'r')
        f_attr = f.attrs
        eos_name = f_attr.get('EOS-name', None)
        # if f_attr['mass2-msol']>1.3:
        #     print('skip')
        #     continue
        # if f_attr['mass2-msol']<1.15:
        #     continue

        par['m1'] = f_attr['mass1-msol'] # grav mass
        par['m2'] = f_attr['mass2-msol'] # grav mass
        par['lambda1'] = f_attr.get('tidal-lambda1', None)
        par['lambda2'] = f_attr.get('tidal-lambda2', None)
        par['s1x'] = f_attr['spin1x']
        par['s1y'] = f_attr['spin1y']
        par['s1z'] = f_attr['spin1z']
        par['s2x'] = f_attr['spin2x']
        par['s2y'] = f_attr['spin2y']
        par['s2z'] = f_attr['spin2z']
        
        if par['s1z'] != 0:
            if par['s2z'] != 0:
                continue
        
        
        par['ecc'] = f_attr['eccentricity']

        if basename(wf_file) in short_wf_files:
            par['t_align_start'] = 200
        elif basename(wf_file) in sacra_wf_files:
            par['t_align_start'] = 1000
        else:
            par['t_align_start'] = 700

        #par['kappa2t'] = f_attr['kappa2t']
        par['kappa2t'] = lalsim.SimNRTunedTidesComputeKappa2T(par['m1']*lal.MSUN_SI, par['m2']*lal.MSUN_SI, par['lambda1'], par['lambda2'])
#omega_orbit_msol = fa['Omega'] / mtotal
        par['f_lower'] = f_attr['f_lower_at_1MSUN'] 
        par_NR_BNS[basename(wf_file)] = par
    json.dump(par_NR_BNS, open("../NR_data/BAM_SACRA_data/NRinfo_nospin.json", 'w'), sort_keys=True, indent=4)


def max_f_merg():
    sacra_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_sacra.json"))
    yes_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_yes.json"))
    maybe_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_maybe.json"))   
    doubts_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_doubts.json"))
    
    full_dict = {} 
    full_dict.update(sacra_dict)
    full_dict.update(yes_dict)
    full_dict.update(maybe_dict)

    for key, value in full_dict.items():
        f_merg = lalsim.SimNRTunedTidesMergerFrequency((value['m1']+value['m2']), value['kappa2t'], value['m1']/value['m2'])*lal.MTSUN_SI*(value['m1']+value['m2'])
        f_isco = lal.C_SI**3/(lal.MSUN_SI*(value['m1']+value['m2'])*lal.G_SI*6**1.5*np.pi)*lal.MTSUN_SI*(value['m1']+value['m2'])
        plt.scatter(f_merg, f_isco)
    plt.xlabel('f_merg')
    plt.ylabel('f_isco')
    #plt.xlim(1200, 2200)
    #plt.ylim(1200, 2200)
    plt.savefig('f_merg_f_isco.png')

def parameter_space_plot():
    sacra_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_nospin.json"))
    
    full_dict = {}
    full_dict.update(sacra_dict)

    values_spin1 = [] 
    values_spin2 = [] 
    values_mass1 = [] 
    values_mass2 = [] 
    values_lambda1 = [] 
    values_lambda2 = [] 
    
    # save all the parameters of all the sets
    for key, value in full_dict.items():
        # if value['m2']>1.3:
        #     continue
        # if value['m2']<1.15:
        #     continue
        values_spin1.append(value['s1z']) 
        values_spin2.append(value['s2z']) 
        values_mass1.append(value['m1']) 
        values_mass2.append(value['m2']) 
        values_lambda1.append(value['lambda1']) 
        values_lambda2.append(value['lambda2']) 
    values_spin1 = np.array(values_spin1)    
    values_spin2 = np.array(values_spin2)
    values_mass1 = np.array(values_mass1)
    values_mass2 = np.array(values_mass2)
    values_lambda1 = np.array(values_lambda1)
    values_lambda2 = np.array(values_lambda2)
    # fig_width_pt = 420.0  # Get this from LaTeX using \showthe\columnwidth
    # inches_per_pt = 1.0/72.27               # Convert pt to inch
    # golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    # fig_width = fig_width_pt*inches_per_pt  # width in inches
    # fig_height = 2.8*fig_width*golden_mean      # height in inches
    # fig_size =  [fig_width,fig_height]
    
    # params = {'backend': 'pdf',
    #             'axes.labelsize': 11,
    #             'legend.fontsize': 9,
    #             'xtick.labelsize': 11,
    #             'ytick.labelsize': 11,
    #             'text.usetex': True,
    #             'figure.figsize': fig_size}
    
    # pylab.rcParams.update(params)
    
    # WF PLOT
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    # PARAM PLOT
    ax1 = fig.add_subplot(gs[0, 0]) 
    #ax1.scatter(values_mass1, values_mass2, alpha=0.1, color='black')
    ax1.scatter(values_mass1/values_mass2, values_mass1+values_mass2, alpha=0.1, color='black')
    ax1.set_xlabel(r'$q$')
    ax1.set_ylabel(r'$M$')
    #ax2 = fig.add_subplot(gs[0, 1]) 
    #ax2.scatter(values_spin1, values_spin2, alpha=0.1, color='black')
    #ax2.set_xlabel(r'$\chi_1$')
    #ax2.set_ylabel(r'$\chi_2$')
    ax3 = fig.add_subplot(gs[0, 1]) 
    ax3.scatter(np.log10(1+values_lambda1/100), np.log10(1+values_lambda2/100), alpha=0.1, color='black')
    ax3.set_xlabel(r'$\lambda_1$')
    ax3.set_ylabel(r'$\lambda_2$')
    plt.savefig('nospin_par_space_2.png')
    
if __name__ == "__main__":
    # max_f_merg()

    # create_dict_NR_BNS_nospin()
    # create_dict_NR_BNS()
    parameter_space_plot()

    nr_dict = json.load(open("../NR_data/BAM_SACRA_data/NRinfo_yes.json"))
    i = 0
    for key, value in nr_dict.items():
            wf_nr = waveform_nr() 
            wf_nr.set_parameters(key, value)
            print(wf_nr.par['m1'])
            print(wf_nr.dt)
            wf_nr.get_h_td()
