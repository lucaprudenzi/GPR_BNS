import lal
from pycbc.types import TimeSeries, FrequencySeries, Array
from pycbc.filter.matchedfilter import match
import pycbc.waveform as wf
import numpy as np
from pycbc.filter import make_frequency_series
import matplotlib.pyplot as plt
from scipy.special import expit

class waveform_base(object):
    def __init__(self):
        self.a = 1

    def h_td_to_amp_phase_freq_td(self):
        """
        From FD hp and hc to FD amp and phase
        """
        hp_td = TimeSeries(self.hp_td, delta_t = self.dt) 
        hc_td = TimeSeries(self.hc_td, delta_t = self.dt)
        self.amp_td = wf.utils.amplitude_from_polarizations(hp_td, hc_td)
        self.phase_td = wf.utils.phase_from_polarizations(hp_td, hc_td, remove_start_phase=False)
        self.freq_td = wf.utils.frequency_from_polarizations(hp_td, hc_td)

    def h_fd_to_amp_phase_fd(self):
        """
        From FD hp and hc to FD amp and phase
        """
        self.h_fd = self.hp_fd+1j*self.hc_fd
        # hp_fd = FrequencySeries(self.hp_fd, delta_f = self.df) 
        # hc_fd = FrequencySeries(self.hc_fd, delta_f = self.df)
        # self.phase_fd = wf.utils.phase_from_frequencyseries(h_fd)
        # self.amp_fd = wf.utils.amplitude_from_frequencyseries(h_fd)

        self.phase_fd = np.unwrap(np.angle(self.h_fd))
        self.amp_fd = np.absolute(self.h_fd)

    def amp_phase_fd_to_h_fd(self):
        """
        From FD amp and phase to FD h
        """
        self.h_fd = self.amp_fd*np.exp(1j*self.phase_fd)
    
    def compute_df(self):
        """ 
        Compute df = 1/T using starting frequency and masses of the system
        """

        Mc = (self.par['m1']*self.par['m2'])**(3./5)/(self.par['m1']+self.par['m2'])**(1./5)
        Mc_SI = Mc*lal.MSUN_SI
        T = 5/256./(np.pi*self.f_lower)**(8/3.)*(lal.C_SI**3/(lal.G_SI*Mc_SI))**(5./3)
        self.T = (T+100) # to be conservative
        self.df = 1./T
        
    def h_fd_ifft(self, h_fd, freq, fstart, fstop):
        """
        IFFT of FD hfd. Tapering applied between fstart and fstop
        """
        df = freq[1]-freq[0]
        h_fd = FrequencySeries(h_fd, delta_f = df)
        
        """ Tapering """
        # Choice 1
        #wind_right = self.TanhTaper(freq, fstop-2, 4, side="tap_at_right")
        #wind_left = self.TanhTaper(freq, fstart+0.5, 1, side="tap_at_left")
        #h_fd = h_fd*wind_left*wind_right
        
        # Choice 2
        wind_left = self.PlanckTaper(freq, fstart, fstart+5, side="tap_at_left")
        wind_right = self.PlanckTaper(freq, fstop-20, fstop, side="tap_at_right")
        h_fd = h_fd*wind_left*wind_right
        
        h_fd = FrequencySeries(h_fd, delta_f = df)
        
        """ Padding to obtain len 2^N """
        N = len(h_fd)
        pow_2 = np.ceil(np.log2(N))
        nZeroPad = int((2**(pow_2))-N) 
        h_fd = np.pad(h_fd, (0,int((2**pow_2)-N)),'constant')
        f_max = freq[-1] 
        new_f_max = f_max+nZeroPad*df
        freq = np.append(freq, np.arange(f_max+df,new_f_max+df/10., df))
        h_fd = FrequencySeries(h_fd, delta_f = df)
         
        """ IFFT """
        h_td = wf.utils.fd_to_td(h_fd)
        time = h_td.get_sample_times()
        h_td = h_td.cyclic_time_shift(-10)
        time = time-time[np.argmax(h_td)]
        h_td = TimeSeries(h_td, delta_t = np.diff(time)[0])
        return h_td, time

    def h_td_fft(self, h_td, time, tmin=0, tmax=0, plot=0):
        """
        FFT of TD htd. Tapering applied between tmin and tmax 
        """

        dt = np.abs(np.diff(time)[0])
        h_td = TimeSeries(h_td, delta_t=dt)
        
        """ Tapering """
        # Choice 1
        #h_td = wf.utils.taper_timeseries(h_td, tapermethod="startend")
        
        # Choice 2
        #wind_right = self.TanhTaper(time, time[-1]-0.005, 0.01, side="tap_at_right")
        #wind_left = self.TanhTaper(time, time[0]+0.001, 0.002, side="tap_at_left")
        #h_td = h_td*wind_right*wind_left
        
        # Choice 3
        wind_right = self.PlanckTaper(time, time[-1]-0.001, time[-1], side="tap_at_right")
        wind_left = self.PlanckTaper(time, time[0], time[0]+0.001, side="tap_at_left")
        h_td = h_td*wind_right*wind_left
        
        h_td = TimeSeries(h_td, delta_t=dt)

        """ Padding (to obtain a td wf of len 2^N) """
        N = len(h_td)
        pow_2 = np.ceil(np.log2(N))
        numZeroPad = int((2**(pow_2))-N) 
        h_td = np.pad(h_td, (0, numZeroPad),'constant')
        h_td = TimeSeries(h_td, delta_t=dt)

        """ FFT """
        h_fd = make_frequency_series(h_td)  
        freq = h_fd.get_sample_frequencies()
        
        """ Manual reconstruction of freq """
        #timePad = np.arange(time[-1]+dt, time[-1]+numZeroPad*dt+dt/10., dt)
        #time = np.append(time, timePad)
        #freq_sampling = 1/dt
        #freq_nyq = freq_sampling/2.
        #T = time[-1]-time[0]+dt
        #df = 1/T
        #freq = np.arange(0, freq_nyq+df/10,df)
        
        return h_fd, freq

    ########################
    # Windows
    #######################

    def PlanckTaper(self, x, x1, x2, side):
        """
        Planck tapering between x1 and x2 (x1<x2)
        Args:
            - x: current argument of the windowed function
            - x1: start of the window
            - x2: end of the window 
        """
        taper_arr = []
        if side == "tap_at_right":
            for i in x:
                if (i <= x1):
                    taper = 1.
                elif (i >= x2):
                    taper = 0.
                else:
                    taper = 1-expit((x1 - x2)/(i - x2)+(x1 - x2)/(i - x1))
                taper_arr.append(taper)
            return np.array(taper_arr)

        if side == "tap_at_left":
            for i in x:
                if (i <= x1):
                    taper = 0.
                elif (i >= x2):
                    taper = 1.
                else:
                    taper = expit((x1 - x2)/(i - x2)+(x1 - x2)/(i - x1))
                taper_arr.append(taper)
            return np.array(taper_arr)

    def TanhTaper(self, x, x_center, width, side):
        if side == "tap_at_right":
            return 1./2*(1-np.tanh(4*(x-x_center)/(width)))
        if side == "tap_at_left":
            return 1./2*(1+np.tanh(4*(x-x_center)/(width)))

if __name__=="__main__":
    # test ifft of taylorf2

    # test fft of taylort2
    
    a = waveform_base()
    t = np.linspace(-1, 3,100)
    y = np.ones(len(t))
    plank_wind = a.PlanckTaper(t, 0, 2, "tap_at_left")
    plt.plot(t, y)
    plt.plot(t, y*plank_wind)
    plt.show()
