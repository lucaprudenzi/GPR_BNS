from waveform_base import waveform_base
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import lalsimulation
import lal

def phaseshift_td(times1, phase1, times2, phase2, t_start_window, t_end_window):
    """
    Find a TD phase shift for phase2 to minimize phase1-phase2 between t_min and t_max
    Minimization over phase and time is made indipendently to obtain a better convergence
    Return shifted time2, phase2
    """
    phase1_interp = scp.interpolate.interp1d(times1, phase1, fill_value='extrapolate') 
    phase2_interp = scp.interpolate.interp1d(times2, phase2, fill_value='extrapolate') 
    
    # INDEPENDENT 
    #1
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, 0, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, 0, t_start_window, t_end_window))
    dt = best.x[0]
    
    #2 
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0]
    
    #3  
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0]
    
    #4 
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0]     

    #5
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0]     
    
    #6
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0]

    #7
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0]  

    #8
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0] 

    #9
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0] 

    #10
    best = scp.optimize.minimize(integral_phaseshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dt, t_start_window, t_end_window))
    dphi = best.x[0] 
    best = scp.optimize.minimize(integral_timeshift_td, [0., ], tol=1e-8, args=(phase1_interp, phase2_interp, dphi, dt, t_start_window, t_end_window))
    dt = best.x[0] 
    
    # TOGETHER
    # best = scp.optimize.minimize(integral_phasetimeshift_td, [0., 0.], args=(phase1_interp, phase2_interp, t_start_window, t_end_window))
    # dt = best.x[0]
    # dphi = best.x[1]
    
    print(dt, dphi)
    return phase2_interp(times2+dt)+dphi

def integral_phaseshift_td(dphi, phase1_interp, phase2_interp, dtold, t_start_window, t_end_window):
    """
    integral of phi1(t)-(phi2(t+deltat)+deltaphi) between fmin and fmax
    """
    i = np.sqrt(scp.integrate.quad(lambda t:np.power((phase1_interp(t) - phase2_interp(t)-dphi),2.), t_start_window, t_end_window)[0])
    return i

def integral_timeshift_td(dt, phase1_interp, phase2_interp, dphi, dtold, t_start_window, t_end_window):
    """
    integral of phi1(t)-(phi2(t+deltat)+deltaphi) between fmin and fmax
    """
    i = np.sqrt(scp.integrate.quad(lambda t:np.power((phase1_interp(t-dtold-dt) - phase2_interp(t)-dphi),2.), t_start_window, t_end_window)[0])
    return i

def integral_phasetimeshift_td(dt_phi, phase1_interp, phase2_interp, t_start_window, t_end_window):
    dt, dphi = dt_phi
    i = np.sqrt(scp.integrate.quad(lambda t:np.power((phase1_interp(t) - phase2_interp(t+dt)-dphi),2), t_start_window, t_end_window)[0])
    # i = scp.integrate.quad(lambda t:np.abs(phase1_interp(t) - phase2_interp(t+dt)-dphi), t_start_window, t_end_window)[0]
    return i

def phaseshift_fd(freq1, phase1, freq2, phase2, f_start_window, f_end_window):
    """
    Find a FD phase shift for phase2 to minimize phase1-phase2 between f_min and f_max
    Return shifted phase2 
    """
    phase1_interp = scp.interpolate.interp1d(freq1, phase1, fill_value='extrapolate') 
    phase2_interp = scp.interpolate.interp1d(freq2, phase2, fill_value='extrapolate') 
    
    best = scp.optimize.minimize(integral_phaseshift_fd, [1., 1.], args=(phase1_interp, phase2_interp, f_start_window, f_end_window))
    m = best.x[0]
    q = best.x[1]

    return phase1-m*freq1-q

def integral_phaseshift_fd(pars, phase1_interp, phase2_interp, f_start_window, f_end_window):
    """
    integral of (phi1(f)-mf-q)-phi2(f) between fmin and fmax
    """
    m, q = pars 
    i = np.sqrt(scp.integrate.quad(lambda f:np.power((phase1_interp(f) - phase2_interp(f)-m*f-q),2.), f_start_window, f_end_window)[0])
    return i

def interpolate_grid(x, y, x_interp):
    #y_interp = interpolate.splrep(x, y, s=0)
    #y_interpolated = interpolate.splev(x_interp, y_interp, der=0)
    y_interp = scp.interpolate.interp1d(x, y, fill_value="extrapolate") 
    y_interpolated = y_interp(x_interp)
    return y_interpolated
        
def sample_parameters():
    m_min, m_max = 1.3, 1.4
    s_min, s_max = -0.05, 0.05 
    lambda_min, lambda_max = 0, 1000
    m1 = np.random.uniform(m_min, m_max)
    m2 = np.random.uniform(m_min, m1)
    lambda1 = np.random.uniform(lambda_min, lambda_max)
    #lambda1 = lambda_from_mass(m1)
    lambda2 = np.random.uniform(lambda_min, lambda_max)
    #lambda2 = lambda_from_mass(m2)
    s1z = np.random.uniform(s_min, s_max)
    s2z = np.random.uniform(s_min, s_max)

    par = {
        'm1' : m1,
        'm2' : m2,
        'lambda1' : lambda1,
        'lambda2' : lambda2,
        's1z' : s1z, 
        's2z' : s2z}
    return par 
    
def lambda_from_mass(m):
    #eos_names = ['ALF2', 'ENG', 'MPA1', 'MS1', 'MS1B', 'SLY'] 
    eos_names = ['MPA1'] 
    eos_name = eos_names[np.random.randint(len(eos_names))]
    eos = lalsimulation.SimNeutronStarEOSByName(eos_name)
    eosFam = lalsimulation.CreateSimNeutronStarFamily(eos)
    m_SI = m*lal.MSUN_SI
    r= lalsimulation.SimNeutronStarRadius(m_SI, eosFam)
    k2 = lalsimulation.SimNeutronStarLoveNumberK2(m_SI, eosFam)
    lambda_tid = 2./3.*k2*(lal.C_SI**2/lal.G_SI*r/m_SI)**5 
    return lambda_tid