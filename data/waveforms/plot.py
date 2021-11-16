def plot_td(wf_bns, wf_nr, wf_hyb):
    fig_width_pt = 420.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = 2.8*fig_width*golden_mean      # height in inches
    fig_size =  [fig_width,fig_height]
    
    params = {'backend': 'pdf',
               'axes.labelsize': 11,
               'legend.fontsize': 9,
               'xtick.labelsize': 11,
               'ytick.labelsize': 11,
               'text.usetex': True,
               'figure.figsize': fig_size}
    
    pylab.rcParams.update(params)

    """
    TIME DOMAIN PLOT
    """
    
    # WF PLOT
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    #fig.suptitle(key.decode('ascii'), fontsize=16)
    ax1 = fig.add_subplot(gs[0, :]) 
    ax1.plot(wf_hyb.time, wf_hyb.hp_td, "--", label='hyb')
    ax1.plot(wf_bns.time, wf_bns.hp_td, label='lal')
    ax1.plot(wf_nr.time, wf_nr.hp_td, alpha=0.3, label='nr')
    
    ax1_1 = ax1.twinx() 
    ax1_1.plot(wf_hyb.time[:-1], wf_hyb.freq_td, label='nr')
    ax1_1.set_ylim(-max(wf_hyb.freq_td),max(wf_hyb.freq_td))
    
    ax1.axvline(wf_hyb.hyb_window_start, color='black')
    ax1.axvline(wf_hyb.hyb_window_end, color='blue')
    ax1.axvline(wf_hyb.tap_window_start, color='green')
    ax1.axvline(wf_hyb.tap_window_end, color='red')
    ax1.set_xlim(left=wf_nr.time[0], right=wf_nr.time[-1]) 
    ax1.set_xlabel(r't[s]')
    ax1.set_ylabel(r'$h$')
    ax1.legend()

    ax1.axvline(wf_hyb.hyb_window_start)
    ax1.axvline(wf_hyb.hyb_window_end)
    ax1.axvline(wf_hyb.tap_window_start)
    ax1.axvline(wf_hyb.tap_window_end)
    #ax1.legend()
    
    # PHASE PLOT
    p1 = compare_wf.interpolate_grid(wf_bns.time, wf_bns.phase_td, wf_hyb.time)
    p2 = compare_wf.interpolate_grid(wf_hyb.time, wf_hyb.phase_td, wf_hyb.time)
    delta_phase_td = p1-p2
    delta_phase_td = delta_phase_td - delta_phase_td[0]
    
    ax2 = fig.add_subplot(gs[1, :]) 
    ax2.plot(wf_hyb.time, delta_phase_td)
    ax2.axvline(wf_hyb.hyb_window_start, color='black')
    ax2.axvline(wf_hyb.hyb_window_end, color='blue')
    ax2.axvline(wf_hyb.tap_window_start, color='green')
    ax2.axvline(wf_hyb.tap_window_end, color='red')
    ax2.set_xlim(left=wf_nr.time[0], right=wf_hyb.tap_window_start)
    ax2.set_ylim(-6,6)
    ax2.set_xlabel(r't[s]')
    ax2.set_ylabel(r'$\Delta\phi$')
    
    a1 = compare_wf.interpolate_grid(wf_bns.time, wf_bns.amp_td, wf_hyb.time)
    a2 = compare_wf.interpolate_grid(wf_hyb.time, wf_hyb.amp_td, wf_hyb.time)
    delta_amp_td = a1/a2
    
    ax3 = fig.add_subplot(gs[2, :]) 
    ax3.plot(wf_hyb.time, delta_amp_td)
    ax3.axvline(wf_hyb.hyb_window_start, color='black')
    ax3.axvline(wf_hyb.hyb_window_end, color='blue')
    ax3.axvline(wf_hyb.tap_window_start, color='green')
    ax3.axvline(wf_hyb.tap_window_end, color='red')
    ax3.set_xlim(left=wf_nr.time[0], right=wf_hyb.tap_window_start)
    ax3.set_ylim(0, 2)
    ax3.set_xlabel(r't[s]')
    ax3.set_ylabel(r'$Amp ratio$')
    
    # PARAM PLOT
    #ax3 = fig.add_subplot(gs[2, 0]) 
    #ax3.scatter(values_mass1, values_mass2, alpha=0.1, color='black')
    #ax3.plot(value['m1'], value['m2'], 'x', color='red')
    #ax3.set_xlabel(r'$m_1$')
    #ax3.set_ylabel(r'$m_2$')
    #ax4 = fig.add_subplot(gs[2, 1]) 
    #ax4.scatter(values_spin1, values_spin2, alpha=0.1, color='black')
    #ax4.plot(value['s1z'], value['s2z'], 'x', color='red')
    #ax4.set_xlabel(r'$\chi_1$')
    #ax4.set_ylabel(r'$\chi_2$')
    #ax5 = fig.add_subplot(gs[2, 2]) 
    #ax5.scatter(values_lambda1, values_lambda2, alpha=0.1, color='black')
    #ax5.plot(value['lambda1'], value['lambda2'], 'x', color='red')
    #ax5.set_xlabel(r'$\lambda_1$')
    #ax5.set_ylabel(r'$\lambda_2$')
    
    # SAVE PLOT
    if key in yes_dict:
        plt.savefig('../NR_data/td_v4TSurr/yes/'+key[:-3]+".png", dpi=500)
    if key in maybe_dict:
        plt.savefig('../NR_data/td_v4TSurr/maybe/'+key[:-3]+".png", dpi=500)
    if key in short_dict:
        plt.savefig('../NR_data/td_v4TSurr/short/'+key[:-3]+".png", dpi=500)
    if key in lambda5000_dict:
        plt.savefig('../NR_data/td_v4TSurr/lambda5000/'+key[:-3]+".png", dpi=500)
    if key in sacra_dict:
        plt.savefig('../NR_data/td_v4TSurr/sacra/'+key[:-3]+".png", dpi=500)
    if key in doubts_dict:
        plt.savefig('../NR_data/td_v4TSurr/doubts/'+key[:-3]+".png", dpi=500)
    if key in thc_dict:
        plt.savefig('../NR_data/td_v4TSurr/thc/'+key[:-3]+".png", dpi=500)
    #plt.show()

def plot_fd(wf_bns, wf_nr, wf_hyb):
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 1)
    #fig.suptitle(key.decode('ascii'), fontsize=16)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.loglog(Mf, a1, label='hyb')
    ax1.loglog(Mf, a2, label='lal')
    ax1.loglog(wf_nr.freq*lal.MTSUN_SI*(value['m1']+value['m2']), wf_nr.amp_fd, label='nr')
    ax1.axvline(wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax1.axvline(1.2*wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax1.axvline(wf_hyb.f_nr_start*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax1.axvline(wf_hyb.f_nr_start_full*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax1.set_xlim(Mf[0], Mf[-1])
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[1, :]) 
    ax2.plot(Mf, delta_amp_fd)
    ax2.axvline(wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax2.axvline(1.2*wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax2.axvline(wf_hyb.f_nr_start*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax2.axvline(wf_hyb.f_nr_start_full*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax2.set_xlim(Mf[0], Mf[-1])

    ax2.set_xscale('log')
    
    # PHASE PLOT 
    p1 = compare_wf.interpolate_grid(wf_hyb.freq, wf_hyb.phase_fd, f)
    p2 = compare_wf.interpolate_grid(wf_bns.freq, wf_bns.phase_fd, f)
    delta_phase_fd = p1-p2

    ax3 = fig.add_subplot(gs[2, :]) 
    ax3.plot(Mf, delta_phase_fd)
    ax3.axvline(wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax3.axvline(1.2*wf_nr.f_merg*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax3.axvline(wf_hyb.f_nr_start*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax3.axvline(wf_hyb.f_nr_start_full*lal.MTSUN_SI*(value['m1']+value['m2']))
    ax3.set_xlim(Mf[0], Mf[-1])
    ax3.set_xscale('log')

    if key in yes_dict:
        plt.savefig('../NR_data/fd_v4TSurr/yes/'+key[:-3]+".png", dpi=500)
    if key in maybe_dict:
        plt.savefig('../NR_data/fd_v4TSurr/maybe/'+key[:-3]+".png", dpi=500)
    if key in short_dict:
        plt.savefig('../NR_data/fd_v4TSurr/short/'+key[:-3]+".png", dpi=500)
    if key in lambda5000_dict:
        plt.savefig('../NR_data/fd_v4TSurr/lambda5000/'+key[:-3]+".png", dpi=500)
    if key in sacra_dict:
        plt.savefig('../NR_data/fd_v4TSurr/sacra/'+key[:-3]+".png", dpi=500)
    if key in doubts_dict:
        plt.savefig('../NR_data/fd_v4TSurr/doubts/'+key[:-3]+".png", dpi=500)
    if key in thc_dict:
        plt.savefig('../NR_data/fd_v4TSurr/thc/'+key[:-3]+".png", dpi=500)
    #plt.show()
    
