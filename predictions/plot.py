from pathlib import Path
import matplotlib
import numpy as np
import json
import matplotlib.pyplot as plt
import os
os.sys.path.append("../data/waveforms/")
import waveform_lal as wf_lal
plt.rcParams['figure.dpi'] = 500 # 200 e.g. is really fine, but slower
################################
# PLOT functions
################################

font = {'weight' : 'normal',
        'size': 10}

matplotlib.rc('font', **font)

def plot_frequency_evolution(gpr):
    fig, ax = plt.subplots(figsize=[5,4])
    if gpr.variable_name == "amp":
        gpr.y_train = np.exp(gpr.y_train)
        gpr.y_predict = np.exp(gpr.y_predict)
        gpr.y_test = np.exp(gpr.y_test)
    if gpr.nodes == True:
        #for i in range(gpr.y_train.shape[1]):
        #    if i == 0:
        #        ax.plot(gpr.freq_interp, (gpr.y_train[:,i]), "x-", color='gray', label="Trainining set")
        #    else:
        #        ax.plot(gpr.freq_interp, (gpr.y_train[:, i]),"x-", color='gray')
        
        for i in range(gpr.y_test.shape[1]):
            if i == 0:
                ax.errorbar(gpr.freq_interp, (gpr.y_predict[:, i]), yerr=gpr.y_predict_std[:, i], color="k", label="Prediction")
                ax.plot(gpr.freq_interp, (gpr.y_predict[:, i]), marker=".", color="black")
                ax.scatter(gpr.freq_interp, (gpr.y_test[:, i]), marker="x", color="red", label="Validation")
            else:
                ax.errorbar(gpr.freq_interp, (gpr.y_predict[:, i]), yerr=gpr.y_predict_std[:, i], color="k")
                ax.plot(gpr.freq_interp, (gpr.y_predict[:, i]), marker=".", color="black")
                ax.scatter(gpr.freq_interp, (gpr.y_test[:, i]), marker="x", color="red")

    if gpr.nodes == False:
        freq_train = [item[-1] for item in gpr.x_train]
        freq_test = [item[-1] for item in gpr.x_test]
        
        ax.scatter(freq_train, gpr.y_train, marker="x", color='gray')
        ax.scatter(freq_test, gpr.y_predict, marker="x", color="green")
        ax.scatter(freq_test, gpr.y_test, marker=".", color="red")
        ax.errorbar(freq_test, gpr.y_predict, yerr=gpr.y_predict_std, ls='none', color="k")
    
    if gpr.variable_name=="amp":
        plt.yscale('log')
    ax.set_ylabel(r"$\Delta$ {}".format(gpr.variable_name))
    ax.set_xlabel(r"$f[Hz]$")
    ax.legend(loc="upper left")
    plt.xscale('log') 
    directory = gpr.savedir
    directory_split = directory.split("/")[-2]
    plt.savefig("1.png")

    # plt.savefig("plot/"+directory_split+".png")
    #plt.show()

def plot_uncertainty(gpr):
    """
    Produce a heatmap for the uncertainty of 
    randomly extracted test points 
    """ 
    
    m1_arr = np.linspace(1.1,2,10)
    m2 = 1
    s1z = 0
    s2z = 0
    lambda1_arr = np.linspace(0, 3000, 10)
    lambda2 = 1000
    errs=np.zeros([len(lambda1_arr),len(m1_arr),len(gpr.freq_interp)])

    for i, lambda1 in enumerate(lambda1_arr):
        for j, m1 in enumerate(m1_arr):

            par = np.array([[m1, m2, s1z, s2z, lambda1, lambda2]])
            y_predict, y_predict_std = gpr.test(x_test=par)
            pardict = {
                    'm1':m1, 
                    'm2':m2,
                    's1z':s1z,
                    's2z':s2z,
                    'lambda1':lambda1,
                    'lambda2':lambda2}
            #x_test, y_test = gpr.load_new_point_otf(pardict)
            errs[i, j, :] = y_predict_std
            #errs[i, j, :] = y_predict-y_test
        print(i)

    rms_errs=np.zeros((len(lambda1_arr),len(m1_arr)))
    max_errs=np.zeros((len(lambda1_arr),len(m1_arr)))

    for i, lambda1 in enumerate(lambda1_arr):
        for j, m1 in enumerate(m1_arr):
            rms_errs[i, j] = 1/len(gpr.freq_interp)*np.sqrt(np.sum(errs[i, j, :]**2))
            max_errs[i, j] = np.max(errs[i, j, :])
    
    L1, M1 = np.meshgrid(lambda1_arr, m1_arr)
    c = plt.contour(M1,L1,rms_errs.T)
    print(rms_errs)
    plt.clabel(c, inline=True)
    
    for i in range(gpr.x_train.shape[0]):
        m1, m2, s1z, s2z, lambda1, lambda2  = gpr.x_train[i,:]
        plt.scatter(m1, lambda1, color='blue')

    plt.xlabel(r'$m1_1$')
    plt.ylabel(r'$\Lambda_1$')
    plt.ylim(0, 3000)
    plt.xlim(1,2)
    plt.show()

def plot_error_estimate(gpr_model):
    """
    Histograms of the errors at each frequency nodes
    """ 
    # Params: hyperparams_path, delta_path_add, delta_name, NR, freq_nodes)
    error = []
    for i, y_train in enumerate(y_train_s): 
        hyperparams_path_full = hyperparams_path+str(i)+".pkl"
        y_predict, y_std, gpr_model = prediction.predict_gpr(x_valid, method=method, hyperparams_path=hyperparams_path_full, x_train = x_train, y_train=y_train)
        error.append(np.abs(y_predict-y_valid_s[i]))
        plt.title(freq[i])
        plt.xlabel("prediction-validation")
        plt.hist(error[i], bins=20)
        plt.show()

def plot_distance():
    full_dict = json.load(open("../data/NR_data/BAM_SACRA_data/NRinfo.json"))
    # find max value for normalization
    max_m1, max_m2, max_s1, max_s2, max_l1, max_l2  = -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf
    for key, value in full_dict.items():
        if value['m1']>max_m1:
            max_m1 = value['m1']
        if value['m2']>max_m2:
            max_m2 = value['m2']
        if value['s1z']>max_s1:
            max_s1 = value['s1z']
        if value['s2z']>max_s2:
            max_s2 = value['s2z']
        if value['lambda1']>max_l1:
            max_l1 = value['lambda1']
        if value['lambda2']>max_l2:
            max_l2 = value['lambda2']
    print(max_m1, max_m2, max_s1, max_s2, max_l1, max_l2) 
    
    distances = {}
    for key1, value1 in full_dict.items():
        distances_inner = {}
        for key2, value2 in full_dict.items():
            m1_contr=((value1['m1']-value2['m1'])/max_m1)**2
            m2_contr=((value1['m2']-value2['m2'])/max_m2)**2
            s1_contr=((value1['s1z']-value2['s1z'])/max_s1)**2
            s2_contr=((value1['s2z']-value2['s2z'])/max_s2)**2
            l1_contr=((value1['lambda1']-value2['lambda1'])/max_l1)**2
            l2_contr=((value1['lambda2']-value2['lambda2'])/max_l2)**2
            d = np.sqrt(m1_contr+m2_contr+s1_contr+s2_contr+l1_contr+l2_contr)
            distances_inner[key2]=d
        distances[key1]=distances_inner

    for key1, value1 in full_dict.items():
        print('key')
        print(key1)
        hist_arr = []
        fig, ax = plt.subplots()
        for key2, value2 in full_dict.items():
            hist_arr.append(distances[key1][key2])
        n, bins, patches = ax.hist(hist_arr, 20, facecolor='blue', alpha=0.5)
        plt.savefig("plot_2/"+key1+"_distances.png")

###########################
# OLD
###########################

#def plot_mll(gpr):
#    ######################################
#    # Plot of the likelihood funtion for grid combination of
#    # 2 different hyperparameters
#    ########################################
#    length_scales = gpr.get_params()['kernel__k2__length_scale']
#    const_value_fit = gpr.get_params()['kernel__k1__constant_value']
#    
#    theta = np.logspace(-1, 3, 100)
#    length = np.logspace(-1, 2, 100)
#    
#    #LML = []
#    #for i in range(len(theta)):
#    #    hyp = [np.log(theta[i]), length_scales[0], length_scales[1], length_scales[2]]
#    #    LML.append(gpr_model.log_marginal_likelihood(hyp)) 
#    #plt.plot(theta, LML) 
#    
#    Theta, Length = np.meshgrid(theta, length)
#    LML = [[gpr.log_marginal_likelihood(np.log([Theta[i, j], Length[i, j]]))
#        for i in range(Theta.shape[0])] for j in range(Length.shape[1])]
#
#    LML = np.array(LML)
#    
#    #vmin, vmax = (-LML).min(), (-LML).max()
#    #print(vmin)
#    #print(vmax)
#    #level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)
#
#    max_x, max_y = np.unravel_index(LML.argmax(), LML.shape)
#    print("max theta", theta[max_x])
#    print("max length", length[max_y])
#    
#    ax = plt.axes(projection='3d')
#    ax.plot_surface(Theta, Length, LML,
#                cmap='viridis', edgecolor='none')
#    ax.set_title('surface');
#
#
#    #CS = plt.contourf(Theta, Length, LML, levels=20)
#    #plt.clabel(CS, fontsize=10)
#    #plt.colorbar()
#    #plt.yscale("log")
#    #plt.xscale("log")
#    #plt.title("Log-marginal-likelihood")
#    #plt.tight_layout()
#    plt.show()
#
#
#        
#def plot2d():
#    ########################
#    # Std in heatmap plot with 2
#    # different parameters along the 2 axis
#    ##########################
#    delta_name = "deltaphase"
#    NR = False
#    freq_nodes = True
#    home = "/home/luca"
#    delta_path_add = "6D_150"
#    hyperparams_path_add = "6D_Mcql1l2s1s2_150_p"
#    hyperparams_path = home+"/Projects/gpr_seobnrv4tsurrogate/predictions/saved_data/"+hyperparams_path_add+"/"
#
#    x_train, y_train_s, x_valid, y_valid_s, freq = load_data.load_delta(method="sklearn",  delta_path_add=delta_path_add, delta_name=delta_name, hyperparams_path_add=hyperparams_path_add, NR=NR, freq_nodes=freq_nodes)
#    print(x_valid)
#    par1_tr = []
#    par2_tr = []
#    #par1_val = np.linspace(0.8, 1.8, 40)
#    #par2_val = np.linspace(1, 1.8, 40)
#    par1_val = np.linspace(-0.5, 0.5, 40)
#    par2_val = np.linspace(-0.5, 0.5, 40)
#
#    x_valid = []
#    y_std_arr = []
#
#    lambda1_red_fix = np.log10(600/100+1)
#    lambda2_red_fix = np.log10(1000/100+1)
#    s1_fix = 0
#    s2_fix = 0
#    Mc_fix = 1.2
#    q_fix = 1.1
#    for par1 in par1_val:
#        for par2 in par2_val:
#            #x_valid.append([par1, par2, lambda1_red_fix, lambda2_red_fix, s1_fix, s2_fix])
#            x_valid.append([Mc_fix, q_fix, lambda1_red_fix, lambda2_red_fix, par1, par2])
#    print(x_train)
#    for x_tr in x_train:
#        print(x_tr)
#        #par1_tr.append(x_tr[0])
#        #par2_tr.append(x_tr[1])
#        par1_tr.append(x_tr[4])
#        par2_tr.append(x_tr[5])
#
#    for i, y_train in enumerate(y_train_s): 
#        hyperparams_path_full = hyperparams_path+str(i)+".pkl"
#        y_predict, y_std, gpr_model = prediction.predict_gpr(x_valid, method="sklearn", hyperparams_path=hyperparams_path_full, x_train = x_train)
#        y_std_arr.append(y_std)
#    
#    for i, y_std in enumerate(y_std_arr):
#        par1_mesh, par2_mesh = np.meshgrid(par1_val, par2_val)
#        plt.title("freq {}".format(freq[i]))
#        std_arr = np.array(y_std)
#        std_arr=std_arr.reshape(len(par1_val),len(par2_val))
#        plt.pcolormesh(par1_mesh, par2_mesh, std_arr)
#        plt.colorbar(label=r"std")
#        plt.scatter(par1_tr, par2_tr, color='red')
#        #plt.xlabel("Mc")
#        #plt.ylabel("q")
#        plt.xlabel("s1")
#        plt.ylabel("s2")
#        plt.show()
#
#
#def sliceplot(gpr_model):
#    #########################
#    # plot of the GPR prediction for 
#    # only 1 parameter (along x)
#    ########################
#    for i, y_train in enumerate(y_train_s): 
#        hyperparams_path_full = hyperparams_path+str(i)+".pkl"
#        y_predict, y_std, gpr_model = prediction.predict_gpr(x_valid, method="sklearn", hyperparams_path=hyperparams_path_full, x_train = x_train)
#        y_predict_s.append(y_predict)
#        y_std_s.append(y_std)
#        gpr_model_s.append(gpr_model)
#        plt.scatter(x_train, y_train)
#        plt.scatter(x_valid, y_predict) 
#        plt.errorbar(x_valid, y_predict, yerr=y_std, marker="", color="k")
#    plt.show()
#
#
