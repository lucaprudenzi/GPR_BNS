import sys
sys.path.append('../')
import gpr_sklearn 
import numpy as np

class wf_correction(object):
    def __init__(
            self, 
            data_path_train_damp, 
            data_path_validation_damp,
            model_path_damp,
            data_path_train_dphase, 
            data_path_validation_dphase,
            model_path_dphase):
        
        # damp 
        self.gpr_damp = gpr_sklearn.GPR_sklearn(data_path_train_damp, data_path_validation_damp, model_path_damp)
        self.gpr_damp.setup_data_sklearn()
        
        # dphase 
        self.gpr_dphase = gpr_sklearn.GPR_sklearn(data_path_train_dphase, data_path_validation_dphase, model_path_dphase)
        self.gpr_dphase.setup_data_sklearn()
    
    def draw_sample_damp_dphase(self, m1, m2, s1z, s2z, lambda1, lambda2):
        
        eta = (m1*m2)/(m1+m2)**2
        q = m1/m2
        Mc = (m1*m2)**(3./5)/(m1+m2)**(1./5)
        lambda1_red = np.log10(lambda1/100+1)
        lambda2_red = np.log10(lambda2/100+1)
        lambda_tilde = 16./13*(\
           (m1+12*m2)*m1**4*lambda1+\
           (m2+12*m1)*m2**4*lambda2)/\
           (m1+m2)**5
        lambda_tilde_red = np.log10(lambda_tilde/100+1)
        chi_eff=(m1*s1z+m2*s2z)/(m1+m2)

        par = np.atleast_2d([Mc, eta, lambda_tilde_red, chi_eff])
        
        # Phase 
        self.gpr_damp.test(par)
        damp = self.gpr_damp.y_test
        
        self.gpr_dphase.test(par)
        dphase = self.gpr_dphase.y_test
        freqs = self.gpr_damp.freq

        return damp, dphase, freqs
