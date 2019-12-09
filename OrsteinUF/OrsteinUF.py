#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#OUF Kernel


import numpy as np
from scipy import integrate
from .kern import Kern
from ...core.parameterization import Param
from ...util.linalg import tdot
from ... import util
from ...util.config import config # for assesing whether to use cythons
from paramz.caching import Cache_this
from paramz.transformations import Logexp
from .stationary import Stationary
from .psi_comp import PSICOMP_RBF, PSICOMP_RBF_GPU
from ...core import Param

class OrsteinUF(Kern):
    
    def __init__(self,input_dim,timescale_H=2.,timescale_F=1.,sigma_a=1.,active_dims=None):
        super(OrsteinUF, self).__init__(input_dim, active_dims, 'OUF')
        assert input_dim == 1 #, "For this kernel we assume input_dim=1"
        self.timescale_H = Param('timescale_H', timescale_H,Logexp())
        self.timescale_F = Param('timescale_F', timescale_F,Logexp())
        self.sigma_a = Param('sigma_a', sigma_a,Logexp())
        self.link_parameters(self.timescale_H,self.timescale_F,self.sigma_a)
   
    def K(self, X, X2=None):
        r_H = self._scaled_dist_H(X, X2)
        r_F = self._scaled_dist_F(X, X2)
        if np.abs(self.timescale_H - self.timescale_F)<1e-3:
            self.timescale_H = self.timescale_H + 0.01
        Ke = 1/(self.timescale_H - self.timescale_F ) * (self.timescale_H * np.exp(-r_H) - self.timescale_F * np.exp(-r_F))
        return 0.5* self.sigma_a * (self.timescale_H**2) * (self.timescale_F**2) * 1/(self.timescale_H + self.timescale_F)* Ke
            
    def _unscaled_dist(self, X, X2=None):
       # """
        #Compute the Euclidean distance between each row of X and X2, or between
        #each pair of rows of X if X2 is None.
        #"""
        #X, = self._slice_X(X)
        if X2 is None:
            Xsq = np.sum(np.square(X),1)
            r2 = -2.*tdot(X) + (Xsq[:,None] + Xsq[None,:])
            util.diag.view(r2)[:,]= 0. # force diagnoal to be zero: sometime numerically a little negative
            r2 = np.clip(r2, 0, np.inf)
            return np.sqrt(r2)
        else:
            #X2, = self._slice_X(X2)
            X1sq = np.sum(np.square(X),1)
            X2sq = np.sum(np.square(X2),1)
            r2 = -2.*np.dot(X, X2.T) + (X1sq[:,None] + X2sq[None,:]) # this forms the square (X-X2)^2, .T to do the multiplication
            r2 = np.clip(r2, 0, np.inf) # this makes sure we don't get any values that are inf, i.e. values are between 0 and inf
            return np.sqrt(r2) # take the sqrt to get the modulus of the difference 
      
# Two different timescales, i.e tau_H and tau_F.

    def _scaled_dist_H(self, X, X2=None):
         return self._unscaled_dist(X, X2)/self.timescale_H
           

    def _scaled_dist_F(self, X, X2=None):
         return self._unscaled_dist(X, X2)/self.timescale_F


    def Kdiag(self, X):
        ret = np.empty(X.shape[0])
        if np.abs(self.timescale_H - self.timescale_F)<1e-3:
            self.timescale_H = self.timescale_H + 0.01
        ret[:] = 0.5* self.sigma_a * (self.timescale_H**2) * (self.timescale_F**2) * 1/(self.timescale_H + self.timescale_F)        
        return ret
   
    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: 
            X2 = X
        r_H = self._scaled_dist_H(X, X2)
        r_F = self._scaled_dist_F(X, X2)
        Ke = 1/(self.timescale_H - self.timescale_F) * (self.timescale_H * np.exp(-r_H) - self.timescale_F * np.exp(-r_F))
        f = self.timescale_H * np.exp(-r_H) - self.timescale_F * np.exp(-r_F)
        g = 0.5*self.sigma_a*(self.timescale_H**2) * (self.timescale_F**2)* 1/(self.timescale_H + self.timescale_F)
        dsigma = 0.5*(self.timescale_H**2) * (self.timescale_F**2)* 1/(self.timescale_H + self.timescale_F )* Ke
        Ke_prim_H = ( np.exp(-r_H) *(1+r_H) * (self.timescale_H -self.timescale_F) - f) / (self.timescale_H -self.timescale_F)**2   
        Ke_prim_F =  (-np.exp(-r_F) *(1+r_F) * (self.timescale_H -self.timescale_F) + f) / (self.timescale_H -self.timescale_F)**2
        dtimescaleH =  0.5 * self.sigma_a * (self.timescale_F**2)* (self.timescale_H**2 + 2*self.timescale_F*self.timescale_H)/(self.timescale_H + self.timescale_F)**2 * Ke + Ke_prim_H * g
        dtimescaleF = 0.5 * self.sigma_a * (self.timescale_H**2)* (self.timescale_F**2 + 2*self.timescale_F*self.timescale_H)/((self.timescale_H + self.timescale_F)**2) * Ke + Ke_prim_F * g  
        self.sigma_a.gradient = np.sum(dsigma*dL_dK)
        self.timescale_H.gradient = np.sum(dtimescaleH*dL_dK)
        self.timescale_F.gradient = np.sum(dtimescaleF*dL_dK)

   