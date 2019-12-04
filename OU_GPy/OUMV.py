#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#OUF Kernel

# make sure to align 

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

class OUMV(Kern):
    
    def __init__(self,input_dim,variance=1.,lengthscale=1.,active_dims=None):
        super(OUMV, self).__init__(input_dim, active_dims, 'OUMV')
        assert input_dim == 1 #, "For this kernel we assume input_dim=1"
        self.variance = Param('variance', variance,Logexp())
        self.lengthscale = Param('lengthscale', lengthscale,Logexp())
        self.link_parameters(self.variance, self.lengthscale)


    def K(self, X, X2=None):
        r_l = self._scaled_dist_l(X, X2)
        Ke = self.variance*np.exp(-r_l)
        return Ke
       
    
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

    def _scaled_dist_l(self, X, X2=None):
         return self._unscaled_dist(X, X2)/self.lengthscale 

    def Kdiag(self, X):
        ret = np.empty(X.shape[0])
        ret[:] = self.variance 
        return ret

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: 
            X2 = X
        r_l = self._scaled_dist_l(X, X2)
        dvar = np.exp(-r_l)
        dl = self.variance * np.exp(-r_l) *r_l/self.lengthscale
        self.variance.gradient = np.sum(dvar*dL_dK)
        self.lengthscale.gradient = np.sum(dl*dL_dK)

