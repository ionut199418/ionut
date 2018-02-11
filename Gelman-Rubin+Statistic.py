
# coding: utf-8

# In[ ]:


#function R = GelmanRubinR(x, m, chainlng, dim)
# Calculates scale reduction factor for Gelman Rubin test
# Input parameters:
# x: data cell array
# m: no of chains run
# chainlng: chain length
# dim: indicates for which parameter in the cell array R will be calculated
#
# Output parameters
# R: scale reduction factor

# Within Chain Variance
#ssq = NaN(m, 1);
#for j = 1:m
 #   ssq(j) = var(x{j}(dim,:)); 
#end
#Wvar = mean(ssq); %or Wvar = (1/m)*sum(ssq);

#% Between Chain Variance
#schain = 0;
#for j = 1:m
 #   schain = schain + mean(x{j}(dim,:)); % sum of all chain means
#end

#mubar2 = (1/m)*schain;

#bs = 0;
#for j = 1:m
 #   bs = bs + sum((mean(x{j}(dim,:))-mubar2)^2);
#end
#Bvar = (chainlng/(m-1))*bs;

#% Estimated variance
#muvar = (1-1/chainlng)*Wvar + (1/chainlng)*Bvar;

#% Potential Scale Reduction Factor
#R = sqrt(muvar/Wvar); 

#end


# In[ ]:


import random
import math
get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import partial
import statistics
import scipy.stats as stats
from __future__ import division
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import binom
get_ipython().magic('matplotlib inline')
get_ipython().magic('precision 4')
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from scipy.stats import beta
from functools import partial
np.random.seed(1234)


# In[ ]:


def rw2(n):# this gives samples from weibull(5,2)# put n=10000,the shape 5 and the scale 2
    x,y=0,0
    distance=[]
    for i in range(1,n+1):
            r=random.weibullvariate(alpha,beta) # step size r
            theta=2.*math.pi*random.random()
            x +=r*math.cos(theta)
            y +=r*math.sin(theta)
            distance.append(r)# save the distances r for the mcmc sampler
    return(np.array(distance))# depending what do you want to do, either save the pairs, or save the distances r  between points
    #return(x,y)
    
alpha=2#float(input("what's the scale?"))
beta=5#float(input("what's the shape?"))
n=10000#int(input("how many steps?"))
#tries=int(input("how many tries?"))#  put tries=1  in order to properly save the list distance
data = rw2(n)# very important because you need the same data, if you put rw2(n) in there you change the data every time


# In[ ]:


# calculate the log likelihood of weibull(a,b)
import scipy.stats 
from scipy.stats import dweibull
def log_likelihood(x,a,b):
     return sum(dweibull.logpdf(x,scale=b,c=a))


# In[ ]:


import scipy.stats as stats
import numpy as np
#n=10000 #number of observations in a sample 
# initial guess for theta(a and b) as array.
def mh(guess):
    #guess = [5.0,2.0]
# Prepare storing MCMC chain as array of arrays.
    A = [guess]
    # define stepsize of MCMC.
    stepsizes = [0.01,0.01]  # array of stepsizes
    accepted  = 0.0
    old_theta=guess# define initial values for theta 
    old_loglik = log_likelihood(data,old_theta[0],old_theta[1])#calculate the first log likelihood
# Metropolis-Hastings with 10,000 iterations.
    for p in range(10000):
    #old_theta  = A[len(A)-1]   old parameter value as array(not that elegant way)
    # Suggest new candidate from Gaussian proposal distribution.
    #new_theta = np.zeros([len(old_theta)])
        new_theta = old_theta + stats.norm(0, stepsizes).rvs() 
    #add the restraints on new_theta
        if new_theta[0]<0 or new_theta[1]<0: 
            continue
        new_loglik = log_likelihood(data,new_theta[0],new_theta[1])
    # Accept new candidate in Monte-Carlo fashing.
        if (new_loglik > old_loglik):
            A.append(new_theta)
            accepted = accepted + 1.0  # monitor acceptance
            old_loglik=new_loglik
            old_theta=new_theta
        else:
            u = random.uniform(0.0,1.0)
            if (u < math.exp(new_loglik - old_loglik)):
                A.append(new_theta)
                accepted = accepted + 1.0  # monitor acceptance
                old_loglik=new_loglik
                old_theta=new_theta
            else:
                A.append(old_theta)
    print("Acceptance rate = "+str(accepted/10000.0))
    return A


# In[ ]:


sampless_shape = [mh([theta,2]) for theta in np.linspace(2,7,5)]# 5 mh chains for shape keeping the scale fixed at 2 and varying 
# the starting points for the shape from 2 to 7


# In[ ]:


sampless_scale=[mh([5,theta]) for theta in np.linspace(1,6,5)]
#5 mh chains for scale keeping the shape fixed at 5 and varying the starting points from 1 to 3


# In[ ]:


#sampless_shape
final_shape_vector=[[p[0] for p in chain] for chain in sampless_shape]


# In[ ]:


final_scale_vector=[[p[1] for p in chain] for chain in sampless_scale]
#final_scale_vector

