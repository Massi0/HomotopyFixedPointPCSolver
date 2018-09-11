#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
# 
# Copyright (c) 2018  Massi Amrouche  <amrouch2@illinois.edu>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
#title           : PredictorCorrectorImpl_.py
#date created    : 2018/09
#notes           :
__author__ = "Massi Amrouche"
__license__ = "GPLv3"
__version__ = "0.0.1"
__maintainer__ = "Massi Amrouche"
__email__ = "amrouch2@illinois.edu"

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

#Headers 
from  HomPCSolver import linalg,np,abc


class PredictorCorrectorSolver(object):

    def __init__(self,ndim,epsPred=1e-3,epsCorr=1e-3, maxSteps=1000):
        self.ndim = ndim
        self.maxSteps = maxSteps
        self.epsPred = epsPred
        self.epsCorr = epsCorr
        self.hStep = self.epsPred *1e1
        self.AbsTol = self.epsPred*1e-1

    @abc.abstractmethod
    def _F(self,x):
        "Define the Function here. returns an array"

    @abc.abstractmethod
    def _DF(self,x):
        "Definition of the Jacobian of the function here. return an array"
    @abc.abstractmethod
    def _stopCriteria(self,x):
        "Definition of the stoping criteria. returns a bool"
        
    def _predictorSteps(self,y0):
        
        err = 0
        d = y0
        while err<self.epsPred:
            F = self._F(y0)
            dF = self._DF(y0)            
            d = linalg.null_space(dF) #kernel of the Jacobian
            #d = d/linalg.norm(d,2) #Normalization of the kernel
            #if d[2,0]<0:
            #    d = -d
            ynext = y0 + self.hStep * d
            err = linalg.norm(F,2)
            y0 = ynext
        
        return (y0,np.transpose(d)) # return the prediction step and a vector orthogonal to the kernel 

    def _correctorSteps(self,y0,b):
        err = 10
       
        while err>self.epsCorr:
            F = self._F(y0)
            dF = self._DF(y0)
            A = np.append(dF,b,axis=0)
            B = A.dot(y0) - np.append(F,[[0]],axis=0)            
            ynew = linalg.solve(A,B)
            err = linalg.norm(F,2)
            y0 = ynew

        return (y0,err)

    def _refineNearEnd(self,t):
        if t>.94:
            self.hStep = self.hStep /10.
            self._isRefined = True
        return
    
    def _solverPC(self,y0):
        i_iter = 0
        self._isRefined = False
        while (not self._stopCriteria(y0)) and (i_iter<self.maxSteps):
            (ypred,b) = self._predictorSteps(y0)
            (ypred,errH) = self._correctorSteps(ypred,b)
            i_iter += 1

            y0 = ypred
            self._refineNearEnd(y0[-1])
            print i_iter,errH,self.hStep,y0[-1]

        return (y0,i_iter)

           
    def _printTest(self,x):
        return self._F(x)
        
        
