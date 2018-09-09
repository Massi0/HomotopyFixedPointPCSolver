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
        self.hStep = self.epsPred *1e-2
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
        
        err = 1

        while err>self.epsPred:
            dF = self._DF(y0)
            
            d = linalg.null_space(dF) #kernel of the Jacobian
            #d = d/linalg.norm(d,2) #Normalization of the kernel
            
            #if d[2,0]<0:
            #    d = -d
            ynext = y0 + self.hStep * d
            F = self._F(ynext)
            err = linalg.norm(F,2)
            y0 = ynext
        
        return (y0,np.transpose(d)) # return the prediction step and a vector orthogonal to the kernel 

    def _correctorSteps(self,y0,b):
        err = 1
       
        while err>self.epsCorr:
            dF = self._DF(y0)
            F = self._F(y0)
            A = np.append(dF,b,axis=0)
            B = A.dot(y0) - np.append(F,[[0]],axis=0)     
            ynew = linalg.solve(A,B)

            err = linalg.norm(F,2)
            y0 = ynew

        return (y0,err)
            
    def _solverPC(self,y0):
        i_iter = 0
        
        while (not self._stopCriteria(y0)) and (i_iter<self.maxSteps):
            (ypred,b) = self._predictorSteps(y0)
            (ypred,errH) = self._correctorSteps(ypred,b)
            i_iter += 1
            if (linalg.norm(ypred-y0)<self.AbsTol) and (1-ypred[2,0]<.8):
                self.hStep = self.hStep*2
            else:
                self.hStep = self.epsPred

            y0 = ypred
            print i_iter,errH,self.hStep,y0[2,0]

        return (y0,i_iter)

           
    def _printTest(self,x):
        return self._F(x)
        
        
