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
#title           : HomotopySolverImpl_.py
#date created    : 2018/09
#notes           :
__author__ = "Massi Amrouche"
__license__ = "GPLv3"
__version__ = "0.0.1"
__maintainer__ = "Massi Amrouche"
__email__ = "amrouch2@illinois.edu"

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

#Headers
from  HomPCSolver import linalg, np, abc, PredictorCorrectorSolver

class HomotopyPCSolver(PredictorCorrectorSolver):
    def __init__(self,xinit,ndim,epsPred=1e-3,epsCorr=1e-3, maxSteps=1000):
        super(HomotopyPCSolver, self).__init__(ndim,epsPred,epsCorr,maxSteps)
        self.xinit = xinit
        self.Id = np.identity(ndim)
        self.ndimA = ndim+1
        self.yeq = np.zeros([ndim+1,1]) #vec of dim n+1
        self.tfinal = -1
    @abc.abstractmethod
    def F(self,x):
        "Define the Function here. returns an array"

    @abc.abstractmethod
    def DF(self,x):
        "Definition of the Jacobian of the function here. return an array"

    def _linearHomotopy(self,x,t):
        #H(x,t) = (1-t)(x-x0)+t(x-F(x))
        #       = (x-x0)+t(x0-F(x))
        return (x-self.xinit)+t*(self.xinit-self.F(x))
    def _JacLinearHomotopy(self,x,t):
        # dH(x,t) = [dH/dx dH/dt]
        #         = [I-t*dF(x) x0-F(x) ]

        return np.append(self.Id-t*self.DF(x),self.xinit-self.F(x),axis=1)

    def _F(self,y):
        """Definition of the Homotopy function"""
        return self._linearHomotopy(y[0:self.ndimA-1],y[self.ndimA-1])
    
    def _DF(self,y):
        """Definition of the Jacobian of the Homotopy"""
        return self._JacLinearHomotopy(y[0:self.ndimA-1],y[self.ndimA-1])

    
    def _stopCriteria(self,y0):
        """Implementaion of the Stopping criteria"""
        return y0[self.ndimA-1]>1

    def isValidEquilibrium(self,ErrTol):
        y = self.yeq
        y[self.ndimA-1] = 1.
        return linalg.norm(self._F(y),2)<ErrTol

    def runSolver(self):
        (y,i_iter) = self._solverPC(np.append(self.xinit,[[0]],axis=0))
        errCode = 0
        if (i_iter>self.maxSteps):
            errCode |= 0b001
        if ~self.isValidEquilibrium(1e-2):
            errCode |= 0b010
        self.yeq = y
        self.tfinal = y[-1]

        return (y,errCode)

    def errorInterpreter(self,errCode):
        header = "HomotopySolver::"
        if errCode & 0b001:
            print header,"Max number of iteration reached"
        if errCode>>1 &0b001:
            y = self.yeq
            y[self.ndimA-1] = 1
            print header,"Not valid equilibrium::H(x,1)=",linalg.norm(self._F(y),2)
            
        

    
        
