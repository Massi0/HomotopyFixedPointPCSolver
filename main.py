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

from HomPCSolver import HomotopyPCSolver 
from HomPCSolver import np,linalg
from demo import pkl,path,LSTMNN
import pdb



class PolyTest(HomotopyPCSolver):
    def __init__(self,xinit,ndim,epsPred=1e-3,epsCorr=1e-3, maxSteps=1000):
        super(PolyTest, self).__init__(xinit,ndim,epsPred,epsCorr, maxSteps)
    def F(self,x):
        #x = x.tolist()
        return np.array([[(x[0,0]**2/3.+2.*x[1,0])],
                         [x[1,0]**3/3+x[0,0]**2/2-2]])

    def DF(self,x):
        return np.array([[2.*x[0,0]/3., 2.],
                         [x[0,0], x[1,0]**2 ]])


if __name__ == '__main__':

    test = 2
    if test == 1:
        x0 = np.array([[0],
                       [0]])
        epsPred = 1e-3
        epsCorr = 1e-9
        ndim = 2
        maxSteps = 1e4
        PT = PolyTest(x0,ndim,epsPred,epsCorr,maxSteps)
        (y,errCode) = PT.runSolver()
        print errCode
        print y
        PT.errorInterpreter(errCode)
        if ~(errCode>>1 &0b010):
            y[-1] = 1
            print "Valid equilibrium::H(x,1)=",linalg.norm(PT._F(y),2)


    if test == 2:
        data_input_file = 'demo/data/save_10v2.pkl'
        input_size = 200

        lstm = LSTMNN(data_input_file)
        ndim = 2*lstm.dimOfOutput()
        x0 = np.random.uniform(-1,1,[ndim,1])
        x0 = np.zeros([ndim,1])
        lstm.setInitStat(x0)
        (t,x,eCode) = lstm.findEquilibria()
        lstm.errorInterpreter(errCode)
        
        
       
