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
from HomPCSolver import np
import pdb


class PolyTest(HomotopyPCSolver):
    def __init__(self,xinit,ndim,epsPred=1e-3,epsCorr=1e-3, maxSteps=1000):
        super(PolyTest, self).__init__(xinit,ndim,epsPred,epsCorr, maxSteps)
    def F(self,x):
        #x = x.tolist()
        return np.array([[(x[0,0]**2+x[1,0]**2)/2-1],
                         [x[1,0]**3/3+x[0,0]**2/2-2]])

    def DF(self,x):
        return np.array([[x[0,0]+x[1,0]],
                         [x[1,0]**2+x[0,0] ]])



if __name__ == '__main__':
    #def run():
    x0 = np.array([[1],
                   [.5]])                        
    PT = PolyTest(x0,2,1e-3,1e-4,1e9)
    (y,errCode) = PT.runSolver()
    print errCode,y
