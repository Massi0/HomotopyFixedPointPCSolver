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

def readFromPickle(ww):
    
    wcc = ww[0]
    bcc = ww[1]
    
    b_i, b_C, b_f, b_o = np.split(bcc, 4, axis=0)
    w_i, w_C, w_f, w_o = np.split(wcc, 4, axis=1)
    
    input_size = w_i.shape[0]-b_i.shape[0]
    
    dim = len(b_o)
    w_xi = w_i[:input_size, :]
    w_hi = w_i[input_size:, :]
    
    w_xC = w_C[:input_size, :]
    w_hC = w_C[input_size:, :]
    
    w_xf = w_f[:input_size, :]
    w_hf = w_f[input_size:, :]
    
    w_xo = w_o[:input_size, :]
    w_ho = w_o[input_size:, :]
                
    w = {'Wo':[],'Wf':[],'Wi':[],'Wg':[],
         'Wo_x':[],'Wf_x':[],'Wi_x':[],'Wg_x':[],
         'bo':[],'bf':[],'bo':[],'bg':[]}
        
    w['Wi_x'] = w_i[:input_size, :].T
    w['Wi'] = w_i[input_size:, :].T
    
    w['Wg_x'] = w_C[:input_size, :].T
    w['Wg'] = w_C[input_size:, :].T
    
    w['Wf_x']  = w_f[:input_size, :].T
    w['Wf'] =  w_f[input_size:, :].T
        
    w['Wo_x']  = w_o[:input_size, :].T
    w['Wo'] = w_o[input_size:, :].T
        
    w['bi'] = np.array([b_i]).T
    w['bo'] = np.array([b_o]).T
    w['bf'] = np.array([b_f]).T
    w['bg'] = np.array([b_C]).T       
        
    return w



if __name__ == '__main__':
    #test 1: Demo: root of polynomials
    #test 2: Demo: Equilibria of LSTM Neural network tested on Penn Tree Bank (PTB) dataset
    #test 3: Demo: Equilibria of LSTM NN during the training tested on PTB dataset
    test = 3
    if test == 1:
        class PolyTest(HomotopyPCSolver):
            def __init__(self,xinit,ndim,epsPred=1e-3,epsCorr=1e-3, hStep=1e-3, maxSteps=1000):
                super(PolyTest, self).__init__(xinit,ndim,epsPred,epsCorr,hStep, maxSteps)
            def F(self,x):
                #x = x.tolist()
                return np.array([[(x[0,0]**2/3.+2.*x[1,0])],
                                 [x[1,0]**3/3+x[0,0]**2/2-2]])

            def DF(self,x):
                return np.array([[2.*x[0,0]/3., 2.],
                                 [x[0,0], x[1,0]**2 ]])



        x0 = np.array([[0],
                       [0]])
        hStep = 1e-3
        epsPred = 1e-5
        epsCorr = 1e-9
        ndim = 2
        maxSteps = 1e4
        PT = PolyTest(x0,ndim,epsPred,epsCorr,hStep,maxSteps)
        (y,errCode) = PT.runSolver()
        print errCode
        print y
        PT.errorInterpreter(errCode)
        if ~(errCode>>1 &0b010):
            y[-1] = 1
            print "Valid equilibrium::H(x,1)=",linalg.norm(PT._PC_F(y),2)

    
    if test == 2:
        data_input_file = 'demo/data/PTB_0/save_10v2.pkl'
        
        epsPred = 1e-3
        epsCorr = 1e-9
        maxSteps = 1e5

        lstm = LSTMNN(data_input_file)
        #lstm.makeOutputsAsInputs()
        ndim = 2*lstm.dimOfOutput()
        #x0 = np.random.uniform(-1,1,[ndim,1])
        x0 = np.zeros([ndim,1])
        lstm.setInitStat(x0)
        (t,x,eCode) = lstm.findEquilibria(epsPred,epsCorr,maxSteps)
        xeq = lstm.refineTheEquilibria(x)
        xeq = np.array([xeq]).T
        lstm.errorInterpreter(eCode)
        print "||F(xeq)-xeq|| = ",linalg.norm(xeq-lstm.F(xeq),2)
        
    if test == 3:
        data_input_file = 'demo/data/PTB_overtime/weights_PTB_overtimev2.pkl'

        epsPred = 1e-3
        epsCorr = 1e-9
        maxSteps = 1e5

        lstm = LSTMNN(None)

        paramOverEpochs = pkl.load(open(data_input_file,'rb'))[0]
        
        eq = []
        i_step = 1
        for w in paramOverEpochs:
            weights = readFromPickle(w)
            lstm.updateWeigths(weights)
            ndim = 2*lstm.dimOfOutput()
            #x0 = np.random.uniform(-1,1,[ndim,1])
            x0 = np.zeros([ndim,1])
            lstm.setInitStat(x0)
            print i_step, ": dim = ",str(ndim)
            (t,x,eCode) = lstm.findEquilibria(epsPred,epsCorr,maxSteps)
            xeq = lstm.refineTheEquilibria(x)
            xeq = np.array([xeq]).T
            lstm.errorInterpreter(eCode)
            print i_step, ": ||F(xeq)-xeq|| = ",linalg.norm(xeq-lstm.F(xeq),2)
            i_step +=1
            eq.append(xeq)
            
