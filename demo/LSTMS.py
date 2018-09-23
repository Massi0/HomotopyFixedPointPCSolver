from demo import HomotopyPCSolver,linalg,np
import pickle as pkl
from os import path

class LSTMNN(HomotopyPCSolver):
    def __init__(self,pickleFN=None,epsPred = 1e-7,epsCorr = 1e-9, maxSteps = 1e5):
        #Default values
        self.w = {'Wo':[],'Wf':[],'Wi':[],'Wg':[],
                  'Wo_x':[],'Wf_x':[],'Wi_x':[],'Wg_x':[],
                  'bo':[],'bf':[],'bo':[],'bg':[]}
        x0 = []

        self.ndim = ndim = 0
        if pickleFN != None:
            ncells = self.loadFromPickle(pickleFN)
            self.ndim = ndim = 2*ncells
            self.ncells = ncells
            self.xinit = x0 = np.zeros([ndim,1])
            self._jac = np.zeros([ndim,ndim])
            self._hout = x0[0:ncells]
            self._cell = x0[ncells:2*ncells]
        
        self.maxSteps = maxSteps
        self.epsPred = epsPred
        self.epsCorr = epsCorr
        
        super(LSTMNN,self).__init__(x0,ndim,epsPred,epsCorr,maxSteps)

    
    def setInitStat(self,x0):
        self.xinit = x0
        ncells = self.ncells
        
        self._hout = self.xinit[0:ncells]
        self._cell = self.xinit[ncells:2*ncells]
        return


    def updateWeigths(self,w):
        hl = ["Wi","Wi_x",
              "Wf","Wf_x",
              "Wg","Wg_x",
              "Wo","Wo_x",
              "bo","bi",
              "bf","bg"]

        for it in hl:
            self.w[it] = w[it]

        ncells = w["bf"].shape[0]
        self.ndim = ndim = 2*ncells
        self.ncells = ncells
        self._jac = x0 = np.zeros([ndim,ndim])
        self._hout = x0[0:ncells]
        self._cell = x0[ncells:2*ncells]

        super(LSTMNN,self).__init__(x0,ndim,self.epsPred,self.epsCorr,self.maxSteps)
        return ncells
        
    def loadFromPickle(self,filename):
        if not path.isfile(filename):
            return -1
        w = pkl.load(open(filename,'rb'))
        wcc = w[0]
        bcc = w[1]
        b_i, b_C, b_f, b_o = np.split(bcc, 4, axis=0)
        w_i, w_C, w_f, w_o = np.split(wcc, 4, axis=1)

        input_size = w_i.shape[0]-b_i.shape[0]
        ncells = b_i.shape[0]
        
        self.w['Wi_x'] = w_i[:input_size, :].T
        self.w['Wi'] = w_i[input_size:, :].T
        
        self.w['Wg_x'] = w_C[:input_size, :].T
        self.w['Wg'] = w_C[input_size:, :].T
        
        self.w['Wf_x']  = w_f[:input_size, :].T
        self.w['Wf'] =  w_f[input_size:, :].T
        
        self.w['Wo_x']  = w_o[:input_size, :].T
        self.w['Wo'] = w_o[input_size:, :].T

        self.w['bi'] = np.array([b_i]).T
        self.w['bo'] = np.array([b_o]).T
        self.w['bf'] = np.array([b_f]).T
        self.w['bg'] = np.array([b_C]).T       

        print "Parameters loaded from "+filename+" | dim(h) = "+ str(ncells) + " | dim(x) = " + str(input_size)
        
        return ncells

    def makeOutputsAsInputs(self):
        hl = [["Wi","Wi_x"],
              ["Wf","Wf_x"],
              ["Wg","Wg_x"],
              ["Wo","Wo_x"]]

        for it in hl:
            self.w[it[0]] = self.w[it[0]]+self.w[it[1]]
        return True
            
    def dimOfOutput(self):
        return self.ncells
    
    def _phi(self,x):
        return 1./(np.exp(-x)+1)
    def _dphi(self,x):
        tmp = self._phi(x)
        return tmp*(1-tmp)
    def _dtanh(self,x):
        return 1-self._tanh(x)**2
    def _tanh(self,x):
        return np.tanh(x)

        
    def _update_units(self,h,w):
        #Definition of the LSTM units + derivatives
        self.o =  self._phi(w['Wo'].dot(h)+w['bo'])
        self.do = w['Wo']*self._dphi(w['Wo'].dot(h)+w['bo'])
        
        self.f =  self._phi(w['Wf'].dot(h)+w['bf'])
        self.df = w['Wf']*self._dphi(w['Wf'].dot(h)+w['bf'])
        
        self.i =  self._phi(w['Wi'].dot(h)+w['bi'])
        self.di = w['Wi']*self._dphi(w['Wi'].dot(h)+w['bi'])

        self.g =  self._tanh(w['Wg'].dot(h)+w['bg'])

    
        self.dg = self.w['Wg']*self._dtanh(self.w['Wg'].dot(h)+self.w['bg'])
        
        return 

    def _mapLstm(self,x,w):
        ncells = self.ncells
        h = x[0:ncells]
        c = x[ncells:2*ncells]
        self._update_units(h,w)
        
        G = self.f*c+self.i*self.g
        F = self.o*self._tanh(G)
        
        return (np.append(F,G,axis=0))


    def _jacLstm(self,x,w):
        nc = self.ncells
        h = x[0:nc]
        c = x[nc:2*nc]
        self._update_units(h,w)
        
        G = self.f*c+self.i*self.g
        F = self.o*self._tanh(G)
                
        C = self.df*c+ self.di*self.g + self.dg*self.i
        #D = np.diag(self.f)
        K = self.o*(1-self._tanh(G)**2)
        #FE = self.do*(1-self._tanh(G)**2)
        self._jac[0:nc,0:nc] = self.do*self._tanh(G)+C*K
        self._jac[0:nc,nc:2*nc] = np.diag(self.f*K)
        self._jac[nc:2*nc,0:nc] = C
        self._jac[nc:2*nc,nc:2*nc] = np.diag(self.f)

        return self._jac                
                
    def F(self,x):
        return self._mapLstm(x,self.w)
        

    def DF(self,x):
        return self._jacLstm(x,self.w)

    def F0(self,x):
        w = self.w['bg']
        self.w['bg'] = self.w['bg']*0
        tmp = self._mapLstm(x,self.w)
        self.w['bg'] = w
        return tmp
    def DF0(self,x):
        w = self.w['bg']
        self.w['bg'] = w*0
        tmp = self._jacLstm(x,self.w)
        self.w['bg'] = w
        return tmp
#*********************************************************************
#Uncomment the following lines to use the Newton-like homotopy (tailored for LSTMs). Otherwise by default the solver use the linear homotopy inherated from the HomotopySolver
#    def lstmHomotopy(self,x,t):
        #H(x,t) = t(F(x)-x)+(1-t)F0(x)
#        return t*(self.F(x)-x)+(1-t)*(self.F0(x)-x)

#    def jacLstmHomotopy(self,x,t):
        #dH(x,t) = t(dF(x)-I)+(1-t)dF0(x)
#        return np.append(t*(self.DF(x)-self.Id)+(1-t)*(self.DF0(x)-self.Id),self.F(x)-self.F0(x),axis=1)

#    def _PC_F(self,y):
#        return self.lstmHomotopy(y[0:self.ndimA-1],y[self.ndimA-1])
    
#    def _PC_DF(self,y):
#        return self.jacLstmHomotopy(y[0:self.ndimA-1],y[self.ndimA-1])
#******************************************************************

    def findEquilibria(self,epsPred = 1e-7,epsCorr = 1e-9, maxSteps = 1e5):
        self.maxSteps = maxSteps
        self.epsPred = epsPred
        self.epsCorr = epsCorr
        self.hStep = np.sqrt(epsPred)
        
        (y,errCode) = self.runSolver()
        xeq = y[:self.ndim+1]
        t = y[-1]
        return (t,xeq,errCode)
