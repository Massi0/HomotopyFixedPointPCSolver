from demo import HomotopyPCSolver,linalg,np
import pickle as pkl
from os import path

class LSTMNN(HomotopyPCSolver):
    def __init__(self,ncells):
        #Default values
        self.ndim = ndim = 2*ncells
        self.ncells = ncells
        self.xinit = x0 = np.zeros(ndim)
        self.epsPred = epsPred = 1e-3
        self.epsCorr = epsCorr = 1e-7
        self.maxSteps = maxSteps = 1e5
        self.w = {'Wo':[],'Wf':[],'Wi':[],'Wg':[],
                  'Wo_x':[],'Wf_x':[],'Wi_x':[],'Wg_x':[],
                  'bo':[],'bf':[],'bo':[],'bg':[]}
        self._jac = np.zeros([ndim,ndim])
        self._hout = x0[0:ncells]
        self._cell = x0[ncells:2*ncells]
        super(LSTMNN,self).__init__(x0,ndim,epsPred,epsCorr,maxSteps)

    def _initLSTM(self):
        ncells = self.ncells
        self.ndim = ndim = 2*ncells
        self._jac = np.zeros([ndim,ndim])
        self._hout = self.xinit[0:ncells]
        self._cell = self.xinit[ncells:2*ncells]

        return

    
    def _phi(self,x):
        return 1./(np.exp(-x)+1)
    def _dphi(self,x):
        tmp = self._phi(x)
        return tmp*(1-tmp)
    def _dtanh(self,x):
        return 1-self._tanh(x)**2
    def _tanh(self,x):
        return np.tanh(x)

    def loadFromPickle(self,filename):
        print "in"
        if not path.isfile(filename):
            return -1
        w = pkl.load(open(filename,'rb'))
        wcc = w[0]
        bcc = w[1]
        print ' fff'
        b_i, b_C, b_f, b_o = np.split(bcc, 4, axis=0)
        w_i, w_C, w_f, w_o = np.split(wcc, 4, axis=1)

        input_size = w_i.shape[0]-b_i.shape[0]
        self.ncells = b_i.shape[0]
        
        self.w['Wi_x'] = w_i[:input_size, :]
        self.w['Wi'] = w_i[input_size:, :]
        
        self.w['Wg_x'] = w_C[:input_size, :]
        self.w['Wg'] = w_C[input_size:, :]
        
        self.w['Wf_x']  = w_f[:input_size, :]
        self.w['Wf'] =  w_f[input_size:, :]
        
        self.w['Wo_x']  = w_o[:input_size, :]
        self.w['Wo'] = w_o[input_size:, :]

        self.w['bi'] = b_i
        self.w['o'] = b_o
        self.w['bf'] = b_f
        self.w['bg'] = b_C

        self._initLSTM()
        return self.ndim

    
    def setInitStat(self,x0):
        self.xinit = x0
        self._initLSTM()
        return
    
    def _update_units(self,h):
        #Definition of the LSTM units + derivatives
        self.o =  self._phi(self.w['Wo'].dot(h)+self.w['bo'])
        self.do = self.w['Wo']*self._dphi(self.w['Wo'].dot(h)+self.w['bo'])
        
        self.f =  self._phi(self.w['Wf'].dot(h)+self.w['bf'])
        self.df = self.w['Wf']*self._dphi(self.w['Wf'].dot(h)+self.w['bf'])
        
        self.i =  self._phi(self.w['Wi'].dot(h)+self.w['bi'])
        self.di = self.w['Wi']*self._dphi(self.w['Wi'].dot(h)+self.w['bi'])

        self.g =  self._tanh(self.w['Wg'].dot(h)+self.w['bg'])
        self.dg = self.w['Wg']*self._dtanh(self.w['Wg'].dot(h)+self.w['bg'])
        
        return 

    def _map_jac_update(self,option=0):
        #option = 1 : update/returns the map and its jacobian
        #option = 2: update/returns the map
        #option = 3: update/returns the jacobian
        #else wont do anything

        if not (option|0b11):
            return

        nc = self.ncells
        h = self._hout
        c = self._cell
        self._update_units(h)
        
        G = self.f*c+self.i*self.g
        F = self.o*self._tanh(G)


        if option&0b10:
            self._hout = F
            self._cell = G

        if option&0b11:
            C = self.df*c+ self.di*self.g + self.dg*self.i
            #D = np.diag(self.f)
            K = self.o*(1-self._tanh(G)**2)
            #FE = self.do*(1-self._tanh(G)**2)
            self._jac[0:nc,0:nc] = self.do*self._tanh(G)+C*K
            self._jac[0:nc,nc:2*nc] = np.diag(self.f*K)
            self._jac[nc:2*nc,0:nc] = C
            self._jac[nc:2*nc,nc:2*nc] = np.diag(self.f)

        return (np.append(self._hout,self._cell,axis=0),self._jac)

        
        
