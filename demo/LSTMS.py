from demo import *



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
                  'bo':[],'bf':[],'bo':[],'bg':[]}
        self._jac = np.zeros([ndim,ndim])
        super(LSTMNN,self).__init__(x0,ndim,epsPred,epsCorr,maxSteps)

    def _phi(self,x):
        return 1./(np.exp(-x)+1)
    def _dphi(self,x):
        tmp = self._phi(x)
        return tmp*(1-tmp)
    def _dtanh(self,x):
        return 1-self._tanh(x)**2
    def _tanh(self,x):
        return np.tanh(x)

    def _update_units(self,h):
        #Definition of the LSTM units + derivatives
        self.o =  self._phi(self.w('Wo').dot(h)+self.w('bo'))
        self.do = self.w('Wo')*self._dphi(self.w('Wo').dot(h)+self.w('bo'))
        
        self.f =  self._phi(self.w('Wf').dot(h)+self.w('bf'))
        self.df = self.w('Wf')*self._dphi(self.w('Wf').dot(h)+self.w('bf'))
        
        self.i =  self._phi(self.w('Wi').dot(h)+self.w('bi'))
        self.di = self.w('Wi')*self._dphi(self.w('Wi').dot(h)+self.w('bi'))

        self.g =  self._tanh(self.w('Wg').dot(h)+self.w('bg'))
        self.dg = self.w('Wg')*self._dtanh(self.w('Wg').dot(h)+self.w('bg'))
        
        return 

    def _func(self,h,c):
        nc = self.ncells
        self._update_units(h)
        G = self.f*c+self.i*self.g
        F = self.o*self._tanh(G)
        C = self.df*c+ self.di*self.g + self.dg*self.i
        #D = np.diag(self.f)
        K = self.o*(1-self._tanh(G)**2)
        #FE = self.do*(1-self._tanh(G)**2)
        self._jac[0:nc,0:nc] = (self.do*(1-self._tanh(G)**2)+C*K)
