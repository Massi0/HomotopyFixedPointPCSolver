#Headers

from HomPCSolver import HomotopyPCSolver 
from HomPCSolver import np,linalg
from demo import pkl,path,LSTMNN
import scipy.io as sio
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

def pruneW(w,th):
    label = {'Wo','Wf','Wi','Wg',
             'Wo_x','Wf_x','Wi_x','Wg_x',
             'bo','bf','bo','bg'}
    s = 0
    maxval = 0
    minval = 0
    nT = 0
    for i in label:
        tmp = np.max(w[i])
        maxval = tmp if tmp>maxval else maxval
        tmp = np.min(w[i])
        minval = tmp if tmp<minval else minval
        
        ind = np.abs(w[i])<th
        s += np.nonzero(np.concatenate(ind))[0].size
        nT += np.size(w[i])
        w[i][ind] = 0
       
        
    print s,' elements discarded | nb elements: ',nT, '| Pourcentage =  ',s/(.01*nT),' | min =', minval, ' | max = ',maxval 
    
    return (w,s/(.01*nT))

if __name__ == '__main__':
    #test 1: Demo: root of polynomials
    #test 2: Demo: Equilibria of LSTM Neural network tested on Penn Tree Bank (PTB) dataset
    #test 3: Demo: Equilibria of LSTM NN during the training tested on PTB dataset
    test = 5
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
            


    if test == 4:
        #data_input_file = 'demo/data/Notingham/OneLayerNaiveSoft80TH_overtime.pkl'
        #output_eq_file = 'demo/data/Notingham/Eq_80_Naive.mat'

        #data_input_file = 'demo/data/randInput/output_N1_02_overtime.pkl'
        #output_eq_file = 'demo/data/randInput/Eq_N1_02_randInput.mat'
        data_input_file = 'demo/data/textTest/test_1_dropout_0_overtime.pkl'
        output_eq_file = 'demo/data/textTest/test_1_dropout_0.mat'



        epsPred = 1e-3
        epsCorr = 1e-9
        maxSteps = 1e5

        lstm = LSTMNN(None)

        paramOverEpochs = pkl.load(open(data_input_file,'rb'))
        
        eq = []
        i_step = 1
        for w in paramOverEpochs:
            w = pruneW(w,.002)
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

        tmp_out = {'xeq':eq}
        sio.savemat(output_eq_file,tmp_out)


    if test == 5:
        #data_input_file = 'demo/data/Notingham/OneLayerNaiveSoft80TH_overtime.pkl'
        #output_eq_file = 'demo/data/Notingham/Eq_80_Naive.mat'

        #output_eq_file = 'demo/data/randInput/Eq_N1_02_randInput.mat'
        data_input_file = 'demo/data/textTest/test_1_dropout_0_overtime.pkl'
        output_eq_file = 'demo/data/textTest/test_1_dropout_0.mat'



        epsPred = 9e-3
        epsCorr = 1e-9
        maxSteps = 5e2

        lstm = LSTMNN(None)

        paramOverEpochs = pkl.load(open(data_input_file,'rb'))
        
        eq = []
        i_step = 1
        plist = [.03,.1,.5,.6,.65,.7,.75,.8,.85,.9,.95]
        w = paramOverEpochs[0]
        pList = []
        for i in plist:
            print i
            weights = readFromPickle(w)
            weights, pPrune = pruneW(weights,i)
            
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
            pList.append(pPrune)

        tmp_out = {'xeq':eq,'prTh':plist,'pPrune':pList}
        sio.savemat(output_eq_file,tmp_out)
