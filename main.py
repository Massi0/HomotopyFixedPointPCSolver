#Headers

from HomPCSolver import HomotopyPCSolver 
from HomPCSolver import np,linalg
from demo import pkl,path,LSTMNN
import scipy.io as scipyio
import pdb

def readFromPickle(data_input_file):

    ww = pkl.load(open(data_input_file,'rb'))[0]
    
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

    data_input_file = 'demo/data/textTest/test_1_dropout_0_overtime.pkl'
    output_eq_file = 'demo/data/textTest/test_1_dropout_0.mat'
        
    epsPred = 9e-3 #Prediction error 
    epsCorr = 1e-9 #Correction error
    maxSteps = 5e2 #Max number of steps 

    thrPrune = .01
    
    lstm = LSTMNN(None) 

    weights = readFromPickle(data_input_file)
    weights, pPrune = pruneW(weights,thrPrune)
    
    lstm.updateWeigths(weights)

    ndim = 2*lstm.dimOfOutput()    
    x0 = np.zeros([ndim,1])
    lstm.setInitStat(x0)
    
    print("dim = ",str(ndim))
    (t,x,eCode) = lstm.findEquilibria(epsPred,epsCorr,maxSteps)

    xeq = lstm.refineTheEquilibria(x)
    xeq = np.array([xeq]).T

    lstm.errorInterpreter(eCode)

    print("||F(xeq)-xeq|| = ",linalg.norm(xeq-lstm.F(xeq),2))

            
    tmp_out = {'xeq':xeq,'prTh':thrPrune}
    scipyio.savemat(output_eq_file,tmp_out)
