import pickle as pkl
import numpy as np
import mat4py as m4p 
import sys

fn = 'weights_AND_biases.pkl'
output = 'TrainedData.dat'
input_size = 200


arg = sys.argv

fn = arg[1]
output = arg[2]
input_size = int(arg[3])

w = pkl.load(open(fn,'rb'))

wcc = w[0]
bcc = w[1]

b_i, b_C, b_f, b_o = np.split(bcc, 4, axis=0)
w_i, w_C, w_f, w_o = np.split(wcc, 4, axis=1)

w_xi = w_i[:input_size, :]
w_hi = w_i[input_size:, :]

w_xC = w_C[:input_size, :]
w_hC = w_C[input_size:, :]

w_xf = w_f[:input_size, :]
w_hf = w_f[input_size:, :]

w_xo = w_o[:input_size, :]
w_ho = w_o[input_size:, :]

open(output,"w").close()
fo = open(output,"a")



data = [w_ho,w_hf,w_hi,w_hC,b_C,b_f,b_o,b_i]
dim = len(b_o)
for mat in data:
        fo.write(str(dim)+'\n')
        np.savetxt(fo,mat,delimiter=' ', newline='\n')

fo.close()
    


#m4p.savemat('trainedExample.mat',data)
