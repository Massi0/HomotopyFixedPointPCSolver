import sys
import pickle as pkl

arg = sys.argv

fn = arg[1]
output = arg[2]

w = pkl.load(open(fn,'rb'))

pkl.dump(w, open(output,"wb"), protocol=2)
