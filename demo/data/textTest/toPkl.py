import pickle as pkl


dict = {'w':'model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix:0',
        'b':'model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias:0'}

dict={'w':u'rnn/output_projection_wrapper/lstm_cell/kernel:0',
      'b':u'rnn/output_projection_wrapper/lstm_cell/bias:0'}

dict={'b': u'rnn/basic_lstm_cell/bias:0',
      'w': u'rnn/basic_lstm_cell/kernel:0'}

fn = 'test_1_dropout_0.pkl'
outfn = 'test_1_dropout_0_overtime.pkl'



if __name__ == '__main__':
    out = []
    
    dat = pkl.load(open(fn,'r'))
    
    for it in dat:
        tmp = []
        tmp.append(it[dict['w']])
        tmp.append(it[dict['b']])
        tmp.append(it['loss'])
        
        out.append(tmp)
        
        
    with open(outfn,'wb') as f:
        pkl.dump(out, f, protocol=2)
