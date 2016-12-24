def train(inps):
    ## inps in list of training set. list of [inputs,target]
    ## inputs is list of np arrays and targets is list of labels
    global Wxh, Whh, Why,Wxhr, Whhr, Whyr, bh, by, bhr, seq_length
    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mWxhr, mWhhr, mWhyr = np.zeros_like(Wxhr), np.zeros_like(Whhr), np.zeros_like(Whyr)
    mbh, mby, mbhr = np.zeros_like(bh), np.zeros_like(by), np.zeros_like(bhr) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/input_size)*seq_length
    for j in range(epoch):
        n=0
        random.shuffle(inps)
        for i in range(len(inps)):
            xs, hs, hsr, ps,loss = forward_propagate(inps[i][0],inps[i][1])
            dWxh, dWhh, dWhy, dWxhr, dWhhr, dWhyr, dbh, dby, dbhr, hs = backward_propagate(inps[i][0],xs, hs, hsr, ps, inps[i][1])
            
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            seq_length=len(inps[i])
            if n % 500 == 0: print ('epoch %d iter %d, loss: %f %f' % (j,n, smooth_loss, loss)) # print progress
        
            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wxh, Whh, Why,Wxhr, Whhr, Whyr, bh, by, bhr], 
                                                                        [dWxh, dWhh, dWhy,dWxhr, dWhhr, dWhyr, dbh, dby,dbhr], 
                                                                        [mWxh, mWhh, mWhy,mWxhr, mWhhr, mWhyr, mbh, mby, mbhr]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        
            p += seq_length # move data pointer
            n += 1 # iteration counter 
        test()
            
        print ("epoch",j)