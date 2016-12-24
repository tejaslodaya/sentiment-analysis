def forward_propagate(inputs,targets):
    ## inputs = list of nparrays
    ## targets = list of target labels..zero indexed
    xs, hs, ys, ps, hsr = {}, {}, {}, {}, {}
    hprev = np.zeros((hidden_size,1))
    hs[-1] = np.copy(hprev)
    hsr[len(inputs)] = np.copy(hprev)
    loss=0
    T=len(inputs)-1
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((input_size,1)) # encode in 1-of-k representation
        xs[t] = inputs[t]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
        
        
    for t in reversed(range(len(inputs))):
        hsr[t] = np.tanh(np.dot(Wxhr, xs[t]) + np.dot(Whhr, hsr[t+1])+ bhr)
        
    for t in range(len(inputs)):
        ys[t] = np.dot(Why,hs[t]) + np.dot(Whyr,hsr[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
    loss += -np.log(ps[T][targets[t],0])
        
    return (xs, hs, hsr, ps,loss)