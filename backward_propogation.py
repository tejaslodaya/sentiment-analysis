def backward_propagate(inputs,xs, hs, hsr, ps, targets):
    ## inputs = list of nparrays
    ## targets = list of target labels..zero indexed
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dWxhr, dWhhr, dWhyr = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby, dbhr = np.zeros_like(bh), np.zeros_like(by) ,np.zeros_like(bh)
    
    dy={}
    T=len(inputs)-1
    for t in range(len(inputs)):
        dy[t] = np.zeros_like(ps[t])
        #if t==T:
        dy[t]=np.copy(ps[t])
        dy[t][targets[t]] -= 1
        dWhy += np.dot(dy[t], hs[t].T)
        dWhyr += np.dot(dy[t], hsr[t].T)
        dby += dy[t]

    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dh = np.dot(Why.T, dy[t]) + dhnext # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
        
    dhnext = np.zeros_like(hs[0])
    for t in range(len(inputs)):
        dh = np.dot(Whyr.T, dy[t]) + dhnext # backprop into h
        dhraw = (1 - hsr[t] * hsr[t]) * dh # backprop through tanh nonlinearity
        dbhr += dhraw
        dWxhr += np.dot(dhraw, xs[t].T)
        dWhhr += np.dot(dhraw, hsr[t+1].T)
        dhnext = np.dot(Whhr.T, dhraw)
    
    for dparam in [dWxh, dWhh, dWhy, dWxhr, dWhhr, dWhyr, dbh, dby, dbhr]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return dWxh, dWhh, dWhy, dWxhr, dWhhr, dWhyr, dbh, dby, dbhr, hs[len(inputs)-1]