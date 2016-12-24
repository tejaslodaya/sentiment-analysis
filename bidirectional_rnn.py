import numpy as np
import pickle
import random
from forward_propagation import *
from backward_propagate import *
from train import *

gensim=pickle.load(open("model.pkl","rb"))
sent=pickle.load(open("sent.pkl","rb"))
labels=pickle.load(open("labels.pkl","rb"))

data=[]
print(labels[0])
for i in range(len(sent)):
    l=(sent[i],labels[i])
    data.append(list(l))

# hyperparameters
input_size = 100
hidden_size = 64# size of hidden layer of neurons
output_size = 5
seq_length = 5 # number of steps to unroll the RNN for
learning_rate = 0.02
epoch=20


def fn(a,b):
    return np.random.uniform(low=-2,high=2,size=(a,b))
    
np.random.randn = fn

# model parameters
Wxh = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(output_size, hidden_size)*0.01 # hidden to output
Wxhr = np.random.randn(hidden_size, input_size)*0.01
Whhr = np.random.randn(hidden_size, hidden_size)*0.01
Whyr = np.random.randn(output_size, hidden_size)*0.01
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((output_size, 1)) # output bias
bhr = np.zeros((hidden_size, 1))

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mWxhr, mWhhr, mWhyr = np.zeros_like(Wxhr), np.zeros_like(Whhr), np.zeros_like(Whyr)
mbh, mby, mbhr = np.zeros_like(bh), np.zeros_like(by), np.zeros_like(bhr) # memory variables for Adagrad
smooth_loss = -np.log(1.0/input_size)*seq_length


def test(inputs=None):
    ## inps in list of training set. list of [inputs,target]
    ## inputs is list of np arrays and targets is list of labels
    if inputs==None:
        inputs=data[int(0.8*len(data)):]
    score=0
    tot=0
    num=0
    prec_den=0
    recall_den=0
    for i in range(len(inputs)):
        xs, hs, hsr, ps,loss = forward_propagate(inputs[i][0],inputs[i][1])
        T=len(inputs[i][0])-1
        pred=np.argmax(ps[T])
        if pred==inputs[i][1][0]:
            score+=1
        print (pred,inputs[i][1][0])
        tot+=1


    print ("Accuracy: ",1.0*score/tot)
    print "Precision: ",1.0*num/prec_den
    try:print "Recall: ",1.0*num/recall_den
    except:pass
    print "Complete accuracy: ",1.0*full/len(inputs)

def predict():
    while True:
        print ("\nEnter a sequence (space separated) (Enter -1 to exit) :")
        seq = input()
        if seq=="-1":
            break
        inputs = map(int,seq.strip().split())
        xs, hs, hsr, ps,loss = forward_propagate(inputs,sorted(inputs))
        for i in range(len(inputs)):
            print (np.argmax(ps[i]))
        print

if __name__=="__main__":
    inn=0
    if inn==0:
        data_length=len(data)
        for i in range(data_length):
            data[i][1]=[data[i][1] for _ in range(len(data[i][0]))]
            for j in range(len(data[i][0])):
                data[i][0][j]=data[i][0][j].reshape((input_size,1))
        train(data[:int(0.8*len(data))])
        test(data[int(0.8*len(data)):])
        predict()