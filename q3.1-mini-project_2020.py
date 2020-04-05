import torch
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################################
###########################################################################################################

def generate_disc_set(nb):
    input = torch.empty(nb,2).uniform_(0,1).type(torch.FloatTensor)
    target = torch.empty(nb,2).fill_(1)
    distance = torch.mul(input,input).sum(dim=1)
    
    for i in range(nb):
        if distance[i] >= (1/(2*np.pi)):
            target[i][0], target[i][1] = 0, 1
        elif distance[i] < (1/(2*np.pi)):
            target[i][0], target[i][1] = 1, 0
            
    return input, target

def sigma(x):
    return torch.tanh(x)
    
def dsigma(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

def loss(x, t):
    return np.log10( float(torch.sum((x-t).pow(2)))/float( x.size(1) ) )

def dloss(x, t):
    return 2*(x-t)/float( x.size(1) )

def forward(w1, b1, w2, b2, w3, b3, w4, b4, x):
    x0 = x
    s1 = torch.mm( x0, w1.T ) + b1
    x1 = sigma( s1 )
    s2 = torch.mm( x1, w2.T ) + b2
    x2 = sigma( s2 )
    s3 = torch.mm( x2, w3.T ) + b3
    x3 = sigma( s3 )
    s4 = torch.mm( x2, w4.T ) + b4
    x4 = sigma( s4 )
    return x0,s1,x1,s2,x2,s3,x3,s4,x4                     
    
def backward(w1, b1, w2, b2, w3, b3, w4, b4,
             t,
             x0, s1, x1, s2, x2, s3, x3, s4, x4):
    dl_db4 = torch.mul( dloss(x4, t), dsigma(s4))
    dl_dw4 = torch.mm( dl_db4.T, x3 )

    dl_db3 = torch.mul( torch.mm(dl_db4, w4), dsigma(s3))
    dl_dw3 = torch.mm( dl_db3.T, x2 )

    dl_db2 = torch.mul( torch.mm(dl_db3, w3), dsigma(s2))
    dl_dw2 = torch.mm( dl_db2.T, x1 )

    dl_db1 = torch.mul( torch.mm(dl_db2, w2), dsigma(s1))
    dl_dw1 = torch.mm( dl_db1.T, x0 )        
    return dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4

def create_parameters(fir_hidden_layer_feature, sec_hidden_layer_units, thr_hidden_layer_units,
                      input_features, output_units):
    epsilon = 1e-6
    w1 = torch.empty(fir_hidden_layer_units, input_units).normal_(0, epsilon)
    b1 = torch.empty(fir_hidden_layer_units).normal_(0, epsilon)
    w2 = torch.empty(sec_hidden_layer_units, fir_hidden_layer_units,).normal_(0, epsilon)
    b2 = torch.empty(sec_hidden_layer_units).normal_(0, epsilon)
    w3 = torch.empty(thr_hidden_layer_units, sec_hidden_layer_units).normal_(0, epsilon)
    b3 = torch.empty(thr_hidden_layer_units).normal_(0, epsilon)
    w4 = torch.empty(output_units, thr_hidden_layer_units).normal_(0, epsilon)
    b4 = torch.empty(output_units).normal_(0, epsilon)
    
    dl_dw1 = torch.empty(w1.size())
    dl_db1 = torch.empty(b1.size())
    dl_dw2 = torch.empty(w2.size())
    dl_db2 = torch.empty(b2.size())
    dl_dw3 = torch.empty(w3.size())
    dl_db3 = torch.empty(b3.size())
    dl_dw4 = torch.empty(w4.size())
    dl_db4 = torch.empty(b4.size())   
    
    return w1, b1, w2, b2, w3, b3, w4 ,b4 ,dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4


###########################################################################################################
###########################################################################################################


#CREATE TRAIN/TEST SET#

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

#CONSTANT ASSIGNING#
#2 IN/OUT#
#25 HIDDEN UNITS FOR 3 LAYERS#

fir_hidden_layer_units = 25
sec_hidden_layer_units = 25
thr_hidden_layer_units = 25
output_units = 2
input_units = train_input.size(1)

#INITIALZE WEIGHTS AND BIAS#

w1, b1, w2, b2, w3, b3, w4 ,b4 , \
dl_dw1, dl_db1, \
dl_dw2, dl_db2, \
dl_dw3, dl_db3, \
dl_dw4, dl_db4 = create_parameters(fir_hidden_layer_units, sec_hidden_layer_units, thr_hidden_layer_units,\
                                   input_units, output_units)

print("Input size: {:4d} x{:4d}, (N*SIZE)\n".format(int( train_input.size(0) ), test_input.size(1) ))
print("w1 size: {:4d} x{:4d}, (SIZE*hidden1)\nb1 size:       {:4d}  (1*hidden1)".
      format(w1.size()[0], w1.size()[1], b1.size()[0]))
print("w2 size: {:4d} x{:4d}, (SIZE*hidden1)\nb1 size:       {:4d}  (1*hidden1)".
      format(w2.size()[0], w2.size()[1], b2.size()[0]))
print("w3 size: {:4d} x{:4d}, (SIZE*hidden1)\nb1 size:       {:4d}  (1*hidden1)".
      format(w3.size()[0], w3.size()[1], b3.size()[0]))
print("w4 size: {:4d} x{:4d}, (SIZE*hidden1)\nb1 size:       {:4d}  (1*hidden1)\n".
      format(w4.size()[0], w4.size()[1], b4.size()[0]))

#RUN 5 EPOCH; MINIBACH=1000#

number_of_epoch = 5
eta = 1e-1 / float( train_input.size(0) )

for e in range(number_of_epoch):
    
    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()
    dl_dw3.zero_()
    dl_db3.zero_()
    dl_dw4.zero_()
    dl_db4.zero_()
    
    train_error = 0
    test_error = 0
    total_loss = 0
    batch = 1000
    
    for i in range(0,len(train_input),batch):
        x0,s1,x1,s2,x2,s3,x3,s4,x4 = forward(w1, b1, w2, b2, w3, b3, w4, b4, train_input[i:i+batch])
        dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4 = backward(w1, b1, w2, b2, w3, b3, w4, b4,\
                                                                                  train_target[i:i+batch],\
                                                                                  x0, s1, x1, s2, x2, s3, x3, s4, x4)
        
        for j in range(batch):
            if x4.argmax(dim=1)[j] != train_target.argmax(dim=1)[i+j]:
                train_error += 1 
        
        w1 = w1 - eta*(dl_dw1)
        b1 = b1 - eta*(dl_db1)
        w2 = w2 - eta*(dl_dw2)
        b2 = b2 - eta*(dl_db2)
        w3 = w3 - eta*(dl_dw3)
        b3 = b3 - eta*(dl_db3)
        w4 = w4 - eta*(dl_dw4)
        b4 = b4 - eta*(dl_db4)

        total_loss += loss(x4, train_target[i:i+batch])
        
    if e%1 == 0:
        total_error = 0
        for i in range(0,len(test_input),batch):
            _,_,_,_,_,_,_,_,nx4 = forward(w1, b1, w2, b2, w3, b3, w4, b4, test_input[i:i+batch])
            for j in range(batch):
                if nx4.argmax(dim=1)[j] != test_target.argmax(dim=1)[i+j]:
                    test_error += 1 
        print("epoch= {:2d}, total loss= {:8f}, train error= {:4f}, test error= {:4f}".\
              format(e, total_loss, train_error/float(train_input.size(0)), test_error/float(test_input.size(0))))

