import torch
import numpy as np
import matplotlib.pyplot as plt

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

#ACTIVATION FUNCTION
def sigma(x):
    return torch.tanh(x)
    
def dsigma(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

def ReLU(x):
    return 0.5*( torch.abs(x) + x )

def dReLU(x):
    return 1. * (x > 0)

#LOSS
def loss(x, t):
    return (x - t).pow(2).sum()

#dl/dx
def dloss(x, t):
    return 2*(x-t)

#FORWARD AND BACKWARD PASSES
def forward(w1, b1, w2, b2, w3, b3, w4, b4, x):
    
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)
    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)
    s3 = w3.mv(x2) + b3
    x3 = sigma(s3)
    s4 = w4.mv(x3) + b4
    x4 = ReLU(s4)    
    
    return x0,s1,x1,s2,x2,s3,x3,s4,x4                     
    
def backward(w1, b1, w2, b2, w3, b3, w4, b4,
             t,
             x0, s1, x1, s2, x2, s3, x3, s4, x4,
             dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4):

    dl_dx4 = dloss(x4, t)
    dl_ds4 = dReLU(s4) * dl_dx4
    dl_dx3 = w4.t().mv(dl_ds4)
    dl_ds3 = dsigma(s3) * dl_dx3
    dl_dx2 = w3.t().mv(dl_ds3)
    dl_ds2 = dsigma(s2) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dsigma(s1) * dl_dx1

    dl_dw4 = dl_ds4.view(-1, 1).mm(x3.view(1, -1))
    dl_db4 = dl_ds4
    dl_dw3 = dl_ds3.view(-1, 1).mm(x2.view(1, -1))
    dl_db3 = dl_ds3
    dl_dw2 = dl_ds2.view(-1, 1).mm(x1.view(1, -1))
    dl_db2 = dl_ds2   
    dl_dw1 = dl_ds1.view(-1, 1).mm(x0.view(1, -1))
    dl_db1 = dl_ds1
    
    return dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4

def create_parameters(fir_hidden_layer_feature, sec_hidden_layer_feature, thr_hidden_layer_feature,
                      input_features, output_feature):
    
    w1 = torch.empty(fir_hidden_layer_feature, input_features)
    w2 = torch.empty(sec_hidden_layer_feature, fir_hidden_layer_feature)
    w3 = torch.empty(thr_hidden_layer_feature, sec_hidden_layer_feature)
    w4 = torch.empty(output_feature, thr_hidden_layer_feature)
    
    epsilon = 1e-1

    b1 = torch.empty(fir_hidden_layer_feature)    
    b2 = torch.empty(sec_hidden_layer_feature)
    b3 = torch.empty(thr_hidden_layer_feature).normal_(0, epsilon)
    b4 = torch.empty(output_feature).normal_(0, epsilon)
    w1 = w1.normal_(0, epsilon)
    b1 = b1.normal_(0, epsilon)
    w2 = w2.normal_(0, epsilon)
    b2 = b2.normal_(0, epsilon)
    w3 = w3.normal_(0, epsilon)
    b3 = b3.normal_(0, epsilon)
    w4 = w4.normal_(0, epsilon)
    b4 = b4.normal_(0, epsilon)
    
    dl_dw1 = torch.empty(w1.size())
    dl_db1 = torch.empty(b1.size())
    dl_dw2 = torch.empty(w2.size())
    dl_db2 = torch.empty(b2.size())
    dl_dw3 = torch.empty(w3.size())
    dl_db3 = torch.empty(b3.size())
    dl_dw4 = torch.empty(w4.size())
    dl_db4 = torch.empty(b4.size())   
    
    return w1, b1, w2, b2, w3, b3, w4 ,b4 ,dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4

def init_gradient(dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4):
    
    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()
    dl_dw3.zero_()
    dl_db3.zero_()
    dl_dw4.zero_()
    dl_db4.zero_()
###########################################################################################################

#CREATE TRAIN/TEST SET#

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

#CONSTANT ASSIGNING#
#2 IN/OUT#
#25 HIDDEN UNITS FOR 3 LAYERS#
#5 EPOCH; MINIBACH=1000#

fir_hidden_layer_feature = 25
sec_hidden_layer_feature = 25
thr_hidden_layer_feature = 25
output_feature = 2
input_features = train_input.size(1)

number_of_epoch = 501
eta = 5e-1 / train_input.size(0)

w1, b1, w2, b2, w3, b3, w4 ,b4 , \
dl_dw1, dl_db1, \
dl_dw2, dl_db2, \
dl_dw3, dl_db3, \
dl_dw4, dl_db4 = create_parameters(fir_hidden_layer_feature, sec_hidden_layer_feature, thr_hidden_layer_feature,input_features, \
                                   output_feature)

print("Input size: {:4d} x{:4d}, (N*SIZE)".format(train_input.size(0), test_input.size()[1] ))
print("w1 size: {:4d} x{:4d}, (hidden1*SIZE), b1 size: {:4d}  (hidden1)".format(w1.size()[0], w1.size()[1], b1.size()[0]))
print("w2 size: {:4d} x{:4d}, (hidden2*SIZE), b2 size: {:4d}  (hidden2)".format(w2.size()[0], w2.size()[1], b2.size()[0]))
print("w3 size: {:4d} x{:4d}, (hidden3*SIZE), b3 size: {:4d}  (hidden3)".format(w3.size()[0], w3.size()[1], b3.size()[0]))
print("w4 size: {:4d} x{:4d}, (output *SIZE), b4 size: {:4d}  (output )\n".format(w4.size()[0], w4.size()[1], b4.size()[0]))

###########################################################################################################

#TRAIN & UPDATE
for e in range(number_of_epoch):
    
    init_gradient(dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4)
    
    train_error = 0
    test_error = 0
    total_loss = 0
    
    for i in range(0,len(train_input)):
        x0,s1,x1,s2,x2,s3,x3,s4,x4 = forward(w1, b1, w2, b2, w3, b3, w4, b4, train_input[i])
        
        if x4.argmax() != train_target.argmax(dim=1)[i]:
            train_error += 1
        total_loss += loss(x4, train_target[i])
        
        
        dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3, dl_dw4, dl_db4 =  backward(w1, b1, w2, b2, w3, b3, w4, b4, train_target[i],\
                                                                                   x0, s1, x1, s2, x2, s3, x3, s4, x4,\
                                                                                   dl_dw1, dl_db1, dl_dw2, dl_db2, \
                                                                                   dl_dw3, dl_db3, dl_dw4, dl_db4)
        w1.sub_(eta*(dl_dw1))
        b1.sub_(eta*(dl_db1))
        w2.sub_(eta*(dl_dw2))
        b2.sub_(eta*(dl_db2))
        w3.sub_(eta*(dl_dw3))
        b3.sub_(eta*(dl_db3))
        w4.sub_(eta*(dl_dw4))
        b4.sub_(eta*(dl_db4))

    if e%1 == 0:
        total_error = 0
        for i in range(0,len(test_input)):
            _,_,_,_,_,_,_,_,nx4 = forward(w1, b1, w2, b2, w3, b3, w4, b4, test_input[i])
            if nx4.argmax() != test_target.argmax(dim=1)[i]:
                test_error += 1 
                
        print("epoch= {:2d}, total loss= {:f}, train error= {:f}, test error= {:f}".\
              format(e, total_loss, train_error, test_error))
