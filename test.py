import numpy as np
from mynn_module import *

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

# define learning rate based on training set size
# define NN model, criterion(MSE_Loss), optimizer(SGD)

nb = len(train_input)
lr = 1e2/nb
mynn = model()
cri = criterion_mse(mynn)
opt = optimizer_sgd(lr,mynn)

# define the sequential structure for NN
# Here I choose: FF(2x25)-sig-FF(25x25)-ReLU-FF(25x25)-sig, where
#                FF (Fully connected), sig (Tanh activation), ReLU (rectified linear unit )

sequence = [ linear(2,25,mynn,cri),
             sigmoid(mynn),
             linear(25,25,mynn,cri),
             relu(mynn),
             linear(25,2,mynn,cri),
             sigmoid(mynn)
           ]

# train w/ full training set multiple times (e=800)
# for every epoch(e):
#     train w/ training set
#     calculate total loss (tloss) by criterion cri(MSE_Loss) and number of training error (nberr)
#     update weights and bias by optimizer opt (SGD)
#
#     do forward pass of testing set
#     calculate number of testing error (tnberr)
#
#     print result every e==50 

for e in range(101):
    tloss,nberr,tnberr = 0,0,0
    
    for i in range(len(train_input)):
        mynn.train(train_input[i],train_target[i],sequence,forward_only=False)
        tloss = tloss + cri.loss(mynn.x[mynn.lt],train_target[i],)
        if mynn.x[mynn.lt].argmax()!=train_target[i].argmax():
            nberr = nberr + 1
        opt.update()

    for i in range(len(test_input)):
        mynn.train(test_input[i],test_target[i],sequence,forward_only=True)
        if mynn.x[mynn.lt].argmax()!=test_target[i].argmax():
            tnberr = tnberr + 1
        
    if e%10==0:
        print("epoch:{:3d}, total loss:{:3f}, number of train error:{:3d}, number of test error:{:3d}"\
              .format(e,np.log10(tloss.item()),nberr,tnberr))
