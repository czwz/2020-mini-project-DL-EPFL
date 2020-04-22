import numpy as np
from mynn_module import *

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

nb = len(train_input)
lr = 5e-1/nb
mynn = model()
cri = criterion_mse(mynn)
opt = optimizer_sgd(lr,mynn)

sequence = [ linear(2,25,mynn,cri),
             sigmoid(mynn),
             linear(25,25,mynn,cri),
             sigmoid(mynn),
             linear(25,2,mynn,cri),
             sigmoid(mynn)
           ]

for e in range(801):
    tloss = 0
    nberr = 0
    for i in range(len(train_input)):
        mynn.train(train_input[i],train_target[i],sequence)

        tloss = tloss + cri.loss(mynn.x[mynn.lt],train_target[i])
        if mynn.x[mynn.lt].argmax()!=train_target[i].argmax():
            nberr = nberr + 1

        opt.update()

    if e%50==0:
        print("epoch:{:3d}, total loss:{:3f}, number of error:{:3d}".format(e,np.log10(tloss.item()),nberr))
