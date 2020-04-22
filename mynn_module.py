import torch

class model:
    def __init__(self):
        self.x = [None]
        self.s = [None]
        self.w = [None]
        self.b = [None]
        self.dl_dw = [None]
        self.dl_db = [None]
        self.dl_ds = [None]
        self.dl_dx = [None]
        self.lt = 0
        self.ln = 0
        self.ini = 0
    def loss(self,x,t):
        return (x-t).pow(2).sum()
    def dloss(self,x,t):
        return 2*(x-t)
    def l_counter(self):
        lt = self.lt
        self.lt = lt + 1
    def train(self,x,t,s):
        if self.ini == 0:
            for i in range(len(s)):
                s[i].ini(x)
            for i in range(len(s)):
                s[i].forward(x, self.ln)
            for i in range(len(s)-1,-1,-1):
                s[i].backward(t, self.ln)
            self.ini = 1
        elif self.ini == 1:
            for i in range(len(s)):
                s[i].forward(x, self.ln)
            for i in range(len(s)-1,-1,-1):
                s[i].backward(t, self.ln)            
                 
class criterion_mse:
    def __init__(self,nn):
        return None
    def loss(self,x,t):
        return (x - t).pow(2).sum()
    def dloss(delf,x,t):
        return 2*(x-t)
                    
class optimizer_sgd:
    def __init__(self,lr,nn):
        self.lr = lr
        self.nn = nn
    def update(self):
        for i in range(1,self.nn.lt):
            self.nn.w[i] = self.nn.w[i] - self.lr*self.nn.dl_dw[i]
            self.nn.b[i] = self.nn.b[i] - self.lr*self.nn.dl_db[i]

class sigmoid:
    def __init__(self, nn):
        self.nn = nn
        return None
    def ini(self,x):
        pass
    def forward(self,x,ln):
        s0 = self.nn.s[ln]
        self.nn.x[ln] = s0.tanh()
    def backward(self,t,ln):
        s0 = self.nn.s[ln]
        self.nn.dl_ds[ln] = 4*(s0.exp() + s0.mul(-1).exp()).pow(-2)

class linear:
    def __init__(self, m, n, nn, criterion):
        self.epsilon = 1e-1
        self.n = n
        self.m = m
        self.nn = nn
        self.cri = criterion
    def ini(self,x):
        w = torch.empty(self.n, self.m).normal_(0, self.epsilon)
        b = torch.empty(self.n).normal_(0, self.epsilon)
        self.nn.x.append( None )
        self.nn.s.append( None )
        self.nn.w.append( w )
        self.nn.b.append( b )
        self.nn.dl_dw.append( torch.empty(w.size()) )
        self.nn.dl_db.append( torch.empty(b.size()) )
        self.nn.dl_ds.append( torch.empty(self.n).fill_(1.0) )
        self.nn.dl_dx.append( torch.empty(self.n).fill_(1.0) )
        self.nn.x
        self.nn.l_counter()        
    def forward(self,x,ln):
        self.nn.x[0] = x
        self.nn.x[ln+1] = self.nn.w[ln+1].mv(self.nn.x[ln]) + self.nn.b[ln+1]
        self.nn.s[ln+1] = self.nn.x[ln+1]
        self.nn.ln = ln + 1
    def backward(self,t,ln):
        if ln==self.nn.lt:
            self.nn.dl_dx[ln] = self.cri.dloss(self.nn.x[ln], t)            
            self.nn.dl_ds[ln] = self.nn.dl_ds[ln] * self.nn.dl_dx[ln]
        elif ln!=self.nn.lt:
            self.nn.dl_dx[ln] = self.nn.w[ln+1].t().mv(self.nn.dl_ds[ln+1])
            self.nn.dl_ds[ln] = self.nn.dl_ds[ln] * self.nn.dl_dx[ln]               
        self.nn.dl_dw[ln] = self.nn.dl_ds[ln].view(-1, 1).mm(self.nn.x[ln-1].view(1, -1))
        self.nn.dl_db[ln] = self.nn.dl_ds[ln]
        self.nn.ln = ln - 1
        
def generate_disc_set(nb):
    input = torch.empty(nb,2).uniform_(0,1).type(torch.FloatTensor)
    target = torch.empty(nb,2).fill_(1)
    distance = torch.mul(input,input).sum(dim=1)
    
    for i in range(nb):
        if distance[i] >= 0.5: 
            target[i][0], target[i][1] = 0, 1
        elif distance[i] < 0.5:
            target[i][0], target[i][1] = 1, 0
            
    return input, target
