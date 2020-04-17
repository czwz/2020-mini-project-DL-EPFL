import torch

class model:
    def __init__(self):
        self.x = [None]
        self.s = [None]
        self.w = [None]
        self.b = [None]
        self.dl_dw = []
        self.dl_db = []
        self.dl_ds = []
        self.dl_dx = []
        self.lt = 0
        self.ln = 0
        
    def loss(self,x,t):
        return (x - t).pow(2).sum()
    def dloss(delf,x,t):
        return 2*(x-t)
    def l_counter(self):
        lt = self.lt
        self.lt = lt + 1

    def train(self,x,t,*argv):
        for i in range(len(argv)):
            argv[i].forward(x, self.ln)
        for i in range(len(argv)):
            argv[i].backward(t, self.ln)
            
class optimizer:
    def __init__(self,lr,nn):
        self.lr = lr
    
    def sgd(self):
        for i in range(lt):
            nn.w[i] = nn.w[i] - lr*nn.dl_dw[i]
            nn.b[i] = nn.b[i] - lr*nn.dl_db[i]
        
class sigmoid:
    def __init__(self, nn):
        return None
    def forward(self,x,ln):
        s0 = s[ln]
        nn.x[ln] = s0.tanh()
    def backward(self,t,ln):
        s0 = nn.s[ln]
        nn.dl_ds[ln] = 4*(s0.exp() + s0.mul(-1).exp()).pow(-2)

class Linear:
    def __init__(self, m, n, nn):
        self.epsilon = 1e-1 
        w = torch.empty(m, n).normal_(0, self.epsilon)
        b = torch.empty(m).normal_(0, self.epsilon)
        nn.x.append( None )
        nn.s.append( None )
        nn.w.append( w )
        nn.b.append( b )
        nn.dl_dw.append( torch.empty(w.size()) )
        nn.dl_db.append( torch.empty(b.size()) ) 
        nn.dl_ds.append( None )
        nn.dl_dx.append( None )
        nn.x
        nn.l_counter()
        
    def forward(self,x,ln):
        nn.x[0] = x
        nn.s[ln+1] = nn.w[ln+1].mv(nn.x[ln]) + nn.b[ln+1]
        ln = ln + 1
    def backward(self,t,ln):
        if ln==nn.lt:
            nn.dl_dx[ln] = nn.dloss(nn.x[ln], t)            
            nn.dl_ds[ln] = nn.dl_ds[ln] * nn.dl_dx[ln]
        elif ln!=nn.lt:
            nn.dl_dx[ln] = nn.w[ln+1].t().mv(nn.dl_ds[ln+1])
            nn.dl_ds[ln] = nn.dl_ds[ln] * nn.dl_dx[ln]               
        nn.dl_dw[ln] = nn.dl_ds[ln].view(-1, 1).mm(nn.x[ln-1].view(1, -1))
        nn.dl_db[ln] = nn.dl_ds[ln]
        ln = ln - 1
