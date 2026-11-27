import numpy as np
import time
import random
import math

from torchvision import datasets # Technically an ml library but I am only using it for the dataset so I think it should be allowed
train = datasets.MNIST(root='./data', train=True, download=True)
x_train = train.data.numpy().reshape(len(train), -1)
y_train = train.targets.numpy()
val = datasets.MNIST(root='./data', train=False, download=True)
x_val = val.data.numpy().reshape(len(val), -1)
y_val = val.targets.numpy()
del datasets # Like it was never there

SG_K = 5.0
def surrogate(v, thr): return 1.0/(1.0 + SG_K*np.abs(v-thr))**2

class layer():
    # Roll operations into one layer class for better parallelization
    def __init__(self, n_in, n_out, beta, w_scale, b=0.0):
        self.w = np.random.randn(n_in, n_out)*w_scale/np.sqrt(n_in)
        self.b = np.full(n_out, b)
        self.beta = beta
        self.thr = 1.0
        # Optim
        self.gW, self.gb = np.zeros_like(self.w), np.zeros(n_out)
        self.m = [np.zeros_like(self.w), np.zeros_like(self.b)]
        self.v = [np.zeros_like(self.w), np.zeros_like(self.b)]
        self.k = 0

    def clear(self, B, n_out):
        self.p = np.zeros((B, n_out))
        self.carry = np.zeros((B, n_out))
        self.h_in, self.h_v, self.h_s = [], [], []

    def fire(self, x):
        self.p = self.p*self.beta + x@self.w + self.b
        v = self.p
        s = (v > self.thr).astype(float)
        self.p = np.where(s > 0, 0.0, self.p)
        self.h_in.append(x)
        self.h_v.append(v)
        self.h_s.append(s)
        return s

    def back(self, t, ds):
        dv = ds*surrogate(self.h_v[t], self.thr) + self.carry
        self.gW += self.h_in[t].T@dv; self.gb += dv.sum(0)
        s_prev = self.h_s[t-1] if t > 0 else 0.0
        self.carry = dv*self.beta*(1.0 - s_prev)
        return dv@self.w.T

    def update(self, lr):
        self.k += 1
        for i, (P, G) in enumerate(((self.w, self.gW), (self.b, self.gb))):
            self.m[i] = 0.9*self.m[i] + 0.1*G; self.v[i] = 0.999*self.v[i] + 0.001*G**2
            P -= lr*(self.m[i]/(1-0.9**self.k))/(np.sqrt(self.v[i]/(1-0.999**self.k)) + 1e-8)
        self.gW[:] = 0; self.gb[:] = 0

class snn():
    def __init__(self, n_in, n_hid, n_out, w1=1.5, w2=1.5, gain=1.5):
        self.n_in, self.n_hid, self.n_out, self.gain = n_in, n_hid, n_out, gain
        self.l1 = layer(n_in, n_hid, 0.8, w1)
        self.l2 = layer(n_hid, n_out, 0.8, w2)

    def run(self, x, T):
        B = len(x); self.T = T
        self.l1.clear(B, self.n_hid); self.l2.clear(B, self.n_out)
        p0 = np.zeros((B, self.n_in)); self.out = np.zeros((B, self.n_out))
        for t in range(T):
            p0 = p0*0.9 + self.gain*x
            s0 = (p0 > 1.0).astype(float); p0 = np.where(s0 > 0, 0.0, p0)
            self.out += self.l2.fire(self.l1.fire(s0))
        return self.out

    def backward(self, y):
        z = self.out - self.out.max(1, keepdims=True); P = np.exp(z)/np.exp(z).sum(1, keepdims=True)
        d = P.copy(); d[np.arange(len(y)), y] -= 1; d /= len(y)
        for t in reversed(range(self.T)):
            self.l1.back(t, self.l2.back(t, d))

    def update(self, lr):
        self.l1.update(lr); self.l2.update(lr)

def fit(xt, yt, xv, yv, n_hid, epochs, T=15, bs=64, base_lr=1e-3, seed=0, **kw):
    np.random.seed(seed); model = snn(xt.shape[1], n_hid, 10, **kw)
    for ep in range(epochs):
        idx = np.random.permutation(len(xt)); corr = 0
        for i in range(0, len(idx), bs):
            b = idx[i:i+bs]; out = model.run(xt[b], T); corr += (out.argmax(1) == yt[b]).sum()
            model.backward(yt[b]); model.update(base_lr*(0.55+0.45*math.cos(math.pi * ep/epochs)))
        print(f"epoch {ep} train acc {corr/len(xt):.3f} test acc {(model.run(xv, T+35).argmax(1) == yv).mean():.3f}", flush=True)
    return model

if __name__ == "__main__":
    model = fit(x_train, y_train, x_val, y_val, n_hid=224, epochs=25)
    with open("model.txt", "w") as f:
        f.write(np.array2string(model.l1.w, threshold=np.inf) + "\n\n")
        f.write(np.array2string(model.l1.b, threshold=np.inf) + "\n\n")
        f.write(np.array2string(model.l2.w, threshold=np.inf) + "\n\n")
        f.write(np.array2string(model.l2.b, threshold=np.inf) + "\n\n")