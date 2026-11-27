import numpy as np
import random
import time

from torchvision import datasets # Technically an ml library but I am only using it for the dataset so I think it should be allowed
train = datasets.MNIST(root='./data', train=True, download=True)
x_train = train.data.numpy().reshape(len(train), -1)
y_train = train.targets.numpy()
val = datasets.MNIST(root='./data', train=False, download=True)
x_val = val.data.numpy().reshape(len(val), -1)
y_val = val.targets.numpy()
del datasets # Like it was never there

print("Loaded Data")

SG_K = 5.0 # surrogate sharpness
def surrogate(v, thr):
    return 1.0 / (1.0 + SG_K*abs(v - thr))**2

class neuron():
    def __init__(self, input_weights, beta, w_scale):
        self.p = 0.0
        self.beta = beta
        self.activation_p = 1.0
        self.w = np.random.randn(input_weights) * w_scale
        self.b = 0.2

        self.h_in, self.h_v, self.h_s = [], [], []
        self.carry = 0.0
        # grads + optimizer state
        self.gw, self.gb = np.zeros(input_weights), 0.0
        self.mw, self.vw = np.zeros(input_weights), np.zeros(input_weights)
        self.mb, self.vb, self.n_updates = 0.0, 0.0, 0

    def fire(self, inputs):
        self.p = self.p*self.beta + np.dot(inputs, self.w) + self.b
        v = self.p
        spiked = v > self.activation_p
        if spiked:
            self.p = 0.0
        self.h_in.append(np.array(inputs, dtype=float)); self.h_v.append(v); self.h_s.append(1.0 if spiked else 0.0)
        return spiked

    def update(self, lr):
        self.n_updates += 1; k = self.n_updates
        self.mw = 0.9*self.mw + 0.1*self.gw;  self.vw = 0.999*self.vw + 0.001*self.gw**2
        self.mb = 0.9*self.mb + 0.1*self.gb;  self.vb = 0.999*self.vb + 0.001*self.gb**2
        self.w -= lr * (self.mw/(1-0.9**k)) / (np.sqrt(self.vw/(1-0.999**k)) + 1e-8)
        self.b -= lr * (self.mb/(1-0.9**k)) / (np.sqrt(self.vb/(1-0.999**k)) + 1e-8)
        self.gw, self.gb = np.zeros_like(self.gw), 0.0

class snn():
    def __init__(self):
        # Neuron layers
        self.l0_n = [neuron(1, beta=0.9, w_scale=0.0) for _ in range(784)]
        for n in self.l0_n:
            n.w = np.array([1.5]); n.b = 0.0
        self.l1_n = [neuron(784, beta=0.8, w_scale=1.0) for _ in range(112)]
        self.l2_n = [neuron(112, beta=0.8, w_scale=0.6) for _ in range(10)]

        # Input voltages
        self.l0_v = [0.0] * len(self.l0_n)
        self.l1_v = [0.0] * len(self.l0_n)
        self.l2_v = [0.0] * len(self.l1_n)
        self.l3_v = [0.0] * len(self.l2_n)
        self.output = np.zeros(len(self.l2_n))
        self.t = 0

    def clear(self):
        for n in self.l0_n + self.l1_n + self.l2_n:
            n.p = 0.0; n.carry = 0.0
            n.h_in, n.h_v, n.h_s = [], [], []
        self.output = np.zeros(len(self.l2_n))
        self.t = 0

    def step(self):
        self.t += 1
        for i, n in enumerate(self.l0_n):
            self.l1_v[i] = 1.0 if n.fire([self.l0_v[i]]) else 0.0
        for i, n in enumerate(self.l1_n):
            self.l2_v[i] = 1.0 if n.fire(self.l1_v) else 0.0
        for i, n in enumerate(self.l2_n):
            self.l3_v[i] = 1.0 if n.fire(self.l2_v) else 0.0
        self.output += self.l3_v
        return [i for i, v in enumerate(self.l3_v) if v != 0.0]

    def backward(self, target, scale=1.0):
        z = self.output - self.output.max()
        P = np.exp(z)/np.exp(z).sum()
        dcount = P.copy(); dcount[target] -= 1.0
        dcount *= scale

        for t in reversed(range(self.t)):
            dl2_in = np.zeros(len(self.l2_v))
            for i, n in enumerate(self.l2_n):
                dv = dcount[i]*surrogate(n.h_v[t], n.activation_p) + n.carry
                n.gw += dv*n.h_in[t]; n.gb += dv
                dl2_in += dv*n.w
                s_prev = n.h_s[t-1] if t > 0 else 0.0
                n.carry = dv*n.beta*(1.0 - s_prev)
            # Layer 1
            for j, n in enumerate(self.l1_n):
                dv = dl2_in[j]*surrogate(n.h_v[t], n.activation_p) + n.carry
                n.gw += dv*n.h_in[t]; n.gb += dv
                s_prev = n.h_s[t-1] if t > 0 else 0.0
                n.carry = dv*n.beta*(1.0 - s_prev)

    def update(self, lr):
        for n in self.l1_n + self.l2_n:
            n.update(lr)

T = 15 # Compute steps
def present(model, x, steps=T):
    model.l0_v = (-1.0*x*(1/255.0)*0.9 + 1.1)
    for _ in range(steps):
        model.step()
    return int(np.argmax(model.output))

def train(EPOCHS=5, lr=0.03):
    random.seed(time.time()); np.random.seed(int(time.time()))
    model = snn()
    for ep in range(EPOCHS):
        correct = 0
        s_correct = 0
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            model.clear()
            pred = present(model, x)
            correct += (pred == y)
            s_correct += (pred == y)
            model.backward(y, scale=1.0/len(x_train))
            if i % 16 == 1:
                print(f"batch: {i}/{len(x_train)} acc: {s_correct/16}, last activations: {model.output}")
                s_correct = 0
        model.update(lr)
        print(f"finished epoch {ep} with accuracy {correct/len(x_train)}")
    return model

def evaluate(model, verbose=False):
    T += 35
    with open("model.txt", "w") as f:
        f.write(str([n.w for n in model.l0_n]) + "\n")
        f.write(str([n.w for n in model.l1_n]) + "\n")
        f.write(str([n.w for n in model.l2_n]) + "\n")
    correct = 0
    for x, y in zip(x_val, y_val):
        model.clear()
        pred = present(model, x)
        if True: print(pred, "p vs a", y, "activations", model.output)
        correct += (pred == y)
    return correct/len(x_val)

model = train(int(time.time()))
print(f"Accuracy = {evaluate(model)}")