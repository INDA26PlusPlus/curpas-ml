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
random.seed(time.time())

class neuron():
    def __init__(self, input_weights):
        self.p = 0.0
        self.v = 0.0
        self.leak = 0.0
        self.activation_p = 1.0
        self.w = [random.random()*0.1 for _ in range(input_weights)]
        self.sign = -1 if random.random() < 0.2 else 1

class snn():
    def __init__(self):
        self.l0_n = [neuron(1) for _ in range(784)]
        self.l1_n = [neuron(784) for _ in range(224)]
        self.l2_n = [neuron(224) for _ in range(10)]

        self.l0_v = -1*(1/255.0)*x_train[0] + 1.0
        self.l1_v = [0.0] * 784
        self.l2_v = [0.0] * 224
        self.l3_v = [0.0] * 10

        self.t = 0

    def step(self):
        self.t += 1
        # Layer 0
        for i, n in enumerate(self.l0_n):
            n.p += self.l0_v[i]*0.125 # Fixed input weight
            if n.p > n.activation_p:
                self.l1_v[i] = n.sign
                n.p = 0.0
            else:
                self.l1_v[i] = 0.0
        # Layer 1
        for i, n in enumerate(self.l1_n):
            for j, v in enumerate(self.l1_v):
                n.p += v*n.w[j]
            if n.p > n.activation_p:
                self.l2_v[i] = n.sign
                n.p = 0.0
            else:
                self.l2_v[i] = 0.0
        # Layer 2
        for i, n in enumerate(self.l2_n):
            for j, v in enumerate(self.l2_v):
                n.p += v*n.w[j]
            if n.p > n.activation_p:
                self.l3_v[i] = 1.0
                n.p = 0.0
            else:
                self.l3_v[i] = 0.0
        
        # Output
        classes = []
        for i, v in enumerate(self.l3_v):
            if v != 0.0:
                classes.append(i)
        
        print(f"On step of time {self.t} output neurons {classes} activated")



model = snn()
while True:
    model.step()
    time.sleep(0.2)
