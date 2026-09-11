import numpy as np
from main import MLP_2_2_1, SingleLayerPerceptron, descale_y, sigmoid


W1 = np.load("mlp_W1.npy")
W2 = np.load("mlp_W2.npy")

mlp = MLP_2_2_1()
mlp.W1 = W1
mlp.W2 = W2

slp = SingleLayerPerceptron(2)
slp.W = np.load("slp_W.npy")

c0 = 3.0
c1 = -8.0
X_min, X_max = -8.0, 3.0

def run():
    print("Интерактивный режим. Введите A B или exit.")
    while True:
        s = input("A B: ")
        if s == "exit":
            break
        a,b = map(float, s.split())
        x = np.array([[a,b]])
        x_n = (x - X_min) / (X_max - X_min) * 2 - 1

        out_mlp,_,_,_ = mlp.forward(x_n)
        out_slp,_ = slp.forward(x_n)

        y_mlp = descale_y(out_mlp)[0,0]
        y_slp = descale_y(out_slp)[0,0]

        cls_mlp = c0 if abs(y_mlp-c0) < abs(y_mlp-c1) else c1
        cls_slp = c0 if abs(y_slp-c0) < abs(y_slp-c1) else c1

        print("MLP:", y_mlp, "класс:", cls_mlp)
        print("SLP:", y_slp, "класс:", cls_slp)

run()
