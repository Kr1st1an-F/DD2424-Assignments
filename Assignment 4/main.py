import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class RNN:
    def __init__(self, k=1, m=100, eta=0.1, seqLength=25, sig=0.01):
        self.m, self.k, self.eta, self.seqLength, self.sig = m, k, eta, seqLength, sig
        self.initWeightsBiases()

    def initWeightsBiases(self):
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.k, 1))
        self.u = np.random.rand(self.m, self.k) * self.sig
        self.w = np.random.rand(self.m, self.m) * self.sig
        self.v = np.random.rand(self.k, self.m) * self.sig


def readData():
    file = open("C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assignment4/goblet_book.txt", 'r')
    Chars = file.read()
    return Chars, set(Chars)


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def synthesize(rnn, h0, x0, n):
    h = h0[:, np.newaxis]
    x = x0
    samples = np.zeros((x0.shape[0], n))

    for t in range(n):
        a = np.dot(rnn.w, h) + np.dot(rnn.u, x) + rnn.b
        h = np.tanh(a)
        o = np.dot(rnn.v, h) + rnn.c
        p = softmax(o.squeeze())
        idx = np.random.choice(len(p), p=p)
        x = np.zeros_like(x0)
        x[idx] = 1
        samples[:, t] = x.ravel()

    return samples


def forward(rnn, h0, x):
    h = np.zeros((rnn.m, x.shape[1] + 1))
    h[:, 0] = h0
    a = np.zeros((rnn.m, x.shape[1]))
    prob = np.zeros(x.shape)

    for t in range(x.shape[1]):
        a[:, t] = (rnn.w @ h[:, t][:, np.newaxis] + rnn.u @ x[:, t][:, np.newaxis] + rnn.b).flatten()
        h[:, t + 1] = np.tanh(a[:, t])
        o = rnn.v @ h[:, t + 1][:, np.newaxis] + rnn.c
        p = softmax(o)
        prob[:, t] = p.flatten()

    return prob, h[:, 1:], a


def backprop(rnn, y, p, h, hPrev, a, x):
    # Initialize gradients
    VGradient = np.zeros_like(rnn.v)
    WGradient = np.zeros_like(rnn.w)
    UGradient = np.zeros_like(rnn.u)
    BGradient = np.zeros_like(rnn.b)
    CGradient = np.zeros_like(rnn.c)
    NextGradient = np.zeros_like(h[:, 0])

    # Backward pass
    for t in reversed(range(x.shape[1])):
        grad_o = p[:, t] - y[:, t]
        VGradient += np.outer(grad_o, h[:, t])
        CGradient += grad_o
        grad_h = np.dot(grad_o, rnn.v) + NextGradient
        grad_a = grad_h * (1 - h[:, t] ** 2)
        WGradient += np.outer(grad_a, h[:, t - 1] if t > 0 else hPrev)
        UGradient += np.outer(grad_a, x[:, t])
        BGradient += grad_a
        NextGradient = np.dot(grad_a, rnn.w)

    # Clip gradients
    Gradients = RNN()
    Gradients.v = VGradient
    Gradients.w = WGradient
    Gradients.u = UGradient
    Gradients.b = BGradient[:, np.newaxis]
    Gradients.c = CGradient[:, np.newaxis]

    return Gradients


def computeLoss(y, p):
    return -np.sum(np.log(np.sum(y * p, axis=0)))


def oneHot(vec, conversor):
    mat = np.zeros((len(conversor), len(vec)))
    indices = [conversor[char] for char in vec]
    mat[indices, np.arange(len(vec))] = 1
    return mat


def adaGrad(oldM, gradient, oldParam, learning_rate):
    updatedM = oldM + np.square(gradient)
    updatedParam = oldParam - (learning_rate / np.sqrt(updatedM + np.finfo(float).eps)) * gradient

    return updatedParam, updatedM


def main():
    np.random.seed(10)
    Chars, UniqueChars = readData()

    CharIndex = {x: idx for idx, x in enumerate(UniqueChars)}
    IndexChar = {idx: x for idx, x in enumerate(UniqueChars)}

    d = len(UniqueChars)  # Dimensionality == number of different characters
    rnn = RNN(k=d)
    hPrev = np.zeros(rnn.m)
    lossList = list()
    smoothLoss = 0
    mList = [0, 0, 0, 0, 0]
    bestRNN = RNN()
    bestLoss = float('inf')


    for epoch in tqdm(range(2)):
        e = 0
        while e <= len(Chars) - rnn.seqLength:
            x = oneHot(Chars[e:e + rnn.seqLength], CharIndex)
            y = oneHot(Chars[e + 1:e + 1 + rnn.seqLength], CharIndex)
            p, h, a = forward(rnn, hPrev, x)
            rnnGradients = backprop(rnn, y, p, h, hPrev, a, x)
            attributes = ['b', 'c', 'u', 'w', 'v']

            # Update gradients and parameters
            for idx, attribute in enumerate(attributes):
                gradients = getattr(rnnGradients, attribute)
                gradients = np.clip(gradients, -5, 5)
                updatedAttribute, updatedMValue = adaGrad(mList[idx], gradients, getattr(rnn, attribute), rnn.eta)
                setattr(rnn, attribute, updatedAttribute)
                mList[idx] = updatedMValue

            currentLoss = computeLoss(y, p)
            smoothLoss = 0.999 * smoothLoss + 0.001 * currentLoss

            if e == 0 and epoch == 0:
                lossList.append(smoothLoss)
                bestRNN = RNN()
                bestLoss = smoothLoss
            elif smoothLoss < bestLoss:
                bestRNN = RNN()
                bestLoss = smoothLoss

            if e % (rnn.seqLength * 100) == 0:
                lossList.append(smoothLoss)

            hPrev = h[:, -1]
            e += rnn.seqLength

        hPrev = np.zeros(hPrev.shape)


    plt.plot(np.arange(len(lossList)) * 100, lossList, label="Training loss")
    plt.xlabel("Update step")
    plt.ylabel("Loss")
    plt.show()

    x0 = oneHot('.', CharIndex)
    print("Lowest loss: " + str(bestLoss))
    hPrev = np.zeros(rnn.m)
    samples = synthesize(bestRNN, hPrev, x0, 1000)
    samples = [IndexChar[int(np.argmax(samples[:, n]))] for n in range(samples.shape[1])]
    print("\n")
    print("".join(samples))
    print("\n")


if __name__ == "__main__":
    main()
