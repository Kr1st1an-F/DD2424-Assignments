import pickle
import numpy as np
import matplotlib.pyplot as plt


# Exercise 1.1
def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def LoadBatch(filename):
    data = unpickle(filename)
    images = data[b'data'].T / 255
    labels = data[b'labels']
    oneHot = np.zeros(shape=(10, len(labels)))
    for i, label in enumerate(labels):
        oneHot[label, i] = 1

    return images, labels, oneHot


image1, label1, oneHot1 = LoadBatch(
    'C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment1/Datasets/data_batch_1')
image2, label2, oneHot2 = LoadBatch(
    'C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment1/Datasets/data_batch_2')
image3, label3, oneHot3 = LoadBatch(
    'C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment1/Datasets/data_batch_3')
image4, label4, oneHot4 = LoadBatch(
    'C:/Users/Kristian/CivilIngenjor 4-5/DD2424- Deep Learning/Assigment1/Datasets/data_batch_4')
image5, label5, oneHot5 = LoadBatch(
    'C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment1/Datasets/data_batch_5')
imageTest, labelTest, oneHotTest = LoadBatch(
    'C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment1/Datasets/test_batch')
batches = unpickle('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment1/Datasets/batches.meta')
labelNames = [label.decode('utf-8') for label in batches[b'label_names']]


# Exercise 1.2
def initWeight(inputD, outputD, false=None):
    #
    scale = 1 / np.sqrt(inputD) if false else 0.01

    np.random.seed(0)
    weight = np.random.normal(size=(outputD, inputD)) * scale

    np.random.seed(0)
    bias = np.random.normal(size=(outputD, 1)) * scale

    return weight, bias


input_dimension = image1.shape[0]
output_dimension = oneHot1.shape[0]
weight, bias = initWeight(input_dimension, output_dimension)


# Exercise 1.3
def softmax(S):
    return np.exp(S) / np.sum(np.exp(S), axis=0)


def EvaluateClassifier(X, weight, bias):
    S = weight @ X + bias
    P = softmax(S)

    return P


# Exercise 1.4
def ComputeCost(X, Y, W, b, lambda_):
    # Compute the predictions
    P = EvaluateClassifier(X, W, b)

    # Compute the loss function term
    loss_cross = sum(-np.log((Y * P).sum(axis=0)))

    # Compute the regularization term
    loss_regularization = lambda_ * (W ** 2).sum()

    # Sum the total cost
    J = loss_cross / X.shape[1] + loss_regularization

    return J


# Exercise 1.5
def ComputeAccuracy(X, y, W, b):
    P = EvaluateClassifier(X, W, b)
    predictions = np.argmax(P, axis=0)
    accuracy = np.sum(predictions == y) / len(y)

    return accuracy


# Exercise 1.6
def ComputeGradients(X, Y, P, W, lambda_):
    n = X.shape[1]
    C = Y.shape[0]
    G = -(Y - P)
    grad_W = (G @ X.T) / n + 2 * lambda_ * W
    grad_b = (G @ np.ones(shape=(n, 1)) / n).reshape(C, 1)

    return grad_W, grad_b


def ComputeGradsNum(X, Y, W, b, lambda_, h=0.000001):
    grad_W = np.zeros(shape=W.shape)
    grad_b = np.zeros(shape=b.shape)
    c = ComputeCost(X, Y, W, b, lambda_)

    for i in range(b.shape[0]):
        b_try = b.copy()
        b_try[i, 0] = b_try[i, 0] + h
        c2 = ComputeCost(X, Y, W, b_try, lambda_)
        grad_b[i, 0] = (c2 - c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = W.copy()
            W_try[i, j] = W_try[i, j] + h
            c2 = ComputeCost(X, Y, W_try, b, lambda_)
            grad_W[i, j] = (c2 - c) / h

    return grad_W, grad_b


X = image1[:, 0:5]
Y = oneHot1[:, 0:5]
lambda_ = 0.01

# Compute the gradients analytically
P = EvaluateClassifier(X, weight, bias)
gradAnalyticalWeight, gradAnalyticalBias = ComputeGradients(X, Y, P, weight, lambda_)

# Compute the gradients numerically
gradNumericallyWeight, gradNumericallyBias = ComputeGradsNum(X, Y, weight, bias, lambda_)

# Absolute error between numerically and analytically computed gradient.
gradDiffWeight = np.abs(gradNumericallyWeight - gradAnalyticalWeight)
gradDiffBias = np.abs(gradNumericallyBias - gradAnalyticalBias)
print('For weights: ' + str(np.mean(gradDiffWeight < 1e-6) * 100))
print('For bias: ' + str(np.mean(gradDiffBias < 1e-6) * 100))

gradSumWeight = np.maximum(np.abs(gradNumericallyWeight) + np.abs(gradAnalyticalWeight), 0.00001)
gradSumBias = np.maximum(np.abs(gradNumericallyBias) + np.abs(gradAnalyticalBias), 0.00001)
print('For weights: ' + str(np.mean(gradDiffWeight / gradSumWeight < 0.000001) * 100))
print('For bias: ' + str(np.mean(gradDiffBias / gradSumBias < 0.000001) * 100))


# Exercise 1.7
def MiniBatchGD(X, Y, y, GDparams, weight, bias=None, lambda_=0):
    n = X.shape[1]
    eta = GDparams['eta']
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']

    # Create a copy of weights to update
    weight = weight.copy()

    # Create a dictionary to store the performance metrics
    metrics = {'epochs': [], 'loss_train': [], 'acc_train': []}

    # Iterate epochs
    for epoch in range(n_epochs):

        # Iterate data batches or splits
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = range(j_start, j_end)
            X_batch = X[:, inds]
            Y_batch = Y[:, inds]
            y_batch = [y[index] for index in inds]

            # Compute gradients and update weights and bias for this batch
            PBatch = EvaluateClassifier(X_batch, weight, bias)
            gradWeight, gradBias = ComputeGradients(X_batch, Y_batch, PBatch, weight, lambda_)
            weight += -eta * gradWeight
            bias += -eta * gradBias


        # Save the performance metrics of the epoch
        metrics['epochs'].append(epoch + 1)
        metrics['trainAccuracy'].append(ComputeAccuracy(X, y, weight, bias))
        metrics['trainLoss'].append(ComputeCost(X, Y, weight, bias, lambda_))

    return weight, metrics


def plotWeights(W, label_names, title=''):
    _, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 2.5))

    for c, ax in enumerate(axes.flatten()):
        # Subset the corresponding output node weights
        image = W[c, :]

        # Show the weights in image format
        minImage = min(image)
        maxImage = max(image)
        image = (image - minImage) / (maxImage - minImage)
        ax.imshow(image.reshape(3, 32, 32).transpose([1, 2, 0]))
        ax.axis('off')
        ax.set_title(label_names[c])


def plotLearningCurve(metrics, title=''):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plt.suptitle('Learning curves ' + title)

    metricDetails = {
        'loss': {'label': 'Loss', 'opt_func': np.argmin},
        'acc': {'label': 'Accuracy', 'opt_func': np.argmax}
    }

    for metric, ax in zip(metricDetails.keys(), axes.flatten()):
        metricInfo = metricDetails[metric]
        train_metrics = metrics[metric + '_train']
        val_metrics = metrics.get(metric + '_val', [])

        # Calculate and plot for both train and validation metrics
        for datasetType, datasetMetrics in [('Train', train_metrics), ('Validation', val_metrics)]:
            if datasetMetrics:
                optimalEpoch = metricInfo['opt_func'](datasetMetrics)
                optimalValue = np.round(datasetMetrics[optimalEpoch], 4)
                label = f'{datasetType}: {optimalValue} at epoch {optimalEpoch + 1}'
                ax.plot(metrics['epochs'], datasetMetrics, label=label)

        ax.set_xlabel("epcoh")
        ax.set_ylabel(metricInfo['label'])
        ax.legend()
        ax.grid(True)


# Press the green button in the gutter to run the script.
lambda1 = 0
lambda2 = 0
lambda3 = 0.1
lambda4 = 1

GDparams1 = {'nBatch': 100, 'nEpochs': 40, 'eta': 0.1, }
GDparams2 = {'nBatch': 100, 'nEpochs': 40, 'eta': 0.001}
GDparams3 = {'nBatch': 100, 'nEpochs': 40, 'eta': 0.001}
GDparams4 = {'nBatch': 100, 'nEpochs': 40, 'eta': 0.001}

# Train the network
weight1, metrics1 = MiniBatchGD(image1, oneHot1, label1, GDparams1, weight, bias, lambda1)
weight2, metrics2 = MiniBatchGD(image1, oneHot1, label1, GDparams2, weight, bias, lambda2)
weight3, metrics3 = MiniBatchGD(image1, oneHot1, label1, GDparams3, weight, bias, lambda3)
weight4, metrics4 = MiniBatchGD(image1, oneHot1, label1, GDparams4, weight, bias, lambda4)

# Plot the learning curve
title = '..'
plotLearningCurve(metrics1, title=title)
#plotLearningCurve(metrics2, title=title)
#plotLearningCurve(metrics3, title=title)
#plotLearningCurve(metrics4, title=title)

# Plot the weights
plotWeights(weight1, labelNames, title=title)
#plotWeights(weight2, labelNames, title=title)
#plotWeights(weight3, labelNames, title=title)
#plotWeights(weight4, labelNames, title=title)
