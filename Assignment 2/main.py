import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(50)


def initNetwork(dataTrain, labelNames):
    weights = list()
    bias = list()
    weights.append(np.random.normal(0, 1 / np.sqrt(dataTrain.shape[0]),
                                    (50, dataTrain.shape[0])))
    weights.append(np.random.normal(0, 1 / np.sqrt(50),
                                    (len(labelNames), 50)))
    bias.append(np.zeros((50, 1)))
    bias.append(np.zeros((len(labelNames), 1)))

    return weights, bias


def findBestN(array, n):
    array_copy = np.copy(array)
    indices = list()
    for i in range(n):
        index = np.argmax(array_copy)
        array_copy[index] = -1
        indices.append(int(index))

    return indices


def forwardPass(dataTrain, weights, bias):
    output = list()
    sList = list()
    output.append(np.copy(dataTrain))
    sList.append(weights[0] @ dataTrain + bias[0])
    for i in range(1, len(weights)):
        output.append(np.maximum(0, sList[-1]))
        sList.append(weights[0] @ output[-1] + bias[0])

    return output, sList


def cyclicalUpdate(t, n_s, etaMin, etaMax):
    cycle = t // (2 * n_s)
    within_cycle = t % (2 * n_s)

    if within_cycle <= n_s:
        return etaMin + within_cycle / n_s * (etaMax - etaMin)
    else:
        return etaMax - (within_cycle - n_s) / n_s * (etaMax - etaMin)


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def computeLoss(data, labels, weights, bias):
    p = softmax(forwardPass(data, weights, bias)[1][-1])
    lCrossSum = np.sum(-np.log(np.sum(labels * p, axis=0)))
    return (1 / data.shape[1]) * lCrossSum


def computeCost(data, labels, weights, bias, lmb):
    loss = computeLoss(data, labels, weights, bias)
    reg = lmb * np.sum([np.sum(np.square(w)) for w in weights])  # Regularization term L2
    return loss + reg


def computeAccuracy(data, labels, weights, bias):
    p = softmax(forwardPass(data, weights, bias)[1][-1])
    prediction = np.argmax(p, axis=0)
    real = np.argmax(labels, axis=0)
    return np.sum(real == prediction) / len(real)


def computeGradsAnalytic(data, labels, weights, lmb, p):
    gradWeights = []
    gradBias = []

    g = -(labels - p)

    for i in range(len(weights) - 1, -1, -1):
        gradWeights.insert(0, (g @ data[i].T) / data[0].shape[1] + 2 * lmb * weights[i])
        gradBias.insert(0, np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])

        if i != 0:
            g = weights[i].T @ g
            g[data[i] <= 0] = 0

    return gradWeights, gradBias


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def loadData(size_val=5000):
    batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    file = unpickle('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Datasets/' + batch[0])
    dataTrain = file['data']
    labelsTrain = file['labels']

    for i in range(1, 5):
        file = unpickle('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Datasets/' + batch[i])
        dataTrain = np.vstack((dataTrain, file['data']))
        labelsTrain = np.hstack((labelsTrain, file['labels']))

    indicesVal = np.random.choice(range(dataTrain.shape[0]), size_val, replace=False)
    dataVal = dataTrain[indicesVal]
    labelsVal = labelsTrain[indicesVal]
    dataTrain = np.delete(dataTrain, indicesVal, axis=0)
    labelsTrain = np.delete(labelsTrain, indicesVal)

    file = unpickle('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Datasets/test_batch')
    dataTest = file['data']
    labelsTest = file['labels']
    file = unpickle('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Datasets/batches.meta')
    labelNames = file['label_names']

    return dataTrain, labelsTrain, dataVal, labelsVal, dataTest, labelsTest, labelNames


def preprocessImages(data, mean, std):
    data = np.float64(data)
    if mean is None and std is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
    data -= mean
    data /= std
    return np.array(data), mean, std


def oneHot(labels, dim):
    labels_mat = np.zeros((dim, len(labels)))
    for i in range(len(labels)):
        labels_mat[labels[i], i] = 1
    return labels_mat


def preprocessData(size_val=5000):
    dataTrain, labelsTrain, dataVal, labelsVal, dataTest, labelsTest, labelNames = loadData(size_val)
    dataTrain, meanTrain, stdTrain = preprocessImages(dataTrain, mean=None, std=None)
    dataTrain = dataTrain.T
    dataVal = preprocessImages(dataVal, meanTrain, stdTrain)[0].T
    dataTest = preprocessImages(dataTest, meanTrain, stdTrain)[0].T
    labelsTrain = oneHot(labelsTrain, len(labelNames))
    labelsVal = oneHot(labelsVal, len(labelNames))
    labelsTest = oneHot(labelsTest, len(labelNames))
    return dataTrain, labelsTrain, dataVal, labelsVal, dataTest, labelsTest, labelNames


def readImage(colors):
    red = np.array(colors[0:1024]).reshape(32, 32) / 255.0
    green = np.array(colors[1024:2 * 1024]).reshape(32, 32) / 255.0
    blue = np.array(colors[2 * 1024:3 * 1024]).reshape(32, 32) / 255.0

    return np.dstack((red, green, blue))  # Combine the three color channels


def getImages(data):
    result = list()
    for i in range(data.shape[0]):
        result.append(readImage(data[i]))

    return result


def plotResults(train, val, mode):
    plt.plot(range(len(train)), train, label="Training " + mode, color="Black")
    plt.plot(range(len(val)), val, label="Validation " + mode, color="Blue")
    plt.xlabel("Epochs")
    plt.ylabel(mode.capitalize())
    plt.legend()
    plt.show()


def train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                  weights, bias, nBatch, eta, n_s, etaMin, etaMax, cycles=2,
                  plotting=False, bestLambda=None, lmbSearch=True):
    if lmbSearch:
        if bestLambda is None:
            lVal = np.random.uniform(-5, -1)  # Random sample from -5 to -1 interval in log10 scale
            lmb = pow(10, lVal)  # Get random lambda
        else:
            lVal = np.random.uniform(bestLambda[0], bestLambda[1])  # Random sample from
            # the interval defined by the two best previous lambdas in log10 scale
            lmb = pow(10, lVal)  # Define lambda
    else:
        lmb = pow(10, bestLambda)
    iterations = cycles * 2 * n_s  # Number of eta updates
    cyclesPerEpoch = dataTrain.shape[1] / nBatch  # Number of eta update cycles per epoch
    nEpoch = iterations / cyclesPerEpoch  # Define number of epochs needed to perform "cycles" updates
    trainingLoss = list()  # Training data loss per epoch
    validationLoss = list()  # Validation data loss per epoch
    trainingCost = list()  # Training data cost per epoch
    validationCost = list()  # Validation data cost per epoch
    accTraining = list()
    accVal = list()
    etaVal = list()  # Value of eta per cycle
    for t in tqdm(range(int(nEpoch))):
        for j in range(int(dataTrain.shape[1] / nBatch)):
            etaVal.append(eta)
            start = j * nBatch
            end = (j + 1) * nBatch
            data, sList = forwardPass(dataTrain[:, start:end], weights, bias)
            deltaW, deltaB = computeGradsAnalytic(data, labelsTrain[:, start:end],
                                                  weights, lmb, softmax(sList[-1]))
            weights = [weights[i] - eta * deltaW[i] for i in range(len(weights))]
            bias = [bias[i] - eta * deltaB[i] for i in range(len(bias))]
            eta = cyclicalUpdate((t * cyclesPerEpoch) + j, n_s, etaMin, etaMax)
        if plotting:
            trainingLoss.append(computeLoss(dataTrain, labelsTrain, weights, bias))
            validationLoss.append(computeLoss(dataVal, labelsVal, weights, bias))
            trainingCost.append(computeCost(dataTrain, labelsTrain, weights, bias, lmb))
            validationCost.append(computeCost(dataVal, labelsVal, weights, bias, lmb))
            accTraining.append(computeAccuracy(dataTrain, labelsTrain, weights, bias))
            accVal.append(computeAccuracy(dataVal, labelsVal, weights, bias))

    # Show results
    if plotting:
        plt.plot(range(len(etaVal)), etaVal)
        plt.xlabel("Update step")
        plt.ylabel(r"$\eta_{value}$")
        plt.show()
        plotResults(accTraining, accVal, "accuracy")
        plotResults(trainingLoss, validationLoss, "loss")
        plotResults(trainingCost, validationCost, "cost")

    return weights, bias, computeAccuracy(dataVal, labelsVal, weights, bias), np.log10(lmb)


def main():
    np.random.seed(8)
    # Read data
    dataTrain, labelsTrain, dataVal, labelsVal, dataTest, labelsTest, labelNames = preprocessData(size_val=5000)
    # Initialize model parameters
    weights, bias = initNetwork(dataTrain, labelNames)
    nBatch = 100
    etaMin = 1e-5
    etaMax = 1e-1
    eta = etaMin
    n_s = 2 * int(dataTrain.shape[1] / nBatch)

    lmbSearch = 8
    nLmb = 3
    bestAcc = np.zeros(lmbSearch)
    bestLmb = np.zeros(lmbSearch)

    if not os.path.isfile('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data1.txt'):
        for i in range(lmbSearch):
            acc, lmb = train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                                     weights, bias, nBatch, eta, n_s, etaMin, etaMax, cycles=2,
                                     plotting=False, bestLambda=None, lmbSearch=True)[2:]
            bestAcc[i] = acc
            bestLmb[i] = lmb
        indices = findBestN(bestAcc, nLmb)
        bestAcc = bestAcc[indices]  # Get the three best accuracies
        bestLmb = bestLmb[indices]  # Get the three best lambdas
        np.savez_compressed('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data1.txt',
                            acc=bestAcc, lmb=bestLmb)
    else:
        dictData = np.load('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data1.txt',
                           allow_pickle=True)
        bestAcc = dictData['acc']
        bestLmb = dictData['lmb']
    #print("Best accuracies -> " + str(bestAcc))
    #print("Best lambdas -> " + str(bestLmb))

    if not os.path.isfile('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data2.txt'):
        improvedAcc = np.zeros(lmbSearch + nLmb)
        improvedLmb = np.zeros(lmbSearch + nLmb)
        improvedAcc[0:nLmb] = bestAcc
        improvedLmb[0:nLmb] = bestLmb
        for i in range(1, lmbSearch):
            acc, lmb = train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                                     weights, bias, nBatch, eta, n_s, etaMin, etaMax, cycles=4,
                                     plotting=False, bestLambda=bestLmb[:2], lmbSearch=True)[2:]
            improvedAcc[i] = acc
            improvedLmb[i] = lmb
        indices = findBestN(improvedAcc, nLmb)
        improvedAcc = improvedAcc[indices]
        improvedLmb = improvedLmb[indices]
        np.savez_compressed('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data2.txt',
                            acc=improvedAcc, lmb=improvedLmb)
    else:
        dict_data = np.load('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data2.txt',
                            allow_pickle=True)
        improvedAcc = dict_data['acc']
        improvedLmb = dict_data['lmb']
    #print("The improved accuracies -> " + str(improvedAcc))
    #print("The improved lambdas -> " + str(improvedLmb))
    # Training with the best found parameters
    indicesVal = np.random.choice(range(dataVal.shape[1]), 4000, replace=False)  # Select random samples in validation
    dataTrain = np.hstack((dataTrain, dataVal[:, indicesVal]))  # Add previous samples to the training set
    labelsTrain = np.hstack((labelsTrain, labelsVal[:, indicesVal]))  # Add correspondent labels
    dataVal = np.delete(dataVal, indicesVal, axis=1)  # Delete selected samples from validation
    labelsVal = np.delete(labelsVal, indicesVal, axis=1)  # Delete correspondent labels
    n_s = 4 * int(dataTrain.shape[1] / nBatch)  # Step size in eta value modification
    weights, bias = train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                                  weights, bias, nBatch, eta, n_s, etaMin, etaMax, cycles=3,
                                  plotting=True, bestLambda=improvedLmb[0], lmbSearch=False)[0:2]
    # Check accuracy over test data
    print("Accuracy on the test data -> " + str(computeAccuracy(dataTest, labelsTest, weights, bias) * 100) + "%")


if __name__ == "__main__":
    main()
