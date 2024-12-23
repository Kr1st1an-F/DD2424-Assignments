import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(50)


def initNetwork(dataTrain, layerNodes, layers=2, he=False, sigma=None):
    weights = list()
    bias = list()
    gamma = list()
    beta = list()
    if he:
        num = 2
    else:
        num = 1

    if sigma is not None:
        weights.append(np.random.normal(0, sigma, (layerNodes[0], dataTrain.shape[0])))
    else:
        weights.append(np.random.normal(0, 1 / np.sqrt(dataTrain.shape[0]), (layerNodes[0], dataTrain.shape[0])))

    bias.append(np.zeros((layerNodes[0], 1)))

    for i in range(1, layers):
        if sigma is not None:
            weights.append(np.random.normal(0, sigma, (layerNodes[i], weights[-1].shape[0])))
        else:
            weights.append(
                np.random.normal(0, np.sqrt(num / weights[-1].shape[0]), (layerNodes[i], weights[-1].shape[0]))
                / num)
        bias.append(np.zeros((layerNodes[i], 1)))

    for layerNode in layerNodes[:-1]:
        gamma.append(np.ones((layerNode, 1)))
        beta.append(np.zeros((layerNode, 1)))

    return weights, bias, gamma, beta


def findBestN(array, n):
    array_copy = np.copy(array)
    indices = list()
    for i in range(n):
        index = np.argmax(array_copy)
        array_copy[index] = -1
        indices.append(int(index))

    return indices


def BatchNormalization(s, mu, var):
    epsilon = np.finfo(float).eps
    inv_sqrt_var = np.diag(1 / np.sqrt(var + epsilon))
    centered_s = s - mu[:, np.newaxis]
    return inv_sqrt_var @ centered_s


def BatchNormalizationBackPass(g, s, mu, var):
    epsilon = np.finfo(float).eps
    sigma1 = 1 / np.sqrt(var + epsilon)
    sigma2 = 1 / np.power(var + epsilon, 1.5)
    g1 = g * sigma1
    g2 = g * sigma2
    D = s - mu[:, np.newaxis]
    c = np.sum(g2 * D, axis=1)[:, np.newaxis]
    g_batch = g1 - (1 / g.shape[1]) * np.sum(g1, axis=1)[:, np.newaxis] - (1 / g.shape[1]) * D * c

    return g_batch


def forwardPass(dataTrain, weights, bias, gamma=None, beta=None, mu=None, var=None, batchNorm=False):
    if not batchNorm:
        output = list()
        sList = list()
        output.append(np.copy(dataTrain))
        for i in range(1, len(weights)):
            output.append(np.maximum(0, sList[-1]))
            sList.append(weights[0] @ output[-1] + bias[0])
        sList.append(weights[-1] @ output[-1] + bias[-1])
        output.append(softmax(sList[-1]))

        return output, sList
    else:
        output = list()
        sList = list()
        sHatList = list()
        muList = list()
        varList = list()
        output.append(np.copy(dataTrain))
        for i in range(len(weights) - 1):
            sList.append(weights[i] @ output[-1] + bias[i])
            if mu is None and var is None:
                mu = np.mean(sList[-1], axis=1)
                var = np.var(sList[-1], axis=1)
            else:
                muList.append(mu[i])
                varList.append(var[i])
            sHatList.append(BatchNormalization(sList[-1], muList[-1], varList[-1]))
            tilde = gamma[i] * sHatList[-1] + beta[i]
            output.append(np.maximum(0, tilde))

        sList.append(weights[-1] @ output[-1] + bias[-1])
        output.append(softmax(sList[-1]))

    return output, sList, sHatList, muList, varList


def cyclicalUpdate(t, n_s, etaMin, etaMax):
    cycle = t // (2 * n_s)
    within_cycle = t % (2 * n_s)

    if within_cycle <= n_s:
        return etaMin + within_cycle / n_s * (etaMax - etaMin)
    else:
        return etaMax - (within_cycle - n_s) / n_s * (etaMax - etaMin)


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def computeLoss(data, labels, weights, bias, gamma=None, beta=None, mu=None, var=None, batchNorm=False):
    p = forwardPass(data, weights, bias, gamma, beta, mu, var, batchNorm)[0][-1]
    lCrossSum = np.sum(-np.log(np.sum(labels * p, axis=0)))
    return (1 / data.shape[1]) * lCrossSum


def computeCost(data, labels, weights, bias, lmb, gamma=None, beta=None, mu=None, var=None, batchNorm=False):
    loss = computeLoss(data, labels, weights, bias, gamma, beta, mu, var, batchNorm)
    reg = lmb * np.sum([np.sum(np.square(w)) for w in weights])  # Regularization term L2
    return loss + reg


def computeAccuracy(data, labels, weights, bias, gamma=None, beta=None, mu=None, var=None, batchNorm=False):
    p = forwardPass(data, weights, bias, gamma, beta, mu, var, batchNorm)[1][-1]
    prediction = np.argmax(p, axis=0)
    real = np.argmax(labels, axis=0)
    return np.sum(real == prediction) / len(real)


def computeGradsAnalytic(data, labels, weights, lmb, p, sList, sHatList=None, gamma=None, mu=None, var=None,
                         batchNorm=False):
    gradWeights = []
    gradBias = []

    if not batchNorm:
        g = -(labels - p)

        for i in reversed(range(len(weights))):
            gradWeights.append((g @ data[i].T) / data[0].shape[1] + 2 * lmb * weights[i])
            gradBias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
            g = weights[i].T @ g
            diag = np.copy(data[i])
            diag[diag > 0] = 1
            g = g * diag
        gradWeights.reverse(), gradBias.reverse()

        return gradWeights, gradBias

    else:
        gradGamma = list()
        gradBeta = list()

        g = -(labels - p)
        gradWeights.append((g @ data[-2].T) / data[0].shape[1] + 2 * lmb * weights[-1])
        gradBias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
        g = weights[-1].T @ g
        diag = np.copy(data[-2])
        diag[diag > 0] = 1
        g = g * diag

        for i in reversed(range(len(weights) - 1)):
            gradGamma.append(np.sum(g * sHatList[i], axis=1)[:, np.newaxis] / data[0].shape[1])
            gradBeta.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
            g = gamma[i] * g
            g = BatchNormalizationBackPass(g, sList[i], mu[i], var[i])
            gradWeights.append((g @ data[i].T) / data[0].shape[1] + 2 * lmb * weights[i])
            gradBias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
            if i > 0:
                g = weights[i].T @ g
                diag = np.copy(data[i])
                diag[diag > 0] = 1
                g = g * diag
        gradWeights.reverse(), gradBias.reverse(), gradGamma.reverse(), gradBeta.reverse()

        return gradWeights, gradBias, gradGamma, gradBeta


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def loadData(size_val=5000):
    batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    file = unpickle('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment3/Datasets/' + batch[0])
    dataTrain = file['data']
    labelsTrain = file['labels']

    for i in range(1, 5):
        file = unpickle('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment3/Datasets/' + batch[i])
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
    plt.plot(range(len(train)), train, label="Validation Loss" + mode, color="Black")
    plt.plot(range(len(val)), val, label="Training Loss" + mode, color="Blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.grid()


def train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                  weights, bias, gamma, beta, nBatch, eta, n_s, etaMin, etaMax, cycles=2,
                  plotting=False, bestLambda=None, lmbSearch=True, alpha=0.9, batchNorm=False):
    if lmbSearch:
        if bestLambda is None:
            lVal = np.random.uniform(-5, -1)  # Random sample from -5 to -1 interval in log10 scale
            lmb = pow(10, lVal)  # Get random lambda
        else:
            dist = abs(bestLambda[0] - bestLambda[1])
            lVal = np.random.uniform(bestLambda[0] - dist, bestLambda[1] + dist)  # Random sample from
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
        meanAverage = list()
        varAverage = list()
        for j in range(int(dataTrain.shape[1] / nBatch)):
            start = j * nBatch
            end = (j + 1) * nBatch
            if not batchNorm:

                data, sList = forwardPass(dataTrain[:, start:end], weights, bias)
                deltaW, deltaB = computeGradsAnalytic(data, labelsTrain[:, start:end],
                                                      weights, lmb, data[-1], sList)

                weights = [weights[i] - eta * deltaW[i] for i in range(len(weights))]
                bias = [bias[i] - eta * deltaB[i] for i in range(len(bias))]
            else:
                data, sList, sHatList, muList, varList = forwardPass(dataTrain[:, start:end], weights, bias, gamma,
                                                                     beta, batchNorm)
                deltaW, deltaB, deltaGamma, deltaBeta = computeGradsAnalytic(data, labelsTrain[:, start:end],
                                                                             weights, lmb, data[-1], sList, sHatList,
                                                                             gamma, muList, varList, batchNorm)

                weights = [weights[i] - eta * deltaW[i] for i in range(len(weights))]
                bias = [bias[i] - eta * deltaB[i] for i in range(len(bias))]
                gamma = [gamma[i] - eta * deltaGamma[i] for i in range(len(gamma))]
                beta = [beta[i] - eta * deltaBeta[i] for i in range(len(beta))]

                if j == 0:
                    for i in range(len(muList)):
                        meanAverage.append(muList[i])
                        varAverage.append(varList[i])
                else:
                    meanAverage = [
                        [alpha * x for x in meanAverage[i]] + [(1 - alpha) * muList[i][j] for i in range(len(muList))]
                        for j in range(len(muList[0]))]
                    varAverage = [
                        [alpha * x for x in varAverage[i]] + [(1 - alpha) * varList[i][j] for i in range(len(varList))]
                        for j in range(len(varList[0]))]

                if t == nEpoch - 1:
                    mu = muList
                    var = varList
            eta = cyclicalUpdate(t * cyclesPerEpoch + j, n_s, etaMin, etaMax)

        if plotting and batchNorm:
            trainingLoss.append(
                computeLoss(dataTrain, labelsTrain, weights, bias, gamma, beta, meanAverage, varAverage, batchNorm))
            validationLoss.append(
                computeLoss(dataVal, labelsVal, weights, bias, gamma, beta, meanAverage, varAverage, batchNorm))
            trainingCost.append(
                computeCost(dataTrain, labelsTrain, weights, bias, lmb, gamma, beta, meanAverage, varAverage,
                            batchNorm))
            validationCost.append(
                computeCost(dataVal, labelsVal, weights, bias, lmb, gamma, beta, meanAverage, varAverage, batchNorm))
            accTraining.append(
                computeAccuracy(dataTrain, labelsTrain, weights, bias, gamma, beta, meanAverage, varAverage, batchNorm))
            accVal.append(
                computeAccuracy(dataVal, labelsVal, weights, bias, gamma, beta, meanAverage, varAverage, batchNorm))

        elif plotting:
            trainingLoss.append(computeLoss(dataTrain, labelsTrain, weights, bias))
            validationLoss.append(computeLoss(dataVal, labelsVal, weights, bias))
            trainingCost.append(computeCost(dataTrain, labelsTrain, weights, bias, lmb))
            validationCost.append(computeCost(dataVal, labelsVal, weights, bias, lmb))
            accTraining.append(computeAccuracy(dataTrain, labelsTrain, weights, bias))
            accVal.append(computeAccuracy(dataVal, labelsVal, weights, bias))

    if plotting:
        plotResults(accTraining, accVal, "accuracy")
        plotResults(trainingLoss, validationLoss, "loss")
        plotResults(trainingCost, validationCost, "cost")

    if batchNorm:
        return weights, bias, gamma, beta, meanAverage, varAverage, computeAccuracy(dataVal, labelsVal, weights, bias,
                                                                                    gamma, beta, meanAverage,
                                                                                    varAverage, batchNorm), np.log10(
            lmb)

    return weights, bias, computeAccuracy(dataVal, labelsVal, weights, bias), np.log10(lmb)


def main():
    np.random.seed(10)
    # np.random.seed(100)

    # Read data
    dataTrain, labelsTrain, dataVal, labelsVal, dataTest, labelsTest, labelNames = preprocessData(size_val=5000)
    # Initialize model parameters
    weights, bias, gamma, beta = initNetwork(dataTrain, [50, 50, 10], 3, he=True, sigma=None)
    nBatch = 100
    etaMin = 1e-5
    etaMax = 1e-1
    eta = etaMin
    n_s = 2 * int(dataTrain.shape[1] / nBatch)

    lmbSearch = 10
    nLmb = 3
    bestAcc = np.zeros(lmbSearch)
    bestLmb = np.zeros(lmbSearch)

    if not os.path.isfile('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data1.txt'):
        for i in range(lmbSearch):
            acc, lmb = train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                                     weights, bias, gamma, beta, nBatch, eta, n_s, etaMin, etaMax, cycles=2,
                                     plotting=False, bestLambda=None, lmbSearch=True, batchNorm=True)[6:]
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
    print("Best accuracies -> " + str(bestAcc))
    #print("Best lambdas -> " + str(bestLmb))

    if not os.path.isfile('C:/Users/Kristian/CivilIngenjor 4-5/DD2424 - Deep Learning/Assigment2/Data/data2.txt'):
        improvedAcc = np.zeros(lmbSearch + nLmb)
        improvedLmb = np.zeros(lmbSearch + nLmb)
        improvedAcc[0:nLmb] = bestAcc
        improvedLmb[0:nLmb] = bestLmb
        for i in range(1, lmbSearch):
            acc, lmb = train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                                     weights, bias, gamma, beta, nBatch, eta, n_s, etaMin, etaMax, cycles=2,
                                     plotting=False, bestLambda=bestLmb[:2], lmbSearch=True, batchNorm=True)[6:]
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
    print("The improved lambdas -> " + str(improvedLmb))
    # Training with the best found parameters
    indicesVal = np.random.choice(range(dataVal.shape[1]), 4000, replace=False)  # Select random samples in validation

    weights, bias, gamma, beta, mu, var = train_network(dataTrain, labelsTrain, dataVal, labelsVal,
                                                        weights, bias, gamma, beta, nBatch, eta, n_s, etaMin, etaMax,
                                                        cycles=3,
                                                        plotting=True, bestLambda=improvedLmb[0], lmbSearch=False,
                                                        batchNorm=True)[:6]
    # Check accuracy over test data
    print("Accuracy on the test data -> " + str(
        computeAccuracy(dataTest, labelsTest, weights, bias, gamma, beta, mu, var, batchNorm=True) * 100) + "%")


if __name__ == "__main__":
    main()
