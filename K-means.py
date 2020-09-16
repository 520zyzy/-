import numpy as np
import matplotlib.pyplot as plt


def dataset():
    dataset = np.random.uniform(-10, 10, (1000, 2))
    return dataset


def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataset, k):
    n = np.shape(dataset)[1]
    centroids = np.mat(np.zeros((k, n)))
    # print(centroids)
    for j in range(n):
        minJ = min(dataset[:, j])
        rangeJ = float(max(dataset[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def Kmeans(dataset, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataset)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataset, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.Inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataset[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if len(ptsInClust) != 0:
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
        print(clusterAssment)
    return centroids, clusterAssment


def resultImage(dataset, centroids, clusterAssment, k):
    # print(centroids)
    x = np.array(centroids[:, 0])[:, 0]
    y = np.array(centroids[:, 1])[:, 0]
    x = np.array(x).tolist()
    y = np.array(y).tolist()
    print(x)
    print(y)

    color = ['b', 'y', 'g', 'k']
    label = ['p', '8', '^', 's']
    plt.scatter(x, y, color='r', marker='D')
    for cent in range(k):
        ptsInClust = dataset[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
        if len(ptsInClust) != 0:
            print("-------" + str(cent) + "-------------")
            print(ptsInClust)
            x = np.array(ptsInClust[:, 0])
            y = np.array(ptsInClust[:, 1])
            plt.scatter(x, y, marker=label[cent], color=color[cent])

    plt.show()


def main():
    Dataset = dataset()
    centroids, clusterAssment = Kmeans(Dataset, 4)
    resultImage(Dataset, centroids, clusterAssment, 4)


if __name__ == '__main__':
    main()
