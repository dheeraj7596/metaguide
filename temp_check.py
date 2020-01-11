import collections
from page_rank import powerIteration

if __name__ == "__main__":
    data_path = "./data/"

    edgeWeights = collections.defaultdict(lambda: collections.Counter())
    edgeWeights[1][2] = 1.0
    edgeWeights[1][3] = 1.0
    edgeWeights[4][2] = 1.0
    edgeWeights[5][2] = 1.0

    edgeWeights[2][1] = 1.0
    edgeWeights[3][1] = 1.0
    edgeWeights[2][4] = 1.0
    edgeWeights[2][5] = 1.0


    wordProbabilities = powerIteration(edgeWeights, rsp=0.15)
    sorted_wordprobs = wordProbabilities.sort_values(ascending=False)
    pass
