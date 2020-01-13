import numpy as np
from page_rank import powerIteration
from scipy import sparse
from fast_pagerank import pagerank
from fast_pagerank import pagerank_power

if __name__ == "__main__":
    data_path = "./data/"

    A = np.array([[0, 1], [0, 2], [3, 1]])
    weights = [1, 1, 1]
    G = sparse.csr_matrix((weights, (A[:, 0], A[:, 1])), shape=(4, 4))
    pr = pagerank(G, p=0.85)

    # edgeWeights = collections.defaultdict(lambda: collections.Counter())
    # edgeWeights[1][2] = 1.0
    # edgeWeights[1][3] = 1.0
    # edgeWeights[4][2] = 1.0
    # edgeWeights[5][2] = 1.0

    # edgeWeights[2][1] = 1.0
    # edgeWeights[3][1] = 1.0
    # edgeWeights[2][4] = 1.0
    # edgeWeights[2][5] = 1.0


    # wordProbabilities = powerIteration(edgeWeights, rsp=0.15)
    # sorted_wordprobs = wordProbabilities.sort_values(ascending=False)
    pass
