import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == "__main__":
    # dblp
    # plt_micro = [0.39, 0.60, 0.63, 0.66, 0.66, 0.66, 0.67, 0.67, 0.67, 0.67]
    # plt_macro = [0.35, 0.57, 0.59, 0.63, 0.63, 0.63, 0.64, 0.64, 0.65, 0.65]

    # books
    # plt_micro = [0.44, 0.50, 0.56, 0.58, 0.59, 0.60, 0.61, 0.61, 0.61, 0.61]
    # plt_macro = [0.46, 0.52, 0.57, 0.59, 0.59, 0.60, 0.62, 0.61, 0.61, 0.61]

    #dblp seeds
    # plt_micro = [0.52, 0.59, 0.66, 0.68, 0.70]
    # plt_macro = [0.51, 0.58, 0.63, 0.66, 0.67]

    # books seeds
    plt_micro = [0.48, 0.56, 0.62, 0.65, 0.67]
    plt_macro = [0.47, 0.55, 0.63, 0.65, 0.66]

    ax = plt.figure().gca()
    plt_x = [1, 2, 3, 4, 5]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.ylim(0, 1)

    # plt_micro = [0.58, 0.59, 0.60, 0.65, 0.61, 0.61, 0.62, 0.62, 0.62]
    # plt_macro = [0.62, 0.70, 0.78, 0.79]

    plt.plot(plt_x, plt_micro, label="Micro F1")
    plt.plot(plt_x, plt_macro, label="Macro F1")
    # plt.xlabel("Number of seedwords", fontsize=22)
    # plt.ylabel("F1 score", fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(prop={'size': 22}, loc="lower right")

    plt.savefig('./books_seeds.png')
    # plt.show()
