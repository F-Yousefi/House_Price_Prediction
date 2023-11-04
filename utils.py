import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def cost_plotter(repo):
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)
    fig.set_dpi(100)
    ax.plot(np.arange(0, len(repo["costs"]), 1) *
            100, np.array(repo["costs"]), color='blue')
    plt.show()


def scikit_truth_weights(x, y, repo):

    reg = LinearRegression().fit(x, y)
    reg.score(x, y)
    w = reg.coef_
    b = reg.intercept_
    print(
        "\n.....\nThe Weights Calsulated by Current Model : \n{0}, \
  \n\nThe Weights Calsulated by Scikit Learn Linear Regression Model : \n{1}\
  \n\nThe distance between the truth weights vs calculated weights: {2}." .format(
            repo["weights"][0], w[0], round(
                np.linalg.norm(
                    w - repo["weights"], axis=1)[0], 5)))
