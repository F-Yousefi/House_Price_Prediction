"""This module contains the class of Linear regression model."""
import numpy as np


class LinearRegression:

    def __init__(self):
        self.__repo = 0
        self.__params = 0
        self.__weights = 0
        self.__intecept = 0

    def score(self, x, y_true):
        y_pred = np.transpose(
            np.dot(
                self.__weights,
                np.transpose(x))) + self.__intecept
        score = 1 - (((y_true - y_pred) ** 2).sum()) / \
            (((y_true - y_true.mean()) ** 2).sum())
        return score

    def predict(self, x):
        return np.transpose(
            np.dot(
                self.__weights,
                np.transpose(x))) + self.__intecept

    def get_report(self):
        return self.__repo

    def set_params(self, w, b, rpo_obj):
        self.__repo = rpo_obj
        self.__weights = w
        self.__intecept = b


class LinearRegressionGD(LinearRegression):

    def fit(self, x, y, alpha=0.7, acceptable_cost=0.1e-5):
        costs = []
        rpo_obj = {}
        w = np.random.rand(1, x.shape[1])
        b = 1
        i = 0
        print("\nLINEAR REGRESSION: training process has been started ...\n")
        while True:
            y_hat = np.transpose(np.dot(w, np.transpose(x))) + b
            def J(y, y_hat): return 1 / (2 * len(y_hat)) * sum((y - y_hat)**2)
            cost = J(y, y_hat)
            def dJw(y, y_hat, x): return 1 / \
                (len(y_hat)) * sum((y_hat - y) * x)

            def dJb(y, y_hat): return 1 / (len(y_hat)) * sum((y_hat - y))
            w_next = w - alpha * dJw(y, y_hat, x)
            b_next = b - alpha * dJb(y, y_hat)
            if abs(np.sum(w_next - w)) > acceptable_cost:
                w, b = w_next, b_next
                i += 1
                if i % 100 == 0 and i >= 100:
                    costs.append(cost)
                    print(
                        "ITERATION: #{1} ,\n COST: {0},\n COEFFICIENT VALUE: {2},\
                        \n INTERCEPT VALUE: {3} \n ====================" .format(
                            round(
                                cost[0], 7), i, np.round(
                                w, 7), np.round(
                                b, 7)))

            else:
                print("\nThe training process has been finished.......\n")
                break

        rpo_obj["costs"] = costs
        rpo_obj["weights"] = w
        rpo_obj["intercept"] = b
        self.set_params(w, b, rpo_obj)

        return self


class LinearRegressionNE(LinearRegression):

    def fit(self, x, y):
        x = np.c_[np.ones((x.shape[0], 1)), x]
        x_transpose = np.transpose(x)
        x_transpose_x = np.dot(x_transpose, x)
        x_transpose_y = np.dot(x_transpose, y)
        rpo_obj = {}

        try:
            theta = np.linalg.solve(x_transpose_x, x_transpose_y)
            rpo_obj["weights"] = theta[1:].T
            rpo_obj["intercept"] = theta[0]
            w, b = theta[1:].T, theta[0]
            self.set_params(w, b, rpo_obj)
            return self

        except np.linalg.LinAlgError:
            return None
