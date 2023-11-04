import pickle
import numpy as np
from sklearn.preprocessing import normalize



class LinearRegression:

    def __init__(self):
        self.__repo = 0
        self.__params = 0
        self.__weights = 0
        self.__intecept = 0

    def fit(self, x, y, alpha=0.7, acceptable_cost=0.1e-5):
        x, y = self.__norm(x, y)
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
                        "ITERATION: #{1} ,\n COST: {0},\n COEFFICIENT VALUE: {2},\n \
                        INTERCEPT VALUE: {3} \n ====================" .format(
                            np.round(
                                cost, 4), i, np.round(
                                w, 7), np.round(
                                b, 7)))

            else:
                break

        rpo_obj["costs"] = costs
        rpo_obj["weights"] = w
        rpo_obj["intercept"] = b
        self.__repo = rpo_obj
        self.__weights = w
        self.__intecept = b
        return self

    def __norm(self, x, y):
        xmax = np.max(x, axis=0)
        ymax = np.max(y, axis=0)
        x = normalize(x, axis=0, norm='max')
        y = normalize(y, axis=0, norm='max')
        self.__params = (x, y, xmax, ymax)

        return x, y

    def predict(self, x):
        x /= self.__params[2]
        p = np.transpose(
            np.dot(
                self.__weights,
                np.transpose(x))) + self.__intecept
        p *= self.__params[3]
        return int(p[0])

    def get_report(self):
        return self.__repo

    def get_params(self):
        return self.__params

    def save(self, path):
        with open('{0}.pickle'.format(path), 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Model Is Saved.{0}".format('{0}.pickle'.format(path)))
