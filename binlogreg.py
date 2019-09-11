import numpy as np
import matplotlib.pyplot as plt
import data


def binlogreg_train(x, y_):
    param_niter = 5000
    param_delta = 0.001
    # param_lambda = 0

    w = np.random.randn(x.shape[1])
    b = 0

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):

        # klasifikacijske mjere
        scores = np.dot(x, w) + b   # N x 1
        # s = W * x + b

        # vjerojatnosti razreda c_1
        probs = softmax(scores)  # N x 1
        # Pi = e^si / sigma(j)(e^sj)

        # gubitak
        loss = -np.sum(cross_entropy(y_, probs)) / len(probs)  # scalar
        # L = -1 / N * sigma(i)(P(Y = yi | xi))

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskim mjerama
        dl_dscores = probs - y_    # N x 1
        # dL / dsi

        # gradijenti parametara
        grad_w = np.dot(dl_dscores, x)    # D x 1
        # dL / dw = dL / dsi * dsi / dw
        grad_b = np.sum(dl_dscores)    # 1 x 1
        # dL / db = dL / dsi * dsi / db

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def softmax(x):
    # brojnik softmaxa
    expscores = np.exp(x)  # N x C
    # nazivnik softmaxa
    sumexp = expscores + 1  # N x 1
    return expscores / sumexp  # N x C


def stable_softmax(x):
    # brojnik softmaxa
    expscores = np.exp(x - np.max(x))  # N x C
    # nazivnik softmaxa
    sumexp = expscores + np.exp(-np.max(x))  # N x 1
    return expscores / sumexp  # N x C


def softmax_simplified(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy(y, probs):
    return y * np.log(probs) + (1 - y) * np.log(1 - probs)


def binlogreg_classify(x, w, b):
    scores = np.dot(x, w) + b
    return softmax(scores)


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)
    # train the model
    w, b = binlogreg_train(X, Y_)
    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = (probs > 0.5).astype(int)
    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_ap(Y_[probs.argsort()])
    print(accuracy, recall, precision, AP)
    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    # show the plot
    plt.show()
