import numpy as np
import matplotlib.pyplot as plt
import data


def logreg_train(x, y_):
    param_niter = 1000
    param_delta = 0.001
    # param_lambda = 0

    n = x.shape[0]
    c = 1 + max(y_)

    w = np.random.randn(x.shape[1], c)
    b = np.zeros(c)

    # gradijentni spust (param_niter iteracija)
    for i in range(param_niter):

        scores = np.dot(x, w) + b   # N x C

        # brojnik softmaxa
        expscores = np.exp(scores)  # N x C
        # nazivnik softmaxa
        sumexp = np.sum(expscores, axis=1, keepdims=True)  # N x 1
        probs = expscores / sumexp  # N x C

        # logaritmirane vjerojatnosti razreda
        logprobs = -np.log(probs[range(n), y_])  # N x C

        # gubitak
        loss = np.sum(logprobs / n)    # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        y = np.zeros([x.shape[0], max(y_) + 1])
        for j in range(0, len(y_)):
            y[j][y_[j]] = 1
        dL_ds = probs - y   # N x C

        # gradijenti parametara
        grad_w = np.transpose(np.dot(np.transpose(dL_ds), x))   # C x D (ili D x C)
        grad_b = np.sum(dL_ds, axis=0)  # C x 1 (ili 1 x C)

        # poboljšani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def logreg_classify(x, w, b):
    scores = np.dot(x, w) + b
    # brojnik softmaxa
    expscores = np.exp(scores)  # N x C
    # nazivnik softmaxa
    sumexp = np.sum(expscores, axis=1, keepdims=True)    # N x 1
    probs = expscores / sumexp  # N x C
    return np.argmax(probs, axis=1)


def logreg_decfun(w, b):
    def classify(x):
        return logreg_classify(x, w, b)
    return classify


if __name__ == "__main__":
    np.random.seed(42)
    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 1000)
    # train the model
    w, b = logreg_train(X, Y_)
    # evaluate the model on the training dataset
    Y = logreg_classify(X, w, b)
    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    print("{}\n{}\n{}".format(accuracy, recall, precision))
    # graph the decision surface
    decfun = logreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    # show the plot
    plt.show()
