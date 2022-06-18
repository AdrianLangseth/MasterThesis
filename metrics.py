import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def ndcg_at_k(k):

    def ndcg(y_true, y_pred):
        """
        :param y_true: ndarray
        :param y_pred: ndarray
        :param k: The k to which to measure
        :return: ndcg score
        """
        print(y_true)
        print(y_pred)
        print(y_true.shape)
        print(y_pred.shape)

        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        a2i = y_pred.argsort()
        # sort_pred = y_pred[a2i[::-1]]
        sort_true = y_true[a2i[::-1]]
        discount = 1 / (np.log2(np.arange(k) + 2))
        opt_true = np.sort(y_true)[::-1]
        dcg = np.sum(sort_true[:k] * discount)
        idcg = np.sum(opt_true[:k] * discount)
        return dcg / idcg
    return ndcg


def MAP(y_true, y_pred, k):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert k <= len(y_true) and k <= len(y_true)
    a2i = y_pred.argsort()  # Sort acc to pred
    sort_true = y_true[a2i[::-1]]  # True sorted according to pred vals
    if not sum(sort_true[:k]):
        return 0
    SAP = 0
    for idx, val in enumerate(sort_true):
        if idx == k:
            break
        if val:
            SAP += sum(sort_true[:idx + 1]) / (idx + 1)

    return SAP / sum(sort_true[:k])


def precision(y_true, y_pred, k):
    """
    y_true will be
    :param y_true: Ndarray of binary indications of relevance
    :param y_pred: ndarray of predicted relevancies
    :param k:
    :return:
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert k <= len(y_true) and k <= len(y_true)
    a2i = y_pred.argsort()  # Sort acc to pred
    sort_true = y_true[a2i[::-1]]  # True sorted according to pred vals
    return sum(sort_true[:k] > 0) / k


def recall(y_true, y_pred, k):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    assert k <= len(y_true) and k <= len(y_true)
    a2i = y_pred.argsort()  # Sort acc to pred
    sort_true = y_true[a2i[::-1]]  # True sorted according to pred vals
    correct_sort = np.sort(y_true)[::-1]

    rel_rec = sum(sort_true[:k] > 0)
    tot_rel = sum(correct_sort[:] > 0)

    return rel_rec / tot_rel


if __name__ == '__main__':
    X = np.random.randint(low=0, high=9, size=(1000, 2))
    y = (X[:, 0] + X[:, 1]) % 2

    x_train, x_val, y_train, y_val = train_test_split(X, y)

    i = tf.keras.layers.Input(shape=(2,))
    x = tf.keras.layers.Dense(4)(i)
    x = tf.keras.layers.Dense(2)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(i, x)

    model.compile(
        optimizer='adam',
        loss="binary_crossentropy",
        metrics=["accuracy", ndcg, MAP, precision, recall],
        # , tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5")],
    )
    model.fit(x_train, y_train, validation_data=(x_val, y_val))
