import numpy as np


def metrics_at_k(actual, predicted, k = 10):
    truncated_predicted = predicted[:k]

    tp = np.sum(np.in1d(truncated_predicted, actual))
    tp_fn = len(actual)
    tp_fp = k

    recall_at_k = tp / (tp_fn + 1e-10)
    precision_at_k = tp / k

    return recall_at_k, precision_at_k


def apk(actual, predicted, k=10):
    if not actual:
        return 0.0

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(list(a), list(p), k) for a,p in zip(actual, predicted)])