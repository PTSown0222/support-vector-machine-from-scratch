"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    x = x.astype(float)
    mean_x = np.mean(x, axis = 0)
    std_x = np.std(x, axis = 0)
    std_x[std_x == 0] = 1.0
    rescale = (x - mean_x) / std_x
    return rescale

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    weights = np.zeros(n_features)
    bias = 0.0
    return {
        "w": weights,
        "b": bias
    }

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    w = params["w"]
    b = params["b"]
    return np.dot(x, w) + b

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    #return np.where(scores >= 0, 1, -1)
    return [1 if score >= 0 else -1 for score in scores]

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    margin = 1 - y * score
    hinge_loss = max(0.0, margin)
    return hinge_loss

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    w = params["w"]
    scores = compute_scores(x, params)
    import numpy as np
    hinge_losses = np.maximum(0.0, 1.0 - y * scores)
    mean_hinge = np.mean(hinge_losses)
    j = mean_hinge + reg_lambda * np.dot(w,w)
    return float(j)

# Step 7 - compute_gradients
"""
m = [0.5, -1.2, 2.0]
[0.5 > 0, -1.2 < 0, 2.0 > 0]
[True, False, True] -> float [1, 0, 1]
y = [1.0, -1.0, 1.0]
mask * y = [1.0 * 1.0, 0.0 * -1.0, 1.0 * 1.0]
= [1.0, 0.0, 1.0]
"""
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].
    w = params["w"]
    b = params["b"]
    n = len(x)
    scores = np.dot(x, w) + b
    m = 1 - y * scores
    mask = (m > 0).astype(float)
    dw = - (x.T @ (mask * y)) / n + 2 * reg_lambda * w
    db = - np.sum(mask * y) / n
    return {
        "dw": dw,
        "db": db,
    }

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w = params["w"]
    b = params["b"]
    dw = grads["dw"]
    db = grads["db"]
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return {
        "w": w,
        "b": b,
    }

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # TODO: fit a linear SVM by repeatedly updating parameters over n_epochs passes.
    n_features = x.shape[1]
    params = {
        "w": np.zeros(n_features),
        "b": 0.0
    }
    for _ in range(n_epochs):
        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)
    return params

# Step 10 - predict_labels
import numpy as np

def predict_labels(x, params):
    # TODO: return an array of {-1, +1} labels, one per row of x, using params['w'] and params['b'].
    scores = compute_scores(x, params)
    predict = predict_from_scores(scores)
    return np.array(predict)

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

