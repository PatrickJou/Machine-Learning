import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.neural_network import MLPClassifier
from plot_learning_curve import plot_learning_curve

def MLP_classifier(X, Y, datasource):

    param_range = range(1,201,20)
    train_scores, test_scores = validation_curve(
    MLPClassifier(random_state=626), X, Y, param_name="hidden_layer_sizes", param_range=param_range,
    scoring="accuracy", n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print("train_scores_mean")
    print(train_scores_mean)
    print("test_scores_mean")
    print(test_scores_mean)
    print(np.argmax(test_scores_mean))
    print(test_scores_mean[np.argmax(test_scores_mean)])
    print(param_range[np.argmax(test_scores_mean)])
    hidden_layer_sizes_value = param_range[np.argmax(test_scores_mean)]

    plt.title("MLP Validation Curve on "+ datasource)
    plt.xlabel("hidden_layer")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2

    plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)

    plt.legend(loc="best")
    plt.show()

    MLP_Learning_Curves(X, Y, datasource, hidden_layer_sizes_value)

def MLP_Learning_Curves(X, Y, datasource, hidden_layer_sizes_value):
    title = "MLP Learning Curves on " + datasource
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=626)
    estimator = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes_value, random_state=626)
    plt = plot_learning_curve(estimator, title, X, Y, ylim=(0.0, 1.05), cv=cv)
    plt.show()
