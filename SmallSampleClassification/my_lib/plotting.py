import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {'train': 'b', 'test': 'r'}


def plot_validation_curve(train_scores, test_scores, train_sizes, expected_score=None, ax=None, stat_error=True):

    ax = _plot_generic_curve(
        train_scores=train_scores,
        test_scores=test_scores,
        x_pars=train_sizes,
        expected_score=expected_score,
        ax=ax,
        stat_error=stat_error
    )

    ax.set_ylabel('score')
    ax.set_xlabel('training sample size')


def plot_learning_curve(train_scores, test_scores, param_range, expected_score=None, ax=None, stat_error=True):

    ax = _plot_generic_curve(
        train_scores=train_scores,
        test_scores=test_scores,
        x_pars=param_range,
        expected_score=expected_score,
        ax=None,
        stat_error=stat_error
    )

    ax.set_ylabel('score')
    ax.set_xlabel('parameter value')


def _plot_generic_curve(train_scores, test_scores, x_pars, expected_score=None, ax=None, stat_error=True):
    train_score_mean, train_score_error, test_score_mean, test_score_error = \
        get_score_means_and_errors(train_scores, test_scores, stat_error=stat_error)

    if ax is None:
        ax = plt.gca()

    ax.plot(
        x_pars,
        train_score_mean,
        color=COLORS['train'],
        linestyle='none',
        marker='o',
        label='train'
    )
    plt.fill_between(
        x_pars,
        train_score_mean + train_score_error,
        train_score_mean - train_score_error,
        alpha=0.15,
        color=COLORS['train']
    )

    ax.plot(
        x_pars,
        test_score_mean,
        color=COLORS['test'],
        linestyle='none',
        marker='s',
        label='test'
    )
    plt.fill_between(
        x_pars,
        test_score_mean + test_score_error,
        test_score_mean - test_score_error,
        alpha=0.15,
        color=COLORS['test']
    )

    if expected_score is not None:
        plt.plot(
            (x_pars[0], x_pars[-1]),
            (expected_score, expected_score),
            linestyle='--',
            color='k'
        )

    ax.legend()

    return ax


def get_score_means_and_errors(train_scores, test_scores, stat_error=True):
    train_score_mean = np.mean(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)

    if stat_error:
        train_score_error = np.std(train_scores, axis=1) / np.sqrt(len(train_scores[0]))
        test_score_error = np.std(test_scores, axis=1) / np.sqrt(len(test_scores[0]))
    else:
        train_score_error = np.std(train_scores, axis=1)
        test_score_error = np.std(test_scores, axis=1)

    return train_score_mean, train_score_error, test_score_mean, test_score_error


def plot_probability_distributions(clf, X, y):
    prob = clf.predict_proba(X)
    plt.hist([p for p, y in zip(prob[:,0], y) if y == 0], bins=np.linspace(0, 1, 25), alpha=0.6, color='r')
    plt.hist([p for p, y in zip(prob[:,0], y) if y == 1], bins=np.linspace(0, 1, 25), alpha=0.6, color='b')