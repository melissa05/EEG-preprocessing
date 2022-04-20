import numpy as np
from matplotlib import pyplot as plt
import pylab as py
from scipy import optimize


def derive_conditions_rois(labels):
    conditions = [s.split('/')[0] for s in labels]
    conditions = list(set(conditions))
    rois = [s.split('/')[1] for s in labels]
    rois = list(set(rois))
    return conditions, rois


def plot_mean_epochs(mean_signals, conditions, rois):

    x_axis = list(range(-200, 802, 2))

    for condition in conditions:
        correct_labels = [s for s in mean_signals.keys() if condition + '/' in s]
        correct_short_labels = [s.split('/')[1] for s in correct_labels]

        for idx, label in enumerate(correct_labels):
            plt.plot(x_axis, mean_signals[label].T, label=correct_short_labels[idx])

        # plt.vlines(170, ymin=min_value, ymax=max_value)

        path = '../images/epochs/' + condition + '.png'
        plt.title(condition)
        plt.legend()
        plt.savefig(path)
        plt.show()

    for roi in rois:

        correct_labels = [s for s in mean_signals.keys() if '/' + roi in s]
        correct_short_labels = [s.split('/')[0] for s in correct_labels]

        for idx, label in enumerate(correct_labels):
            plt.plot(x_axis, mean_signals[label].T, label=correct_short_labels[idx])

        # plt.vlines(170, ymin=min_value, ymax=max_value)

        path = '../images/epochs/' + roi + '.png'
        plt.title(roi)
        plt.legend()
        plt.savefig(path)
        plt.show()


def get_fitted_normal_distribution(data, number_bins=100):
    # Equation for Gaussian
    def f(x, a, b, c):
        return a * py.exp(-(x - b) ** 2.0 / (2 * c ** 2))

    # Generate data from bins as a set of points
    x = [0.5 * (data[1][i] + data[1][i + 1]) for i in range(len(data[1]) - 1)]
    y = data[0]

    popt, pcov = optimize.curve_fit(f, x, y)

    x_fit = py.linspace(x[0], x[-1], number_bins)
    y_fit = f(x_fit, *popt)

    return x_fit, y_fit


def plot_distribution(array_data, path):

    bins = np.linspace(array_data.min(), array_data.max(), 100)
    data = py.hist(array_data, bins=bins)

    x_fit, y_fit = get_fitted_normal_distribution(data, number_bins=len(bins))
    plt.plot(x_fit, y_fit, lw=4, color="r")

    plt.title((path.rsplit('.', 1)[0]).rsplit('/', 1)[1])
    plt.savefig(path)
    plt.show()
