from matplotlib import pyplot as plt


def derive_conditions_rois(labels):
    conditions = [s.split('/')[0] for s in labels]
    conditions = list(set(conditions))
    rois = [s.split('/')[1] for s in labels]
    rois = list(set(rois))
    return conditions, rois


def plot_mean_epochs(mean_signals, conditions, rois):

    x_axis = list(range(-200, 802))
    for condition in conditions:
        correct_labels = [s for s in mean_signals.keys() if condition + '/' in s]
        correct_short_labels = [s.split('/')[1] for s in correct_labels]

        print(mean_signals)
        for idx, label in enumerate(correct_labels):
            plt.plot(x_axis, mean_signals[label], label=correct_short_labels[idx])
            print(len(mean_signals[label]))
            exit(1)

        # plt.vlines(170, ymin=min_value, ymax=max_value)

        path = '../image/epochs/' + condition + '.png'
        plt.title(condition)
        plt.legend()
        # plt.savefig(path)
        plt.show()

    for roi in rois:

        correct_labels = [s for s in mean_signals.keys() if '/' + roi in s]
        correct_short_labels = [s.split('/')[0] for s in correct_labels]

        for idx, label in enumerate(correct_labels):
            plt.plot(x_axis, mean_signals[label], label=correct_short_labels[idx])

        # plt.vlines(170, ymin=min_value, ymax=max_value)

        path = '../image/epochs/' + roi + '.png'
        plt.title(roi)
        plt.legend()
        # plt.savefig(path)
        plt.show()