# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:24:51 2016
last update: 2017/09/13

@author: Markus Eisenbach, Ronny Stricker
"""
import itertools
import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import pylab


def plot_two_axes(x_values, y_plots1, y_styles1, y_plots2, y_styles2,
                  title, x_label, y_label1, y_label2,
                  color1='b', color2='r'):
    fig, ax1 = plt.subplots()
    for y_plot, style in zip(y_plots1, y_styles1):
        ax1.plot(x_values[:min(len(x_values), len(y_plot))], y_plot,
                 color1 + style)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label1, color=color1)
    # Make the y-axis label and tick labels match the line color.
    for tl in ax1.get_yticklabels():
        tl.set_color(color1)
    # second plot
    ax2 = ax1.twinx()
    for y_plot, style in zip(y_plots2, y_styles2):
        ax2.plot(x_values[:min(len(x_values), len(y_plot))], y_plot,
                 color2 + style)
    ax2.set_ylabel(y_label2, color=color2)
    # Make the y-axis label and tick labels match the line color.
    for tl in ax2.get_yticklabels():
        tl.set_color(color2)
    plt.title(title)
    plt.show()


def get_statistics_binary(ground_truth,
                          class_scores,
                          n_supporting_points=None,
                          additional_thresholds=()):
    """
    Compute TP, FP statistics for binary classification problem.
    Statistics are computed for all (or subsampled) possible threshold values.

    Parameters
    ----------
    ground_truth : numpy array
        One dimensional numpy array containung ground truth class label (0, 1)
    class_scores : numpy array
        Two dimensional array with class probabilities (sum of one).
        First dim: different classes (2)
        Second dim: number of samples
    n_supporting_points : int
        How many threshold values should be used to compute statistics.
        None - all possible threshold values.
    additional_thresholds : {list, tuple}
        Additional thresholds to use.

    Returns
    -------
    tp_list : numpy array
        TP values for every threshold value
    fp_list : numpy array
        FP values for every threshold value
    thresholds : numpy array
        array with all threshold values
    NP : numpy array
        number of positive and negative samples (Array with two elements)

    """

    # extract "scores" of second output neuron (class 1) for each class
    n_scores = np.sort(class_scores[ground_truth == 0, 1])
    p_scores = np.sort(class_scores[ground_truth == 1, 1])

    # generate list of thresholds
    thresholds = []

    if n_supporting_points is None:
        # try every output value as threshold
        # generate sorted list of unique output values (also include 0 and 1)
        thresholds = sorted(np.unique(np.concatenate((np.array([0.0, 1.0]),
                            n_scores, p_scores))))
    else:
        # apply subsampling

        # generate lists of unique output values for both classes
        unique_n = np.unique(n_scores)
        unique_p = np.unique(p_scores)

        # we want to sample from the class with less samples first of all
        scores = []
        if n_scores.shape[0] < p_scores.shape[0]:
            scores = [unique_n, unique_p]
        else:
            scores = [unique_p, unique_n]

        # include zero and 1 into threshold values
        thresholds = [0, 1]
        # sample from first class (with less samples)
        thresholds += scores[0][np.round(np.linspace(0,
                                                     scores[0].shape[0]-1,
                                                     n_supporting_points/2-1))
                                .astype(int)].tolist()
        # sample the remaining values from the second class
        thresholds += scores[1][np.round(np.linspace(0,
                                                     scores[1].shape[0]-1,
                                                     n_supporting_points -
                                                     len(thresholds)))
                                .astype(int)].tolist()
        # we need a sorted list. Also we can throw away samples that occure
        # in the list twice
        thresholds = sorted(np.unique(thresholds))

    # add additional thresholds
    thresholds.extend(additional_thresholds)
    thresholds = sorted(thresholds)


    # count number of class samples
    N = float(len(n_scores))
    P = float(len(p_scores))

    # initialize tp and fp lists
    tp_list = np.empty(len(thresholds))
    fp_list = np.empty(len(thresholds))
    p_index = 0
    n_index = 0
    p_len = len(p_scores)
    n_len = len(n_scores)
    # calculate tp and fp value for every threshold value
    for i, t in enumerate(thresholds):
        # We need to count the samples in the p_scores and n_scores arrays that
        # are below the threshold in order to compute the TP and FP samples
        #
        # Since the arrays are sorted, we only need the find the first value,
        # that is above the threshold (index = number of samples below
        # threshold)
        # Furtheremore, the number of samples can only increase with increasing
        # threshold -> we only need to compare the values starting from the
        # last index

        while p_index < p_len and p_scores[p_index] < t:
            p_index += 1

        while n_index < n_len and n_scores[n_index] < t:
            n_index += 1
        tp_list[i] = p_index
        fp_list[i] = n_index

    # compute inverse
    tp_list = P-tp_list
    fp_list = N-fp_list

    return tp_list, fp_list, np.array(thresholds), np.array([N, P])


def roc_measures(statistics_binary):
    tp, fp, thr, NP = statistics_binary
    N = NP[0]
    P = NP[1]
    fn = P - tp
    tn = N - fp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tpr = tp / float(P)
        fpr = fp / float(N)
        accuracy = (tp + tn) / (P + N)
        balanced_error_rate = 0.5 * (fn / P + fp / N)

        matthews_correlation_coefficient = (tp * tn - fp * fn) / \
            np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        g_mean = np.sqrt((tp / P) * (tn / N))
    # resolve nan
    accuracy = np.nan_to_num(accuracy)
    balanced_error_rate[np.isnan(balanced_error_rate)] = 1.0
    matthews_correlation_coefficient = np.nan_to_num(
        matthews_correlation_coefficient)
    g_mean = np.nan_to_num(g_mean)
    # find maximum (position)
    acc_max_idx = np.argmax(accuracy)
    ber_min_idx = np.argmin(balanced_error_rate)
    mcc_max_idx = np.argmax(matthews_correlation_coefficient)
    gm_max_idx = np.argmax(g_mean)
    # area under curve
    auc = 0.0
    if not isinstance(tpr, list):
        tpr = tpr.tolist()
    if not isinstance(fpr, list):
        fpr = fpr.tolist()
    if tpr[0] < tpr[-1]:
        tpr = [0.0] + tpr + [1.0]
        fpr = [0.0] + fpr + [1.0]
    else:
        tpr = [1.0] + tpr + [0.0]
        fpr = [1.0] + fpr + [0.0]
    for idx in range(len(tpr) - 1):
        t1 = tpr[idx]
        t2 = tpr[idx + 1]
        f1 = fpr[idx]
        f2 = fpr[idx + 1]
        mean_t = 0.5 * (t1 + t2)
        f_dif = abs(f1 - f2)
        area = f_dif * mean_t
        auc += area
    # create dict
    roc_measure_dict = {}
    roc_measure_dict['true_positive_rate'] = np.array(tpr)
    roc_measure_dict['false_positive_rate'] = np.array(fpr)
    roc_measure_dict['threshold'] = thr
    roc_measure_dict['accuracy'] = accuracy
    roc_measure_dict['balanced_error_rate'] = balanced_error_rate
    roc_measure_dict['matthews_correlation_coefficient'] = \
        matthews_correlation_coefficient
    roc_measure_dict['g_mean'] = g_mean
    roc_measure_dict['best_accuracy'] = \
        (accuracy[acc_max_idx], thr[acc_max_idx], acc_max_idx)
    roc_measure_dict['best_balanced_error_rate'] = \
        (balanced_error_rate[ber_min_idx], thr[ber_min_idx], ber_min_idx)
    roc_measure_dict['best_matthews_correlation_coefficient'] = \
        (matthews_correlation_coefficient[mcc_max_idx],
         thr[mcc_max_idx], mcc_max_idx)
    roc_measure_dict['best_g_mean'] = \
        (g_mean[gm_max_idx], thr[gm_max_idx], gm_max_idx)
    roc_measure_dict['auc_roc'] = auc
    return roc_measure_dict


def pr_measures(statistics_binary):
    tp, fp, thr, NP = statistics_binary
    P = NP[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = tp / (tp + fp)
        recall = tp / float(P)
        f1_score = 2.0 * (precision * recall) / (precision + recall)
    g_measure = np.sqrt(precision * recall)
    break_even_point_dist = np.abs(precision - recall)
    # resolve nan
    precision = np.nan_to_num(precision)
    f1_score = np.nan_to_num(f1_score)
    g_measure = np.nan_to_num(g_measure)
    break_even_point_dist[np.isnan(break_even_point_dist)] = 1.0
    # find maximum (position)
    f1_max_idx = np.argmax(f1_score)
    gm_max_idx = np.argmax(g_measure)
    bep_min_idx = np.argmin(break_even_point_dist)
    break_even_point = 0.5 * (precision[bep_min_idx] + recall[bep_min_idx])
    # area under curve
    auc = 0.0
    if not isinstance(precision, list):
        precision = precision.tolist()
    if not isinstance(recall, list):
        recall = recall.tolist()
    if recall[0] < recall[-1]:
        if recall[0] != 0.0:
            recall = [0.0] + recall + [1.0]
            precision = [1.0] + precision + [0.0]
        else:
            recall = recall + [1.0]
            precision = precision + [0.0]
    else:
        if recall[-1] != 0.0:
            recall = [1.0] + recall + [0.0]
            precision = [0.0] + precision + [1.0]
        else:
            recall = [1.0] + recall
            precision = [0.0] + precision
    for idx in range(len(precision) - 1):
        p1 = precision[idx]
        p2 = precision[idx + 1]
        r1 = recall[idx]
        r2 = recall[idx + 1]
        mean_p = 0.5 * (p1 + p2)
        r_dif = abs(r1 - r2)
        area = r_dif * mean_p
        auc += area
    # create dict
    pr_measure_dict = {}
    pr_measure_dict['precision'] = np.array(precision)
    pr_measure_dict['recall'] = np.array(recall)
    pr_measure_dict['threshold'] = thr
    pr_measure_dict['f1_score'] = f1_score
    pr_measure_dict['g_measure'] = g_measure
    pr_measure_dict['best_f1_score'] = \
        (f1_score[f1_max_idx], thr[f1_max_idx], f1_max_idx)
    pr_measure_dict['best_g_measure'] = \
        (g_measure[gm_max_idx], thr[gm_max_idx], gm_max_idx)
    pr_measure_dict['break_even_point'] = \
        (break_even_point, thr[bep_min_idx], bep_min_idx)
    pr_measure_dict['auc_pr'] = auc
    return pr_measure_dict


def confusion_matrix(statistics_binary, threshold):
    tp, fp, thr, NP = statistics_binary
    N = NP[0]
    P = NP[1]
    fn = P - tp
    tn = N - fp
    confusion_matrices = np.array([[tp, fn], [fp, tn]])
    thr_idx = np.argmin(np.abs(thr - threshold))
    return confusion_matrices[:, :, thr_idx]


def prepare_roc():
    t = np.arange(0.01, 0.991, 0.01)
    f = np.arange(0.01, 0.991, 0.01)
    T, F = np.meshgrid(t, f)
    ber = 0.5 * ((1 - T) + F)
    ber[ber > 0.55] = 0.55

    pylab.rcParams['figure.figsize'] = (6, 6)
    plt.figure(figsize=(15, 7))
    ax = plt.subplot(111)
    plt.axis([0, 1, 0, 1], 'equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    CS = plt.contour(F, T, ber, 10, colors='gray', linestyles='dotted')
    plt.clabel(CS, fontsize=9, inline=1)
    CS_line, = plt.plot(0, 0, 'gray', linestyle='dotted',
                        label='BER contour lines')
    return CS_line


def roc_curve_and_measures(roc_stat, name, color):
    tpr = roc_stat['true_positive_rate']
    fpr = roc_stat['false_positive_rate']
    ber_minimum, ber_min_thr, ber_min_idx = roc_stat[
        'best_balanced_error_rate']
    acc_maximum, acc_max_thr, acc_max_idx = roc_stat['best_accuracy']
    mcc_maximum, mcc_max_thr, mcc_max_idx = roc_stat[
        'best_matthews_correlation_coefficient']
    gm_maximum, gm_max_thr, gm_max_idx = roc_stat['best_g_mean']
    auc_roc = roc_stat['auc_roc']
    ROC, = plt.plot(fpr, tpr, color=color, linestyle='-', label=name)
    GM, = plt.plot(fpr[gm_max_idx], tpr[gm_max_idx], color=color,
                   linestyle='', marker='^',
                   label='Best G-mean = {:.4}, $\\tau$ = {:.4}'.format(
                   gm_maximum, gm_max_thr))
    MCC, = plt.plot(fpr[mcc_max_idx], tpr[mcc_max_idx], color=color,
                    linestyle='', marker='*',
                    label='Best matthews correlation coefficient' +
                    ' = {:.4}, $\\tau$ = {:.4}'.format(mcc_maximum,
                                                       mcc_max_thr))
    ACC, = plt.plot(fpr[acc_max_idx], tpr[acc_max_idx], color=color,
                    linestyle='', marker='d',
                    label='Best accuracy = {:.4}, $\\tau$ = {:.4}'.format(
                    acc_maximum, acc_max_thr))
    BER, = plt.plot(fpr[ber_min_idx], tpr[ber_min_idx], color=color,
                    linestyle='', marker='o',
                    label='Best balanced_error_rate ' +
                    '= {:.4}, $\\tau$ = {:.4}'.format(ber_minimum,
                                                      ber_min_thr))
    auc_line, = plt.plot(0, 0, color='w', linestyle='',
                         label='Area under curve (AUC-ROC) = {:.4}'.format(
                             auc_roc))
    return [ROC, BER, ACC, MCC, GM, auc_line]


def default_names(names, n):
    if not(isinstance(names, list)):
        if names is None:
            names = []
        else:
            names = [names]
    n_names = len(names)
    if n_names < n:
        for i in range(n_names, n):
            names.append('approach ' + str(i + 1))
    return names


def default_colors(colors, n):
    if not(isinstance(colors, list)):
        if colors is None:
            colors = []
        else:
            colors = [colors]
    n_colors = len(colors)
    default = ['b', 'r', 'g', 'm', 'c', 'gray', 'k']
    if n_colors < n:
        for i in range(n_colors, n):
            idx = i % len(default)
            colors.append(default[idx])
    for i in range(n):
        if not(colors[i] in default):
            idx = i % len(default)
            colors[i] = default[idx]
    return colors


def roc_binary(roc_stats, names=None, colors=None):
    CS_line = prepare_roc()
    handles = []
    styles = {}
    if not(isinstance(roc_stats, list)):
        roc_stats = [roc_stats]
    names = default_names(names, len(roc_stats))
    colors = default_colors(colors, len(roc_stats))
    for roc_stat, name, color in zip(roc_stats, names, colors):
        handle_list = roc_curve_and_measures(roc_stat, name, color)
        handles += handle_list
        for i in range(1, 5):
            styles[handle_list[i]] = HandlerLine2D(numpoints=1)
    handles += [CS_line]
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.title('Receiver operator characteristic (ROC)')
    plt.legend(handles=handles, handler_map=styles,  # prop={'size':10}
               loc='center left', bbox_to_anchor=(1.025, 0.5))
    plt.show()


def prepare_det(axes_scale_min_decade):
    axes_scale_min = 10 ** axes_scale_min_decade
    m = np.logspace(axes_scale_min_decade, 0, 100)
    f = np.logspace(axes_scale_min_decade, 0, 100)
    M, F = np.meshgrid(m, f)
    ber = 0.5 * (M + F)
    ber[ber > 0.55] = 0.55
    ber[np.isnan(ber)] = 0.55
    pylab.rcParams['figure.figsize'] = (6, 6)

    plt.figure(figsize=(15, 7))
    ax = plt.subplot(111)
    plt.axis([axes_scale_min, 1, axes_scale_min, 1], 'equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    CS = plt.contour(f, m, ber, 10, colors='gray', linestyles='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.clabel(CS, fontsize=9, inline=1)
    CS_line, = plt.loglog(axes_scale_min, axes_scale_min, 'gray',
                          linestyle='dotted', label='BER contour lines')
    return CS_line


def det_curve_and_measures(roc_stat, name, color, axes_scale_min_decade):
    axes_scale_min = 10 ** axes_scale_min_decade
    fnr = 1.0 - roc_stat['true_positive_rate']
    fpr = roc_stat['false_positive_rate']
    fpr[fpr < axes_scale_min] = axes_scale_min
    fnr[fnr < axes_scale_min] = axes_scale_min
    ber_minimum, ber_min_thr, ber_min_idx = \
        roc_stat['best_balanced_error_rate']
    acc_maximum, acc_max_thr, acc_max_idx = roc_stat['best_accuracy']
    mcc_maximum, mcc_max_thr, mcc_max_idx = \
        roc_stat['best_matthews_correlation_coefficient']
    gm_maximum, gm_max_thr, gm_max_idx = roc_stat['best_g_mean']
    DET, = plt.loglog(fpr, fnr, color=color, linestyle='-', label=name)
    GM, = plt.loglog(fpr[gm_max_idx], fnr[gm_max_idx], color=color,
                     linestyle='', marker='^',
                     label='Best G-mean = {:.4}, $\\tau$ = {:.4}'.format(
                     gm_maximum, gm_max_thr))
    MCC, = plt.loglog(fpr[mcc_max_idx], fnr[mcc_max_idx], color=color,
                      linestyle='', marker='*',
                      label='Best matthews correlation coefficient' +
                      ' = {:.4}, $\\tau$ = {:.4}'.format(mcc_maximum,
                                                         mcc_max_thr))
    ACC, = plt.loglog(fpr[acc_max_idx], fnr[acc_max_idx], color=color,
                      linestyle='', marker='d',
                      label='Best accuracy = {:.4}, $\\tau$ = {:.4}'.format(
                      acc_maximum, acc_max_thr))
    BER, = plt.loglog(fpr[ber_min_idx], fnr[ber_min_idx], color=color,
                      linestyle='', marker='o',
                      label='Best balanced_error_rate' +
                      ' = {:.4}, $\\tau$ = {:.4}'.format(ber_minimum,
                                                         ber_min_thr))
    return [DET, BER, ACC, MCC, GM]


def det_binary(roc_stats, names=None, colors=None):
    axes_scale_min_decade = -3
    CS_line = prepare_det(axes_scale_min_decade)
    handles = []
    styles = {}
    if not(isinstance(roc_stats, list)):
        roc_stats = [roc_stats]
    names = default_names(names, len(roc_stats))
    colors = default_colors(colors, len(roc_stats))
    for roc_stat, name, color in zip(roc_stats, names, colors):
        handle_list = det_curve_and_measures(roc_stat, name, color,
                                             axes_scale_min_decade)
        handles += handle_list
        for i in range(1, 5):
            styles[handle_list[i]] = HandlerLine2D(numpoints=1)
    handles += [CS_line]
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('Miss rate (MR)')
    plt.title('Detection error tradeoff (DET)')
    plt.legend(handles=handles, handler_map=styles,  # prop={'size':10}
               loc='center left', bbox_to_anchor=(1.025, 0.5))
    plt.show()


def prepare_pr():
    p = np.arange(0.01, 0.991, 0.01)
    r = np.arange(0.01, 0.991, 0.01)
    P, R = np.meshgrid(p, r)
    f1 = 2.0 * (P * R) / (P + R)

    pylab.rcParams['figure.figsize'] = (6, 6)
    plt.figure(figsize=(15, 7))
    ax = plt.subplot(111)
    plt.axis([0, 1, 0, 1], 'equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

    CS = plt.contour(R, P, f1, 9, colors='gray', linestyles='dotted')
    plt.clabel(CS, fontsize=9, inline=1)
    CS_line, = plt.plot(0, 0, 'gray', linestyle='dotted',
                        label='F$_1$-score contour lines')
    return CS_line


def pr_curve_and_measures(pr_stat, name, color):
    precision = pr_stat['precision']
    recall = pr_stat['recall']
    f1_maximum, f1_max_thr, f1_max_idx = pr_stat['best_f1_score']
    gm_maximum, gm_max_thr, gm_max_idx = pr_stat['best_g_measure']
    bep_maximum, bep_max_thr, bep_max_idx = pr_stat['break_even_point']
    auc_pr = pr_stat['auc_pr']
    PR, = plt.plot(recall[recall > 0], precision[recall > 0], color=color,
                   linestyle='-', label=name)
    BGM, = plt.plot(recall[gm_max_idx], precision[gm_max_idx], color=color,
                    linestyle='', marker='*',
                    label='Best G-measure = {:.4}, $\\tau$ = {:.4}'.format(
                    gm_maximum, gm_max_thr))
    BEP, = plt.plot(recall[bep_max_idx], precision[bep_max_idx], color=color,
                    linestyle='', marker='d',
                    label='Break even point = {:.4}, $\\tau$ = {:.4}'.format(
                    bep_maximum, bep_max_thr))
    BF1, = plt.plot(recall[f1_max_idx], precision[f1_max_idx], color=color,
                    linestyle='', marker='o',
                    label='Best F$_1$-score = {:.4}, $\\tau$ = {:.4}'.format(
                    f1_maximum, f1_max_thr))
    auc_line, = plt.plot(0, 0, color='w', linestyle='',
                         label='Area under curve (AUC-PR) = {:.4}'.format(
                             auc_pr))
    return [PR, BF1, BGM, BEP, auc_line]


def pr_binary(pr_stats, names=None, colors=None):
    CS_line = prepare_pr()
    handles = []
    styles = {}
    if not(isinstance(pr_stats, list)):
        pr_stats = [pr_stats]
    names = default_names(names, len(pr_stats))
    colors = default_colors(colors, len(pr_stats))
    for pr_stat, name, color in zip(pr_stats, names, colors):
        handle_list = pr_curve_and_measures(pr_stat, name, color)
        handles += handle_list
        for i in range(1, 4):
            styles[handle_list[i]] = HandlerLine2D(numpoints=1)
    handles += [CS_line]
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(handles=handles, handler_map=styles,  # prop={'size':10}
               loc='center left', bbox_to_anchor=(1.025, 0.5))
    plt.show()


def threshold_from_measure(measure, roc_stat, pr_stat):
    curve = 'roc' if measure in roc_stat else 'pr'
    thr_from_measure = {'roc': roc_stat, 'pr': pr_stat}
    threshold = thr_from_measure[curve][measure][1]
    return threshold


def color_map_from_color(color):
    lookup = {'b': plt.cm.Blues, 'blue': plt.cm.Blues,
              'r': plt.cm.Reds, 'red': plt.cm.Reds,
              'g': plt.cm.Greens, 'green': plt.cm.Greens,
              'm': plt.cm.pink_r, 'magenta': plt.cm.pink_r,
              'c': plt.cm.cool, 'cyan': plt.cm.cool,
              'gray': plt.cm.Greys, 'k': plt.cm.Greys,
              'black': plt.cm.Greys}
    return lookup[color]


def plot_confusion_matrix(cm, classes,
                          percent=False,
                          title='Confusion matrix',
                          cmap=plt.cm.YlOrRd):
    # source: http://scikit-learn.org/stable/auto_examples/model_selection/
    #         plot_confusion_matrix.html
    # + few modifications

    plt.figure()

    if percent:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100

    max_value = np.max(cm.sum(axis=1))
    plt.imshow(cm, interpolation='nearest', cmap=cmap,
               vmin=0.0, vmax=max_value)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = max_value * 0.7
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt_fmt = '{:d}' if not percent else '{:.2f}'
        plt.text(j, i, txt_fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_curves_binary(ground_truths, class_scores_list,
                       names=None, colors=None,
                       n_supporting_points=None):
    if not(isinstance(ground_truths, list)):
        ground_truths = [ground_truths]
    if not(isinstance(class_scores_list, list)):
        class_scores_list = [class_scores_list]
    stats_st = []
    stats_roc = []
    stats_pr = []
    for ground_truth, class_scores in zip(ground_truths, class_scores_list):
        statistics_binary = get_statistics_binary(ground_truth, class_scores,
                                                  n_supporting_points)
        roc_stat = roc_measures(statistics_binary)
        pr_stat = pr_measures(statistics_binary)
        stats_st.append(statistics_binary)
        stats_roc.append(roc_stat)
        stats_pr.append(pr_stat)

    roc_binary(stats_roc, names, colors)
    det_binary(stats_roc, names, colors)
    pr_binary(stats_pr, names, colors)

    names = default_names(names, len(class_scores_list))
    colors = default_colors(colors, len(class_scores_list))

    for statistics_binary, roc_stat, pr_stat, name, color \
            in zip(stats_st, stats_roc, stats_pr, names, colors):
        # threshold = threshold_from_measure('best_f1_score',
        threshold = threshold_from_measure('best_balanced_error_rate',
                                           roc_stat, pr_stat)
        cm = confusion_matrix(statistics_binary, threshold)
        plot_confusion_matrix(cm, ['1', '0'], True,
                              title='Confusion matrix ' + str(name),
                              cmap=color_map_from_color(color))
