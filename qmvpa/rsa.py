""" representational similarity analysis
"""

import numpy as np
from scipy.stats.stats import pearsonr


def within_RSMs(Xs):
    """ given a list of Xs, compute all within-subject RSM
    """
    # compute RSM
    num_subjects = len(Xs)
    rsms = [inter_RSM(Xs[s], Xs[s]) for s in range(num_subjects)]
    return rsms


def correlate_2RSMs(rsm1, rsm2):
    """compute the correlation between 2 RSMs
    """
    # compute the linear correlation of 2 vectorized RSMs
    r, _ = pearsonr(np.reshape(rsm1, (-1,)), np.reshape(rsm2, (-1,)))
    return r


def correlate_RSMs(rsms):
    """ given a list of rsms, compute all pairwise corr
    """
    num_subjects = len(rsms)
    corr_rsms = np.zeros((num_subjects, num_subjects))
    for i in range(num_subjects):
        for j in range(num_subjects):
            corr_rsms[i, j] = correlate_2RSMs(
                rsms[i], rsms[j])
    return corr_rsms


def inter_RSM(m1, m2):
    """ compute the RSM for 2 activation matrices
        input: mi (n_feature_dim x n_examples)
        action: compute corr(col_i(m1), col_j(m2)) for all i and j
    """
    assert(np.shape(m1)[1] == np.shape(m2)[1])
    n_examples = np.shape(m1)[1]
    # compute the correlation matrix of hidden activity
    corr_mat = np.zeros((n_examples, n_examples))
    for i in range(n_examples):
        for j in np.arange(0, i+1, 1):
            corr_mat[i, j] = pearsonr(m1[:, i], m2[:, j])[0]
            if all(m1[:, i] == 0) or all(m2[:, j] == 0):
                corr_mat[i, j] = 0
    # fillin the upper triangular part by symmetry
    irows, icols = np.triu_indices(np.shape(corr_mat)[0], 1)
    corr_mat[irows, icols] = corr_mat[icols, irows]
    return corr_mat


def inter_RSMs(matrix_list):
    """ compute group-level intersubj_RSM
        similar to the two subject case...
        when comparing k subjects to 1 subject, average the k
    """
    matrix_array = np.array(matrix_list)
    len(matrix_list)
    cross_subj_RSM = []
    for loo_idx in range(len(matrix_list)):
        mean_Hs = np.mean(
            matrix_array[np.arange(len(matrix_list)) != loo_idx], axis=0)
        cross_subj_RSM.append(inter_RSM(matrix_array[loo_idx], mean_Hs))
    cross_subj_RSM = np.mean(cross_subj_RSM, axis=0)
    return cross_subj_RSM


""" util functions """


def moving_window_avg(matrix, axis, win_size):
    """ moving window averaging along axis 0
    """
    n0, n1 = np.shape(matrix)
    if axis == 0:
        n_new = int(n0 / win_size)
        matrix_avg = np.zeros((n_new, n1))
        for i in range(n_new):
            idx_start = i * win_size
            idx_end = (i + 1) * win_size
            matrix_avg[i, :] = np.mean(matrix[idx_start: idx_end, :], axis=0)
    elif axis == 1:
        # just transpose the matrix ...
        raise ValueError('Oops you haven\'t implement it!')
    else:
        raise ValueError('axis should be 0 or 1!')
    return matrix_avg
