""" Helper functions for matrix factorization analysis
"""
import numpy as np
from brainiak.funcalign.srm import SRM
from sklearn.decomposition import PCA


def fit_srm(n_components, data_train, data_test):
    """Fit the shared response model
    Parameters
    ----------
    n_components: k
    data_train: 3d array (n_subj, n_features, n_examples/tps)
    data_test: 3d array (n_subj, n_features, n_examples/tps)

    Returns
    -------
    data_test_shared: 3d array (n_subj, n_components, n_examples/tps)
        the transformed test set
    srm: the fitted model
    """
    #
    srm = SRM(features=n_components)
    # fit SRM on the training set
    data_train_shared = srm.fit_transform(data_train)
    # transform the hidden activity (on the test set) to the shared space
    data_test_shared = srm.transform(data_test)
    return data_train_shared, data_test_shared, srm


def trace_var_exp_srm(data_train, n_components_list):
    """Trace the variance explained curve, over the number of components
    """
    pass


def procrustes_align(X_new, S_target):
    """One-step deterministic SRM
    Parameters
    ----------
    X_new:
        an activation trajectory
    S_target:
        pre-computed shared response as the alignment target

    Returns
    -------
    W:
        the transformation matrix from X to S
    X_aligned:
        the aligned trajectory
    """
    U, s_vals, VT = np.linalg.svd(X_new @ S_target.T, full_matrices=False)
    W = U @ VT
    X_aligned = W.T @ X_new
    return W, X_aligned


def fit_pca(n_components, data_train, data_test):
    """Fit PCA
    Parameters
    ----------
    n_components: k
    data_train: 2d array (n_features, n_examples/tps)
    data_test: 2d array (n_features, n_examples/tps)

    Returns
    -------
    data_test_shared: 3d array (n_components, n_examples/tps)
        the transformed test set
    pca: the fitted model
    """
    pca = PCA(n_components=n_components)
    # tranpose the data to make the format consistent with SRM fit
    pca.fit(data_train.T)
    data_test_pca = pca.transform(data_test.T).T
    return data_test_pca, pca
