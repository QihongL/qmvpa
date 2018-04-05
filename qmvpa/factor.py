""" Helper functions for matrix factorization analysis
"""
from brainiak.funcalign.srm import SRM
from sklearn.decomposition import PCA


def fit_srm(n_srm_features, data_train, data_test):
    """Fit the shared response model
    Parameters
    ----------
    n_srm_features: k
    data_train: 3d array (n_subj, n_features, n_examples/tps)
    data_test: 3d array (n_subj, n_features, n_examples/tps)

    Returns
    -------
    data_test_shared: 3d array (n_subj, n_srm_features, n_examples/tps)
        the transformed test set
    srm: the fitted model
    """
    #
    srm = SRM(features=n_srm_features)
    # fit SRM on the training set
    srm.fit(data_train)
    # transform the hidden activity (on the test set) to the shared space
    data_test_shared = srm.transform(data_test)
    return data_test_shared, srm


def fit_pca(n_pca_features, data_train, data_test):
    """Fit PCA
    Parameters
    ----------
    n_pca_features: k
    data_train: 2d array (n_features, n_examples/tps)
    data_test: 2d array (n_features, n_examples/tps)

    Returns
    -------
    data_test_shared: 3d array (n_pca_features, n_examples/tps)
        the transformed test set
    pca: the fitted model
    """
    pca = PCA(n_components=n_pca_features)
    # tranpose the data to make the format consistent with SRM fit
    pca.fit(data_train.T)
    data_test_pca = pca.transform(data_test.T).T
    return data_test_pca, pca
