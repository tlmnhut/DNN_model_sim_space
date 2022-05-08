import numpy as np
from scipy.spatial import procrustes
from sklearn.cross_decomposition import PLSCanonical
from scipy.stats import pearsonr

import sys
sys.path.append('../')
from compare_2OI import compute_sim_mat, upper_tri
import cca_core


def measure_change(original_mat, changed_mat, method, labels, idx_label):
    """
    Metric to measure the similarity between two feature matrices.
    Args:
        original_mat: feature matrix, shape num_sample x num_feature
        changed_mat: feature matrix, shape num_sample x num_feature
        method: either 'procrustes', 'rsa' (2nd-order isomorphism), or 'cca' (canonical correlation analysis)
    Returns: a scalar score of similarity.
    """

    if method == 'procrustes':
        try:
            changed_mat = np.hstack([changed_mat,
                                     np.zeros([changed_mat.shape[0], original_mat.shape[1] - changed_mat.shape[1]])])
            mtx1, mtx2, disparity = procrustes(original_mat, changed_mat)  # square error
        except ValueError:  # in some case the feature matrix has all 1 elements  # TODO
            return None
        return np.sqrt(disparity / len(original_mat))  # root mean square error

    elif method == 'rsa':
        # compute average concepts
        original_concepts = get_concept(feature_mat=original_mat, labels=labels, idx_label=idx_label)
        changed_concepts = get_concept(feature_mat=changed_mat, labels=labels, idx_label=idx_label)

        # compute similarity matrix
        original_sim_mat = compute_sim_mat(feat_mat=original_concepts)
        changed_sim_mat = compute_sim_mat(feat_mat=changed_concepts)

        # compute spearman score
        spearman_score = pearsonr(upper_tri(original_sim_mat), upper_tri(changed_sim_mat))[0] ** 2
        return spearman_score

    elif method == 'plsca':
        smaller_dim = np.min([changed_mat.shape[1], original_mat.shape[1]])
        plsca = PLSCanonical(n_components=smaller_dim)
        plsca.fit(original_mat, changed_mat)
        score = plsca.score(original_mat, changed_mat)
        return score

    elif method == 'cca':
        # CCA require transpose matrices: num_feature x num_sample
        original_mat, changed_mat = original_mat.T, changed_mat.T

        # it is recommended to do SVD as a preprocess step
        original_mat = original_mat - np.mean(original_mat, axis=1, keepdims=True)
        changed_mat = changed_mat - np.mean(changed_mat, axis=1, keepdims=True)
        U1, s1, V1 = np.linalg.svd(original_mat, full_matrices=False)
        U2, s2, V2 = np.linalg.svd(changed_mat, full_matrices=False)
        n_dim_original, n_dim_changed = int(1.0*original_mat.shape[0]), int(1.0*changed_mat.shape[0])
        if n_dim_original == 0:
            n_dim_original = original_mat.shape[0]
        if n_dim_changed == 0:
            n_dim_changed = changed_mat.shape[0]
        original_mat = np.dot(s1[:n_dim_original] * np.eye(n_dim_original), V1[:n_dim_original])
        changed_mat = np.dot(s2[:n_dim_changed] * np.eye(n_dim_changed), V2[:n_dim_changed])

        # compute CCA score
        svcca_results = cca_core.get_cca_similarity(original_mat, changed_mat, epsilon=1e-10, verbose=False)
        return np.mean(svcca_results["cca_coef1"])

    else:
        return None


def get_concept(feature_mat, labels, idx_label):
    num_classes = len(idx_label)
    avg_features = np.zeros([num_classes, feature_mat.reshape(feature_mat.shape[0], -1).shape[1]])
    for i in idx_label.keys():
        mask = (labels == i) * 1
        filter_mat = np.take(feature_mat, np.nonzero(mask), axis=0)[0]
        filter_mat = filter_mat.reshape(filter_mat.shape[0], -1)
        avg_features[i] = np.mean(filter_mat, axis=0)
    return avg_features

