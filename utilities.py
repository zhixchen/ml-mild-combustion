'''
MODULE: utilities.py
@Author:
    G. D'Alessio [1,2]
    [1]: Université Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Bruxelles, Belgium
    [2]: CRECK Modeling Lab, Department of Chemistry, Materials and Chemical Engineering, Politecnico di Milano
@Contacts:
    giuseppe.dalessio@ulb.ac.be
@Details:
    This module contains a set of functions which are useful for reduced-order modelling with PCA.
    A detailed description is available under the definition of each function.
@Additional notes:
    This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    Please report any bug to: giuseppe.dalessio@ulb.ac.be
'''


import numpy as np
from numpy import linalg as LA
import functools


__all__ = ["unscale", "uncenter", "center", "scale", "center_scale", "PCA_fit", "accepts", "readCSV", "allowed_centering","allowed_scaling", "outlier_removal_leverage", "outlier_removal_orthogonal"]

# ------------------------------
# Functions (alphabetical order)
# ------------------------------


def center(X, method, return_centered_matrix=False):
    '''
    Computes the centering factor (the mean/min value [mu]) of each variable of all data-set observations and
    (eventually) return the centered matrix.
    - Input:
    X = original data matrix -- dim: (observations x variables)
    method = "string", it is the method which has to be used. Two choices are available: MEAN or MIN
    return_centered_matrix = boolean, choose if the script must return the centered matrix (optional)
    - Output:
    mu = centering factor for the data matrix X
    X0 = centered data matrix (optional)
    '''
    # Main
    if not return_centered_matrix:
        if method.lower() == 'mean':
            mu = np.mean(X, axis = 0)
        elif method.lower() == 'min':
            mu = np.min(X, axis = 0)
        else:
            raise Exception("Unsupported centering option. Please choose: MEAN or MIN.")
        return mu
    else:
        if method.lower() == 'mean':
            mu = np.mean(X, axis = 0)
            X0 = X - mu
        elif method.lower() == 'min':
            mu = np.min(X, axis = 0)
            X0 = X - mu
        else:
            raise Exception("Unsupported centering option. Please choose: MEAN or MIN.")
        return mu, X0


def center_scale(X, mu, sig):
    '''
    Center and scale a given multivariate data-set X.
    Centering consists of subtracting the mean/min value of each variable to all data-set
    observations. Scaling is achieved by dividing each variable by a given scaling factor. Therefore, the
    i-th observation of the j-th variable, x_{i,j} can be
    centered and scaled by means of:
    \tilde{x_{i,j}} = (x_{i,j} - mu_{j}) / (sig_{j}),
    where mu_{j} and sig_{j} are the centering and scaling factor for the considered j-th variable, respectively.
    AUTO: the standard deviation of each variable is used as a scaling factor.
    PARETO: the squared root of the standard deviation is used as a scaling f.
    RANGE: the difference between the minimum and the maximum value is adopted as a scaling f.
    VAST: the ratio between the variance and the mean of each variable is used as a scaling f.
    '''
    TOL = 1E-16
    if X.shape[1] == mu.shape[0] and X.shape[1] == sig.shape[0]:
        X0 = X - mu
        X0 = X0 / (sig + TOL)
        return X0
    else:
        raise Exception("The matrix to be centered & scaled and the centering/scaling vectors must have the same dimensionality.")


def outlier_removal_leverage(X, eigens, centering, scaling):
    '''
    This function removes the multivariate outliers (leverage) eventually contained
    in the training dataset, via PCA. In fact, examining the data projection
    on the PCA manifold (i.e., the scores), and measuring the score distance
    from the manifold center, it is possible to identify the so-called
    leverage points. They are characterized by very high distance from the
    center of mass, once detected they can easily be removed.
    Additional info on outlier identification and removal can be found here:

    Jolliffe pag 237 --- formula (10.1.2):

    dist^{2}_{2,i} = sum_{k=p-q+1}^{p}(z^{2}_{ik}/l_{k})
    where:
    p = number of variables
    q = number of required PCs
    i = index to count the observations
    k = index to count the PCs

    '''
    #center and scale the input matrix before PCA
    mu_X = center(X, centering)
    sigma_X = scale(X, scaling)

    X_tilde = center_scale(X, mu_X, sigma_X)

    #Leverage points removal:
    #Compute the PCA scores. Override the eventual number of PCs: ALL the
    #PCs are needed, as the outliers are given by the last PCs examination
    all_eig = X.shape[1]-1
    PCs, eigval = PCA_fit(X_tilde, all_eig)
    scores = X_tilde @ PCs
    TOL = 1E-16


    scores_dist = np.empty((X.shape[0],), dtype=float)
    #For each observation, compute the distance from the center of the manifold
    for ii in range(0,X.shape[0]):
    #     t_sq = 0
    #     lam_j = 0
    #     for jj in range(eigens, scores.shape[1]):
    #         t_sq += scores[ii,jj]**2
    #         lam_j += eigval[jj]
    #     scores_dist[ii] = t_sq/(lam_j + TOL)
        scores_dist[ii] = 0
        for jj in range(eigens, scores.shape[1]):
            scores_dist[ii] += scores[ii, jj] ** 2 / (eigval[jj] + TOL)


    #Now compute the distance distribution, and delete the observations in the
    #upper 3% (done in the while loop) to get the outlier-free matrix.

    #Divide the distance vector in 100 bins
    n_bins = 100
    min_interval = np.min(scores_dist)
    max_interval = np.max(scores_dist)

    delta_step = (max_interval - min_interval) / n_bins

    counter = 0
    bin = np.empty((len(scores_dist),))
    var_left = min_interval

    #Find the observations in each bin (find the idx, where the classes are
    #the different bins number)
    while counter <= n_bins:
        var_right = var_left + delta_step
        mask = np.logical_and(scores_dist >= var_left, scores_dist < var_right)
        bin[np.where(mask)] = counter
        counter += 1
        var_left += delta_step

    unique, counts = np.unique(bin, return_counts=True)
    cumulativeDensity = 0
    new_counter = 0

    while cumulativeDensity < 0.98:
        cumulative_ = counts[new_counter]/X.shape[0]
        cumulativeDensity += cumulative_
        new_counter += 1

    new_mask = np.where(bin > new_counter)
    X = np.delete(X, new_mask, axis=0)

    return X, bin, new_mask



def outlier_removal_orthogonal(X, eigens, centering, scaling):
    '''
    This function removes the multivariate outliers (orthogonal out) eventually contained
    in the training dataset, via PCA. In fact, examining the reconstruction error
    it is possible to identify the so-called orthogonal outliers. They are characterized
    by very high distance from the manifold (large rec error), once detected they can easily
    be removed.
    Additional info on outlier identification and removal can be found here:

    Hubert, Mia, Peter Rousseeuw, and Tim Verdonck. Computational Statistics & Data Analysis 53.6 (2009): 2264-2274.

    '''
    #center and scale the input matrix before PCA
    mu_X = center(X, centering)
    sigma_X = scale(X, scaling)

    X_tilde = center_scale(X, mu_X, sigma_X)

    #Orthogonal outliers removal:
    PCs, eigval = PCA_fit(X_tilde, eigens)

    epsilon_rec = X_tilde - X_tilde @ PCs @ PCs.T
    # sq_rec_oss = np.power(epsilon_rec, 2)
    sq_rec_oss = np.sqrt(np.sum(np.power(epsilon_rec, 2), axis=1))

    #Now compute the distance distribution, and delete the observations in the
    #upper 3% (done in the while loop) to get the outlier-free matrix.

    #Divide the distance vector in 100 bins
    n_bins = 100
    min_interval = np.min(sq_rec_oss)
    max_interval = np.max(sq_rec_oss)

    delta_step = (max_interval - min_interval) / n_bins

    counter = 0
    bin_id = np.empty((len(epsilon_rec),))
    var_left = min_interval

    #Find the observations in each bin (find the idx, where the classes are
    #the different bins number)
    while counter <= n_bins:
        var_right = var_left + delta_step
        mask = np.logical_and(sq_rec_oss >= var_left, sq_rec_oss < var_right)
        bin_id[np.where(mask)[0]] = counter
        counter += 1
        var_left += delta_step

    #Compute the classes (unique) and the number of elements per class (counts)
    unique, counts = np.unique(bin_id, return_counts=True)
    #Declare the variables to build the CDF to select the observations belonging to the
    #98% of the total
    cumulativeDensity = 0
    new_counter = 0

    while cumulativeDensity < 0.98:
        cumulative_ = counts[new_counter]/X.shape[0]
        cumulativeDensity += cumulative_
        new_counter += 1

    new_mask = np.where(bin_id > new_counter)
    X = np.delete(X, new_mask, axis=0)

    return X, bin_id, new_mask


def PCA_fit(X, n_eig):
    '''
    Perform Principal Component Analysis on the dataset X,
    and retain 'n_eig' Principal Components.
    The covariance matrix is firstly calculated, then it is
    decomposed in eigenvalues and eigenvectors.
    Lastly, the eigenvalues are ordered depending on their
    magnitude and the associated eigenvectors (the PCs) are retained.
    - Input:
    X = CENTERED/SCALED data matrix -- dim: (observations x variables)
    n_eig = number of principal components to retain -- dim: (scalar)
    - Output:
    evecs: eigenvectors from the covariance matrix decomposition (PCs)
    evals: eigenvalues from the covariance matrix decomposition (lambda)
    !!! WARNING !!! the PCs are already ordered (decreasing, for importance)
    because the eigenvalues are also ordered in terms of magnitude.
    '''
    if n_eig < X.shape[1]:
        C = np.cov(X, rowvar=False) #rowvar=False because the X matrix is (observations x variables)

        evals, evecs = LA.eig(C)
        mask = np.argsort(evals)[::-1]
        evecs = evecs[:,mask]
        evals = evals[mask]

        evecs = evecs[:, 0:n_eig]

        return evecs, evals

    else:
        raise Exception("The number of PCs exceeds the number of variables in the data-set.")


def readCSV(path, name):
    try:
        print("Reading training matrix.." + path + "/" + name)
        X = np.genfromtxt(path + "/" + name, delimiter= ',')
    except OSError:
        print("Could not open/read the selected file: " + name)
        exit()

    return X


def scale(X, method, return_scaled_matrix=False):
    '''
    Computes the scaling factor [sigma] of each variable of all data-set observations and
    (eventually) return the scaled matrix.
    - Input:
    X = original data matrix -- dim: (observations x variables)
    method = "string", it is the method which has to be used. Four choices are available: AUTO, PARETO, VAST or RANGE≥
    return_scaled_matrix = boolean, choose if the script must return the scaled matrix (optional)
    - Output:
    sig = scaling factor for the data matrix X
    X0 = centered data matrix (optional)
    '''
    # Main
    TOL = 1E-16
    if not return_scaled_matrix:
        if method.lower() == 'auto':
            sig = np.std(X, axis = 0)
        elif method.lower() == 'pareto':
            sig = np.sqrt(np.std(X, axis = 0))
        elif method.lower() == 'vast':
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
        elif method.lower() == 'range':
            maxima = np.max(X, axis = 0)
            minima = np.min(X, axis = 0)
            sig = maxima - minima
        else:
            raise Exception("Unsupported scaling option. Please choose: AUTO, PARETO, VAST or RANGE.")
        return sig
    else:
        if method.lower() == 'auto':
            sig = np.std(X, axis = 0)
            X0 = X / (sig + TOL)
        elif method.lower() == 'pareto':
            sig = np.sqrt(np.std(X, axis = 0))
            X0 = X / (sig + TOL)
        elif method.lower() == 'vast':
            variances = np.var(X, axis = 0)
            means = np.mean(X, axis = 0)
            sig = variances / means
            X0 = X / (sig + TOL)
        elif method.lower() == 'range':
            maxima = np.max(X, axis = 0)
            minima = np.min(X, axis = 0)
            sig = maxima - minima
            X0 = X / (sig + TOL)
        else:
            raise Exception("Unsupported scaling option. Please choose: AUTO, PARETO, VAST or RANGE.")
        return sig, X0

def split_for_validation(X, validation_quota):
    '''
    Split the data into two matrices, one to train the model (X_train) and the
    other to validate it.
    - Input:
    X = matrix to be split -- dim: (observations x variables)
    validation_quota = percentage of observations to take as validation
    - Output:
    X_train = matrix to be used to train the reduced model 
    X_test = matrix to be used to test the reduced model
    '''

    nObs = X.shape[0]
    nVar = X.shape[1]

    nTest = int(nObs * validation_quota)

    np.random.shuffle(X)

    X_test = X[:nTest,:]
    X_train = X[nTest+1:,:]

    return X_train, X_test


def uncenter(X_tilde, mu):
    '''
    Uncenter a standardized matrix.
    - Input:
    X_tilde: centered matrix -- dim: (observations x variables)
    mu: centering factor -- dim: (1 x variables)
    - Output:
    X0 = uncentered matrix -- dim: (observations x variables)
    '''
    if X_tilde.shape[1] == mu.shape[0]:
        X0 = np.zeros_like(X_tilde, dtype=float)
        for i in range(0, len(mu)):
            X0[:,i] = X_tilde[:,i] + mu[i]
        return X0
    else:
        raise Exception("The matrix to be uncentered and the centering vector must have the same dimensionality.")
        exit()


def unscale(X_tilde, sigma):
    '''
    Unscale a standardized matrix.
    - Input:
    X_tilde = scaled matrix -- dim: (observations x variables)
    sigma = scaling factor -- dim: (1 x variables)
    - Output:
    X0 = unscaled matrix -- dim: (observations x variables)
    '''
    TOL = 1E-16
    if X_tilde.shape[1] == sigma.shape[0]:
        X0 = np.zeros_like(X_tilde, dtype=float)
        for i in range(0, len(sigma)):
            X0[:,i] = X_tilde[:,i] * (sigma[i] + TOL)
        return X0
    else:
        raise Exception("The matrix to be unscaled and the scaling vector must have the same dimensionality.")
        exit()


# ------------------------------
# Decorators (alphabetical order)
# ------------------------------

def accepts(*types):
    """
    Checks argument types.
    """
    def decorator(f):
        assert len(types) == f.__code__.co_argcount
        @functools.wraps(f)
        def wrapper(*args, **kwds):
            for (a, t) in zip(args, types):
                assert isinstance(a, t), "The input argument %r must be of type <%s>" % (a,t)
            return f(*args, **kwds)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

def allowed_centering(func):
    '''
    Checks the user input for centering criterion.
    Exit with error if the centering is not allowed.
    '''
    def func_check(dummy, x):
        if x.lower() != 'mean' and x.lower() != 'min':
            raise Exception("Centering criterion not allowed. Supported options: 'mean', 'min'. Exiting with error..")
            exit()
        res = func(dummy, x)
        return res
    return func_check


def allowed_scaling(func):
    '''
    Checks the user input for scaling criterion.
    Exit with error if the scaling is not allowed.
    '''
    def func_check(dummy, x):
        if x.lower() != 'auto' and x.lower() != 'pareto' and x.lower() != 'range' and x.lower() != 'vast':
            raise Exception("Scaling criterion not allowed. Supported options: 'auto', 'vast', 'pareto' or 'range'. Exiting with error..")
            exit()
        res = func(dummy, x)
        return res
    return func_check