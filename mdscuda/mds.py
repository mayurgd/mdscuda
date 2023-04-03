import time
import warnings
import numpy as np

from numba import cuda
from numbers import Integral
from scipy.stats.stats import pearsonr
from sklearn.base import BaseEstimator
from scipy.spatial.distance import squareform
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.utils._param_validation import Interval, StrOptions

from mdscuda.minkowski import minkowski_pairs
from mdscuda.utils import bits, np_type, idx, euclidean_pairs_gpu


@cuda.jit("void(float{}[:], float{}[:])".format(bits, bits))
def b_gpu(d, delta):
    """Calculates B matrix from SMACOF algorithm and overwrites d with B matrix.
    Diagonal entries are calculated in x_gpu function.

    Args:
        d (cuda device array): matrix of pairwise distances of current iteration of SMACOF algorithm in longform
        delta (cuda device array): original distance matrix in longform
    """
    i = cuda.grid(1)
    if i < d.shape[0]:
        if d[i] != 0:
            d[i] = -delta[i] / d[i]


@cuda.jit("void(float{}[:, :], float{}[:])".format(bits, bits))
def x_gpu(x, b):
    """Computes matrix multiplication B*x and overwrites x with B*x/n where n is number of samples.
    Diagonal entries of B matrix are computed within this function before matrix multiplication occurs.

    Args:
        x (cuda device array): embedding for current iteration of SMACOF algorithm
        b (cuda device array): B matrix in longform (with no diagonal entries)
    """
    n = x.shape[0]
    m = x.shape[1]
    i, j = cuda.grid(2)
    if i < n and j < m:

        # k < i
        tmp = 0
        for k in range(i):
            tmp += b[idx(k, i, n)] * x[k, j]

        # k == i
        bii = 0
        for l in range(i):
            bii -= b[idx(l, i, n)]
        for l in range(i + 1, n):
            bii -= b[idx(i, l, n)]
        tmp += bii * x[i, j]

        # k > i
        for k in range(i + 1, n):
            tmp += b[idx(i, k, n)] * x[k, j]

        cuda.syncthreads()

        x[i, j] = tmp / n


@cuda.jit("void(float{}[:], float{}[:])".format(bits, bits))
def squared_diff_gpu(d, delta):
    """Computes squares of pairwise differences of d and delta as a first step towards computing sigma.
    Overwrites d with result.

    Args:
        d (cuda device array): distance matrix for current iteration of SMACOF algorithm in longform
        delta (cuda device array): original distance matrix in longform
    """
    i = cuda.grid(1)
    if i < d.shape[0]:
        tmp = d[i] - delta[i]
        d[i] = tmp * tmp


@cuda.jit("void(float{}[:], int32)".format(bits))
def sum_iter_gpu(d, s):
    """Performs one iteration of sum reduction on d with stride s.

    Args:
        d (cuda device array): array to be summed
        s (int): stride
    """
    i = cuda.grid(1)
    if i < s and i + s < d.shape[0]:
        d[i] += d[i + s]


# TODO: copying d[0] to host is extremely slow! any way around this?
def sigma(d, delta, blocks, tpb):
    """Calculates and returns sigma

    Args:
        d (cuda device array): pairwise distance matrix for current iteration of SMACOF algorithm in longform
        delta (cuda device array): original pairwise distance matrix in longform
        blocks (int): number of blocks
        tpb (int): threads per block

    Returns:
        float: sigma value
    """
    # tick = time.perf_counter()
    squared_diff_gpu[blocks, tpb](d, delta)
    # print('sigma diff', time.perf_counter() - tick)
    # tick = time.perf_counter()
    s = 1
    while s < d.shape[0]:
        s *= 2
    s = s // 2
    while s >= 1:
        sum_iter_gpu[int(s / tpb + 1), tpb](d, s)
        s = s // 2
    # print('sigma sum', time.perf_counter() - tick)
    return d[0]


def smacof(x, delta, max_iter, verbosity):
    """Performs SMACOF algorithm

    Args:
        x (cuda device array): initial embedding matrix
        delta (cuda device array): original pairwise distance matrix in longform
        max_iter (int): max number of iterations of SMACOF algorithm
        verbosity (int): 0 for silent, 1 to print sigma after every initialization , 2 to print sigma after every iteration (slows performance)

    Returns:
        [type]: [description]
    """
    rows = x.shape[0]
    cols = x.shape[1]

    block_dim = (16, 16)
    grid_dim = (int(rows / block_dim[0] + 1), int(rows / block_dim[1] + 1))
    grid_dim_x = (int(rows / block_dim[0] + 1), int(cols / block_dim[1] + 1))

    tpb = 256
    grids = int(rows * (rows - 1) // 2 / tpb + 1)

    stream = cuda.stream()
    x2 = cuda.to_device(np.asarray(x, dtype=np_type), stream=stream)
    delta2 = cuda.to_device(np.asarray(delta, dtype=np_type), stream=stream)
    d2 = cuda.device_array(rows * (rows - 1) // 2, dtype=np_type)

    for iter in range(max_iter):

        if verbosity >= 2:  # this overwrites d2
            euclidean_pairs_gpu[grid_dim, block_dim](x2, d2)
            # tick = time.perf_counter()
            sig = sigma(d2, delta2, grids, tpb)
            # print('sig', time.perf_counter() - tick)
            # todo: break condition.
            print("it: {}, sigma: {}".format(iter, sig))

        # tick = time.perf_counter()
        euclidean_pairs_gpu[grid_dim, block_dim](x2, d2)
        # print('euc', time.perf_counter() - tick)
        # tick = time.perf_counter()
        b_gpu[grids, tpb](d2, delta2)
        # print('b', time.perf_counter() - tick)
        # tick = time.perf_counter()
        x_gpu[grid_dim_x, block_dim](x2, d2)
        # print('bx', time.perf_counter() - tick)

    euclidean_pairs_gpu[grid_dim, block_dim](x2, d2)
    sig = sigma(d2, delta2, grids, tpb)

    if verbosity >= 2:
        print("it: {}, sigma: {}".format(iter + 1, sig))

    x = x2.copy_to_host(stream=stream)
    return (x, sig, iter)


def mds_fit(
    dissimilarities,
    n_components=2,
    init=None,
    n_init=4,
    max_iter=300,
    verbose=0,
    random_state=None,
):
    """Compute multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.
    init : ndarray of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.
    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, default=0
        Level of verbosity.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.
    """
    tick = time.perf_counter()

    dissimilarities = check_array(dissimilarities)
    random_state = check_random_state(random_state)
    dissimilarities = check_symmetric(dissimilarities, raise_exception=True)

    n_samples = dissimilarities.shape[0]
    dissimilarities = squareform(dissimilarities)

    if init is not None:
        if not n_init == 1:
            warnings.warn(
                "Explicit initial positions passed: "
                "performing only one init of the MDS instead of %d" % n_init
            )
            n_init = 1

        n_components = init.shape[1]
        if n_samples != init.shape[0]:
            raise ValueError(
                "init matrix should be of shape (%d, %d)" % (n_samples, n_components)
            )

    for k in range(n_init):

        if init is None:
            x = random_state.uniform(0, 100, size=(n_samples, n_components))
        else:
            x = init

        x, sig, iter = smacof(x, dissimilarities, max_iter, verbose)

        if verbose >= 1:
            print(
                "init {} lasted {} iterations. final sigma: {}".format(
                    k + 1, iter + 1, sig
                )
            )

        if k == 0:
            best = (x, sig)
        elif sig < best[1]:
            best = (x, sig)

    if verbose >= 1:
        print("best sigma: {}".format(best[1]))
        print("mds total runtime: {} seconds".format(time.perf_counter() - tick))

    return best[0]


class MDS(BaseEstimator):
    """Multidimensional scaling.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.
    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, default=0
        Level of verbosity.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.
        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "dissimilarity": [StrOptions({"euclidean", "precomputed"})],
    }

    def __init__(
        self,
        n_components=2,
        n_init=4,
        max_iter=300,
        verbose=0,
        random_state=None,
        dissimilarity="euclidean",
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.dissimilarity = dissimilarity

    def fit(
        self,
        X,
        y=None,
        init=None,
        calc_r2=False,
    ):
        """
        Compute the position of the points in the embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # parameter will be validated in `fit_transform` call
        self.fit_transform(X, init=init, calc_r2=calc_r2)
        return self

    def fit_transform(
        self,
        X,
        y=None,
        init=None,
        calc_r2=False,
    ):
        """
        Fit the data from `X`, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        """
        self._validate_params()
        X = self._validate_data(X)

        sqform = X.shape[0] == X.shape[1]
        if sqform and self.dissimilarity != "precomputed":
            warnings.warn(
                "The MDS API has changed. ``fit`` now constructs an"
                " dissimilarity matrix from data. To use a custom "
                "dissimilarity matrix, set "
                "``dissimilarity='precomputed'``."
            )

        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix_ = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix_ = minkowski_pairs(X, p=2, sqform=False)

        self.embedding_ = mds_fit(
            dissimilarities=self.dissimilarity_matrix_,
            n_components=self.n_components,
            init=init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        if calc_r2:
            if sqform:
                X = squareform(X)  # converts to longform for r2 calculation
            self.r2 = (
                pearsonr(minkowski_pairs(self.embedding_, sqform=False), X)[0] ** 2
            )
        return self.embedding_
