import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from .mat_utils import fwht

import logging
logger = logging.getLogger('HDX')

def HD_x(D_by_sqrt_d, pad, x):
    # (d^{-1/2}) * [Dx 0 ... 0]
    HDx = np.concatenate([D_by_sqrt_d * x, pad])
    # (d^{-1/2}) * H [Dx 0 ... 0]
    fwht(HDx)
    return HDx


class Densify:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'nthreads' in kwargs:
            self.nthreads = kwargs['nthreads']
        else:
            self.nthreads = 1
        self.kwargs['nthreads'] = self.nthreads
        return

    def set_params(self, **kwargs):
        self.nthreads = None
        for k in kwargs:
            self.kwargs[k] = kwargs[k]

    def fit(self, X, y=None):
        nrows, self.ncols = X.shape
        # Generate a random diagonal sign matrix
        D = (
            np.random.binomial(n=1, p=0.5, size=self.ncols).astype(float) * 2.0
            - 1.0
        )
        # Pad each point to have some power of 2 size
        lncols = np.log2(self.ncols)
        self.new_ncols = (
            self.ncols if int(lncols) == lncols
            else np.power(2, int(lncols) + 1)
        )
        logger.debug(
            f'Padding {self.ncols} features to {self.new_ncols} with 0'
        )
        self.pad_vec = np.zeros(self.new_ncols - self.ncols)
        # Caching the 1/sqrt(d) operation inside the D sign vector
        self.D_by_sqrt_d = D / np.sqrt(float(self.new_ncols))
        return self

    def transform(self, X):
        nrows, ncols = X.shape
        assert ncols == self.ncols, (
            f'The dimension mismatch (expected: {self.ncols}, '
            f'obtained: {ncols})'
        )
        assert hasattr(self, 'pad_vec')
        assert hasattr(self, 'D_by_sqrt_d')
        feval = lambda x: HD_x(self.D_by_sqrt_d, self.pad_vec, x)
        logger.debug(
            f'Densifying {nrows} points with {self.nthreads} threads ...'
        )
        HD_X = np.array(Parallel(n_jobs=self.nthreads, require='sharedmem')(
            delayed(feval)(x) for x in X
        ))
        logger.debug('Output shape: %s' % str(HD_X.shape))
        return HD_X

    def get_params(self, deep=False):
        return self.kwargs
