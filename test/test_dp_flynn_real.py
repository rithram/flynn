import argparse
from itertools import product
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
import sys
from sklearn.metrics import get_scorer

from utils.evaluations import get_app_data
from test_dp_flynn_syn import temp_seed, DPFlyNNFL, save_list2file

import logging
logger = logging.getLogger('DP-FBFC-REAL')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout)
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--nthreads', help='Number of parallel workers', type=int,
        default=1
    )
    data_choices = [
        'mnist', 'cifar10', 'fashion_mnist', 'cifar100',
        'higgs', 'numerai28.6', 'connect-4', 'APSFailure',
    ]
    parser.add_argument(
        '-d', '--data_name', help='The data set to evaluate on',
        choices=data_choices, default=data_choices[0]
    )
    parser.add_argument(
        '-L', '--label', nargs=2, required=True, type=int,
        help='The label pair to extract to analyze a binary problem (use -1 for all labels)',
    )
    parser.add_argument(
        '-R', '--nreps', help='Number of repetitions', type=int, default=5
    )
    parser.add_argument(
        '-f', '--exp_factor', help='the expansion factor HP',
        type=float, default=20.,
    )
    parser.add_argument(
        '-k', '--wta_ratio', help='the WTA ratio',
        type=float, default=0.025,
    )
    parser.add_argument(
        '-c', '--conn_sparsity', help='the connection sparsity',
        type=float, default=0.1,
    )
    rounds_option = ['geom', 'lin']
    parser.add_argument(
        '-r', '--rounds_option', help='Values of #rounds to try',
        choices=rounds_option, default='geom',
    )
    parser.add_argument(
        '-T', '--step_size_for_T', help='Step size for T values',
        type=int, default=25,
    )
    eval_opts = ['non-private', 'all']
    parser.add_argument(
        '-e', '--evaluation', help='What methods to evaluate',
        choices=eval_opts, default='all',
    )

    args = parser.parse_args()
    dname = args.data_name + (
        '-all' if args.label[0] == args.label[1] == -1
        else f'-{args.label[0]}v{args.label[1]}'
    )
    X1, y1, X2, y2, binary = get_app_data(args.data_name, args.label)
    nrows, ncols = X1.shape
    nclasses = len(np.unique(y1))
    EPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    NRNDS = [
        4, 8, 16, 32, 64, # , 128, 256, 512
    ] if args.rounds_option == 'geom' else np.arange(
        args.step_size_for_T, 601, step=args.step_size_for_T, dtype=np.int
    ).tolist() + [4, 8, 16, 32]
    gamma_list = np.arange(0.1, 1.0, step=0.1).tolist()
    nthreads = args.nthreads
    NREPS = args.nreps
    EF = args.exp_factor
    WTAR = args.wta_ratio
    CS = args.conn_sparsity
    # HP:
    # mnist: EF: 30, CS: 0.00625, WTAR: 0.1
    res_list = []
    res_file = (
        f'DP-{dname}-{nrows}x{ncols}x{nclasses}-classes-{nthreads}-parties'
        f'-{NREPS}-reps-EF-{EF:.1f}-CS-{CS:.1f}-WTAR-{WTAR:.4f}.csv'
    )
    res_cols = ['gamma', 'EPS', 'T', 'MEAN', 'STD', 'MIN', 'MAX']
    fargs = {
        'expansion_factor': EF,
        'connection_sparsity': CS,
        'wta_ratio': WTAR,
        'wta_nnz': int(WTAR * EF * ncols),
        'binary': False,
        'c': 1.0 - gamma_list[0],
        'nthreads': nthreads,
    }
    baccs = []
    res_dict = {
        'seed': [],
        **{(g, np.inf, EF*ncols): [] for g in gamma_list},
        **{(g, eps, T): [] for g, eps, T in product(gamma_list, EPS, NRNDS)}
    }

    with temp_seed(5489):
        fargs['random_state'] = np.random.randint(99999)
    res_dict['seed'] += [fargs['random_state']]
    logger.debug(f"Training fbfc with random seed {fargs['random_state']} ...")
    fbfc = DPFlyNNFL(**fargs)
    fbfc.fit(X1, y1)
    for g in gamma_list:
        fbfc.nonbinary_bf_c = 1.0 - g
        fbfc.reduce_bfs()
        bacc = get_scorer('balanced_accuracy')(fbfc, X2, y2)
        res_dict[(g, np.inf, EF*ncols)] += [bacc]
        logger.info(f'- Non-private @ gamma-{g:.1f}: {bacc:.4f}')
    save_list2file(res_dict, res_file, res_cols)
    if args.evaluation == 'non-private':
        sys.exit(0)
    ntrials = 0
    for eps, T in product(EPS, NRNDS):
        for g in gamma_list:
            fbfc.nonbinary_bf_c = 1.0 - g
            for rep in range(NREPS):
                fbfc.dp = {'eps': eps, 'T': T, 'c0': 0.}
                fbfc.reduce_bfs()
                bacc = get_scorer('balanced_accuracy')(fbfc, X2, y2)
                res_dict[(g, eps, T)] += [bacc]
                logger.info(f'- Private(eps: {eps:.2f}, T={T}) @ gamma-{g:.1f}: {bacc:.4f}')
        save_list2file(res_dict, res_file, res_cols)
