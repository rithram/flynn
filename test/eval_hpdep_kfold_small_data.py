#! /usr/bin/env python

import argparse
from datetime import datetime
from itertools import product
import logging
logger = logging.getLogger('HPDEP')

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
from numpy import geomspace, linspace
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold as SkFold
from sklearn.preprocessing import normalize
from openml.datasets import get_dataset
from joblib import Parallel, delayed
from tqdm import tqdm

from flynn_hpo import gen_pipeline_from_val, get_blank_history
from utils.densify import Densify
from utils.evaluations import eval_est_kfold_cv


def print_dset_shape(X, y):
    print('X:\n  Type: %s\n  Shape: %s' % (str(type(X)), str(X.shape)))
    print('y:\n  Type: %s\n  Shape: %s' % (str(type(y)), str(y.shape)))
    print(np.unique(y, return_counts=True))


def prep_data(X, y, nthreads=1):
    nX = normalize(X)
    dnX = Densify(**{'nthreads': nthreads}).fit(nX).transform(nX)
    return [(nX, y), (dnX, y)]


##### OPENML #####
def get_openml_set(did):
    d = get_dataset(did)
    print('fetched \'%s\' with target column \'%s\'' % (d.name, d.default_target_attribute))
    X, y, c, a = d.get_data(target=d.default_target_attribute, dataset_format='array')
    assert not np.any(c)
    ret = prep_data(X, y)
    print_dset_shape(ret[0][0], ret[0][1])
    print_dset_shape(ret[1][0], ret[1][1])
    return ret


##### DIGITS #####
def get_digits():
    X, y = load_digits(return_X_y=True)
    ret = prep_data(X, y)
    print_dset_shape(ret[0][0], ret[0][1])
    print_dset_shape(ret[1][0], ret[1][1])
    return ret


if __name__ == '__main__':
    hp_choices = ['ef', 'cs', 'wn', 'gamma']
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--n_parallel', help='Number of parallel workers', type=int, default=1
    )
    parser.add_argument(
        '-F', '--n_folds', help='Number of folds', type=int, default=10
    )
    parser.add_argument(
        '-e', '--exp_factor_lb', help='Lower bound on the expansion factor HP', type=float,
    )
    parser.add_argument(
        '-E', '--exp_factor_ub', help='Upper bound on the expansion factor HP', type=float,
    )
    parser.add_argument(
        '-s', '--conn_spar_lb', help='Lower bound on the connection sparsity HP', type=float,
    )
    parser.add_argument(
        '-S', '--conn_spar_ub', help='Upper bound on the connection sparsity HP', type=float,
    )
    parser.add_argument(
        '-w', '--wta_nnz_lb', help='Lower bound on the winner-take-all NNZ HP', type=int,
    )
    parser.add_argument(
        '-W', '--wta_nnz_ub', help='Upper bound on the winner-take-all NNZ HP', type=int,
    )
    parser.add_argument(
        '-g', '--gamma_lb', help='Lower bound for \'gamma\' in the bloom filter', type=float,
    )
    parser.add_argument(
        '-G', '--gamma_ub', help='Upper bound for \'gamma\' in the bloom filter', type=float,
    )
    parser.add_argument(
        '-H', '--hp2tune', help='The hyper-parameter to tune while fixing the rest', choices=hp_choices,
    )
    parser.add_argument(
        '-n', '--nvals_for_hp', help='The number of values to try for the hyper-parameter', type=int
    )
    args = parser.parse_args()
    data_list = [
        ('digits', get_digits),
        ('letters', lambda: get_openml_set(did=6)),
        ('segment', lambda: get_openml_set(did=36)),
        ('gina_prior2', lambda: get_openml_set(did=1041)),
        ('USPS', lambda: get_openml_set(did=41082)),
        ('madeline', lambda: get_openml_set(did=41144)),
    ]
    # Generate HP grid
    dns_hp = [0]
    ef_hps = geomspace(
        args.exp_factor_lb, args.exp_factor_ub,
        num=args.nvals_for_hp, endpoint=True
    ).tolist() if args.hp2tune == 'ef' else [
        args.exp_factor_lb, args.exp_factor_ub
    ]
    cs_hps = geomspace(
        args.conn_spar_lb, args.conn_spar_ub,
        num=args.nvals_for_hp, endpoint=True
    ).tolist() if args.hp2tune == 'cs' else [
        args.conn_spar_lb, args.conn_spar_ub
    ]
    wn_hps = geomspace(
        args.wta_nnz_lb, args.wta_nnz_ub,
        num=args.nvals_for_hp, endpoint=True
    ).astype(np.int64).tolist() if args.hp2tune == 'wn' else [
        args.wta_nnz_lb, args.wta_nnz_ub
    ]
    nb_c_lb, nb_c_ub = 1. - args.gamma_ub, 1. - args.gamma_lb
    c_hps = linspace(
        nb_c_lb, nb_c_ub, num=args.nvals_for_hp, endpoint=True
    ).tolist() if args.hp2tune == 'gamma' else [nb_c_lb, nb_c_ub]
    c_hps.append(None)
    if nb_c_ub == 1.0: c_hps.remove(1.0)
    print('Trying the following HPs:')
    print('EF: ', ef_hps)
    print('CS: ', cs_hps)
    print('WN: ', wn_hps)
    print('DN: ', dns_hp)
    print('gammas: ', [0 if c is None else 1 - c  for c in c_hps])
    today = datetime.today()
    timestamp = (
        str(today.year) + str(today.month) + str(today.day) + '.' +
        str(today.hour) + str(today.minute) + str(today.second)
    )
    print('-'*30)
    print('Experiment timestamp: %s' % timestamp)
    print('-'*30)
    res_dir = 'results/hpdep/' + timestamp
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print('-' * 30)
    print('Results to be saved in', res_dir, 'directory')
    print('-' * 30)
    # get CV splitter
    skf = SkFold(n_splits=args.n_folds, shuffle=True, random_state=5489)
    # lambda for parallel processing
    efunc = lambda g: eval_est_kfold_cv(g[1], dset[g[0][0]], skf)
    for data_name, dfunc in data_list:
        print('=' * 30)
        print('Processing', data_name, '...')
        print('-' * 30)
        dset = dfunc()
        ndims = [dset[0][0].shape[1], dset[1][0].shape[1]]
        output_fname_prefix = res_dir + '/' + data_name + '.' + args.hp2tune
        res_fname = output_fname_prefix + '.hpdep.csv'
        print('Saving HP grid results in \'%s\'' % res_fname)
        # skip ef-wn combination when the FlyHash is not at least 0.5 sparse
        grid = [
            (
                (dns, ef, cs, wn, c),
                gen_pipeline_from_val(
                    [dns, ef, cs, wn], ndims[dns],
                    nthreads=1, skip_densify=True, c=c
                )
            ) for dns, ef, cs, wn, c in product(
                dns_hp, ef_hps, cs_hps, wn_hps, c_hps
            ) if int(ef * ndims[dns]) >= 2 * wn
        ]
        print('Will try a total of %i combinations' % len(grid))
        print('Executing with %i jobs' % args.n_parallel)
        results = Parallel(n_jobs=args.n_parallel)(
            delayed(efunc)(g) for g in tqdm(grid, desc='HP', total=len(grid))
        )
        assert len(results) == len(grid)
        res_df = get_blank_history()
        res_df['c'] = []
        for g, r in zip(grid, results):
            dns, ef, cs, wn, c = g[0]
            acc, f1a, f1i = r[0]
            res_df['densify'].append(False if dns == 0 else True)
            res_df['exp_factor'].append(ef)
            res_df['conn_sparsity'].append(cs)
            res_df['wta_nnz'].append(wn)
            res_df['c'].append(1.0 if c is None else c)
            res_df['acc'].append(acc)
            res_df['f1macro'].append(f1a)
            res_df['f1micro'].append(f1i)
        print('Saving HP results')
        pd.DataFrame.from_dict(res_df).to_csv(res_fname, index=False)
    print('=' * 30)
    print('Results saved in \'%s\'' % res_dir)
