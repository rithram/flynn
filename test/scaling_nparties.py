#!/usr/bin/env python

import argparse
from datetime import datetime
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)
from timeit import default_timer
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from fh.flynn import FHBloomFilter as FlyNN
from utils.evaluations import get_datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--n_parties', help='Maximum number of parties',
        type=int, default=1
    )
    parser.add_argument(
        '-B', '--max_batch_size', help='Maximum batch size', type=int,
        default=4096,
    )
    parser.add_argument(
        '-R', '--n_reps', help='Number of repetitions', type=int, default=5,
    )
    args = parser.parse_args()
    today = datetime.today()
    timestamp = (
        str(today.year) + str(today.month) + str(today.day) + '.' +
        str(today.hour) + str(today.minute) + str(today.second)
    )
    print('-'*30)
    print('Experiment timestamp: %s' % timestamp)
    print('-'*30)
    res_dir = 'results/scaling/' + timestamp
    print('Saving results/logs in \'%s\' ...' % res_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    max_batch_size = args.max_batch_size
    nreps = args.n_reps
    dnames = [
        'digits', 'letter', 'mnist', 'fashion_mnist', 'cifar10', 'cifar100'
    ]
    results_dict = {'dname': []}
    nthreads = 1
    while nthreads <= args.n_parties:
        results_dict['Time(T=%i)' % nthreads] = []
        if nthreads > 1:
            results_dict['Speedup(T=%i)' % nthreads] = []
        nthreads *= 2
    hp_per_data = {
        'digits': [0, 256, 0.3, 32],
        'letter': [0, 1447, 0.5, 221, 0.9],
        'mnist': [0, 217.0, 0.025, 17, 0.49],
        'fashion_mnist': [0, 138, 0.105, 8, 0.2],
        'cifar10': [0, 217, 0.026, 26, 0.49],
        'cifar100': [0, 217, 0.026, 26, 0.49],
    }
    for dname in dnames:
        data_head = 'PROCESSING ' + dname
        hline = '=' * len(data_head)
        print(hline)
        print(data_head)
        print(hline)
        results_dict['dname'].append(dname)
        # get dataset
        X, y, _, _ = get_datasets(dname)
        X = normalize(X)
        nrows, ndims = X.shape
        print('[%s] Shape: %i x %i' % (dname, nrows, ndims))
        # obtain hyper-parameters
        hp = hp_per_data[dname]
        assert not (hp is None)
        assert hp[0] == 0
        exp_factor = hp[1]
        conn_sparsity = hp[2]
        wta_ratio = min(0.5, (float(hp[3]) / (exp_factor * ndims)))
        flynn_args = {
            'expansion_factor': exp_factor,
            'connection_sparsity': conn_sparsity,
            'wta_ratio': wta_ratio,
            'wta_nnz': hp[3],
            'batch_size': max_batch_size,
        }
        if len(hp) == 5:
            flynn_args['binary'] = False
            flynn_args['c'] = hp[4]
        # perform multiple repetitions with fixed seed
        rep_res = {}
        nthreads = 1
        while nthreads <= args.n_parties:
            rep_res[nthreads] = []
            nthreads *= 2
        for i in range(nreps):
            rep_seed = np.random.randint(5489)
            print('Rep seed:', rep_seed)
            # loop over different number of threads
            nthreads = 1
            while nthreads <= args.n_parties:
                flynn_args['nthreads'] = nthreads
                flynn_args['random_state'] = rep_seed
                print('*' * 10)
                flynn = FlyNN(**flynn_args)
                print('[%s] Starting training ...' % dname)
                start_time = default_timer()
                flynn.fit(X, y)
                stop_time = default_timer()
                training_time = stop_time - start_time
                print(
                    '[%s]    ... completed in %g seconds with %i threads'
                    % (dname, training_time, nthreads)
                )
                print('*' * 10)
                rep_res[nthreads].append(training_time)
                nthreads *= 2
        for nthreads in rep_res:
            time_key = 'Time(T=%i)' % nthreads
            time_mean = float(np.mean(np.array(rep_res[nthreads])))
            time_std = float(np.std(np.array(rep_res[nthreads])))
            results_dict[time_key].append('{:.2f} +- {:.2f}'.format(time_mean, time_std))
            if nthreads > 1:
                speedup_key = 'Speedup(T=%i)' % nthreads
                speedups = np.array(rep_res[1]) / np.array(rep_res[nthreads])
                sm = np.mean(speedups)
                ss = np.std(speedups)
                results_dict[speedup_key].append('{:.2f} +- {:.2f}'.format(sm, ss))
    results_df = pd.DataFrame.from_dict(results_dict)
    print(results_df.head())
    fname = res_dir + '/scaling_nthreads.results.' + timestamp + '.csv'
    print('Saving results in %s' % fname)
    results_df.to_csv(fname, index=False)
