# FlyNN: Fly Nearest Neighbor Classifier

Code for the "Federated Nearest Neighbor Classification with a Colony of Fruit-Flies" paper appearing at the AAAI 2022 conference on Artificial Intelligence.


Table of contents:
- [Setting up environment](#setting-up-environment)
  - [Prerequisites](#prerequisites)
  - [Installing requirements](#installing-requirements)
- [Running experiments](#running-experiments)
  - [Running experiments on hyper-parameter dependence](#running-experiments-on-hyper-parameter-dependence)
  - [Running comparison to baselines with synthetic data](#running-comparison-to-baselines-with-synthetic-data)
  - [Running comparison to baselines with OpenML data](#running-comparison-to-baselines-with-openml-data)
  - [Scaling with number of parties](#scaling-with-number-of-parties)
  - [Differential privacy effect](#differential-privacy-effect)
- [Citation](#citation)


## Setting up environment

This section details the setup of the compute environment for executing the provided scripts.

### Prerequisites

- `python3.8`
- `pip`
- `virtualenv`

### Installing requirements

```
$ mkdir flynn
$ virtualenv -p /usr/bin/python3.8 flynn
$ source flynn/bin/activate
(flynn) $ export PYTHONPATH=`pwd`
(flynn) $ pip install --upgrade pip
(flynn) $ pip install -r requirements.txt
```

## Running experiments

This section provides the precise commandline arguments for the various scripts to generate results for the different experiments conducted.

### Running experiments on hyper-parameter dependence

#### Evaluation script options
```
(flynn) $ python test/eval_hpdep_kfold_small_data.py --help
usage: eval_hpdep_kfold_small_data.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-e EXP_FACTOR_LB]
                                      [-E EXP_FACTOR_UB] [-s CONN_SPAR_LB] [-S CONN_SPAR_UB]
                                      [-w WTA_NNZ_LB] [-W WTA_NNZ_UB] [-g GAMMA_LB] [-G GAMMA_UB]
                                      [-H {ef,cs,wn,gamma}] [-n NVALS_FOR_HP]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -e EXP_FACTOR_LB, --exp_factor_lb EXP_FACTOR_LB
                        Lower bound on the expansion factor HP
  -E EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_LB, --conn_spar_lb CONN_SPAR_LB
                        Lower bound on the connection sparsity HP
  -S CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the connection sparsity HP
  -w WTA_NNZ_LB, --wta_nnz_lb WTA_NNZ_LB
                        Lower bound on the winner-take-all NNZ HP
  -W WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all NNZ HP
  -g GAMMA_LB, --gamma_lb GAMMA_LB
                        Lower bound for 'gamma' in the bloom filter
  -G GAMMA_UB, --gamma_ub GAMMA_UB
                        Upper bound for 'gamma' in the bloom filter
  -H {ef,cs,wn,gamma}, --hp2tune {ef,cs,wn,gamma}
                        The hyper-parameter to tune while fixing the rest
  -n NVALS_FOR_HP, --nvals_for_hp NVALS_FOR_HP
                        The number of values to try for the hyper-parameter
```

#### Exact commandline arguments

Set `<NTHREADS>` as per the resources (number of threads, memory) available.

##### FlyHash dimension `m`
```
(flynn) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 4 -E 2048 -s 0.1 -S 0.3 -w 8 -W 32 -g 0. -G 0.5 -H ef -n 10
```

##### Projection density `s`
```
(flynn) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 256 -E 1024 -s 0.1 -S 0.8 -w 8 -W 32 -g 0. -G 0.5 -H cs -n 10
```

##### FlyHash NNZ `\rho`
```
(flynn) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 256 -E 1024 -s 0.1 -S 0.3 -w 4 -W 256 -g 0. -G 0.5 -H wn -n 10
```

##### FlyNN decay rate `\gamma`
```
(flynn) $ python test/eval_hpdep_kfold_small_data.py -t <NTHREADS> -F 10 -e 256 -E 1024 -s 0.1 -S 0.3 -w 8 -W 32 -g 0.1 -G 0.8 -H gamma -n 10
```

### Running comparison to baselines with synthetic data

Set the `<NTHREADS>` based on the compute resources available.

#### Synthetic binary data
We will be trying the following configurations:
- `<NDIMS> = 100, <NNZ_PER_ROW> = 10, <NSAMPLES> = 1000`
- `<NDIMS> = 100, <NNZ_PER_ROW> = 20, <NSAMPLES> = 1000`
- `<NDIMS> = 100, <NNZ_PER_ROW> = 30, <NSAMPLES> = 1000`
- `<NDIMS> = 100, <NNZ_PER_ROW> = 40, <NSAMPLES> = 1000`

##### Evaluation script options
```
(flynn) $ python test/eval_synthetic_binary_data.py --help
usage: eval_synthetic_binary_data.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-c MAX_CALLS]
                                     [-r N_RANDOM_STARTS] [-f EXP_FACTOR_UB]
                                     [-s CONN_SPAR_UB] [-w WTA_NNZ_UB] [-S N_ROWS]
                                     [-d N_COLS] [-L N_CLASSES] [-C N_CLUSTERS_PER_CLASS]
                                     [-W NNZ_PER_ROW] [-R N_REPS]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -c MAX_CALLS, --max_calls MAX_CALLS
                        Maximum number of calls for GP
  -r N_RANDOM_STARTS, --n_random_starts N_RANDOM_STARTS
                        Number of random start points in GP
  -f EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the connection sparsity HP
  -w WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all ratio HP
  -S N_ROWS, --n_rows N_ROWS
                        Number of rows in the data set
  -d N_COLS, --n_cols N_COLS
                        Number of columns in the data set
  -L N_CLASSES, --n_classes N_CLASSES
                        Number of classes in the data set
  -C N_CLUSTERS_PER_CLASS, --n_clusters_per_class N_CLUSTERS_PER_CLASS
                        Number of clusters per class
  -W NNZ_PER_ROW, --nnz_per_row NNZ_PER_ROW
                        Number of NNZ per row in the binary data set
  -R N_REPS, --n_reps N_REPS
                        Number of repetitions
```
##### Exact commandline arguments
```
(flynn) $ python test/eval_synthetic_binary_data.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -S <NSAMPLES> -d <NDIMS> -L 5 -C 3 -W <NNZ_PER_ROW> -R 30
```

#### Synthetic continuous data
We will be trying the following configurations:
- `<NDIMS> = 100, <NSAMPLES> = 1000`

##### Evaluation script options
```
(flynn) $ python test/eval_synthetic_data.py --help
usage: eval_synthetic_data.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-c MAX_CALLS]
                              [-r N_RANDOM_STARTS] [-f EXP_FACTOR_UB] [-s CONN_SPAR_UB]
                              [-w WTA_NNZ_UB] [-S N_ROWS] [-d N_COLS] [-L N_CLASSES]
                              [-C N_CLUSTERS_PER_CLASS] [-R N_REPS]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -c MAX_CALLS, --max_calls MAX_CALLS
                        Maximum number of calls for GP
  -r N_RANDOM_STARTS, --n_random_starts N_RANDOM_STARTS
                        Number of random start points in GP
  -f EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the connection sparsity HP
  -w WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all ratio HP
  -S N_ROWS, --n_rows N_ROWS
                        Number of rows in the data set
  -d N_COLS, --n_cols N_COLS
                        Number of columns in the data set
  -L N_CLASSES, --n_classes N_CLASSES
                        Number of classes in the data set
  -C N_CLUSTERS_PER_CLASS, --n_clusters_per_class N_CLUSTERS_PER_CLASS
                        Number of clusters per class
  -R N_REPS, --n_reps N_REPS
                        Number of repetitions
```
##### Exact commandline arguments
```
(flynn) $ python test/eval_synthetic_data.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -S <NSAMPLES> -d <NDIMS> -L 5 -C 3 -R 30
```

### Running comparison to baselines with OpenML data

OpenML data sets:
- `<MIN_DATA_DIM> = 20, <MAX_DATA_DIM> = 1000, <MAX_DATA_SAMPLES> = 20000`

Based on resources available:
- Set `--n_parallel/-t <NTHREADS>` based on number of threads `<NTHREADS>` available to process that many data sets in parallel.

#### Running `kNNC` baseline
##### Evaluation script options
```
(flynn) $ python test/knn_baseline.py --help
usage: knn_baseline.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-n MIN_DATA_DIM]
                       [-x MAX_DATA_DIM] [-S MAX_DATA_SAMPLES] [-K MAX_K]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -K MAX_K, --max_k MAX_K
                        Maximum k for kNNC
```
##### Exact commandline arguments
```
(flynn) $ python test/knn_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -K 1
(flynn) $ python test/knn_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES>
```

#### Running `SBFC` baseline
##### Evaluation script options
```
(flynn) $ python test/sbf_baseline.py --help
usage: sbf_baseline.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-n MIN_DATA_DIM]
                       [-x MAX_DATA_DIM] [-S MAX_DATA_SAMPLES]
                       [-E EXPANSION_FACTOR_UB]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -E EXPANSION_FACTOR_UB, --expansion_factor_ub EXPANSION_FACTOR_UB
                        Upper bound on the factor with which to project up
```
##### Exact commandline arguments
```
(flynn) $ python test/sbf_baseline.py -t <NTHREADS> -F 10 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -E 2048.0
```

#### Running `FlyNN`
##### Evaluation script options
```
(flynn) $ python test/flynn_hpo.py --help
usage: flynn_hpo.py [-h] [-t N_PARALLEL] [-F N_FOLDS] [-c MAX_CALLS] [-r N_RANDOM_STARTS]
                    [-f EXP_FACTOR_UB] [-s CONN_SPAR_UB] [-w WTA_NNZ_UB] [-D DONE_SET_RE] [-N NON_BINARY]
                    [-n MIN_DATA_DIM] [-x MAX_DATA_DIM] [-S MAX_DATA_SAMPLES] [-B MAX_BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARALLEL, --n_parallel N_PARALLEL
                        Number of parallel workers
  -F N_FOLDS, --n_folds N_FOLDS
                        Number of folds
  -c MAX_CALLS, --max_calls MAX_CALLS
                        Maximum number of calls for GP
  -r N_RANDOM_STARTS, --n_random_starts N_RANDOM_STARTS
                        Number of random start points in GP
  -f EXP_FACTOR_UB, --exp_factor_ub EXP_FACTOR_UB
                        Upper bound on the expansion factor HP
  -s CONN_SPAR_UB, --conn_spar_ub CONN_SPAR_UB
                        Upper bound on the projection density HP
  -w WTA_NNZ_UB, --wta_nnz_ub WTA_NNZ_UB
                        Upper bound on the winner-take-all NNZ HP
  -D DONE_SET_RE, --done_set_re DONE_SET_RE
                        Regex for the files corresponding to datasets already processed
  -N NON_BINARY, --non_binary NON_BINARY
                        Whether to use gamma=0 (binary, default) or gamma>0 (non-binary)
  -n MIN_DATA_DIM, --min_data_dim MIN_DATA_DIM
                        Minimum data dimensionality on OpenML
  -x MAX_DATA_DIM, --max_data_dim MAX_DATA_DIM
                        Maximum data dimensionality on OpenML
  -S MAX_DATA_SAMPLES, --max_data_samples MAX_DATA_SAMPLES
                        Maximum number of samples in data on OpenML
  -B MAX_BATCH_SIZE, --max_batch_size MAX_BATCH_SIZE
                        Maximum batch size for FlyNN training
```
- Set `--max_batch_size/-B <MAX_BATCH_SIZE>` based on amount of memory available; large values lead to larger memory overheads but faster execution times.
##### Exact commandline arguments for `FlyNN, \gamma=0`
```
(flynn) $ python test/flynn_hpo.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES>
```
##### Exact commandline arguments for `FlyNN, \gamma > 0`
- Set `--non_binary/-N` to `True` to evaluate (non-binary) `FlyNN`; by default, the script evaluates (binary) `FlyNN` respectively. Specifically
```
(flynn) $ python test/flynn_hpo.py -t <NTHREADS> -F 10 -c 60 -r 15 -f 2048.0 -s 0.5 -w 256 -n <MIN_DATA_DIM> -x <MAX_DATA_DIM> -S <MAX_DATA_SAMPLES> -N True
```

### Scaling with number of parties

##### Evaluation script options
```
(flynn) $ python test/scaling_nparties.py -h
usage: scaling_nparties.py [-h] [-t N_PARTIES] [-B MAX_BATCH_SIZE] [-R N_REPS]

optional arguments:
  -h, --help            show this help message and exit
  -t N_PARTIES, --n_parties N_PARTIES
                        Maximum number of parties
  -B MAX_BATCH_SIZE, --max_batch_size MAX_BATCH_SIZE
                        Maximum batch size
  -R N_REPS, --n_reps N_REPS
                        Number of repetitions
```
- Set `--max_batch_size/-B <MAX_BATCH_SIZE>` based on amount of memory available; large values lead to larger memory overheads but faster execution times.

##### Exact commandline arguments
```
(flynn) $ python test/scaling_nparties.py -t 16 -B <MAX_BATCH_SIZE> -R 10
```

### Differential privacy effect

#### Synthetic data

##### Evaluation script options
```
(flynn) $ python test/test_dp_flynn_syn.py --help
usage: test_dp_flynn_syn.py [-h] [-t NTHREADS] [-S NROWS] [-d NCOLS] [-L NCLASSES] [-C NCLUSTERS_PER_CLASS]
                            [-R NREPS] [-f EXP_FACTOR] [-k WTA_RATIO] [-c CONN_SPARSITY] [-r {geom,lin}]
                            [-T STEP_SIZE_FOR_T] [-e {non-private,all}]

optional arguments:
  -h, --help            show this help message and exit
  -t NTHREADS, --nthreads NTHREADS
                        Number of parallel workers
  -S NROWS, --nrows NROWS
                        Number of rows in the data set
  -d NCOLS, --ncols NCOLS
                        Number of columns in the data set
  -L NCLASSES, --nclasses NCLASSES
                        Number of classes in the data set
  -C NCLUSTERS_PER_CLASS, --nclusters_per_class NCLUSTERS_PER_CLASS
                        Number of clusters per class
  -R NREPS, --nreps NREPS
                        Number of repetitions
  -f EXP_FACTOR, --exp_factor EXP_FACTOR
                        the expansion factor HP
  -k WTA_RATIO, --wta_ratio WTA_RATIO
                        the WTA ratio
  -c CONN_SPARSITY, --conn_sparsity CONN_SPARSITY
                        the connection sparsity
  -r {geom,lin}, --rounds_option {geom,lin}
                        Values of #rounds to try
  -T STEP_SIZE_FOR_T, --step_size_for_T STEP_SIZE_FOR_T
                        Step size for T values
  -e {non-private,all}, --evaluation {non-private,all}
                        What methods to evaluate
```

##### Exact commandline arguments

We utilize all following 

- <NROWS> = 10000, 100000
- <NCOLS> = 30
- <EF> = 10, 20
- <CS> = 0.1
- <WR> = 0.05, 0.1

```
(flynn) $ python test/test_dp_flynn_syn.py -t 2 -S <NROWS> -d <NCOLS> -L 2 -C 5 -R 10 -f <EF> -k <WR> -c <CS> -r lin -T 50 -e all
```

#### MNIST 3v8

##### Evaluation script options
```
(flynn) $ python test/test_dp_flynn_real.py --help
usage: test_dp_flynn_real.py [-h] [-t NTHREADS]
                             [-d {mnist,cifar10,fashion_mnist,cifar100,higgs,numerai28.6,connect-4,APSFailure}]
                             -L LABEL LABEL [-R NREPS] [-f EXP_FACTOR] [-k WTA_RATIO] [-c CONN_SPARSITY]
                             [-r {geom,lin}] [-T STEP_SIZE_FOR_T] [-e {non-private,all}]

optional arguments:
  -h, --help            show this help message and exit
  -t NTHREADS, --nthreads NTHREADS
                        Number of parallel workers
  -d {mnist,cifar10,fashion_mnist,cifar100,higgs,numerai28.6,connect-4,APSFailure}, --data_name {mnist,cifar10,fashion_mnist,cifar100,higgs,numerai28.6,connect-4,APSFailure}
                        The data set to evaluate on
  -L LABEL LABEL, --label LABEL LABEL
                        The label pair to extract to analyze a binary problem (use -1 for all labels)
  -R NREPS, --nreps NREPS
                        Number of repetitions
  -f EXP_FACTOR, --exp_factor EXP_FACTOR
                        the expansion factor HP
  -k WTA_RATIO, --wta_ratio WTA_RATIO
                        the WTA ratio
  -c CONN_SPARSITY, --conn_sparsity CONN_SPARSITY
                        the connection sparsity
  -r {geom,lin}, --rounds_option {geom,lin}
                        Values of #rounds to try
  -T STEP_SIZE_FOR_T, --step_size_for_T STEP_SIZE_FOR_T
                        Step size for T values
  -e {non-private,all}, --evaluation {non-private,all}
                        What methods to evaluate
```

##### Exact commandline arguments

We utilize all following 

- <EF> = 30
- <CS> = 0.00625
- <WR> = 0.1

```
(flynn) $ python test/test_dp_flynn_real.py -t 2 -d mnist -L 3 8 -R 10 -f <EF> -k <WR> -c <CS> -r lin -T 50 -e all
```


## Citation

Please use the following citation for the paper:
```
Ram, Parikshit and Sinha, Kaushik. "Federated Nearest Neighbor Classification with a Colony of Fruit-Flies." To appear in the Proceedings of the 36th AAAI Conference on Artificial Intelligence. 2022.
```
or
```
@inproceedings{ram2022federated,
  title={Federated Nearest Neighbor Classification with a Colony of Fruit-Flies},
  author={Ram, Parikshit and Sinha, Kaushik},
  booktitle={To appear in Proceedings of the 36th AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
