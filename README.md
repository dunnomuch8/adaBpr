# AdaBpr
Adaboost framework implementation for Bayesian Personalized Ranking considering large dataset. 
## Prerequisites
* [lightfm](https://lyst.github.io/lightfm/docs/home.html#installation): BPR algorithm. To substitude the bpr package, modify the function bpr() in adaboost.py. 
* [joblib](https://joblib.readthedocs.io/en/latest/index.html): Parallel computing.
* Others: scipy, numpy

## Running the tests
First, set the number of threads and batchsize in adaboost.py and eval.py.
```
BATCH_SIZE=5000
NUM_THREADS=8
```

The script consist of 3 commands: train, test, rec(generate the recommendation list). 

Read the usage details: 

```
$ python main.py -h
```

The optional argument --save_time specifies the 2 modes implemented.

saveMemory:
* saving the factorized matrix of each single model
* space required :(n_users+n_items)*dim*num_model
* return a list of models, each containing U,V (+bu,bv), alpha

saveTime:
* saving the whole user-item rating of final ensemble model
* space required:n_users x n_items
* return a list of ONLY ONE model, containing Rate

Time required with mode saveMemory is less than that with mode saveTime at first, but it grows with each iteration.

## TODO
### Training with new data with resumed ensemble model
View previous all models as one single model.

Or, dont't do this. Resort to better solutions/algorithms considering timestamp. 