# BioNAS
NAS for bio-inspired learning rules, we incorporate different feedback alignment techniques from [Biotorch](https://github.com/jsalbert/biotorch)

# Reproducing experiments

To search on cifar10, run 
```
python train_search.py --gpu=0 --batch_size=BATCH_SIZE --epochs=50
```

To train the resulting architecture which can be found in the log file after searching, run:

```
python train.py PARAMS
```

Refer to the hyperparameters used in the submitted paper


The baseline code has been borrowed from [DARTS](https://github.com/quark0/darts) and [EG-NAS](https://github.com/caicaicheng/EG-NAS), by changing the operations.
