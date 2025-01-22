# LightMRGFD: A Lightweight and Effective model for Multi-relation Graph Anomaly Detection


## Dependencies

- Pytorch 2.1.2
- DGL 2.4.0
- sklearn
- xgboost 2.1.3
- imblearn
- Numpy


***

## Dataset

The datasets are in the "datasets" folder.First unzip these datasets.

***

## Hwo to run



Train LightMRGFD on DBLP (5%): 
```
python performance.py --dataset DBLP --train_ratio 0.05 --lr 0.02 \
--max_depth 12 --T 200 --theta 1 --device cuda
```
