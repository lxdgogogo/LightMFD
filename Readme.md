# LightMRGFD: A Lightweight and Effective model for Multi-relation Graph Anomaly Detection


## Dependencies

- Pytorch 2.3.0
- DGL 2.0.0
- sklearn
- xgboost 2.1.0
- imblearn
- Numpy


***

## Dataset

The datasets are in the "datasets" folder. Run `unzip /datasets/Amazon.zip` and `unzip /datasets/yelp.zip` to unzip the datasetrs.

***

## Hwo to run

Train LightMRGFD on Amazon (40%): 
```
python performance.py --dataset amazon --train_ratio 0.4 --lr 0.02 \
--max_depth 12 --T 200 --theta 1 --device cpu
```

Train LightMRGFD on Yelp (5%): 
```
python performance.py --dataset amazon --train_ratio 0.05 --lr 0.02 \
--max_depth 12 --T 200 --theta 1 --device cpu
```
