In this repository, we are interested in studying deep learning techniques for multi-step time series forecasting, specifically for non-stationary signals. This work is largely inspired from the  NeurIPS 2019 paper "Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models". Our contributions consist of :

- Exploring other penalty functions $\Omega$ such as $L^1$, Huber and asymmetric penalties.
- Extending the architecture to LSTM sequence to sequence model.
- Extending the datasets on which the deep learning models (with the DILATE loss) have been trained and tested.
- Provide a link between the soft version of the DTW and DILATE tangled loss $DILATE^T$, and Entropic Regularization in optimal transport.

![](https://github.com/vincent-leguen/DILATE/blob/master/fig2.png)

```
@incollection{leguen19dilate,
title = {Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models},
author = {Le Guen, Vincent and Thome, Nicolas},
booktitle = {Advances in Neural Information Processing Systems},
pages = {4191--4203},
year = {2019}
}


```
