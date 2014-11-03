# HEP_ML

Library for machine learning in high energy physics.

Contains some classifiers and useful utilities for physicists.

Classifiers to fight correlation:
* `uBoost` 
* `meanadaboost` (one particular case is `knn-AdaBoost`)
* different variations of `GradientBoosting` (one of special interest in `FlatnessLoss`)
Some other ways were also tested (in `hep_ml` experiment submodule)

Classifiers implement [sklearn](http://scikit-learn.org/) interface for classifiers and may interact 
with other parts of scikit-learn library.

`reports` module helps draw different information about trained classifiers 
(learning curves, roc curves, correlation curves, output distribution, distribution of passed variables).

`supplementaryclassifiers` contains wrapper for sklearn classifiers 
(which by default use all provided features in training). 

`metrics` module extends `sklearn.metrics` with new metrics of quality 
(which have the same interface as sklearn's metrics).  
It also contains several metrics of non-uniformity of predictions (correlation metrics), 
but they have more complicated API rather than quality metrics, because need additional information
(about 'uniform' variables, i.e. mass).

Correlation metrics include `SDE`, `Theil`, `CvM`, `KS`. Quality metrics contain `AMS` and `punzy` variations 
(and can compute metrics at optimal cut). 

`toyMC` module contains some simple way for over-sampling 
(generating data which distributed like one your already have).

`grid_search` module has improved version of sklearn.grid_search - it does not check 
all possible combinations, but uses come intellectual optimization.