# lhcb_trigger_ml
Friendly machine learning for LHCb experiment. 
Project should enable one to train and compare classifiers on some training dataset.

The programming language is python,
the analysis is performed in __IPython__ notebooks - commonly used in machine learning interactive shell for python, 
which is good for development, analysis and presenting results (plots, histograms and so on)


# Brief demos:
* [Dalitz Demo](http://nbviewer.ipython.org/github/anaderi/lhcb_trigger_ml/blob/master/IPythonWorkflow/DalitzDemo.ipynb) (several uniforming classifiers on dataset from uBoost paper)
* [Decay of tau into three muons](http://nbviewer.ipython.org/github/anaderi/lhcb_trigger_ml/blob/master/IPythonWorkflow/TauIntoMuons.ipynb)
* [Generation of toy Monte-Carlo](http://nbviewer.ipython.org/github/anaderi/lhcb_trigger_ml/blob/master/IPythonWorkflow/Demo_ToyMonteCarlo.ipynb)
* Any other notebook from repository can be viewed: just paste its link to [nbviewer](http://nbviewer.ipython.org)  

# Main points
* working on uniform classifiers - the classifiers with low correlation of predictions and mass (or some other variable(s))
  * measures of uniformity (`MSE`, `Theil`, `CvM`, `KS`)
  * __uBoost__ optimized implementation inside
  * __uniformGradientBoosting__ (with different losses, specially __FlatnessLoss__ is very interesting)
* parameter optimization  <br />
  See `grid_search` module, there is a simulated annealing-like optimization of parameters on dataset, 
  this optimization can be performed on cluster.
* plots, plots, plots <br />
  See `reports` module, it is a good way to visualize learning curves, roc curves, flatness of predictions on variables.
* there is also procedure to generate toy Monte-Carlo in `toymc` module <br />
  (generates new set of events based on the set of events we already have with same distribution) 
  and special notebook 'ToyMonteCarlo' to demonstrate and analyze its results. 
* parallelism <br />
  ClassifiersDict from `reports` can train classifiers on IPython cluster, <br />
  __uBoost__ is quite slow, and it has built-in parallelism option: different BDTs inside uBoost can be trained parallelly in cluster.

###Getting started
To run most the notebooks, only IPython and some python libraries are needed.

To run example notebooks on some machine, one should have
* [IPython](http://ipython.org/install.html)
* Some python libraries that can be installed using any package manager for python
  (`apt-get` will work too, but Ubuntu repo contains quite old versions of libraries),
  better use [pip](http://pip-installer.org)
  

The libraries you need are `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `rootpy`, `root-numpy`
and maybe something else, basically the packages are installed via command-line:
<pre>sudo pip install numpy scipy pandas scikit-learn matplotlib rootpy root-numpy</pre>
IPython can be installed via pip as well
<pre>sudo pip install "ipython[notebook]" </pre>

To use the repository, clone it with git
<pre>git clone https://github.com/anaderi/lhcb_trigger_ml.git
cd lhcb_trigger_ml
sudo pip install -e .
</pre>

To run IPython, there is shell script: hep_ml/runIpython.sh

In order to work with .root files, you need CERN ROOT (make sure you have it by typing `root` in the console) 
with pyROOT package.
