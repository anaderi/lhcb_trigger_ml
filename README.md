#lhcb_trigger_ml
LHCb trigger based on machine learning research

Project should enable one to compare several classifiers by their characteristics based on some training dataset.

The programming language is python,
the analysis is performed in IPython notebooks - commonly used in machine learning interactive shell for python, which is good for development, analysis and presenting results (plots, histograms and so on)

At this moment project contains several notebooks which use python classifiers, 
and some functions to import data from ROOT files, these one need ROOT and python wrappers for it.

There is also procedure to generate toy Monte-Carlo (generates new set of events based on the set of events we already have with same distribution) and special notebook 'ToyMonteCarlo' to demonstrate and analyze its results. 

###Getting this to work
To run most the notebooks, only IPython and some python libraries are needed.

To run example notebooks on some machine, one should have
* IPython, see ipython.org/install.html
* Some python libraries that can be installed using any package manager for python
  (apt-get will work too, but Ubuntu repo contains quite old versions of libraries),
  better use pip: 
  pip-installer.org 
  

The libraries you need are
* numpy 
* scipy
* pandas
* scikit-learn 
* rootpy  
* root-numpy

and maybe something else, basically the packages are installed via command-line:
 <pre>sudo pip install numpy scipy pandas scikit-learn rootpy root-numpy</pre>
IPython can be installed via pip as well
 <pre> sudo pip install ipython</pre>
 To run IPython, there is shell script in IpythonWorkflow/ subfolder

In order to work with ROOT files, you need CERN ROOT, make sure you have it by typing 'root' in the console


###Roadway:
We are going to publish notebook on some server to provide easy access from any machine.