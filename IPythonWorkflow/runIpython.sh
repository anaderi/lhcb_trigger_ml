#source ~/cernroot/bin/thisroot.sh
cd `dirname $0` 
echo "$PWD"
ipython notebook --pylab=inline
