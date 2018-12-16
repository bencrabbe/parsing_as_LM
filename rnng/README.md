How to RNNGF
=========

Installation and Requirements
--------------

* Install python 3
* Install Dynet 

I strongly recommend to use [ Intel MKL library ](https://software.intel.com/en-us/mkl) and to manually compile dynet against this library. This allows multiprocessing and considerable speedups at runtime.


How to train a model
----------------------

* Training supposes you have a treebank such as the Penn Treebank, split into 3 files such as : `train.mrg`,`dev.mrg`,`test.mrg`. Trees are supposed to be encoded in one line s-expression format. 
* The parser requires brown clusters that are to be trained with [ Percy Liang's tool ](https://github.com/percyliang/brown-cluster) on the `train.mrg` file. Let's call the resulting file `ptb.brown`. The brown clusters are used by the parser to predict word probabilities using the class based softmax factorisation described by [(Baltescu and Blunsom 2014)](https://arxiv.org/pdf/1412.7119.pdf)
* Training can be supplied an optional configuration file allowing to control hyperparameters. There is a `default.conf` file in the project directory that can be used as a starting point.
* Training is performed with the following command : 

```
python rnngf.py -m mymodel -t train.mrg -d dev.mrg -b ptb.brown -c default.conf
```
The trainer will store the model parameters in `mymodel`. Training takes time. About one day with the default parameters and multiple processors.


How to test a model
---------------

* Testing supposes you have already trained a model, let's call it `mymodel`.

* The test text is supposed to be one sentence / line and tokens are separated by whitespace. Let's call the test text `prince.raw`

Then : 

```
python -m mymodel -p prince.raw -s -B 400
```

parses the `prince.raw` file with a beam of size 400 and ouptuts stats relevant for analysis. Parsing is slow and benefits from MKL multithreading.


