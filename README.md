QuSoliton: Quantum Soliton toolbox
==================================

[C. Conti](https://github.com/nonlinearxwaves)

QuSoliton is open-source software for simulating the dynamics of quantum solitons.
QuSoliton uses Numpy, Cupy packages as numerical backend, and graphical output is provided by Matplotlib.
QuSoliton is based on the positive P-representation and include nonlocal solitons
QuSoliton is freely available for use and/or modification, and it can be used on all Unix-based platforms and on Windows.
Being free of any licensing fees, QuSoliton is ideal for exploring classical and quantum soliton dynamics for students and researchers.

Support
-------


Installation
------------
Cupy installation requires fine tuning w.r.t. to the install version of CUDA
For example, with CUDA 11.5,
```bash
pip install cupy-cuda115
```
See https://docs.cupy.dev/en/stable/install.html


```bash
pip install qusoliton
```

to get the minimal installation.

```bash
python3 setup.py build #optional
python3 setup.py install
```

To install without cupy-cuda

```bash
python3 setup.py install --no-cuda
```



Documentation
-------------


Contribute
----------


Citing QuSoliton
------------

If you use QuSoliton in your research, please cite XXX
