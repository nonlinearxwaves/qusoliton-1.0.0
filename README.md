QuSoliton: Quantum Soliton toolbox
==================================

[C. Conti](https://github.com/nonlinearxwaves)

QuSoliton is open-source software for simulating the dynamics of quantum solitons.
QuSoliton uses Numpy, Cupy packages as numerical backend, and graphical output is provided by Matplotlib.
QuSoliton is based on the positive P-representation and include nonlocal interactions.
QuSoliton is freely available for use and/or modification, and it can be used on all Unix-based platforms and on Windows.
Being free of any licensing fees, QuSoliton is ideal for exploring classical and quantum soliton dynamics for students and researchers.

QuSoliton is an outcome of the European Project PhoQus (H2020 Program grant number 820392)

Local Installation
------------------


```bash
pip install qusoliton
```

to install for editing and debugging

clone the project and run in the local folder
```bash
python3 -m pip install -e . -v
```


Local Installation with CUDA and cupy
-------------------------------------
Cupy installation requires fine tuning w.r.t. to the install version of CUDA
For example, with CUDA 11.5,
```bash
pip install cupy-cuda115yy
```
See https://docs.cupy.dev/en/stable/install.html


After installing cupy proceed as follows

With cuda (requires cupy) clone the project and run in the local folder
```bash
python3 -m pip install -e .[cuda] -v
```
CREATION OF PACKAGE in PyPi
---------------------------
The package in PyPi is created by running 
in the folder qusoliton-1.0.0 parte of the folder qusoliton

```bash
python3 -m build --skip-dependency-check
```

by following the tutorial
https://packaging.python.org/en/latest/tutorials/packaging-projects/

NB: the package is not available yet I have some issue with cuda (update 9 july 2023)


DOCUMENTATION
-------------

To run test files

```bash
python3 test_3D_diffraction.py
```


Citing QuSoliton
------------

If you use QuSoliton in your research, please cite [arXiv:2202.10741](https://arxiv.org/abs/2202.10741)
See the references in qusoliton.bib

