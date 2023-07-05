QuSoliton: Quantum Soliton toolbox
==================================

[C. Conti](https://github.com/nonlinearxwaves)

QuSoliton is open-source software for simulating the dynamics of quantum solitons.
QuSoliton uses Numpy, Cupy packages as numerical backend, and graphical output is provided by Matplotlib.
QuSoliton is based on the positive P-representation and include nonlocal interactions.
QuSoliton is freely available for use and/or modification, and it can be used on all Unix-based platforms and on Windows.
Being free of any licensing fees, QuSoliton is ideal for exploring classical and quantum soliton dynamics for students and researchers.

QuSoliton is an outcome of the European Project PhoQus (H2020 Program grant number 820392)

Installation
------------
Cupy installation requires fine tuning w.r.t. to the install version of CUDA
For example, with CUDA 11.5,
```bash
pip install cupy-cuda115yy
```
See https://docs.cupy.dev/en/stable/install.html


```bash
pip install qusoliton
```

to install for editing and debugging


```bash
python3 setup.py --no-cuda develop # faster
# or alternative commands with pip
pip install qusoliton --install-option="--no-cuda" -e .
pip install qusoliton -e <local project path>
pip install qusoliton --editable <local project path>
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


To install without cupy-cuda and develop local copy

```bash
python3 setup.py develop --no-cuda 
```
same without cuda



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

