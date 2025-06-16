# pyNICAM-DC

pyNICAM-DC is FORTRAN translation of the dynamical core of Non-hydrostatic ICosahedral Atmospheric Model (NICAM) to Python

For further details on NICAM and its achievements, please visit https://nicam.jp/.

## Credits
pyNICAM-DC is based on [NICAM-DC](http://r-ccs-climate.riken.jp/nicam-dc/) developed by Japan Agency for Marine-Earth Science and Technology (JAMSTEC), Atmosphere and Ocean Research Institute (AORI) at The University of Tokyo, and RIKEN / Advanced Institute for Computational Science (AICS).

## Set up environment via Conda 

### Linux (including WSL)
```
conda create -n pynicam python=3.11 -c conda-forge
conda install -c conda-forge mpich mpi4py jax==0.4.34 jaxlib==0.4.34 toml matplotlib zarr==2.15 jupyterlab ipykernel dask xarray git
```

### Mac
coming soon
