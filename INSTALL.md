# Installing / running pyNICAM-DC

The Python dependency declared by the package itself is just `numpy` (see
`setup.py` / `requirements.txt`). The lists below describe the **full
environment** needed to actually run the model and the optional tooling.

## For a basic run
- python 3.11 or 3.12 (others may work but are not tested)
- mpich
- mpi4py
- toml
- xarray
- dask
- zarr (2.15 recommended; needs a downgrade from 3.xx)

## For quicklook / plotting
- matplotlib
- mpl_toolkits
- cartopy
- scipy
- jupyterlab
- ipykernel

## To run with jax (experimental)
- jax
- jaxlib
- if on macOS: jax-metal (experimental), e.g.
  `jax==0.4.34 jaxlib==0.4.34 jax-metal==0.1.1`

## Other notes
- May run faster with micromamba.
- May need Xcode on macOS.

### Example setup (Tomoki's MacBook, 2025/05/23)
```sh
micromamba create -n jax_mpi python=3.11 -c conda-forge
micromamba activate jax_mpi
micromamba install -c conda-forge mpich mpi4py
pip uninstall -y jax jaxlib jax-metal && \
  pip install --upgrade jax==0.4.34 jaxlib==0.4.34 jax-metal==0.1.1
micromamba install -n jax_mpi -c conda-forge toml matplotlib dask xarray \
  cartopy scipy zarr==2.15
micromamba install -n jax_mpi -c conda-forge jupyterlab ipykernel
python -m ipykernel install --user --name=jax_mpi --display-name "Python (jax_mpi)"
```
