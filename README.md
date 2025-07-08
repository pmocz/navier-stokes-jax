# navier-stokes-jax

Philip Mocz (2025)

A simple Navier-Stokes solver in JAX to be used to solve inverse problems


## Virtual Environment

```console
module purge
module load python/3.11
python -m venv --system-site-packages $VENVDIR/navier-stokes-jax-venv
source $VENVDIR/navier-stokes-jax-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Run Locally

```console
python navier-stokes-jax.py
```


## Submit job (Rusty)

```console
sbatch sbatch_rusty.sh
```

## Plot results

```console
python plot_checkpoints.p
```

