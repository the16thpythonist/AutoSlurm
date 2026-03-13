# Single-Task Jobs

Execute a single task on a compute node:

```bash
aslurmx -cn haicore_1gpu cmd python train.py
```

This will execute `python train.py` using a single GPU on HAICORE.

![Single Job](../images/single_job.png)

!!! tip
    The SLURM job files are written to `./.aslurm/` and submitted with
    `sbatch`. To only create the files without submitting (e.g. for testing), use
    the `--dry-run` / `-d` flag.

## Overwrites

You can overwrite any filler value from the config using the `-o` flag:

```bash
aslurmx -cn haicore_1gpu -o conda_env=my_env cmd python train.py
```

Multiple overwrites are comma-separated:

```bash
aslurmx -cn haicore_1gpu -o conda_env=my_env,time=01:00:00 cmd python train.py
```

To find out what fillers you can overwrite, inspect the `default_fillers` section
in the config files:

```bash
aslurmx config edit <config_name>
```

## Excluding Nodes

Exclude specific nodes from allocation:

```bash
aslurmx -cn config --exclude=node01,node02 cmd python train.py
```
