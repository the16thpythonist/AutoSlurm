# Getting Started

## Installation

Install AutoSlurmX as a pip package:

```bash
pip install git+https://github.com/aimat-lab/AutoSlurm.git
```

The `aslurmx` command will then be available. A legacy `aslurm` command is also included.

## Requirements

- Python 3.9+
- Access to an HPC cluster with SLURM

## Your First Job

Once installed, submit a simple single-task job:

```bash
aslurmx -cn haicore_1gpu cmd python train.py
```

This generates a SLURM batch script and submits it via `sbatch`.

!!! tip
    Use `--dry-run` / `-d` to generate the script without submitting, so you can inspect it first:
    ```bash
    aslurmx -cn haicore_1gpu -d cmd python train.py
    ```
    Scripts are written to `./.aslurm/`.

## Listing Available Configs

See all available cluster configurations:

```bash
aslurmx config list
```

This prints a rich table showing each config's partition, time, memory, CPUs, and GRES settings.

## Virtual Environment Auto-Discovery

AutoSlurmX automatically detects virtual environments (`.venv`, `venv`, `.env`, `env`, `virtualenv`) in your current working directory and activates them in the job script. You can override this with:

```bash
aslurmx -cn config -o venv=/path/to/venv cmd python train.py
```

## Hostname-to-Config Mapping

If you don't specify a config with `-cn`, AutoSlurmX auto-detects the config
based on the current hostname using regex patterns defined in
`~/.config/auto_slurm/general_config.yaml`. This lets you simply run:

```bash
aslurmx cmd python train.py
```
