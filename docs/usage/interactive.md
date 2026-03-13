# Interactive Jobs

Interactive jobs give you shell access on a compute node — useful for debugging,
exploring data, running tools that need interactive input, or testing your
environment before submitting batch jobs. AutoSlurmX submits a SLURM job that
keeps a node allocated, then you attach a bash session to it.

Under the hood, AutoSlurmX submits a job that runs an infinite sleep loop,
keeping the node allocated until you cancel it.

## Quick Start

### 1. Submit the interactive job

```bash
aslurmx -cn haicore_1gpu interactive
```

AutoSlurmX will print the job ID and the exact commands you need:

```
🚀 Interactive job submitted with SLURM ID 123456

Attach a bash shell to the job using:
    srun --jobid 123456 --pty bash

When finished, cancel the job with:
    scancel 123456
```

### 2. Wait for the job to start

The job may spend time in the PENDING state while SLURM allocates resources.
Check its status with:

```bash
squeue --job 123456
```

Look for the `ST` (state) column — `PD` means pending, `R` means running.
You can only attach once the job is in the **RUNNING** state.

!!! tip
    Use `squeue -u $USER` to see all your jobs at once.

### 3. Attach your shell

Once the job is running:

```bash
srun --jobid 123456 --pty bash
```

You now have a bash session on the compute node with the resources defined by
your config (GPUs, memory, CPUs).

### 4. Cancel when done

```bash
scancel 123456
```

!!! warning "Always cancel interactive jobs when finished"
    An idle interactive job wastes cluster resources and counts against your
    allocation. Unlike batch jobs, interactive jobs do not terminate on their
    own — they run until you cancel them or the time limit is reached.

## Customizing Your Session

Interactive jobs inherit all resources from your cluster config (GPUs, CPUs,
memory, time limit, partition). You can override any of these with the `-o` flag.

### Extending the time limit

Configs often default to short time limits (e.g., 1 hour). For longer interactive
sessions, override the time:

```bash
aslurmx -cn haicore_1gpu -o time=04:00:00 interactive
```

!!! warning "Default time limits"
    If your config has `time=01:00:00`, SLURM will kill the interactive job
    after 1 hour — even if you're in the middle of work. Always check (or
    override) the time limit for interactive sessions.

### Overriding other resources

```bash
# More memory and a specific partition
aslurmx -cn haicore_1gpu -o mem=32G,partition=dev interactive

# Different conda environment
aslurmx -cn haicore_1gpu -o conda_env=my_debug_env interactive
```

See [Configuration](../configuration.md) for the full list of available fillers,
or inspect your config directly with `aslurmx config edit <config_name>`.

## Options

All [global CLI options](../cli-reference.md) apply to interactive jobs:

| Option | Description |
|---|---|
| `-cn`, `--config-name` | Config to use. If omitted, auto-detected from hostname |
| `-o`, `--overwrite-fillers` | Override config values (e.g., `time=04:00:00,mem=32G`) |
| `-d`, `--dry-run` | Generate the script without submitting — useful for inspection |
| `-x`, `--exclude` | Exclude specific nodes (e.g., `--exclude=node01,node02`) |
| `--archive-path` | Where to store the generated scripts (default: `.aslurm/`) |

!!! tip "Preview before submitting"
    Use `--dry-run` to inspect the generated script before actually submitting:
    ```bash
    aslurmx -cn haicore_1gpu -d interactive
    ```
    The script will be saved to `.aslurm/main_interactive.sh`.

## Features

### Hostname auto-detection

If you have [hostname-to-config mapping](../configuration.md#hostname-to-config-mapping)
configured, you can omit `-cn`:

```bash
aslurmx interactive
```

### Virtual environment auto-discovery

AutoSlurmX automatically detects virtual environments (`.venv`, `venv`, `.env`,
`env`, `virtualenv`) in your current working directory and activates them in the
interactive job's environment. Override with `-o venv=/path/to/venv`.
