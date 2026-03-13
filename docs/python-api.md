# Python API

For programmatic job submission from Python scripts or notebooks, use the
`ASlurmSubmitter` class. It wraps the AutoSlurmX CLI internally, providing the
same features — config loading, GPU assignment, sweep expansion, script
generation — through a Python interface.

**Typical workflow:**

1. Create an `ASlurmSubmitter` with your cluster config
2. Queue commands with `add_command()`
3. Optionally check `count_jobs()` to preview how many SLURM jobs will be created
4. Call `submit()` to generate scripts and submit them via `sbatch`

## Basic Usage

```python
from auto_slurm.aslurmx import ASlurmSubmitter

submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=4,
)

for lr in [0.001, 0.01, 0.1]:
    submitter.add_command(f'python train.py --learning_rate={lr}')

submitter.submit()
```

This queues 3 commands and submits them in a single SLURM job (since
`batch_size=4` and there are only 3 commands). A `tqdm` progress bar is
displayed during submission.

!!! note "CLI output is suppressed"
    The `submit()` method redirects stdout internally. You will see the `tqdm`
    progress bar but not the per-job CLI output that `aslurmx cmd` normally
    prints.

## Constructor Parameters

```python
ASlurmSubmitter(
    config_name: str,
    batch_size: int = 1,
    gpus_per_task: int | None = None,
    num_gpus: int | None = None,
    randomize: bool = False,
    dry_run: bool = False,
    parallel: bool = False,
    overwrite_fillers: dict[str, str] = {},
    exclude: str | None = None,
    archive_path: str = os.getcwd(),
    logger: logging.Logger = NULL_LOGGER,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config_name` | `str` | *(required)* | Name of the cluster config (e.g., `'haicore_4gpu'`). Must match a YAML file in the config directories. |
| `batch_size` | `int` | `1` | Number of commands per SLURM job. `1` = one job per command (max parallelism). Higher values = fewer jobs but longer per-job runtime. |
| `gpus_per_task` | `int \| None` | `None` | GPUs to assign per command via `CUDA_VISIBLE_DEVICES`. When set, commands within a batch are distributed across GPUs in round-robin fashion. |
| `num_gpus` | `int \| None` | `None` | Total GPUs available. If `None`, inferred from the config's `gres` field. Only relevant when `gpus_per_task` is set. |
| `randomize` | `bool` | `False` | Shuffle command order before batching. Useful for distributing computational load evenly. |
| `dry_run` | `bool` | `False` | Generate SLURM scripts without submitting. Scripts are saved to `archive_path`. |
| `parallel` | `bool` | `False` | Run commands concurrently within each batch (joined with `&`). When `False`, commands run sequentially (joined with `;`). Only applies when `gpus_per_task` is `None`. |
| `overwrite_fillers` | `dict[str, str]` | `{}` | Override config filler values. Keys are filler names, values are the replacements. |
| `exclude` | `str \| None` | `None` | Comma-separated list of nodes to exclude (passed to `sbatch --exclude`). |
| `archive_path` | `str` | `os.getcwd()` | Directory where generated SLURM scripts are stored. |
| `logger` | `logging.Logger` | `NULL_LOGGER` | Logger instance for tracking operations. Defaults to a no-op logger. |

!!! tip "`batch_size` vs CLI `--max-tasks`"
    The CLI uses `--max-tasks` (default 4) to split commands across jobs. The
    Python API uses `batch_size` (default 1). They serve the same purpose but
    have different defaults — adjust `batch_size` accordingly.

## Methods

### `add_command(command: str) -> None`

Adds a shell command to the internal queue. Commands are stored in order and
submitted when `submit()` is called.

```python
submitter.add_command('python train.py --lr=0.01')
submitter.add_command('python train.py --lr=0.001')
```

Commands can include [sweep syntax](usage/sweeps.md), which is expanded at
submission time:

```python
# This single add_command expands to 4 commands (2x2 grid)
submitter.add_command(
    'python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>'
)
```

### `count_jobs() -> int`

Returns the number of SLURM jobs that will be created when `submit()` is called.
Accounts for sweep expansion.

```python
submitter = ASlurmSubmitter('config', batch_size=4)
submitter.add_command('python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>')

print(submitter.count_jobs())  # 1 (4 expanded commands / batch_size 4)
```

This does not modify the command queue.

### `submit() -> None`

Processes all queued commands, groups them into batches, and submits each batch
as a SLURM job. Shows a `tqdm` progress bar during submission.

The submission pipeline:

1. Expand sweep syntax in all queued commands
2. Optionally randomize command order
3. Split into batches of `batch_size`
4. For each batch, generate SLURM scripts and run `sbatch` (unless `dry_run=True`)

## Examples

### Dry run to preview jobs

```python
submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=4,
    dry_run=True,
)

for lr in [1e-3, 1e-4, 1e-5]:
    for bs in [128, 256, 512]:
        submitter.add_command(f'python train.py --lr={lr} --bs={bs}')

print(f"Will create {submitter.count_jobs()} SLURM jobs")  # 3
submitter.submit()  # Scripts created but not submitted
```

### Overriding config fillers

```python
submitter = ASlurmSubmitter(
    config_name='haicore_1gpu',
    batch_size=1,
    overwrite_fillers={
        'time': '04:00:00',
        'mem': '32G',
        'partition': 'gpu_8',
    },
)
```

### GPU assignment

When `gpus_per_task` is set, commands are distributed across GPUs in round-robin
fashion. Commands assigned to the same GPU run sequentially; different GPUs run
in parallel.

```python
submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=8,
    gpus_per_task=1,
)

# 8 commands distributed across 4 GPUs:
#   GPU 0: cmd0, cmd4 (sequential)
#   GPU 1: cmd1, cmd5 (sequential)
#   GPU 2: cmd2, cmd6 (sequential)
#   GPU 3: cmd3, cmd7 (sequential)
for i in range(8):
    submitter.add_command(f'python train.py --seed={i}')

submitter.submit()
```

### Multi-GPU per task

```python
submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=2,
    gpus_per_task=2,  # Each command gets 2 GPUs
)

submitter.add_command('python train.py --model=large')
submitter.add_command('python train.py --model=xlarge')
submitter.submit()
# Command 0: CUDA_VISIBLE_DEVICES=0,1
# Command 1: CUDA_VISIBLE_DEVICES=2,3
```

### Sweep syntax

Sweep syntax in `add_command()` is expanded at submission time, just like on the
CLI. See [Sweeps](usage/sweeps.md) for full syntax documentation.

```python
submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=4,
)

# Grid search: expands to 6 commands
submitter.add_command(
    'python train.py --lr=<{1e-3, 1e-4}> --bs=<{128, 256, 512}>'
)

print(submitter.count_jobs())  # 2 (6 commands / batch_size 4, rounded up)
submitter.submit()
```

### Parallel execution within batches

```python
submitter = ASlurmSubmitter(
    config_name='haicore_1gpu',
    batch_size=4,
    parallel=True,  # Commands run concurrently (joined with &)
)

for i in range(4):
    submitter.add_command(f'python preprocess.py --shard={i}')

submitter.submit()
```

!!! note
    The `parallel` parameter only applies when `gpus_per_task` is `None`. When
    `gpus_per_task` is set, commands are always distributed across GPUs with
    their own parallelism model.

### Custom archive path and logging

```python
import logging

logger = logging.getLogger('my_experiment')
logger.setLevel(logging.INFO)

submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=4,
    archive_path='/scratch/user/experiment_01',
    logger=logger,
)
```

### Excluding nodes

```python
submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=4,
    exclude='node01,node02',
)
```

## Errors and Exceptions

| Exception | When |
|---|---|
| `FileNotFoundError` | `config_name` does not match any config file |
| `subprocess.CalledProcessError` | `sbatch` fails during submission (not raised in `dry_run` mode) |
| `ValueError` | Sweep syntax mixes `<[...]>` and `<{...}>`, or paired lists have different lengths |
