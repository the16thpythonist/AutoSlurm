# Hyperparameter Sweeps

Instead of manually writing out every command variation, AutoSlurmX provides
**sweep syntax** — a shorthand that expands a single command template into many
commands, each with different parameter values. This is ideal for hyperparameter
searches, ablation studies, or any scenario where you need to run the same script
with varying arguments.

Sweep expansion is **pure text substitution** on the command string. It is not
argument-aware — the `<[...]>` and `<{...}>` markers can appear anywhere in
the command, and each marker is replaced with a concrete value. The expanded
commands are then distributed across SLURM jobs automatically.

!!! danger "Always quote sweep syntax"
    Wrap sweep expressions in **single quotes** `''`:
    ```bash
    aslurmx -cn config cmd python train.py lr='<{1e-3,1e-4}>'
    ```
    Without quotes, bash interprets `<` and `>` as **I/O redirection** and
    `{...}` as **brace expansion**, which will silently produce wrong commands
    or errors. If you see unexpected "No such file or directory" errors, missing
    quotes are the likely cause.

## Paired Lists `<[...]>`

Values across parameters are **zipped together positionally** — the first value
of each parameter is combined into one command, then the second value of each,
and so on (like columns in a table).

```bash
aslurmx -cn horeka_4gpu cmd python train.py \
   lr='<[1e-3,1e-4,1e-5,1e-6]>' \
   batch_size='<[1024,512,256,128]>'
```

This creates **4 commands**:

| Task | Expanded command |
|------|-----------------|
| 1 | `python train.py lr=1e-3 batch_size=1024` |
| 2 | `python train.py lr=1e-4 batch_size=512` |
| 3 | `python train.py lr=1e-5 batch_size=256` |
| 4 | `python train.py lr=1e-6 batch_size=128` |

!!! warning "All paired lists must have the same length"
    If the lists have different lengths, AutoSlurmX raises a `ValueError`:
    ```bash
    # ERROR: 4 learning rates but only 2 batch sizes
    aslurmx -cn config cmd python train.py \
       lr='<[1e-3,1e-4,1e-5,1e-6]>' batch_size='<[1024,512]>'
    ```

## Grid Search `<{...}>`

**Every possible combination** of values is generated (the cartesian product).
Use this when you want to explore all interactions between parameters.

```bash
aslurmx -cn horeka_4gpu cmd python train.py \
   lr='<{1e-3,1e-4,1e-5,1e-6}>' \
   batch_size='<{1024,512,128}>'
```

This creates **12 commands** (4 learning rates x 3 batch sizes):

| Task | Expanded command |
|------|-----------------|
| 1 | `python train.py lr=1e-3 batch_size=1024` |
| 2 | `python train.py lr=1e-3 batch_size=512` |
| 3 | `python train.py lr=1e-3 batch_size=128` |
| 4 | `python train.py lr=1e-4 batch_size=1024` |
| 5 | `python train.py lr=1e-4 batch_size=512` |
| ... | ... |
| 12 | `python train.py lr=1e-6 batch_size=128` |

With the default `--max-tasks=4`, these 12 commands are automatically split
across **3 SLURM jobs** (4 tasks each). See [Automatic Splitting](multi-task.md#automatic-splitting-across-jobs)
for details.

!!! tip "Preview before submitting"
    Use `--dry-run` to see exactly how many commands are generated and how
    they are split across jobs, without actually submitting:
    ```bash
    aslurmx -cn horeka_4gpu -d cmd python train.py \
       lr='<{1e-3,1e-4}>' batch_size='<{1024,512}>'
    ```

## Rules and Constraints

1. **You cannot mix `<[...]>` and `<{...}>` in the same command.** AutoSlurmX
   raises a `ValueError` if both syntaxes appear in a single command:
    ```bash
    # ERROR: mixing paired list and grid search
    aslurmx -cn config cmd python train.py \
       lr='<[1e-3,1e-4]>' batch_size='<{1024,512}>'
    ```

2. **Paired lists must all have the same length** (see warning above).

3. **Expansion is per-command.** If you have multiple `cmd` markers, each
   command's sweeps are expanded independently:
    ```bash
    # These are two separate commands, each with its own sweep
    aslurmx -cn config \
       cmd python train.py lr='<{1e-3,1e-4}>' \
       cmd python eval.py metric='<{accuracy,f1}>'
    # Result: 4 commands (2 train + 2 eval)
    ```

4. **Sweep values should be simple tokens** — numbers, short strings, file paths.
   Values should not contain the literal sequences `]>` or `}>` as these are
   used as delimiters.

## Values Containing Commas

If a sweep value itself contains a comma, wrap it in **double quotes** inside
the sweep markers. The double quotes prevent the comma from being interpreted
as a value separator:

```bash
aslurmx -cn config cmd python train.py \
   --features='<{"partial_charges", "partial_charges,x", "x"}>' \
   --lr='<{1e-3,1e-4}>'
```

This creates **6 commands** (3 feature sets x 2 learning rates). The value
`"partial_charges,x"` is treated as a single value, not split on the comma.

## Combining with Other Features

### With `cmdNx` repetition

Command repetition (`cmdNx`) happens **before** sweep expansion. The repeated
command is expanded, then the sweep markers in each copy are resolved:

```bash
# cmd2x with a 2-value grid: 2 repetitions * 2 values = 4 commands
aslurmx -cn config cmd2x python train.py --lr='<{0.1,0.01}>'
```

This produces:

1. `python train.py --lr=0.1`
2. `python train.py --lr=0.01`
3. `python train.py --lr=0.1`
4. `python train.py --lr=0.01`

### With `--max-tasks` / `-mt`

Controls how many expanded commands fit in a single SLURM job. The default
is 4. If your sweep generates more, they are split automatically:

```bash
# 12 grid combinations, 6 per job = 2 jobs
aslurmx -cn config --max-tasks=6 cmd python train.py \
   lr='<{1e-3,1e-4,1e-5}>' batch_size='<{1024,512,256,128}>'
```

### With `--same` / `-s`

Forces all expanded commands into a **single job**, regardless of `--max-tasks`:

```bash
# All 12 combinations in one job
aslurmx -cn config --same cmd python train.py \
   lr='<{1e-3,1e-4,1e-5,1e-6}>' batch_size='<{1024,512,128}>'
```

### In the Python API

Sweep syntax works in `ASlurmSubmitter.add_command()` as well. Commands are
expanded at submission time:

```python
from auto_slurm.aslurmx import ASlurmSubmitter

submitter = ASlurmSubmitter(
    config_name='haicore_4gpu',
    batch_size=4,
)

# This single add_command produces 4 expanded commands
submitter.add_command(
    "python train.py --lr=<{0.1, 0.01}> --bs=<{16, 32}>"
)

# count_jobs() accounts for expansion: 4 commands / batch_size 4 = 1 job
print(submitter.count_jobs())  # 1

submitter.submit()
```

See [Python API](../python-api.md) for full `ASlurmSubmitter` documentation.
