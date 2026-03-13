# Chain Jobs

Many HPC clusters impose time limits on SLURM jobs. To run tasks that exceed
these limits, AutoSlurmX supports **chain jobs** — an automatic mechanism where
each job picks up exactly where the previous one left off, forming an infinite
chain until your work is complete.

!!! warning "Working directory matters"
    Resume files are written to `.aslurm/` relative to the **current working
    directory**. Do not change the working directory while your task is running,
    or change it back before writing the resume file. If the directory doesn't
    match, AutoSlurmX will not find the resume file and the chain will break
    silently.

## How It Works

1. Your script monitors elapsed time using a `RunTimer` (created by `start_run`)
2. When the time limit approaches, your script saves a checkpoint and calls
   `write_resume_file(...)` with the command to resume
3. After all tasks in the job finish (`wait`), the generated SLURM script checks
   for `.resume` files in `.aslurm/`
4. If any are found, AutoSlurmX automatically submits a follow-up job via `sbatch`
   that reads and executes those resume commands

This cycle repeats indefinitely — each resumed job can itself write new resume
files, forming a chain that runs until no task requests further resumption.

![Single Resume Job](../images/single_resume_job.png)

## End-to-End Walkthrough

### Step 1: Write your script with timer and resume logic

```python
# file: main.py

import argparse
from auto_slurm.helpers import start_run, write_resume_file

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--start_iter", type=int, default=0)
args = parser.parse_args()

if args.checkpoint_path is not None:
    pass  # ... Load checkpoint here ...

# Set time_limit to LESS than your SLURM wall time (see tip below)
timer = start_run(time_limit=9)  # 9 hours (for a 10-hour SLURM job)

max_iter = 1000
for i in range(args.start_iter, max_iter):
    print(f"Training iteration {i}")

    # ... Do work (training step, etc.) ...

    if timer.time_limit_reached() and i < max_iter - 1:
        # Time limit approaching and still work to do
        # => Save checkpoint and write resume file

        # ... Save model checkpoint here ...

        write_resume_file(
            "python main.py --checkpoint_path my_checkpoint.pt"
            f" --start_iter {i + 1}"
        )
        break  # IMPORTANT: exit the loop after writing resume file
```

!!! tip "Leave a buffer for checkpointing"
    Set `time_limit` to **less** than your SLURM `--time` wall time. For example,
    if your job has a 10-hour wall time, use `time_limit=9` to leave an hour for
    saving the checkpoint and writing the resume file. How much buffer you need
    depends on how long your checkpoint save takes.

!!! danger "Don't forget the `break`"
    After calling `write_resume_file`, you **must** exit the loop (or return from
    the function). Without the `break`, your script will continue running past
    the time limit and may be killed by SLURM mid-operation, potentially
    corrupting your checkpoint.

### Step 2: Launch with AutoSlurmX

```bash
aslurmx -cn haicore_1gpu cmd python main.py
```

That's it. AutoSlurmX handles the rest — when the first job ends and a resume
file exists, it automatically submits the follow-up job.

!!! tip "Preview before submitting"
    Use `--dry-run` to inspect the generated main and resume scripts before
    actually submitting:
    ```bash
    aslurmx -cn haicore_1gpu -d cmd python main.py
    ```
    Check both scripts in `.aslurm/` to verify the chain mechanism looks correct.

## Multi-Task Chain Jobs

Chain jobs work with multi-task jobs too. Each task independently decides whether
to write a resume file — AutoSlurmX keeps spawning chain jobs as long as **at
least one task** writes a resume file.

![Multi Chain Job](../images/multi_chain_job.png)

### How multi-task resume works

- Each task is identified by its **task index** (`SLURM_SUBMIT_TASK_INDEX`),
  which is automatically set by AutoSlurmX
- Resume files are named `.aslurm/<JOB_ID>_<TASK_INDEX>.resume`, so there
  are no naming collisions between tasks
- In the resumed job, **all original task slots are re-executed**. Tasks that
  wrote a resume file run their resumed command; tasks that did *not* write a
  resume file will attempt to read a non-existent `.resume` file and fail for
  that slot
- If only some of your tasks need to continue, make sure tasks that are done
  simply do not call `write_resume_file` — they will naturally drop out

!!! note
    If you need all tasks to either resume or not, have each task write a resume
    file with a no-op command (e.g., `echo "Task already complete"`) to avoid
    errors from missing resume files.

## API Reference

### `start_run(time_limit=48)`

Starts a timer and returns a `RunTimer` object.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `time_limit` | `int` | `48` | Time limit in **hours**. The timer triggers after this many hours have elapsed. |

**Returns:** `RunTimer` instance.

### `RunTimer`

| Method | Returns | Description |
|---|---|---|
| `time_limit_reached()` | `bool` | Returns `True` if elapsed time exceeds the configured limit |
| `reset()` | `None` | Resets the timer to the current time (useful after loading a checkpoint) |

### `write_resume_file(command)`

Writes a `.resume` file to `.aslurm/` containing the command to execute in the
follow-up job.

| Parameter | Type | Description |
|---|---|---|
| `command` | `str` | The full shell command to resume this task |

**Raises:** `RuntimeError` if `SLURM_JOB_ID` is not set (i.e., not running
inside a SLURM job).

Import both from `auto_slurm.helpers`:

```python
from auto_slurm.helpers import start_run, write_resume_file
```

!!! note "Prerequisite"
    `auto_slurm` must be installed in the Python environment that your SLURM job
    activates (the conda env or virtualenv configured in your cluster config).

## Common Pitfalls

| Pitfall | Consequence | Solution |
|---|---|---|
| Forgetting `break` after `write_resume_file` | Script continues past time limit, SLURM kills it mid-operation | Always `break` or `return` immediately after |
| `time_limit` too close to SLURM wall time | No time left to save checkpoint before SLURM kills the job | Set `time_limit` at least 30-60 min below wall time |
| Changing working directory during the run | Resume file written to wrong location, chain breaks | Stay in the original `cwd`, or `cd` back before writing |
| Testing locally (outside SLURM) | `write_resume_file` raises `RuntimeError` | Guard with `if "SLURM_JOB_ID" in os.environ:` or use try/except |
| Missing resume file in multi-task job | Failed `cat` on non-existent file for that task slot | Have completed tasks write a no-op resume, or accept the error |
