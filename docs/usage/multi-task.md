# Multi-Task Jobs

Execute multiple independent scripts on a single node by supplying multiple `cmd` markers:

```bash
aslurmx -cn horeka_4gpu                    \
   cmd python train.py --config conf0.yaml \
   cmd python train.py --config conf1.yaml \
   cmd python train.py --config conf2.yaml \
   cmd python train.py --config conf3.yaml
```

All 4 tasks run in parallel, each automatically assigned one GPU.

![Multi Job](../images/multi_job.png)

## Command Repetition (`cmdNx`)

To run the same command multiple times, use the `cmdNx` syntax (both `cmdN` and `cmdNx` are supported):

```bash
# Run training 5 times with different random seeds
aslurmx -cn horeka_4gpu cmd5x python train.py --seed=$RANDOM
```

You can mix regular `cmd` with `cmdNx`:

```bash
# Setup once, train 3 times, evaluate once
aslurmx -cn config cmd python setup.py cmd3x python train.py cmd python eval.py
```

## GPU Assignment

By default, each task uses one GPU. Override this with `--gpus-per-task` / `-gpt`:

```bash
# Each task gets 2 GPUs
aslurmx -cn bwuni_4gpu_h100 --gpus-per-task=2 \
   cmd python train.py \
   cmd python train.py
```

For non-GPU jobs, use `--max-tasks` / `-mt` to control how many tasks run per job:

```bash
aslurmx -cn config --max-tasks=8 cmd python preprocess.py
```

## Forcing All Commands Into One Job

Use `--same` / `-s` to put all commands into a single job regardless of the `max-tasks` limit:

```bash
aslurmx -cn config --same \
   cmd python task1.py \
   cmd python task2.py \
   cmd python task3.py
```

## Automatic Splitting Across Jobs

Each config specifies a maximum number of tasks per job (via `--max-tasks`, default: 4).
If you supply more commands than that limit, they are automatically split across multiple jobs.

![Split Job](../images/split_job.png)
