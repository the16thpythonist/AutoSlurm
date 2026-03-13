# CLI Reference

```
aslurmx [OPTIONS] COMMAND [ARGS]...
```

## Global Options

| Option | Description |
|---|---|
| `-cn`, `--config-name TEXT` | Config name to use for scheduling |
| `-o`, `--overwrite-fillers TEXT` | Comma-separated key=value pairs (e.g. `time=01:00:00,mem=16G`) |
| `-s`, `--same` | Put all commands into the same job |
| `-gpt`, `--gpus-per-task INT` | Number of GPUs per task |
| `-ng`, `--num-gpus INT` | Total number of GPUs to use |
| `-mt`, `--max-tasks INT` | Maximum tasks per job (default: 4) |
| `--archive-path PATH` | Where to store generated SLURM scripts |
| `-d`, `--dry-run` | Generate scripts without submitting |
| `-x`, `--exclude TEXT` | Nodes to exclude from allocation |
| `-v`, `--version` | Show version |
| `--help` | Show help |

## Commands

### `cmd`

Schedule commands in SLURM. Supports `cmdNx` repetition syntax.

```bash
# Single command
aslurmx -cn config cmd python train.py

# Multiple commands
aslurmx -cn config cmd python a.py cmd python b.py

# Repeat a command 5 times
aslurmx -cn config cmd5x python train.py
```

### `interactive`

Start an interactive shell job on a compute node.

```bash
aslurmx -cn haicore_1gpu interactive
```

### `config list`

List all available configs in a formatted table.

```bash
aslurmx config list
```

### `config edit`

Open a config file in your editor.

```bash
aslurmx config edit haicore_1gpu
```

### `config where`

Show config storage locations.

```bash
aslurmx config where
```
