# CLAUDE.md

## Project Summary

AutoSlurm is a command-line tool that automatically generates and submits SLURM job scripts for High Performance Computing (HPC) clusters. It simplifies launching jobs by:

- Generating SLURM batch scripts from reusable templates
- Supporting single-task and multi-task parallel jobs with automatic GPU assignment
- Enabling hyperparameter sweeps with `<[...]>` (paired lists) and `<{...}>` (grid search) syntax
- Implementing chain jobs for long-running tasks that exceed time limits
- Providing interactive mode for shell access to compute nodes

## Project Structure

```
AutoSlurm/
├── auto_slurm/
│   ├── aslurm.py            # Main CLI entry point and orchestrator
│   ├── config.py            # Pydantic configuration models
│   ├── helpers.py           # Chain job utilities (RunTimer, write_resume_file)
│   ├── tests.py             # Unit tests
│   ├── general_config.yaml  # User configuration template
│   └── configs/             # Pre-configured cluster templates (18+ configs)
├── examples/                # Usage examples
├── pyproject.toml           # Package metadata and dependencies
└── README.md                # Documentation
```

## Development Setup

For all code execution, activate the virtual environment first:

```bash
source .venv/bin/activate
```

## Running Tests

```bash
source .venv/bin/activate
python -m pytest auto_slurm/tests.py
```
