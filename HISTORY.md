# History

## 0.1.0 (2024-09-22)

Initial release.

- SLURM batch script generation from Jinja2 templates with Hydra-based configuration
- Pre-configured cluster templates (HoreKa, JUWELS, etc.)
- Single-task and multi-task parallel jobs with automatic GPU assignment
- Chain/resume jobs for long-running tasks that exceed time limits
- Interactive mode for shell access to compute nodes
- Hyperparameter sweep syntax: `<[...]>` (paired lists) and `<{...}>` (grid search)
- Dry-run mode (`--dry`) to inspect generated scripts without submitting
- Custom local configuration files with `appdirs`-based discovery
- Legacy CLI entry point: `aslurm`
- New Click-based CLI entry point: `aslurmx`
  - Multi-command short syntax
  - Automatic hostname detection for cluster config selection
  - Programmatic `ASlurmSubmitter` interface with progress bar support
  - `config edit` and `config where` commands
  - `--exclude` option for filtering sweep combinations
- Resume/chain job support in `aslurmx` (template-based with Jinja2 block inheritance)
- Test suite with unit, CLI, and integration tests
