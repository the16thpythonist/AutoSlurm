# Configuration

AutoSlurmX uses YAML config files that define cluster-specific SLURM settings
(partition, time, memory, GPUs, etc.) and Jinja2 templates to render the actual
bash scripts.

## Managing Configs

List all available configs:

```bash
aslurmx config list
```

This prints a rich table showing each config's partition, time, memory, CPUs, and GRES settings.

Inspect or edit a config:

```bash
aslurmx config edit haicore_1gpu
```

See where config files are stored:

```bash
aslurmx config where
```

## Config Locations

Configs are loaded from two locations (higher priority first):

1. **`~/.config/auto_slurm/configs/`** — your custom configs
2. **`configs/`** folder shipped with the package

## Custom Templates

AutoSlurmX uses Jinja2 templates to generate SLURM bash scripts. You can
override the default templates by placing your own in
`~/.config/auto_slurm/templates/`. Custom templates take priority over the
defaults shipped with the package.

## Hostname-to-Config Mapping

You can configure automatic config selection based on hostname in
`~/.config/auto_slurm/general_config.yaml`. This maps hostname regex patterns
to config names, so you can skip the `-cn` flag entirely:

```bash
# Without -cn, the config is selected based on hostname
aslurmx cmd python train.py
```

!!! tip
    Templates for other node types and HPC clusters can easily be
    added by adapting one of the existing configs. Feel free to submit
    new job templates via pull request.
