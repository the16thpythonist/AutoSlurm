# AutoSlurmX

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-0.1.0-green)

**AutoSlurmX** automatically generates SLURM job scripts based on reusable
Jinja2 templates and submits them for you. This includes support for:

- **Multi-task multi-GPU jobs** with automatic GPU assignment
- **Infinite chain jobs** that automatically resume when time limits are reached
- **Hyperparameter sweeps** with paired list and grid search syntax
- **Python API** for programmatic job submission

The available default templates focus on HPC clusters available at the
Karlsruhe Institute of Technology and beyond, but creating templates
for other HPC clusters is straightforward.

!!! note
    If things do not work as expected, if you have questions, or if you
    have ideas for new features, please add an issue to the repository!

## Quick Example

```bash
# Single task on one GPU
aslurmx -cn haicore_1gpu cmd python train.py

# Multi-task: 4 tasks across 4 GPUs
aslurmx -cn horeka_4gpu \
   cmd python train.py --config conf0.yaml \
   cmd python train.py --config conf1.yaml \
   cmd python train.py --config conf2.yaml \
   cmd python train.py --config conf3.yaml

# Hyperparameter sweep (grid search)
aslurmx -cn horeka_4gpu cmd python train.py \
   lr='<{1e-3,1e-4,1e-5}>' batch_size='<{512,256}>'
```
