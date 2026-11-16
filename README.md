# EHRDiB

Code for **Evidence-Conditioned Completion and Synthesis of Electronic Health Records**.

EHRDiB combines a shared variational representation of heterogeneous EHR features with an evidence-conditioned diffusion bridge in latent space. It supports record completion and synthesis, Bernoulli decoding, calibrated threshold decoding, and evaluation of fidelity, diversity, predictive utility, and disclosure risk.

## Workflow and repository scope

The model workflow starts from visit-level PyTorch exports (`.pt`):

```text
Visit export (.pt)
  -> model input conversion and caching
  -> VAE training
  -> latent diffusion bridge training
  -> sampling, calibration, and decoding
  -> evaluation
```

Raw-data preprocessing is maintained in the separate `mimic_data` and `duke_data` projects. The converters in `datasets/` retain the exported feature definitions, fitted transformations, and patient partitions.

Supported data views:

| Dataset | Export | Representation |
| --- | --- | --- |
| MIMIC-IV | `mimic.pt` | Diagnosis, procedure, medication, and categorical/numerical clinical measurements |
| Duke Health (DH) | `duke_continuous.pt` | Diagnosis, medication, and numerical clinical measurements |
| Duke Health Discrete (DH-D) | `duke_binned.pt` | The same Duke Health visits with clinical measurements discretized using binning |

The repository contains model code, dataset converters, evaluation code, experiment configurations, and Slurm scripts. Visit exports, trained weights, and patient records are supplied separately.

## Installation

Use Python 3.10. The supplied Conda environment targets Linux with an NVIDIA GPU and CUDA 12.1:

```bash
conda env create -f environment.yaml
conda activate ehrdib
ehrdib --help
```

For an existing Python 3.10 environment, install the package and its dependencies with:

```bash
python -m pip install -e .
ehrdib --help
```

The CLI supports `--device cpu` and `--device cuda:0`. Select a PyTorch installation appropriate for the intended device. `python -m ehrdib` and `python main.py` also expose the workflow commands.

## Configure the data

Edit `paths.data_dir` in the selected YAML configuration before running an experiment. The following are path fragments to update within the existing configuration files.

For MIMIC-IV:

```yaml
paths:
  data_dir: /path/to/mimic_data/{seed}/mimic.pt
```

For Duke Health, point to the directory containing both exports:

```yaml
paths:
  data_dir: /path/to/duke_data
```

Select `--representation num` for DH or `--representation cat` for DH-D. A Duke configuration can also point directly to the corresponding export file.

Relative paths are resolved from the YAML file's directory. Paths support environment-variable expansion and the `{seed}` placeholder. For MIMIC-IV, the experiment seed must match the selected upstream export. Duke experiments retain the patient partition saved in the selected export.

MIMIC-IV access requires the appropriate PhysioNet credentials and data use agreement. Duke Health records contain protected health information and are not distributed with this repository.

### Evaluation data dependencies

MIMIC-IV predictive-utility evaluation constructs downstream labels using the original hospital diagnosis tables and the Phecode mapping file. The export's `metadata.evaluation_paths` must identify these resources through `data_dir` and `phecode_dir`. Make these paths available on the evaluation machine. Model training and generation use the exported tensors.

## Run an experiment

Check configuration, paths, device availability, and any existing prepared artifacts:

```bash
ehrdib check --config configs/mimic/completion.yaml --device cpu
ehrdib check --config configs/duke/completion.yaml --representation num --device cpu
```

The check command does not train a model. A full run prepares or reuses the model-input cache, trains both model stages, generates records, and evaluates the configured decoders.

### MIMIC-IV

```bash
ehrdib run --config configs/mimic/completion.yaml --device cuda:0
ehrdib run --config configs/mimic/synthesis.yaml --device cuda:0
```

### Duke Health

```bash
# DH: numerical clinical measurements
ehrdib run --config configs/duke/completion.yaml --representation num --device cuda:0
ehrdib run --config configs/duke/synthesis.yaml --representation num --device cuda:0

# DH-D: binned clinical measurements
ehrdib run --config configs/duke/completion.yaml --representation cat --device cuda:0
ehrdib run --config configs/duke/synthesis.yaml --representation cat --device cuda:0
```

Use `--device cpu` for CPU execution. Training sizes and update counts are configured in the YAML files. Use `--resume` to continue an existing run with compatible settings. `--seed` selects an experiment seed; it does not rebuild raw data or create a new upstream patient partition.

### Run stages separately

For a new experiment, the following commands perform the stages individually:

```bash
ehrdib prepare --config configs/mimic/synthesis.yaml
ehrdib train --config configs/mimic/synthesis.yaml --device cuda:0
ehrdib calibrate --config configs/mimic/synthesis.yaml --device cuda:0
ehrdib sample --config configs/mimic/synthesis.yaml --device cuda:0
ehrdib evaluate --config configs/mimic/synthesis.yaml
```

For Duke configurations, also supply `--representation num` or `--representation cat`. `decoding.methods` selects the output decoders; synthesis configurations include both `bernoulli` and `calibrated_threshold`. Run `ehrdib <command> --help` for the available overrides.

### Generate from supplied evidence

After training, generate records from a partial evidence CSV compatible with the saved feature schema:

```bash
ehrdib generate --config configs/mimic/synthesis.yaml \
  --evidence /path/to/evidence.csv \
  --output /path/to/generated.csv \
  --device cuda:0
```

Each evidence row must provide the designated condition required by the saved schema. Use `ehrdib generate --help` for decoder, seed, and reference-data options.

## Outputs

Default outputs are stored relative to the project root:

| Path | Contents |
| --- | --- |
| `data/<dataset>/<seed>/` | Converted model-input caches |
| `results/<name>/<seed>/` | Saved run configuration, checkpoints, and generation artifacts |
| `reports/<name>/<seed>/eval_report.json` | Fidelity, diversity, utility, and privacy metrics |
| `logs/` | Workflow logs |

Duke run names receive a `_num` or `_cat` suffix when the representation is selected. Configurations under `configs/<dataset>/exp/` add the configuration filename to the run and report paths. Custom output paths can be configured in the YAML files.

`report_template.json` illustrates the report structure; `log_template.txt` illustrates workflow logging.

## Slurm execution

The scripts under `scripts/mimic/` and `scripts/duke/` are cluster-specific examples. Update their project, Conda, resource, and log paths for the target cluster, and create the configured log directory before submission.

```bash
mkdir -p logs
sbatch scripts/mimic/job.sh configs/mimic/synthesis.yaml
sbatch scripts/duke/job.sh configs/duke/synthesis.yaml
```

The parallel submission scripts select the three seed configurations for a task:

```bash
bash scripts/mimic/parallel_job.sh synthesis
bash scripts/duke/parallel_job.sh synthesis
```

Pass `completion` to submit completion experiments. The current scripts default to completion when no task is supplied.

## Repository layout

```text
configs/          Dataset and task configurations, including seed variants
datasets/         MIMIC-IV and Duke Health conversion and evaluation modules
ehrdib/           VAE, diffusion bridge, training, generation, and shared utilities
scripts/          Cluster job and parallel submission scripts
main.py           Source-tree workflow entry point
environment.yaml  Linux/CUDA Conda environment
pyproject.toml    Package metadata, dependencies, and CLI entry point
```

The current configuration set covers the main completion/synthesis runs and seed variants. Evidence-fraction sweeps, ablation configurations, and manuscript figure-generation scripts are maintained separately and will be organized in subsequent updates.
