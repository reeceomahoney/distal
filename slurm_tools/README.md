# slurm-tools

CLI for submitting SLURM jobs via SSH and a web GUI for monitoring them.

## Install

```bash
uv pip install git+https://github.com/reeceomahoney/slurm-tools.git
```

## Configuration

Place a `configs/slurm.yaml` in your working directory:

```yaml
host: my-cluster
remote_path: /data/user/project
command: >-
  singularity run --nv container.sif make train
time: 6
gpu: h100
ngpu: 1
cpus: 16
mem: 8G
```

If no config file exists, all options fall back to dataclass defaults and can be set entirely via CLI flags.

## Usage

### Submit a job

```bash
slurm run                          # uses configs/slurm.yaml
slurm run --gpu l40s --time 3      # override specific fields
slurm run --command "make eval"    # override command
slurm run --dry_run true           # print the sbatch script without submitting
```

This rsyncs the project to the remote host (respecting `.gitignore`), then submits via `sbatch`.

### Web GUI

```bash
slurm gui           # start the monitoring server on localhost:5000
slurm gui stop      # stop it
```

The GUI shows GPU availability across nodes, running/completed jobs, log streaming, and supports cancelling jobs.

## Config reference

| Field         | Default | Description                        |
| ------------- | ------- | ---------------------------------- |
| `host`        | **required** | SSH host alias for the cluster     |
| `remote_path` | **required** | Absolute path on the remote host   |
| `command`     | **required** | Shell command to run in the job    |
| `time`        | `6`     | Job time limit in hours            |
| `gpu`         | `h100`  | GPU type (e.g. h100, l40s)        |
| `ngpu`        | `1`     | Number of GPUs                     |
| `cpus`        | `16`    | CPUs per node                      |
| `mem`         | `8G`    | Memory per CPU                     |
| `dry_run`     | `false` | Print sbatch script without submit |
