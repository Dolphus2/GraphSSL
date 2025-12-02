# GraphSSL - HPC Setup Guide

This guide covers setting up and running GraphSSL on the DTU High Performance Computing (HPC) cluster.

## Prerequisites

Begin by following this guide to log on to the DTU HPC: https://www.hpc.dtu.dk/?page_id=4317

### Check Available Space

Before starting, ensure you have enough storage space:

```bash
# Check your home directory quota
getquota_zhome.sh

# View detailed directory sizes
du --apparent -d 2 -h .

# If low on space, switch to tempory scratch storage
cd $BLACKHOLE
# or concider writing an email to hpc support requesting more space
```

## Environment Setup

### 1. Clone Repository

Clone the repository using SSH with agent forwarding:

```bash
git clone git@github.com:Dolphus2/GraphSSL.git
cd GraphSSL
```

### 2. Load Required Modules

Load Python and CUDA modules:

```bash
module load python3/3.12.9
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X
```

**Tip:** You can check available module versions with:
```bash
module avail python3
module avail cuda
```

To switch between module versions:
```bash
module swap module_name/version
```

### 3. Create Virtual Environment

#### Option A: Using venv (Standard)

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

#### Option B: Using uv (Faster Alternative)

If you have `uv` installed, you can create a virtual environment more quickly:

```bash
# Install uv if not already installed. Check website for installation details.
<https://docs.astral.sh/uv/getting-started/installation/>

# Create virtual environment with uv
uv venv .venv
# The python version is specified by the .python-version file.

# To manually specify a Python version: 
uv venv .venv --python python3.12
```

**Note:** `uv` is a fast Python package installer and resolver. It's compatible with standard virtual environments and can significantly speed up package installation.

### 4. Auto-load Modules (Recommended)

Add the module load commands to your activation script so they load automatically:

```bash
# Edit the activate script
nano .venv/bin/activate
```

Add these lines at the **top** of the file (after the shebang):

```bash
module load python3/3.12.9
module load cuda/12.0
module load cudnn/v8.3.2.44-prod-cuda-11.X
```

This ensures the correct modules are loaded whenever you activate the environment.

### 5. Activate Environment

```bash
source .venv/bin/activate
```

You should see `(.venv)` before your prompt:
```
(.venv) gbarlogin1(s123456)$
```

### 6. Install Dependencies

#### Using pip (Standard)

Install from requirements.txt to correctly register --find-links:

```bash
python -m pip install -r requirements.txt
```

Then install the package in editable mode:

```bash
python -m pip install -e .
```

Or 

#### Using uv (Faster)

If you created your environment with `uv`:

As with standard pip

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Then install in editable mode
uv pip install -e .
```

Normally the option below would work as well, but not in this case since pyg-lib is not
in the package registry and needs to be install using --find-links. 

```bash
# Install automatically from pyproject.toml
uv sync
```


**Important:** Do NOT use the `--user` flag inside a virtual environment.

### Troubleshooting: No Space Left

If you encounter:
```
Building wheels failed: [Errno 28] No space left on device
```

Clear the pip cache and retry:

```bash
pip cache purge
python -m pip install -r requirements.txt
```

## Running Jobs

### Interactive Nodes (Debugging Only)

**WARNING:** Do NOT run code on the login node! It slows down jobs for all users.

For debugging, use interactive nodes:

```bash
# Available interactive nodes:
# - a100
# - sxm2  
# - volta

# Connect to a volta node
voltash
```

Once on an interactive node:

```bash
# Check GPU usage and available GPUs
nvidia-smi

# Select a specific GPU (e.g., GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Run your code
python -m graphssl.main --epochs 10
```

**Note:** Interactive nodes have limited resources. Use them for debugging only, not for training.

### Submit Batch Jobs (Recommended)

For actual training, submit batch jobs using LSF:

```bash
# Submit a job
bsub < src/graphssl/run_hpc.sh

# Check job status
bstat -u s123456

# View job output
tail -f logs/graphssl_<JOBID>.out
tail -f logs/graphssl_<JOBID>.err
```

Edit `src/graphssl/run_hpc.sh` to customize:
- Memory requirements (`-R "rusage[mem=32GB]"`)
- GPU type (`-q gpuv100`)
- Time limit (`-W 24:00`)
- Training parameters

Example job script structure:
```bash
#!/bin/bash
#BSUB -J graphssl_train
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=32GB]"
#BSUB -W 24:00

cd /path/to/GraphSSL
source .venv/bin/activate

python -m graphssl.main \
    --hidden_channels 128 \
    --epochs 100 \
    --batch_size 1024
```

## Useful HPC Commands

### Job Management
```bash
# Submit job
bsub < script.sh

# Check job status
bstat -u s123456

# Kill a job
bkill <job_id>

# View job details
bjobs -l <job_id>
```

### Storage Management
```bash
# Check quota
getquota_zhome.sh

# Check directory size
du -sh /path/to/directory

# Detailed size breakdown
du --apparent -d 2 -h
```

### Module Management
```bash
# List available modules
module avail

# Load a module
module load module_name/version

# Unload a module
module unload module_name

# List loaded modules
module list

# Switch module versions
module swap module_name/old_version module_name/new_version
```

## Virtual Environment Best Practices

1. **Always activate before use:**
   ```bash
   source .venv/bin/activate
   ```

2. **Never use `--user` flag** inside virtual environments

3. **Deactivate when done:**
   ```bash
   deactivate
   ```

4. **Keep environments isolated:** Install all dependencies within the environment, not as system modules

5. **Add module loads to activate script** for convenience (see step 4 above)

## Testing Your Setup

Quick test to verify everything works:

```bash
# Activate environment
source .venv/bin/activate

# Test imports
python -m graphssl.test_pipeline

# Quick training run (2 epochs)
graphssl --epochs 2 --batch_size 256
```

## Support

For HPC-specific issues, consult:
- DTU HPC Documentation: https://www.hpc.dtu.dk/
- HPC Support: hpc@dtu.dk

For GraphSSL issues:
- See main [README.md](README.md)
- See [QUICKSTART.md](QUICKSTART.md)
