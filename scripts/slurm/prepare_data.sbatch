#!/bin/bash
# =============================================================================
# PLGrid (Athena) data preparation job: longitudinal pairing + MONAI Stage 1.
#
#   Single job (pairing + preprocessing):
#     sbatch scripts/slurm/prepare_data.sbatch
#
#   Pairing once, then preprocessing sharded over an array (recommended for full CT-RATE):
#     jid=$(sbatch --parsable --export=ALL,STAGES=pair scripts/slurm/prepare_data.sbatch)
#     sbatch --dependency=afterok:$jid --array=0-7 --export=ALL,STAGES=preprocess \
#            scripts/slurm/prepare_data.sbatch
#
#   Fill in --account (`hpc-grants` on the login node lists your grants), and create the
#   log directory before submitting: mkdir -p logs
#
# Preprocessing is CPU/IO-bound. If your grant has a CPU allocation, submit to a
# CPU partition (e.g. plgrid) without --gres to save GPU hours.
# =============================================================================
#SBATCH --job-name=ct-prepare
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=<your-grant>-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# ----------------------------------------------------------------------------- modules
# Versions differ between clusters; check with: module spider Python / module spider CUDA
module purge
module load GCCcore/12.3.0 Python/3.11.3 CUDA/12.1.1

# ----------------------------------------------------------------------------- paths
: "${SCRATCH:?SCRATCH is not set - are you on a PLGrid cluster?}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/vlm-token-reduction}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/config.yaml}"
STAGES="${STAGES:-pair preprocess}"
SPLITS="${SPLITS:-train val test}"

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$SCRATCH/.cache/torch"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
export CT_RATE_ROOT="${CT_RATE_ROOT:-$SCRATCH/data/CT-RATE}"
export PROJECT_OUTPUT="${PROJECT_OUTPUT:-$SCRATCH/vlm_token_reduction}"
# Compute nodes have no outbound internet: everything must be downloaded beforehand.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$PROJECT_OUTPUT"

# ----------------------------------------------------------------------------- env
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

echo "=== Job ${SLURM_JOB_ID:-local} (array task ${SLURM_ARRAY_TASK_ID:-none}) on $(hostname) ==="
echo "Python: $(which python) | stages: $STAGES | splits: $SPLITS"
echo "CT_RATE_ROOT=$CT_RATE_ROOT"
echo "PROJECT_OUTPUT=$PROJECT_OUTPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-4} - 1 ))

for stage in $STAGES; do
  case "$stage" in
    pair)
      echo ">>> [1/2] Building longitudinal pairs"
      python -m src.data.pair_longitudinal --config "$CONFIG"
      ;;
    preprocess)
      echo ">>> [2/2] MONAI Stage 1 preprocessing (splits: $SPLITS)"
      # shellcheck disable=SC2086
      python -m src.data.preprocess_monai --config "$CONFIG" \
        --splits $SPLITS \
        --num-workers "$NUM_WORKERS" \
        --shard-id "${SLURM_ARRAY_TASK_ID:-0}" \
        --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
      ;;
    *)
      echo "Unknown stage '$stage' (expected: pair, preprocess)" >&2
      exit 1
      ;;
  esac
done

echo "=== Finished at $(date) ==="
