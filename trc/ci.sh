set -e
source "$(dirname "$0")/get_index.sh"
idle_gpu=$(get_gpu 4)
if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi
cuda_dev=$idle_gpu
CUDA_VISIBLE_DEVICES=$cuda_dev \
# Get list of changed files (added, modified, renamed, copied) excluding deleted
# Filter out files in trc/ directory
FILES=$(git diff --name-only  upstream/main  ':!trc')
echo $FILES
if [ -z "$FILES" ]; then
    echo "No changed files to check."
    exit 0
fi
# Run pre-commit on the files
pre-commit run --all-files
CUDA_VISIBLE_DEVICES=$cuda_dev \
python3 -m pytest --run-slow -s tests
