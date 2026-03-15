source "$(dirname "$0")/get_index.sh"
idle_gpu=$(get_gpu 4)
if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi
cuda_dev=$idle_gpu
echo "running on device $cuda_dev"
CUDA_VISIBLE_DEVICES=$cuda_dev \
pre-commit run --all-files
#pytest -s tests/test_serving_catalog.py --loops 1 --run-slow
 
