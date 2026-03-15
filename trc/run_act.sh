source "$(dirname "$0")/get_index.sh"
idle_gpu=$(get_gpu 4)
if [ -z "$idle_gpu" ]; then
    echo "No GPUs found that are 0% utilized, using < 4MiB memory, and error-free."
    exit 1
fi
cuda_dev=$idle_gpu
echo "running on device $cuda_dev"
act --container-options "--runtime=nvidia --gpus device=$cuda_dev" --artifact-server-path /home/yoni/xGr/artifacts \
-P self-hosted=ghcr.io/catthehacker/ubuntu:act-latest
