function get_index() {
  users=(
    "sefi"
    "emir"
    "batya"
    "shira"
    "haim"
    "yoni"
    "david"
    "sagiv"
)
# Find index of user
index=-1
for i in "${!users[@]}"; do
    if [ "${users[$i]}" = "$USER" ]; then
        index=$i
        break
    fi
done
echo  $index
}

function get_gpu() {
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
mem_gpu=4
if [ "$1" != "" ]; then
    mem_gpu=$1
fi
idle_gpu=-1

# Loop through each GPU index
for (( i=0; i<$gpu_count; i++ ))
do
    # 1. Get detailed info for just this GPU
    # We capture the standard table output to see the "ERR!" flags
    gpu_status=$(nvidia-smi -i $i)

    # 2. Extract utilization for this GPU
    util=$(nvidia-smi -i $i --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    # 2. Extract memory usage for this GPU
    mem_used=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)

    # 3. CRITERIA CHECK:
    # - Is utilization 0?
    # - Is memory used < 2 MiB?
    # - Does the status NOT contain the word "ERR!"?
    if [ "$util" -eq 0 ] && [ "$mem_used" -lt $mem_gpu ] && ! echo "$gpu_status" | grep -q "ERR!"; then
        idle_gpu=$i
        break
    fi
done
echo $idle_gpu
}

# lock_gpu <gpu_id>
# Allocates a small amount of memory on the given GPU to "lock" it.
# This prevents other scripts using get_gpu from selecting it.
# It starts a background process and prints its PID.
# The caller is responsible for killing this PID to unlock the GPU.
# Requires python and pytorch to be installed.
function lock_gpu() {
    local gpu_id=$1
    if [ -z "$gpu_id" ] || [ "$gpu_id" -lt 0 ]; then
        echo "Invalid GPU ID: $gpu_id" >&2
        return 1
    fi
    # Allocate 4MB, which is > 4MB threshold in get_gpu, to mark it as busy.
    local mem_to_alloc_mb=4

    local python_cmd="
import torch, time, sys
try:
    _ = torch.empty(${mem_to_alloc_mb} * 1024 * 1024 // 4, dtype=torch.float32, device='cuda:0')
    time.sleep(31536000) # Sleep for a year, waiting to be killed.
except Exception as e:
    print(f'ERROR: Failed to lock GPU ${gpu_id}: {e}', file=sys.stderr)
    exit(1)
"
    # The python process will print to stderr on failure.
    CUDA_VISIBLE_DEVICES=$gpu_id python -c "$python_cmd" > /dev/null 2>&1 &
    local lock_pid=$!

    # Wait a moment to see if the python process started successfully.
    sleep 0.5
    if ps -p $lock_pid > /dev/null; then
        echo $lock_pid
    else
        echo "" # Return empty string on failure
        return 1
    fi
}

# unlock_gpu <pid>
# Kills the background process that is holding the GPU memory.
function unlock_gpu() {
    local pid=$1
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
}
