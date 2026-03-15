function get_index() {
  users=(
    "sefi"
    "emir"
    "batya"
    "shira"
    "haim"
    "yoni"
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