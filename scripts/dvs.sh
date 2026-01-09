# /bin/bash

pids=()

cleanup() {
    echo "Interrupt signal captured, terminating all training processes..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
    done
    wait
    echo "All training processes have been terminated."
    exit 1
}

trap cleanup SIGINT SIGTERM


CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/noisy/cf10_dvs/vggsnn.yaml --penalty 5e-4 &


pid=$!
pids+=("$pid")

wait

echo "All experiments completed."
