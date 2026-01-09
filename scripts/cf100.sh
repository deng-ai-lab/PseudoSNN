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


CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/noisy/cf100/r18.yaml --penalty 1e-2 &

pid=$!
pids+=("$pid")


CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/noisy/cf100/vggsnn.yaml --penalty 1e-2 &

pid=$!
pids+=("$pid")


CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/noisy/cf100/r19.yaml --penalty 1e-2 &

pid=$!
pids+=("$pid")


CUDA_VISIBLE_DEVICES=0 python3 main.py --config configs/noisy/cf100/sew_r19.yaml --penalty 1e-2 &

pid=$!
pids+=("$pid")


wait

echo "All experiments completed."
