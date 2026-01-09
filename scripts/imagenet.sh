# /bin/bash

# --- 配置区 ---
# 1. 设置您希望使用的GPU卡号，用逗号隔开。例如，使用0、1、2、3号共4张卡。
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 2. 设置使用的GPU数量，需要与上面的卡号数量一致。
N_GPUS=4
MASTER_PORT=29502
# --- 配置区结束 ---

pids=()

cleanup() {
    echo "Interrupt signal captured, terminating all training processes..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            # kill的-PUP选项会杀掉该pid以及其所有子进程，torchrun会创建多个子进程
            kill -9 -$pid
        fi
    done
    wait
    echo "All training processes have been terminated."
    exit 1
}

trap cleanup SIGINT SIGTERM

echo "Starting distributed training on $N_GPUS GPUs..."

# 使用 torchrun 启动分布式训练
# --nproc_per_node 指定了在当前节点（服务器）上启动的进程数，通常等于GPU数量
torchrun --nproc_per_node=$N_GPUS --master_port $MASTER_PORT main.py --penalty 1e-2 --seed 0 &

pid=$!
pids+=("$pid")

wait

echo "All experiments completed."