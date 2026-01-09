import datetime
import errno
import logging
import os
import random
import time
from collections import defaultdict, deque
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

import neuron


def set_seed(seed: int = 3407):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# =============== Logging Utilities ===============

def setup_logger(output_dir: str) -> logging.Logger:
    """Configure and return a logger instance with both file and console handlers.
    
    Args:
        output_dir: Directory to save log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s]%(message)s',
        datefmt=r'%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.log'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    
    return logger

# =============== Metrics Utilities ===============

class SmoothedValue:
    """Track a series of values and provide access to smoothed statistics over a window."""
    
    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Synchronize statistics across distributed processes."""
        if not is_dist_avail_and_initialized():
            return
            
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self) -> float:
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self) -> float:
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self) -> float:
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        return self.deque[-1] if self.deque else 0.0

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )

class MetricLogger:
    """Track and log training metrics."""
    
    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def log_every(self, iterable, print_freq: int, header: Optional[str] = None):
        """Log training progress and metrics periodically.
        
        Args:
            iterable: Iterable object to process
            print_freq: Frequency of logging
            header: Optional header string for logs
        """
        i = 0
        header = header or ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        log_msg = self.delimiter.join([
            header,
            '[{current_iter' + space_fmt + '}/{total_iter}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}' if torch.cuda.is_available() else ''
        ]).strip()

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                log_args = {
                    'current_iter': i,
                    'total_iter': len(iterable),
                    'eta': eta_string,
                    'meters': str(self),
                    'time': str(iter_time),
                    'data': str(data_time)
                }
                
                if torch.cuda.is_available():
                    log_args['memory'] = torch.cuda.max_memory_allocated() / MB
                
                print(log_msg.format(**log_args))
                
            i += 1
            end = time.time()
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str}')

# =============== Evaluation Metrics ===============

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """Compute top-k accuracy.
    
    Args:
        output: Model output tensor
        target: Target tensor
        topk: Tuple of k values to compute accuracy for
        
    Returns:
        List[float]: List of accuracies for each k value
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        if len(target.shape) > 1:
            target = target.max(dim=1)[1]

        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

# =============== File System Utilities ===============

def mkdir(path: str) -> None:
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# =============== Distributed Training Utilities ===============

def setup_for_distributed(is_master: bool) -> None:
    """Configure printing behavior for distributed training."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    """Get the total number of distributed processes."""
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def get_rank() -> int:
    """Get the rank of current process."""
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0

def save_on_master(*args, **kwargs) -> None:
    """Save model only on the main process."""
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args) -> None:
    """Initialize distributed training mode.
    
    Args:
        args: Arguments object containing distributed training parameters
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    
    # 设置分布式训练超时和重试参数
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    
    try:
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(seconds=1800)  # 30分钟超时
        )
        setup_for_distributed(args.rank == 0)
        print(f'| distributed init (rank {args.rank}): SUCCESS')
    except Exception as e:
        print(f'| distributed init (rank {args.rank}): FAILED - {e}')
        args.distributed = False
        return


# =============== Parameter Splitting ===============

def split_params(model, paras=([], [], [])):
    for n, module in model._modules.items():
        if isinstance(module, neuron.PseudoNeuron):
            for name, para in module.named_parameters():
                if 'logit' in name:
                    paras[0].append(para)
                else:
                    paras[2].append(para)
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
        elif isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.modules.conv._ConvNd):
            paras[1].append(module.weight)
            if module.bias is not None:
                paras[2].append(module.bias)
        elif len(list(module.children())) > 0:
            paras = split_params(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
    return paras


# =============== Reset Network ===============

def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()
