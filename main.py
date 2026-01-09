import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
import os
import time
import yaml
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist
import torch.distributed.optim
import torch.nn.functional as F
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter

import models
import neuron
from utils.data import load_dataset
import utils.misc as utils


class Trainer:
    def __init__(self, args):
        self.args = args
        
        utils.init_distributed_mode(args)
        
        self.device = torch.device(args.device)
        self.max_test_acc1 = 0.
        self.test_acc5_at_max_test_acc1 = 0.
        self.train_tb_writer = None
        self.test_tb_writer = None

        self.output_dir = self._get_output_dir()
        if self.output_dir:
            utils.mkdir(self.output_dir)
        
        self.logger = utils.setup_logger(self.output_dir)

        if utils.is_main_process():
            self.logger.info(args)

        self._load_data()
        
        self._create_models()
        
        self._setup_optimizer_and_loss()
        
        self._setup_distributed()
        
        if args.resume:
            self._load_checkpoint(args.resume)
        
    def _get_output_dir(self) -> str:
        # Create descriptive path components
        training_config = f"{self.args.model}_b{self.args.batch_size}_e{self.args.epochs}_lr{self.args.lr}"

        if self.args.use_noise:
            model_config = f"{self.args.noise_type}_p{self.args.noise_prob}"
            model_config += f"_T{self.args.init_T}_s{self.args.scale}"
            model_config += f"_penalty{self.args.penalty}"
        else:
            model_config = f"T{self.args.init_T}"
        
        output_dir = os.path.join(self.args.output_dir, f"{self.args.data_type}", training_config, model_config, f"{args.seed}")
            
        return output_dir
        
    def _load_data(self):
        self.args.data_path = os.path.expanduser(self.args.data_path)
        if utils.is_main_process():
            self.logger.info(f"Loading data from {self.args.data_path}")

        dataset_train, dataset_test, self.num_classes, self.input_size = load_dataset(
            self.args.data_type.upper(),
            self.args.data_path,
            self.args.auto_aug,
            self.args.cutout,
        )
        if utils.is_main_process():
            self.logger.info(f'dataset_train: {len(dataset_train)}, \
                        dataset_test: {len(dataset_test)}')

        if self.args.distributed:
            self.train_sampler = distributed.DistributedSampler(dataset_train)
            self.test_sampler = distributed.DistributedSampler(dataset_test, shuffle=False)
            batch_size = self.args.batch_size // utils.get_world_size()
        else:
            self.train_sampler = None
            self.test_sampler = None
            batch_size = self.args.batch_size

        self.data_loader = DataLoader(
            dataset=dataset_train, 
            batch_size=batch_size, 
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            pin_memory=True, 
            num_workers=self.args.workers
        )
        self.data_loader_test = DataLoader(
            dataset=dataset_test, 
            batch_size=batch_size, 
            shuffle=False,
            sampler=self.test_sampler,
            pin_memory=True, 
            num_workers=self.args.workers
        )

    def _create_models(self):
        model_dict = models.noisy if self.args.use_noise else models.vanilla
        
        model_kwargs = {
            'num_classes': self.num_classes,
        }

        if self.args.use_noise:
            noise_kwargs = {
                'noise_type': self.args.noise_type,
                'noise_prob': self.args.noise_prob,
                'init_T': self.args.init_T,
                'min_T': self.args.min_T,
                'max_T': self.args.max_T, 
                'scale': self.args.scale
            }
            model_kwargs.update(noise_kwargs)
        else:
            model_kwargs.update({
                'T': self.args.init_T
            })

        model_found = False
        for model_type in ['preact_resnet', 'sew_resnet', 'resnet', 'vgg']:
            module = getattr(model_dict, model_type, None)
            if module and self.args.model in module.__dict__:
                self.model = module.__dict__[self.args.model](**model_kwargs)
                model_found = True
                break
                
        if not model_found:
            raise ValueError(f"Model not found: {self.args.model}")

        if utils.is_main_process():
            self.logger.info(f"Model architecture:\n{self.model}")

        if self.args.pretrained:
            if utils.is_main_process():
                self.logger.info(f"Loading pretrained model from {self.args.pretrained}")
            checkpoint = torch.load(self.args.pretrained, map_location='cpu', weights_only=False)
            new_state_dict = {}
            for k, v in checkpoint['model'].items():
                new_state_dict[k.replace('module.', '')] = v

            self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.to(self.device)

        if self.args.use_noise:
            self.model.register_calculator(self.input_size)
        
    def _setup_optimizer_and_loss(self):
        self.criterion = nn.CrossEntropyLoss()

        params = utils.split_params(self.model)
        params = [
            {'params': params[0], 'weight_decay': 0},
            {'params': params[1], 'weight_decay': self.args.weight_decay},
            {'params': params[2], 'weight_decay': 0},
        ]

        if self.args.adam:
            if utils.is_main_process():
                self.logger.info(f"Using Adam optimizer.")
            self.optimizer = torch.optim.AdamW(
                params, lr=self.args.lr
            )
        else:
            if utils.is_main_process():
                self.logger.info(f"Using SGD optimizer.")
            self.optimizer = torch.optim.SGD(
                params, lr=self.args.lr, momentum=self.args.momentum
            )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, eta_min=0, T_max=self.args.epochs)
        
        self.rate_scheduler = models.MutiStepNoisyRateScheduler(
            init_p=0.8, reduce_ratio=0.8, milestones=[0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
            num_epoch=self.args.epochs, start_epoch=self.args.start_epoch
        ) # TODO: check it
        
        self.scaler = amp.GradScaler() if self.args.amp else None
        
    def _setup_distributed(self):
        self.model_without_ddp = self.model
        if self.args.distributed:
            if self.args.sync_bn:
                if utils.is_main_process():
                    self.logger.info("Using SyncBatchNorm.")
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args.gpu], 
                find_unused_parameters=True,
            )
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module

    def _load_checkpoint(self, path):
        if utils.is_main_process():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.model_without_ddp.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            self.args.start_epoch = checkpoint['epoch']
            self.max_test_acc1 = checkpoint['max_test_acc1']
            self.test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

        if self.args.distributed:
            # Synchronize all processes
            dist.barrier()
            # Broadcast model parameters
            for param in self.model.parameters():
                dist.broadcast(param.data, 0)
            # Broadcast optimizer state
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        dist.broadcast(v, 0)
            # Broadcast other variables
            for name in ['start_epoch', 'max_test_acc1', 'test_acc5_at_max_test_acc1']:
                value = torch.tensor(getattr(self.args if name == 'start_epoch' else self, name), device=self.device)
                dist.broadcast(value, 0)
                if name == 'start_epoch':
                    self.args.start_epoch = value.item()
                else:
                    setattr(self, name, value.item())
        
    def train(self):
        if self.args.test_only:
            self.model.eval()
            test_metrics = self.evaluate()
            if utils.is_main_process():
                acc1 = test_metrics['acc1']
                acc5 = test_metrics['acc5']
                self.logger.info(f'max_test_acc1 {acc1:.4f}, test_acc5_at_max_test_acc1 {acc5:.4f}')
            return
            
        if self.args.tb and utils.is_main_process():
            self._setup_tensorboard()
            
        if utils.is_main_process():
            self.logger.info("Start training")
        start_time = time.time()
        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            save_max = False

            if self.args.data_type == 'imagenet' and self.args.use_noise:
                self.rate_scheduler(epoch, self.model)
            
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)
            
            self.model.train()
            train_metrics = self.train_one_epoch(epoch)
            
            self.model.eval()
            test_metrics = self.evaluate()

            if hasattr(self.model_without_ddp, 'verbose'):
                print(self.model_without_ddp.verbose())
            
            self.lr_scheduler.step()
            
            if utils.is_main_process():
                self._log_metrics(epoch, train_metrics, test_metrics)
                if test_metrics['acc1'] > self.max_test_acc1:
                    self.max_test_acc1 = test_metrics['acc1']
                    self.test_acc5_at_max_test_acc1 = test_metrics['acc5']
                    save_max = True
            
                if self.output_dir:
                    self._save_checkpoint(epoch, save_max)
            
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                
                self.logger.info(
                    f'Epoch {epoch}: Training time {total_time_str}, '
                    f'max_test_acc1 {self.max_test_acc1:.4f}, '
                    f'test_acc5_at_max_test_acc1 {self.test_acc5_at_max_test_acc1:.4f}'
                )
                
                if self.args.use_noise:
                    self.logger.info(f"Average timesteps: {self.model_without_ddp.calculator.calc_timesteps()}")

    def calibrate(self):
        if self.args.calib_epochs == 0:
            return
        
        # Use _load_checkpoint to handle distributed loading
        self._load_checkpoint(os.path.join(self.output_dir, 'checkpoint_max_test_acc1.pth'))

        # modified learning rate
        self.optimizer.param_groups[0]['lr'] = 0
        self.optimizer.param_groups[1]['lr'] = 1e-3
        self.optimizer.param_groups[2]['lr'] = 1e-3

        original_penalty = self.args.penalty
        self.args.penalty = 0.
        
        for epoch in range(self.args.epochs, self.args.epochs + self.args.calib_epochs):
            save_max = False
            
            # calib weight 
            self.model.train()
            for m in self.model.modules():
                if isinstance(m, neuron.PseudoNeuron):
                    m.eval()
            train_metrics = self.train_one_epoch(epoch)

            self.model.eval()
            test_metrics = self.evaluate()
            
            if utils.is_main_process():
                self._log_metrics(epoch, train_metrics, test_metrics)
                if test_metrics['acc1'] > self.max_test_acc1:
                    self.max_test_acc1 = test_metrics['acc1']
                    self.test_acc5_at_max_test_acc1 = test_metrics['acc5']
                    save_max = True
            
                if self.output_dir:
                    self._save_checkpoint(epoch, save_max)

                self.logger.info(
                    f'Epoch {epoch} (recalibrate): '
                    f'max_test_acc1 {self.max_test_acc1}, '
                    f'test_acc5_at_max_test_acc1 {self.test_acc5_at_max_test_acc1}'
                )

        self.args.penalty = original_penalty

    def _setup_tensorboard(self):
        purge_step = self.args.start_epoch
        self.train_tb_writer = SummaryWriter(self.output_dir + '/train', purge_step=purge_step)
        self.test_tb_writer = SummaryWriter(self.output_dir + '/test', purge_step=purge_step)
        with open(self.output_dir + '/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(self.args))
            
        if utils.is_main_process():
            self.logger.info(f'purge_step_train={purge_step}, purge_step_te={purge_step}')
        
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], test_metrics: Dict[str, float]):
        self.train_tb_writer.add_scalar('train_loss', train_metrics['loss'], epoch)
        self.train_tb_writer.add_scalar('train_acc1', train_metrics['acc1'], epoch)
        self.train_tb_writer.add_scalar('train_acc5', train_metrics['acc5'], epoch)
        
        self.test_tb_writer.add_scalar('test_loss', test_metrics['loss'], epoch)
        self.test_tb_writer.add_scalar('test_acc1', test_metrics['acc1'], epoch)
        self.test_tb_writer.add_scalar('test_acc5', test_metrics['acc5'], epoch)

        if self.args.use_noise:
            self.test_tb_writer.add_scalar('timesteps', test_metrics['timesteps'], epoch)
            self.test_tb_writer.add_scalar('flops', test_metrics['flops'], epoch)

        
    def _save_checkpoint(self, epoch: int, save_max: bool):
        checkpoint = {
            'model': self.model_without_ddp.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': self.args,
            'max_test_acc1': self.max_test_acc1,
            'test_acc5_at_max_test_acc1': self.test_acc5_at_max_test_acc1,
        }
        
        utils.save_on_master(
            checkpoint,
            os.path.join(self.output_dir, 'checkpoint_latest.pth'))
            
        if save_max:
            utils.save_on_master(
                checkpoint,
                os.path.join(self.output_dir, 'checkpoint_max_test_acc1.pth'))
                
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:        
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value:.2f}'))
        
        header = f'Epoch: [{epoch}]'
        
        for image, target in metric_logger.log_every(self.data_loader, self.args.print_freq, header):
            start_time = time.time()

            image, target = image.to(self.device), target.to(self.device)
            if self.args.data_type == 'cifar10_dvs_aug':
                image = image.permute(1, 0, 2, 3, 4)
            
            utils.reset_net(self.model)
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    if self.args.use_noise:
                        output, cost = self.model(image)
                        loss = self.criterion(output, target) + self.args.penalty * cost
                    else:
                        output = self.model(image)
                        loss = self.criterion(output, target)
            else:
                if self.args.use_noise:
                    output, cost = self.model(image)
                    loss = self.criterion(output, target) + self.args.penalty * cost
                else:
                    output = self.model(image)
                    loss = self.criterion(output, target)
            
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.args.clip_grad_norm > 0.:
                    self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.args.clip_grad_norm > 0.:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                self.optimizer.step()
            
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            
            metric_logger.meters['loss'].update(loss.item(), n=batch_size)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['lr'].update(self.optimizer.param_groups[0]["lr"])
            metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        
        metric_logger.synchronize_between_processes()
        
        return {
            'loss': metric_logger.loss.global_avg,
            'acc1': metric_logger.acc1.global_avg,
            'acc5': metric_logger.acc5.global_avg,
        }
        
    def evaluate(self) -> Dict[str, float]:
        metric_logger = utils.MetricLogger(delimiter="  ")
        
        with torch.no_grad():
            for image, target in metric_logger.log_every(self.data_loader_test, self.args.print_freq, header='Test:'):
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                if self.args.data_type == 'cifar10_dvs_aug':
                    image = image.permute(1, 0, 2, 3, 4)
                
                utils.reset_net(self.model)
                output = self.model(image)
                loss = self.criterion(output, target)
                
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                batch_size = image.shape[0]
                
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                
        metric_logger.synchronize_between_processes()

        result = {
            'loss': metric_logger.loss.global_avg,
            'acc1': metric_logger.acc1.global_avg,
            'acc5': metric_logger.acc5.global_avg
        }
        
        if self.args.use_noise:
            result.update({
                'timesteps': self.model_without_ddp.calculator.calc_timesteps(),
                'flops': self.model_without_ddp.calculator.calc_flops()
            })
            
        return result


def parse_args():
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    config_parser.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )
    
    parser = argparse.ArgumentParser(description='Training')
    
    # Dataset parameters
    dataset_group = parser.add_argument_group('Dataset parameters')
    dataset_group.add_argument('--data_type', default='ImageNet', help='Dataset type')
    dataset_group.add_argument('--data_path', default='~/datasets/ImageNet', help='Dataset path')
    dataset_group.add_argument('--cache_dataset', action="store_true", help="Cache the datasets for quicker initialization")
    
    # Model parameters
    model_group = parser.add_argument_group('Model parameters')
    model_group.add_argument('--model', default='sew_resnet19', help='Model architecture')
    model_group.add_argument('--pretrained', default='', type=str, help='Pretrained model path')
    
    # Neuron parameters
    model_group.add_argument('--use_noise', type=bool, default=True, help='Use noise-injected neuron')
    model_group.add_argument('--noise_type', default='uniform', type=str, choices=['uniform', 'gaussian'], help='Type of noise to add during training')
    model_group.add_argument('--noise_prob', default=0.5, type=float, help='Probability of adding noise during training')
    model_group.add_argument('--init_T', default=8.0, type=float, help='Initial timestep value')
    model_group.add_argument('--min_T', default=2, type=int, help='Minimum timestep value')
    model_group.add_argument('--max_T', default=16, type=int, help='Maximum timestep value')
    model_group.add_argument('--scale', default=1.0, type=float, help='Initial scale value')
    
    # Training parameters
    training_group = parser.add_argument_group('Training parameters')
    training_group.add_argument('--seed', default=3407, type=int, help='Random seed')
    training_group.add_argument('--device', default='cuda', help='Device to use')
    training_group.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    training_group.add_argument('--epochs', default=300, type=int, help='Number of total epochs to run')
    training_group.add_argument('--calib_epochs', default=0, type=int, help='Number of epochs to calibrate BN stats')
    training_group.add_argument('--workers', default=16, type=int, help='Number of data loading workers')
    training_group.add_argument('--print_freq', default=10, type=int, help='Print frequency')
    training_group.add_argument('--test_only', action="store_true", help="Only test the model")
    training_group.add_argument('--amp', default=True, type=bool, help='Use mixed precision training')
    
    # Optimization parameters
    optim_group = parser.add_argument_group('Optimization parameters')
    optim_group.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')

    optim_group.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay factor')
    optim_group.add_argument('--lr_steps', nargs='+', type=int, default=[100, 150], help='Epoch milestones for LR decay')
    
    optim_group.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    optim_group.add_argument('--weight_decay', default=0, type=float, help='Weight decay')
    optim_group.add_argument('--adam', action='store_true', help='Use Adam optimizer instead of SGD')
    optim_group.add_argument('--penalty', type=float, default=1e-2, help='Penalty on the calculation burden')
    optim_group.add_argument('--clip_grad_norm', type=float, default=0, help='Gradient clipping norm value')
    
    # Data augmentation parameters
    aug_group = parser.add_argument_group('Data augmentation parameters')
    aug_group.add_argument('--auto_aug', type=bool, default=True, help='Use auto augmentation')
    aug_group.add_argument('--cutout', type=bool, default=True, help='Use cutout')
    
    # Distributed training parameters
    dist_group = parser.add_argument_group('Distributed training parameters')
    dist_group.add_argument('--world_size', default=1, type=int,  help='Number of distributed processes')
    dist_group.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    dist_group.add_argument('--sync_bn', action="store_true", help="Use sync batch norm")
    
    # Output parameters
    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--output_dir', default='.', help='Path where to save')
    output_group.add_argument('--resume', default='', help='Resume from checkpoint')
    output_group.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Start epoch')
    output_group.add_argument('--tb', default=True, type=bool, help='Use TensorBoard to record logs')
                        
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    utils.set_seed(args.seed)
    
    trainer = Trainer(args)

    trainer.train()
    if not args.test_only:
        trainer.calibrate()

    exit()
    