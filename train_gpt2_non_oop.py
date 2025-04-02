from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import tiktoken
from tqdm import tqdm 
import time
import os
import inspect
import psutil
import wandb
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import hydra
import logging
from omegaconf import OmegaConf
import argparse

from dataloader_basic import FW_EduDataloader
from hellaswag import render_example, iterate_examples

def print_module_params(model, trainable_only=False, include_non_parameterized_modules=False):
    total_params = 0
    for name, module in model.named_modules():
        params_in_module = 0
        if trainable_only:
            for param in module.parameters():
                params_in_module += param.numel()
        else:
            for buffer in module.buffers(recurse=False):
                params_in_module += buffer.numel()
            for param in module.parameters(recurse=False):
                params_in_module += param.numel()
        if params_in_module > 0 or include_non_parameterized_modules:
            print(f"Module: {name}, Params: {params_in_module}")
            total_params += params_in_module
    print(f"Total Parameters: {total_params:,}")

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()

def get_lr(it, warmup_steps, max_steps, min_lr_ratio, max_lr):
    min_lr = max_lr * 0.1
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def configure_optimizers(model, weight_decay, lr, device):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_param = [p for n, p in param_dict.items() if p.dim() >= 2]
    no_decay_param = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_param, 'weight_decay': weight_decay},
        {'params': no_decay_param, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused='cuda' in device)
    return optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GPT-2 Pre-training")
    parser.add_argument("--config_path", help="Path to the configuration file", type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    print(OmegaConf.to_yaml(config))
    device = config.device

    logging.basicConfig(level=logging.INFO)
    logger_train = logging.getLogger("Train")
    logger_eval = logging.getLogger("Eval")
    logger_ddp = logging.getLogger("DDP")
    logger_optim = logging.getLogger("Optimizer")

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "CUDA required for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        logger_ddp.info(f"Rank {ddp_rank} / {ddp_world_size} on {device}")
    else:
        ddp_rank = 0
        ddp_world_size = 1
        master_process = True

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    torch.set_float32_matmul_precision('high')

    if master_process:
        wandb.login(key=config.wandb_key)
        run = wandb.init(project=config.wandb_project, name=config.wandb_name)
        text_table = wandb.Table(columns=["Generated Text"])

    train_loader = FW_EduDataloader(data_root=config.data.data_root,
                                    batch_size=config.data.train_batch_size,
                                    seq_length=config.data.sequence_length,
                                    process_rank=ddp_rank,
                                    num_process=ddp_world_size,
                                    split=config.data.train_split,
                                    shard_on_ram=config.data.train_shard_on_ram,
                                    master_process=master_process)

    val_loader = FW_EduDataloader(data_root=config.data.data_root,
                                  batch_size=config.data.val_batch_size,
                                  seq_length=config.data.sequence_length,
                                  process_rank=ddp_rank,
                                  num_process=ddp_world_size,
                                  split=config.data.val_split,
                                  shard_on_ram=config.data.val_shard_on_ram,
                                  master_process=master_process)

    model = hydra.utils.instantiate(config.model).to(device)

    if config.training.use_compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model

    print_module_params(raw_model)

    del raw_model

    optimizer = configure_optimizers(
        model=model,
        weight_decay=config.training.weight_decay,
        lr=config.training.max_lr,
        device=device
    )

    # --- Training Preparation ---
    B = config.data.train_batch_size
    T = config.data.sequence_length
    assert config.training.total_batch_size % (B*T*ddp_world_size) == 0
    grad_accum_steps = config.training.total_batch_size // (B*T*ddp_world_size)

    log_dir = os.path.join(config.log_dir, config.wandb_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_{config.wandb_name}.txt')
    result_file = os.path.join(log_dir, f'result_{config.wandb_name}.txt')
    eval_file = os.path.join(log_dir, f'eval_{config.wandb_name}.txt')
    for f in [log_file, result_file, eval_file]:
        open(f, 'w').close()

    for step in range(config.training.max_steps):
        t0 = time.time()
        last_step = (step == config.training.max_steps - 1)

        if step % config.training.eval_frequency == 0 or last_step:
            model.eval()
            val_loader.reset()
            val_loss_accum = 0.0
            with torch.no_grad():
                for _ in range(config.training.val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        _, val_loss = model(x, y)
                    val_loss_accum += val_loss.detach() / config.training.val_loss_steps

            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                logger_eval.info(f"Validation loss: {val_loss_accum.item():.4f}")
                run.log({"validation loss": val_loss_accum.item()}, step=step)
                with open(log_file, 'a') as f:
                    f.write(f"val,{step},{val_loss_accum.item()}\n")
                if step > 0 and (step % config.training.save_checkpoint_frequency == 0 or last_step):
                    checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
                    torch.save({
                        'model': model.state_dict(),
                        'config': model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }, checkpoint_path)
                    logger_eval.info(f"Checkpoint saved to {checkpoint_path}")

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in tqdm(range(grad_accum_steps), desc=f"Step {step}"):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        lr = get_lr(step, config.training.warmup_steps, config.training.max_steps,
                    config.training.min_lr_ratio, config.training.max_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tok_per_sec = tokens_processed / (t1 - t0)

        if master_process:
            logger_train.info(f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | tok/sec: {tok_per_sec:.2f} | dt: {dt:.2f} ms")
            run.log({"loss": loss_accum.item(), "lr": lr, "norm": norm, "tok_per_sec": tok_per_sec}, step=step)
            with open(log_file, 'a') as f:
                f.write(f"train,{step},{loss_accum.item()},{lr},{norm},{tok_per_sec},{dt}\n")

    if ddp:
        destroy_process_group()
