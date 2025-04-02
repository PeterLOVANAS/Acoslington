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

from dataloader_basic import FW_EduDataloader
from hellaswag import render_example, iterate_examples


def print_module_params(model, trainable_only=False, include_non_parameterized_modules=False):
    """
    Prints the name and parameter count for each module in a PyTorch model.
    """
    total_params = 0
    print(f"Model Module Parameter Counts (Trainable Only: {trainable_only}, Include Non-Parameterized Modules: {include_non_parameterized_modules}):")
    for name, module in model.named_modules():
        params_in_module = 0
        if trainable_only:
            for param in module.parameters(): # Iterates only over trainable parameters if trainable_only=True
                params_in_module += param.numel()
        else:
            for buffer in module.buffers(recurse=False): # Count buffers (non-trainable parameters)
                params_in_module += buffer.numel()
            for param in module.parameters(recurse=False): # Count trainable parameters
                params_in_module += param.numel()


        if params_in_module > 0 or include_non_parameterized_modules: # Print if module has parameters or if explicitly requested
            trainable_str = " (trainable)" if trainable_only else ""
            print(f"Module: {name}, Params: {params_in_module}{trainable_str}")
            total_params += params_in_module

    trainable_note = " (trainable only)" if trainable_only else ""
    print(f"\nTotal Parameters{trainable_note}: {total_params:,}")

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def get_lr(it, warmup_steps, max_steps, min_lr_ratio, max_lr):
    min_lr = max_lr * 0.1
    # Linear warmup
    if it < warmup_steps:
      return max_lr * (it + 1) / warmup_steps # +1 to times to zero

    if it > max_steps:
      return min_lr

    # IN between those steps, use cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



class Trainer():
  def __init__(self, config):
      self.config = config
      self.device = self.config.device

      logging.basicConfig(level=logging.INFO)
      self.logger_train = logging.getLogger("Train")
      self.logger_eval = logging.getLogger("Eval")
      self.logger_ddp = logging.getLogger("DDP")
      self.logger_optim = logging.getLogger("Optimizer")

  def configure_optimizers(self, model, weight_decay, lr, device):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # params in 2d or more will be decayed by weight_decay, otherwise no decay
    decay_param = [p for n, p in param_dict.items() if p.dim() >= 2]
    no_decay_param = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
      {'params' : decay_param, 'weight_decay' : weight_decay},
      {'params' : no_decay_param, 'weight_decay' : 0.0}
    ]

    num_decay_params = sum(p.numel() for p in decay_param) # p.numel() means number of elements in the tensor
    num_nodecay_params = sum(p.numel() for p in no_decay_param)
    self.logger_optim.info(f"num decayed param tensor = {len(decay_param)} | num nodecay param tensor = {len(no_decay_param)}")
    self.logger_optim.info(f"num decayed params = {num_decay_params} | num nodecay params = {num_nodecay_params}")

    # Create optimizer
    fused_avail = 'fused' in inspect.signature(torch.optim.AdamW).parameters # parameter of the signature of the function (check if AdamW has fused parameter)
    use_fused = fused_avail and 'cuda' in device # if fused is available and device is cuda
    self.logger_optim.info(f"Using fused optimizer: {use_fused}")
    optimizer = torch.optim.AdamW(optim_groups, lr = lr , betas=(0.9, 0.95), eps=1e-8, fused = use_fused)
    return optimizer

  def train(self):

      # --- DDP ---
      ddp = int(os.environ.get('RANK', -1)) != -1
      if ddp:
        assert torch.cuda.is_available(), "CUDA required for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # for logging and checkpoint saving (master process = cuda:0 if have multiple GPUs)
        self.logger_ddp.info(f"Rank {ddp_rank} / {ddp_world_size} on {device}")
      else:
        master_process = True

      torch.manual_seed(self.config.seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed(self.config.seed)
      torch.set_float32_matmul_precision('high')


      # --- Wandb ---
      if master_process:
          wandb.login(key=self.config.wandb_key)
          run = wandb.init(project=self.config.wandb_project, name=self.config.wandb_name)
          text_table = wandb.Table(columns=["Generated Text"])


      # --- Data ---
      train_loader = FW_EduDataloader(data_root=self.config.data.data_root,
                                      batch_size=self.config.data.train_batch_size,
                                      seq_length=self.config.data.sequence_length, 
                                      process_rank=ddp_rank,  
                                      num_process=ddp_world_size, 
                                      split=self.config.data.train_split, 
                                      shard_on_ram=self.config.data.train_shard_on_ram,
                                      master_process=master_process)

      val_loader = FW_EduDataloader(data_root=self.config.data.data_root,
                                    batch_size=self.config.data.val_batch_size,
                                    seq_length=self.config.data.sequence_length, 
                                    process_rank=ddp_rank,  
                                    num_process=ddp_world_size, 
                                    split=self.config.data.val_split, 
                                    shard_on_ram=self.config.data.val_shard_on_ram,
                                    master_process=master_process)

      # --- Model ---
      model = hydra.utils.instantiate(self.config.model).to(self.device)

      if self.config.training.use_compile:
        t0 = time.time()
        model = torch.compile(model)
        t1 = time.time()
        self.logger_train.info(f"Compile time: {t1-t0:.2f} s")

      if ddp:
        t0 = time.time()
        model = DDP(model, device_ids = [ddp_local_rank])
        t1 = time.time()
        self.logger_ddp.info(f"DDP wrapping time: {t1-t0:.2f} s")

      raw_model = model.module if ddp else model # get the raw model (without DDP wrapper)
      
      print_module_params(raw_model)

      # --- Optimizer ---
      optimizer = self.configure_optimizers(
        model = model,
        weight_decay=self.config.training.weight_decay,
        lr=self.config.training.max_lr,
        device=self.device
      )

      # --- Checkpointing, logging ---
      log_dir = os.path.join(self.config.log_dir, self.config.wandb_name)
      os.makedirs(log_dir, exist_ok=True)
      checkpoint_dir = os.path.join(log_dir, "checkpoints")
      os.makedirs(checkpoint_dir, exist_ok=True)

      log_file = os.path.join(log_dir, f'log_{self.config.wandb_name}.txt')
      result_file = os.path.join(log_dir, f'result_{self.config.wandb_name}.txt')
      eval_file = os.path.join(log_dir, f'eval_{self.config.wandb_name}.txt')
      with open(log_file, 'w') as f:
        pass

      with open(result_file, 'w') as f:
        pass

      with open(eval_file, 'w') as f:
        pass


      # --- Training ---
      B = self.config.data.train_batch_size
      T = self.config.data.sequence_length
    

      # --- Calculate Gradient accumulation steps---
      assert self.config.training.total_batch_size % (B*T*ddp_world_size) == 0, "Batch size must be divisible by B*T*ddp_world_size or all tokens that REALLY get processed in one iteration of all GPUs existing at once"
      grad_accum_steps = self.config.training.total_batch_size // (B*T * ddp_world_size)  # e.g. 32 steps before update (accumulated gradient for 32 steps).  Each process will do B*T
      if master_process:
        self.logger_train.info(f"Total desired batch size: {self.config.training.total_batch_size} tokens per update step")
        self.logger_train.info(f"Gradient accumulation steps: {grad_accum_steps}")
        self.logger_train.info(f"Virtual batch size: {grad_accum_steps * B} samples per update step")


      for step in range(self.config.training.max_steps):
        t0 = time.time()
        last_step = (step == self.config.training.max_steps - 1) # result is bool
        if step % self.config.training.eval_frequency == 0 or last_step:
            model.eval()
            val_loader.reset() 
            with torch.no_grad():
              val_loss_accum = 0.0
              for _ in range(self.config.training.val_loss_steps):  # average the loss across 20 steps
                x,y = val_loader.next_batch()
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.autocast(device_type = self.device, dtype= torch.bfloat16):
                  logits, val_loss = model(x,y)
                val_loss = val_loss / self.config.training.val_loss_steps
                val_loss_accum += val_loss.detach()

            if ddp:
              dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)
            
            if master_process:
              self.logger_eval.info(f"Validation loss: {val_loss_accum.item():.4f}")
              run.log({"validation loss": val_loss_accum.item()}, step=step) # log the validation loss to wandb
              with open(log_file, "a") as f:
                f.write(f"val,{step},{val_loss_accum.item()}\n")
              
              if step > 0 and (step % self.config.training.save_checkpoint_frequency == 0 or last_step):
                  # optionally write model checkpoints
                  checkpoint_path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
                  checkpoint = {
                      'model': model.state_dict(),
                      'config': model.config,
                      'step': step,
                      'val_loss': val_loss_accum.item()
                  }
                  # you might also want to add optimizer.state_dict() and
                  # rng seeds etc., if you wanted to more exactly resume training
                  torch.save(checkpoint, checkpoint_path)
                  self.logger_eval.info(f"Checkpoint saved to {checkpoint_path}")

        if step > 0 and step % self.config.eval.gen_frequency == 0: #and use_compile == False:
            model.eval() # for inference mode
            # prefix token
            enc = tiktoken.get_encoding(self.config.eval.model_arch) # get tokenizer
            tokens = enc.encode(self.config.eval.prefix_text) # encode the prefix text
            tokens = torch.tensor(tokens, dtype = torch.long) # e.g. [123, 1, 3, 45]
            tokens = tokens.unsqueeze(0).repeat(self.config.eval.num_return_seq, 1) # (num_return_seq, seq_len) => e.g. (5, 4)
            x = tokens.to(self.device) # move tokens to GPU (num_return_seq, seq_len)
            sample_rng = torch.Generator(device=self.device) # random number generator for sampling
            sample_rng.manual_seed(self.config.eval.gen_seed + ddp_rank) # set seed for random number generator

            # Generate text
            # (num_return_seq, seq_len) = (batch_size, seq_len)
            while x.size(1) < self.config.eval.max_length:
              
              with torch.no_grad(): # not call .backward() in any of these modules, don't cache tensor

                logits, loss = model(x) # (num_return_seq, seq_len, vocab_size)

                # take logits of last position 
                logits_last = logits[:, -1, :]

                prob = F.softmax(logits_last, dim=-1) # on vocab_size dim

                # top-k sampling
                topk_prob, topk_idx = torch.topk(prob, k = 50, dim=-1) # (num_return_seq, 50)  topk on last vocab_size dim

                ix = torch.multinomial(topk_prob, num_samples=1) # (num_return_seq, 1)  sample from topk_prob: torch.multinomial() select from topk_prob based on the probability distribution

                xcol = torch.gather(topk_idx, -1, ix) # (num_return_seq, 1)  select from topk_idx based on the index ix 

                x = torch.cat((x, xcol), dim=1) # seq_len is dim=1

            # turn token index into text
            for i in range(self.config.eval.num_return_seq):
              tokens = x[i, :self.config.eval.max_length].tolist()
              decoded = enc.decode(tokens)

              if master_process:
                self.logger_eval.info(f"rank {ddp_rank} sample {i} : {decoded}")
                text_table.add_data(decoded) # add the generated text to the table
                with open(result_file, 'a') as f:
                  f.write(f"{step},{i},{decoded}\n")
                run.log({f"Generated_Text_{step} ": text_table}, step=step) # log the generated text to wandb
        
        if step % self.config.eval.hellaswag_frequency == 0 and self.config.eval.eval_hellaswag:
          # self.eval_hellaswag(model, ddp_rank, ddp_world_size, step, run, ddp, master_process, eval_file)
          num_correct_norm = 0
          num_total = 0
          for i, ex in enumerate(iterate_examples('val')):
            if i % ddp_world_size != ddp_rank:
              continue
            
            # render examples into tokens and labels
            _, tokens, mask, label = render_example(ex)
            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
            model.eval()
            with torch.no_grad():
              with torch.autocast(device_type = self.device, dtype = torch.bfloat16):
                logits , loss = model(tokens)
              
              pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

          if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
            dist.all_reduce(num_total, op = dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op = dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
          acc_norm = num_correct_norm / num_total

          if master_process:
            self.logger_eval.info(f"Hellaswag acc: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
            run.log({"hellaswag_acc": acc_norm}, step=step) # log the accuracy to wandb
            with open(eval_file, 'a') as f:
              f.write(f"{step},{val_loss_accum.item()},{num_correct_norm},{num_total},{acc_norm:.4f}\n")

      
        # --- Training step ---
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in tqdm(range(grad_accum_steps), desc=f"Step {step}"):
          x,y = train_loader.next_batch()
          x = x.to(self.device)
          y = y.to(self.device)
          with torch.autocast(device_type = self.device, dtype= torch.bfloat16):
            logits, loss = model(x,y)
          
          loss = loss / grad_accum_steps # sum the already normailzed loss (each real batch) by grad_accum_steps
          loss_accum += loss.detach() # detach the loss tensor from the computation graph

          if ddp:
            model.require_backward_grad_sync = True if micro_step == grad_accum_steps - 1 else False # only sync the gradient in the last micro_step (sync the gradient across all GPUs means averaging gradient before update)

          loss.backward()  # deposit (added) gradient as we passed in each micro_step 

        if ddp:
          dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG) # average the loss across all GPUs


          norm = torch.nn.utils.clip_grad_norm(model.parameters(), self.config.training.grad_clip) # calculate global norm of all gradients for each param and clip it
          
          # LR schedule
          lr = get_lr(step, self.config.training.warmup_steps, self.config.training.max_steps, self.config.training.min_lr_ratio, self.config.training.max_lr)
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr

          optimizer.step()

          torch.cuda.synchronize() # wait for all GPU operations to finish
          t1 = time.time()
          dt = (t1-t0)*1000 # in ms
          token_processed = (train_loader.batch_size * train_loader.seq_length * grad_accum_steps) * ddp_world_size # times to the total number of GPUs
          tok_per_sec = token_processed / (t1-t0)

          if master_process: # master process would print the log (cuda:0)
            self.logger_train.info(f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | tok/sec: {tok_per_sec:.2f} tokens/s | dt: {dt:.2f} ms")
            run.log({"loss": loss_accum.item(), "lr": lr, "norm": norm, "tok_per_sec": tok_per_sec}, step=step) # log the loss, lr, norm, tok_per_sec to wandb
            with open(log_file, "a") as f:
              f.write(f"train,{step},{loss_accum.item()},{lr},{norm},{tok_per_sec},{dt}\n")
    
      if ddp:
          destroy_process_group()



if __name__ == "__main__":
  from omegaconf import OmegaConf, DictConfig
  import argparse

  # Load config
  parser = argparse.ArgumentParser("GPT-2 Pre-training")
  parser.add_argument("--config_path", help="Path to the configuration file", type=str, required=True)
  args = parser.parse_args()

  config = OmegaConf.load(args.config_path)
  print(OmegaConf.to_yaml(config))

  trainer = Trainer(config)
  trainer.train()
