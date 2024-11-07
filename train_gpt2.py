import os
from dataclasses import dataclass
from datetime import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from holoviews.operation.datashader import inspect
from torch.distributed._composable.replicate import DDP
from torch.nn.modules.module import T
import math
import tiktoken

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd%config.n_head == 0
        self.c_attn = nn.Linear(self.config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k ,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3)
        q = q.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_head, C // self.n_head).permute(0, 2, 1, 3)

        '''att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v'''

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight   #sharing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, 0, std)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0, std)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T<=self.config.block_size, f"cannot deal with sequence of length {T}, since the longest is {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        pos_emb = self.transformer.wpr(pos)
        tok_emb = self.transformer.wte(idx)

        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss



    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_param = [p for _, p in param_dict.items() if p.dim()>=2]
        no_decay_param = [p for _, p in param_dict.items() if p.dim()<2]
        optim_groups = [{'params': decay_param, 'weight_decay': weight_decay}, {'params': no_decay_param, 'weight_decay': 0.0}]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_available and 'cuda' in device



        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused=used_fused)
        return optimizer

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        self.current_position += B*T*self.num_processes
        x = buf[:-1].view(self.B, self.T, -1)
        y = buf[1:].view(self.B, self.T, -1)
        if (self.current_position + B*T*self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0<= decay_ratio <=1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == '__main__':
    from torch.distributed import init_device_mesh, destroy_process_group, init_process_group
    import torch.distributed as dist
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group('nccl')
        ddp_rank = int(os.environ.get('RANK'))
        ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
        ddp_world_size = int(os.environ.get('WORLD_SIZE'))
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        #device = "cuda"
        torch.cuda.manual_seed(1337)

    total_batch_size = 524288
    B = 16
    T = 1024
    assert total_batch_size % (B *T *ddp_world_size) ==0
    grad_accum_steps = total_batch_size / (B * T* ddp_world_size)
    train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size)
    torch.set_float32_matmul_precision('high')

    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    #optimizer = torch.optim.Adamw(model.parameters(), betas=(0.9, 0.95), lr=3e-4, eps = 1e-8)
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
    for i in range(100):
        t0 = time.time()
        loss_accum = 0.0

        optimizer.zero_grad()
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type = device, dtype = torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.set_requires_gradient_sync = (micro_step == grad_accum_steps -1)
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = train_loader.B + train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f'step {i:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} |norm {norm:.4e} | dt: {dt*1000} ms | tokens_per_sec {tokens_per_sec:.2f}')

    if ddp:
        destroy_process_group()



