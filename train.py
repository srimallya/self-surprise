# ============================
# Log-Decayed Causal Attention Character LM + Self-Surprise Modeling
# Corpus: input.txt
#
# Core attention idea:
#   Standard attention:
#       score(i, j) = Q_i K_j / sqrt(d)
#
#   Log-decayed attention:
#       score(i, j) = Q_i K_j / sqrt(d) - alpha_h * log(1 + i - j)
#
# Meaning:
#   nearby context is sharp by default
#   older context fades by design
#   old tokens can still win if semantic/content score is strong
#
# New self-correction idea:
#   The model does not directly minimize self-surprise.
#   That would reward overconfident garbage. Humanity already tried that with ideology.
#
#   Instead, the model learns to PREDICT surprise/uncertainty as an auxiliary signal:
#       hidden state h_t -> token logits
#       hidden state h_t -> expected NLL / expected entropy / change probability
#
#   We log:
#       train_ce
#       val_ce
#       generalization_gap = val_ce - train_ce
#       val_entropy
#       val_calibration_gap = val_ce - val_entropy
#       generated_self_surprise
#       generated_entropy
#       generated_entropy_amplification = generated_self_surprise - val_entropy
#       generated_surprise_error = generated_self_surprise - predicted_surprise
#       current-vs-EMA KL on generated contexts
#
#   Optional auxiliary loss:
#       confidence_loss = MSE(predicted_surprise, detached_token_nll)
#       entropy_loss    = MSE(predicted_entropy, detached_entropy)
#       change_loss     = BCE(predicted_change, detached_change_target)
#
#   The auxiliary loss is detached from targets, so it teaches representation of uncertainty
#   without telling the LM head to become a smug deterministic parrot.
#
# Supports attention_method:
#   "dot"        -> standard causal attention
#   "log_decay"  -> causal attention with learned per-head logarithmic distance tax
#   "all"        -> train dot and log_decay side-by-side
#
# Adds:
#   - EMA teacher for optional self-distillation and generated KL diagnostics
#   - privileged-prefix teacher context for self-distillation
#   - CE + optional KL self-distillation
#   - entropy-filtered KD
#   - learned per-head decay alpha
#   - diagnostics for alpha and long-range attention mass
#   - uncertainty heads: surprise, entropy, change
#   - generated self-surprise evaluation
#   - lagged correlation between self-surprise and future val gap
#
# No semantic-window cosplay. Just time with friction and a little self-doubt.
# ============================

import math
import os
import copy
import csv
import shutil
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# Hyperparameters
# ============================

config = dict(
    # data
    input_path="input.txt",
    val_frac=0.10,

    # sequence
    block_size=256,
    hint_len=128,

    # training
    batch_size=8,
    max_steps=30000,
    eval_interval=200,
    eval_iters=50,
    learning_rate=3e-4,
    weight_decay=0.01,
    grad_clip=1.0,

    # model
    n_layer=8,
    n_head=8,
    n_embd=512,
    dropout=0.20,

    # attention
    attention_method="log_decay",  # "dot", "log_decay", "all"

    # log decay attention
    decay_alpha_init=0.10,
    decay_alpha_scale=0.5,
    use_alpha_clamp=True,
    alpha_min=0.0,
    alpha_max=1.5,

    # Attention diagnostics
    long_range_fraction=0.50,

    # persistence
    ckpt_dir="checkpoints_log_decay_self_surprise",
    csv_log_path="checkpoints_log_decay_self_surprise/metrics.csv",
    plot_dir="checkpoints_log_decay_self_surprise/plots",
    attention_video_path="checkpoints_log_decay_self_surprise/attention_evolution.mp4",
    enable_post_training_plots=True,
    enable_attention_video=False,
    attention_video_fps=2,
    attention_video_probe_batch=1,
    attention_video_max_rows=2,
    attention_video_max_heatmap_size=256,

    # self-distillation
    use_self_distill=False,
    distill_weight=0.15,
    distill_temperature=2.0,
    ema_decay=0.995,
    entropy_filter=True,
    teacher_entropy_margin=0.05,

    # uncertainty / self-surprise modeling
    use_uncertainty_heads=True,
    use_uncertainty_loss=True,
    uncertainty_weight=0.05,
    surprise_weight=1.0,
    entropy_pred_weight=0.5,
    change_weight=0.25,

    # A token becomes a "change" target when current token NLL is this much
    # above predicted/rolling expectation. Detached target; not a truth oracle.
    change_surprise_margin=0.50,

    # Generated rollout diagnostics
    enable_self_surprise_eval=True,
    self_surprise_eval_batches=8,
    self_surprise_prefix_len=128,
    self_surprise_rollout_tokens=64,
    self_surprise_temperature=0.8,
    self_surprise_top_k=90,

    # optional synthetic CE gate, off by default.
    # If enabled, trains on generated next tokens with weight exp(-alpha * surprise),
    # using detached confidence. Keep small unless you enjoy inventing collapse machines.
    use_gated_synthetic_ce=False,
    synthetic_ce_weight=0.02,
    synthetic_gate_alpha=1.0,

    # lagged correlation settings
    lag_corr_windows=(1, 2, 5, 10),

    # sampling
    sample_tokens=80,
    top_k=90,
    temperature=0.8,
    sample_prefix="",

    # reproducibility
    seed=1337,
)


# ============================
# Setup
# ============================

torch.manual_seed(config["seed"])


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()
print(f"using device: {device}")

path = config["input_path"]
if not os.path.exists(path):
    raise FileNotFoundError(
        f"input.txt not found at {path}. Set config['input_path'] correctly. "
        "The machine cannot train on vibes. Tragic, really."
    )

with open(path, "r", encoding="utf-8") as f:
    text = f.read()

if len(text) < config["block_size"] + config["hint_len"] + 2:
    raise ValueError(
        "Corpus is too small for block_size + hint_len. Feed the tiny beast more text."
    )

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)


def decode(t):
    return "".join([itos[int(i)] for i in t])


data = encode(text)

n = int((1.0 - config["val_frac"]) * len(data))
train_data = data[:n]
val_data = data[n:]

if len(val_data) < config["block_size"] + config["hint_len"] + 2:
    print(
        "warning: validation split is tiny. Eval may be noisy, because apparently statistics also requires food."
    )


# ============================
# Data
# ============================


def get_batch(split, batch_size=None, block_size=None):
    src = train_data if split == "train" else val_data
    B = batch_size or config["batch_size"]
    T = block_size or config["block_size"]

    if len(src) <= T + 1:
        raise ValueError(f"{split} split too small for block_size.")

    ix = torch.randint(0, len(src) - T - 1, (B,))

    x = torch.stack([src[i:i + T] for i in ix]).to(device)
    y = torch.stack([src[i + 1:i + 1 + T] for i in ix]).to(device)

    return x, y


def get_prefix_distill_batch(split, context_len, hint_len):
    src = train_data if split == "train" else val_data
    total = hint_len + context_len + 1

    if len(src) <= total:
        raise ValueError(
            f"{split} split too small for hint_len + block_size. "
            "The teacher cannot see privileged context if there is no context. Cruel, but mathematical."
        )

    ix = torch.randint(0, len(src) - total, (config["batch_size"],))
    seq = torch.stack([src[i:i + total] for i in ix]).to(device)

    x_student = seq[:, hint_len:hint_len + context_len]
    y_student = seq[:, hint_len + 1:hint_len + context_len + 1]
    x_teacher = seq[:, :hint_len + context_len]

    return x_student, y_student, x_teacher


# ============================
# Utility
# ============================


def top_k_filter(logits, k):
    if k is None or k <= 0:
        return logits

    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    cutoff = v[..., -1, None]

    return torch.where(
        logits < cutoff,
        torch.full_like(logits, -1e10),
        logits,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def safe_float(x):
    if x is None:
        return ""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().item()
    return float(x)


def pearson_corr(xs, ys):
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)

    if vx <= 1e-12 or vy <= 1e-12:
        return None

    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


# ============================
# Attention
# ============================


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        block_size,
        attention_method,
        decay_alpha_init,
        decay_alpha_scale,
        use_alpha_clamp,
        alpha_min,
        alpha_max,
        long_range_fraction,
    ):
        super().__init__()

        assert n_embd % n_head == 0
        assert attention_method in ["dot", "log_decay"]

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size
        self.attention_method = attention_method

        self.decay_alpha_scale = decay_alpha_scale
        self.use_alpha_clamp = use_alpha_clamp
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.long_range_fraction = long_range_fraction

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        pos = torch.arange(block_size)
        dist = pos[:, None] - pos[None, :]
        dist = dist.clamp(min=0).float()

        self.register_buffer(
            "log_distance",
            torch.log1p(dist).view(1, 1, block_size, block_size),
        )

        init = float(decay_alpha_init) / max(float(decay_alpha_scale), 1e-8)
        raw_init = math.log(math.exp(init) - 1.0) if init > 1e-6 else -10.0
        self.raw_alpha = nn.Parameter(torch.full((n_head,), raw_init))

        self.last_diag = {}
        self.capture_attention_map = False
        self.last_attention_map = None

    def get_alpha(self):
        alpha = F.softplus(self.raw_alpha) * self.decay_alpha_scale

        if self.use_alpha_clamp:
            alpha = alpha.clamp(self.alpha_min, self.alpha_max)

        return alpha

    def split_heads(self, x):
        B, T, C = x.size()
        return x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    def merge_heads(self, y):
        B, H, T, D = y.size()
        return y.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(self, x):
        B, T, C = x.size()

        k = self.split_heads(self.key(x))
        q = self.split_heads(self.query(x))
        v = self.split_heads(self.value(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        if self.attention_method == "log_decay":
            alpha = self.get_alpha().view(1, self.n_head, 1, 1)
            decay_bias = -alpha * self.log_distance[:, :, :T, :T]
            scores = scores + decay_bias

        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf"),
        )

        att = F.softmax(scores, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = self.merge_heads(y)
        y = self.resid_drop(self.proj(y))

        with torch.no_grad():
            diag = {}

            if self.attention_method == "log_decay":
                a = self.get_alpha()
                diag["alpha_mean"] = float(a.mean().detach().cpu())
                diag["alpha_min"] = float(a.min().detach().cpu())
                diag["alpha_max"] = float(a.max().detach().cpu())

            dist = torch.arange(T, device=x.device)[:, None] - torch.arange(T, device=x.device)[None, :]
            dist = dist.clamp(min=0)
            threshold = max(1, int(T * self.long_range_fraction))
            long_mask = (dist >= threshold).view(1, 1, T, T)
            long_mass = (att * long_mask.float()).sum() / (B * self.n_head * T)
            diag["long_attn_mass"] = float(long_mass.detach().cpu())

            dist_f = dist.float().view(1, 1, T, T)
            mean_dist = (att * dist_f).sum(dim=-1).mean()
            diag["mean_attn_dist"] = float(mean_dist.detach().cpu())

            self.last_diag = diag

            if self.capture_attention_map:
                self.last_attention_map = att.mean(dim=(0, 1)).detach().cpu()

        return y


# ============================
# Transformer Block
# ============================


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        block_size,
        attention_method,
        decay_alpha_init,
        decay_alpha_scale,
        use_alpha_clamp,
        alpha_min,
        alpha_max,
        long_range_fraction,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=dropout,
            block_size=block_size,
            attention_method=attention_method,
            decay_alpha_init=decay_alpha_init,
            decay_alpha_scale=decay_alpha_scale,
            use_alpha_clamp=use_alpha_clamp,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            long_range_fraction=long_range_fraction,
        )

        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================
# GPT Character LM
# ============================


class GPTLogDecaySelfSurpriseLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layer,
        n_head,
        n_embd,
        dropout,
        block_size,
        attention_method,
        decay_alpha_init,
        decay_alpha_scale,
        use_alpha_clamp,
        alpha_min,
        alpha_max,
        long_range_fraction,
        use_uncertainty_heads=True,
    ):
        super().__init__()

        assert attention_method in ["dot", "log_decay"]

        self.block_size = block_size
        self.attention_method = attention_method
        self.use_uncertainty_heads = use_uncertainty_heads

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            h=nn.ModuleList([
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    dropout=dropout,
                    block_size=block_size,
                    attention_method=attention_method,
                    decay_alpha_init=decay_alpha_init,
                    decay_alpha_scale=decay_alpha_scale,
                    use_alpha_clamp=use_alpha_clamp,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    long_range_fraction=long_range_fraction,
                )
                for _ in range(n_layer)
            ]),
            ln_f=nn.LayerNorm(n_embd),
        ))

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        if use_uncertainty_heads:
            self.surprise_head = nn.Sequential(
                nn.Linear(n_embd, n_embd // 2),
                nn.GELU(),
                nn.Linear(n_embd // 2, 1),
            )
            self.entropy_head = nn.Sequential(
                nn.Linear(n_embd, n_embd // 2),
                nn.GELU(),
                nn.Linear(n_embd // 2, 1),
            )
            self.change_head = nn.Sequential(
                nn.Linear(n_embd, n_embd // 2),
                nn.GELU(),
                nn.Linear(n_embd // 2, 1),
            )
        else:
            self.surprise_head = None
            self.entropy_head = None
            self.change_head = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_aux=False):
        B, T = idx.shape
        assert T <= self.block_size, f"T={T} exceeds block_size={self.block_size}"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        h = self.transformer.ln_f(x)
        logits = self.lm_head(h)

        loss_lm = None
        if targets is not None:
            loss_lm = F.cross_entropy(
                logits.reshape(B * T, -1),
                targets.reshape(B * T),
            )

        aux = {}
        if self.use_uncertainty_heads:
            # softplus makes expected NLL/entropy non-negative.
            aux["pred_surprise"] = F.softplus(self.surprise_head(h)).squeeze(-1)
            aux["pred_entropy"] = F.softplus(self.entropy_head(h)).squeeze(-1)
            aux["change_logit"] = self.change_head(h).squeeze(-1)
            aux["change_prob"] = torch.sigmoid(aux["change_logit"])

        if return_aux:
            return logits, loss_lm, aux

        return logits, loss_lm

    def get_attention_diagnostics(self):
        values = {}
        counts = {}

        for layer_idx, block in enumerate(self.transformer.h):
            diag = block.attn.last_diag

            for key, val in diag.items():
                values[key] = values.get(key, 0.0) + float(val)
                counts[key] = counts.get(key, 0) + 1

                layer_key = f"L{layer_idx}_{key}"
                values[layer_key] = float(val)
                counts[layer_key] = 1

        return {key: values[key] / max(counts[key], 1) for key in values}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k_val=0):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond, targets=None)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = top_k_filter(logits, top_k_val)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ============================
# Model Builders
# ============================


AVAILABLE_METHODS = ["dot", "log_decay"]


def model_block_size():
    if config["use_self_distill"]:
        return config["block_size"] + config["hint_len"]
    return config["block_size"]


def make_model(attention_method):
    return GPTLogDecaySelfSurpriseLM(
        vocab_size=vocab_size,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        block_size=model_block_size(),
        attention_method=attention_method,
        decay_alpha_init=config["decay_alpha_init"],
        decay_alpha_scale=config["decay_alpha_scale"],
        use_alpha_clamp=config["use_alpha_clamp"],
        alpha_min=config["alpha_min"],
        alpha_max=config["alpha_max"],
        long_range_fraction=config["long_range_fraction"],
        use_uncertainty_heads=config["use_uncertainty_heads"],
    ).to(device)


def build_experiment_models():
    method = config["attention_method"]

    if method == "all":
        names = AVAILABLE_METHODS
    elif method in AVAILABLE_METHODS:
        names = [method]
    else:
        raise ValueError(f"Unknown attention_method: {method}")

    models = {name: make_model(name) for name in names}

    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        for name, model in models.items()
    }

    return models, optimizers


def build_ema_teachers(models):
    teachers = {}

    for name, model in models.items():
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teachers[name] = teacher

    return teachers


@torch.no_grad()
def update_ema_teacher(teacher, student, decay):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


# ============================
# Loss helpers
# ============================


def token_entropy_from_logits(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def token_nll_from_logits(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def mean_kl_from_logits(student_logits, teacher_logits, temperature=1.0):
    T = temperature
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return token_kl.mean() * (T * T)


def distill_kl_loss(
    student_logits,
    teacher_logits,
    temperature=2.0,
    entropy_filter=True,
    teacher_entropy_margin=0.05,
):
    T = temperature

    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)

    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)

    with torch.no_grad():
        teacher_entropy = token_entropy_from_logits(teacher_logits, temperature=T)
        student_entropy = token_entropy_from_logits(student_logits, temperature=T)

        if entropy_filter:
            mask = teacher_entropy < (student_entropy - teacher_entropy_margin)
        else:
            mask = torch.ones_like(token_kl, dtype=torch.bool)

        mask_f = mask.float()
        denom = mask_f.sum().clamp_min(1.0)

    kd = (token_kl * mask_f).sum() / denom
    kd = kd * (T * T)

    stats = {
        "teacher_entropy": float(teacher_entropy.mean().detach().cpu()),
        "student_entropy": float(student_entropy.mean().detach().cpu()),
        "kd_keep": float(mask_f.mean().detach().cpu()),
    }

    return kd, stats


def uncertainty_aux_loss(logits, targets, aux):
    if not config["use_uncertainty_heads"] or not aux:
        zero = logits.sum() * 0.0
        return zero, {}

    with torch.no_grad():
        token_nll = token_nll_from_logits(logits, targets)
        token_entropy = token_entropy_from_logits(logits)

        # surprise_error is relative to the detached predicted surprise.
        # This makes the change head learn "more surprising than expected",
        # without pushing the LM logits toward a fake target.
        pred_surprise_detached = aux["pred_surprise"].detach()
        surprise_error = token_nll - pred_surprise_detached
        change_target = (surprise_error > config["change_surprise_margin"]).float()

    surprise_loss = F.mse_loss(aux["pred_surprise"], token_nll.detach())
    entropy_loss = F.mse_loss(aux["pred_entropy"], token_entropy.detach())
    change_loss = F.binary_cross_entropy_with_logits(aux["change_logit"], change_target)

    total = (
        config["surprise_weight"] * surprise_loss
        + config["entropy_pred_weight"] * entropy_loss
        + config["change_weight"] * change_loss
    )

    stats = {
        "aux_surprise_loss": float(surprise_loss.detach().cpu()),
        "aux_entropy_loss": float(entropy_loss.detach().cpu()),
        "aux_change_loss": float(change_loss.detach().cpu()),
        "aux_token_nll": float(token_nll.mean().detach().cpu()),
        "aux_token_entropy": float(token_entropy.mean().detach().cpu()),
        "aux_pred_surprise": float(aux["pred_surprise"].mean().detach().cpu()),
        "aux_pred_entropy": float(aux["pred_entropy"].mean().detach().cpu()),
        "aux_change_rate": float(change_target.mean().detach().cpu()),
        "aux_change_prob": float(aux["change_prob"].mean().detach().cpu()),
        "aux_surprise_error": float(surprise_error.mean().detach().cpu()),
    }

    return total, stats


# ============================
# Eval / Sampling / Self-surprise diagnostics
# ============================


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses_lm = []
        entropies = []
        pred_surprises = []
        pred_entropies = []
        surprise_errors = []
        change_probs = []

        for _ in range(config["eval_iters"]):
            xb, yb = get_batch(split)
            logits, loss_lm, aux = model(xb, yb, return_aux=True)
            losses_lm.append(loss_lm.item())

            token_entropy = token_entropy_from_logits(logits)
            entropies.append(token_entropy.mean().item())

            if aux:
                token_nll = token_nll_from_logits(logits, yb)
                pred_surprises.append(aux["pred_surprise"].mean().item())
                pred_entropies.append(aux["pred_entropy"].mean().item())
                surprise_errors.append((token_nll - aux["pred_surprise"]).mean().item())
                change_probs.append(aux["change_prob"].mean().item())

        lm = sum(losses_lm) / len(losses_lm)
        entropy = sum(entropies) / len(entropies)

        out[split] = dict(
            lm=lm,
            entropy=entropy,
            calibration_gap=lm - entropy,
            pred_surprise=(sum(pred_surprises) / len(pred_surprises)) if pred_surprises else None,
            pred_entropy=(sum(pred_entropies) / len(pred_entropies)) if pred_entropies else None,
            surprise_error=(sum(surprise_errors) / len(surprise_errors)) if surprise_errors else None,
            change_prob=(sum(change_probs) / len(change_probs)) if change_probs else None,
        )

    model.train()
    return out


@torch.no_grad()
def generated_self_surprise_diagnostics(model, teacher=None):
    if not config["enable_self_surprise_eval"]:
        return {}

    was_training = model.training
    model.eval()
    if teacher is not None:
        teacher.eval()

    prefix_len = min(config["self_surprise_prefix_len"], config["block_size"] - 2)
    rollout = config["self_surprise_rollout_tokens"]

    all_nll = []
    all_entropy = []
    all_pred_surprise = []
    all_pred_entropy = []
    all_change_prob = []
    all_kl = []

    for _ in range(config["self_surprise_eval_batches"]):
        prefix, _ = get_batch("val", batch_size=config["batch_size"], block_size=prefix_len)

        generated = model.generate(
            prefix.clone(),
            max_new_tokens=rollout,
            temperature=config["self_surprise_temperature"],
            top_k_val=config["self_surprise_top_k"],
        )

        # Evaluate the generated suffix under teacher-forced generated contexts.
        # inputs predict next token, so suffix positions are after prefix.
        seq = generated[:, -min(generated.size(1), model.block_size):]
        x = seq[:, :-1]
        y = seq[:, 1:]

        logits, _, aux = model(x, y, return_aux=True)
        nll = token_nll_from_logits(logits, y)
        entropy = token_entropy_from_logits(logits)

        all_nll.append(nll.mean().item())
        all_entropy.append(entropy.mean().item())

        if aux:
            all_pred_surprise.append(aux["pred_surprise"].mean().item())
            all_pred_entropy.append(aux["pred_entropy"].mean().item())
            all_change_prob.append(aux["change_prob"].mean().item())

        if teacher is not None:
            teacher_logits, _ = teacher(x, targets=None)
            kl = mean_kl_from_logits(logits, teacher_logits, temperature=1.0)
            all_kl.append(kl.item())

    gen_nll = sum(all_nll) / len(all_nll)
    gen_entropy = sum(all_entropy) / len(all_entropy)
    pred_surprise = (sum(all_pred_surprise) / len(all_pred_surprise)) if all_pred_surprise else None
    pred_entropy = (sum(all_pred_entropy) / len(all_pred_entropy)) if all_pred_entropy else None
    change_prob = (sum(all_change_prob) / len(all_change_prob)) if all_change_prob else None
    ema_kl = (sum(all_kl) / len(all_kl)) if all_kl else None

    out = dict(
        gen_self_surprise=gen_nll,
        gen_entropy=gen_entropy,
        gen_pred_surprise=pred_surprise,
        gen_pred_entropy=pred_entropy,
        gen_surprise_error=(gen_nll - pred_surprise) if pred_surprise is not None else None,
        gen_entropy_pred_error=(gen_entropy - pred_entropy) if pred_entropy is not None else None,
        gen_change_prob=change_prob,
        gen_ema_kl=ema_kl,
    )

    if was_training:
        model.train()

    return out


@torch.no_grad()
def sample_text(model, prefix=None, steps=None):
    was_training = model.training
    if prefix is None:
        prefix = config["sample_prefix"]

    if steps is None:
        steps = config["sample_tokens"]

    if len(prefix) == 0:
        start_id = torch.randint(0, vocab_size, (1, 1), device=device)
    else:
        start_id = encode(prefix).unsqueeze(0).to(device)

    out = model.generate(
        start_id,
        max_new_tokens=steps,
        temperature=config["temperature"],
        top_k_val=config["top_k"],
    )[0].tolist()

    print(decode(out))

    if was_training:
        model.train()


# ============================
# Training Steps
# ============================


def train_one_model_step(model, optimizer, xb, yb, teacher=None):
    logits, loss_lm, aux = model(xb, yb, return_aux=True)

    aux_loss, aux_stats = uncertainty_aux_loss(logits, yb, aux)
    loss = loss_lm

    if config["use_uncertainty_loss"]:
        loss = loss + config["uncertainty_weight"] * aux_loss

    synthetic_stats = {}
    if config["use_gated_synthetic_ce"]:
        synthetic_loss, synthetic_stats = gated_synthetic_ce_loss(model, teacher)
        loss = loss + config["synthetic_ce_weight"] * synthetic_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    optimizer.step()

    if teacher is not None:
        update_ema_teacher(teacher=teacher, student=model, decay=config["ema_decay"])

    diag = model.get_attention_diagnostics()

    return dict(
        loss=loss.item(),
        lm=loss_lm.item(),
        aux=aux_loss.item() if isinstance(aux_loss, torch.Tensor) else float(aux_loss),
        grad=float(grad_norm),
        diag=diag,
        aux_stats=aux_stats,
        synthetic_stats=synthetic_stats,
    )


def train_one_model_step_self_distill(model, teacher, optimizer):
    context_len = config["block_size"]
    hint_len = config["hint_len"]

    x_student, y_student, x_teacher = get_prefix_distill_batch(
        split="train",
        context_len=context_len,
        hint_len=hint_len,
    )

    student_logits, ce_loss, aux = model(x_student, y_student, return_aux=True)

    with torch.no_grad():
        teacher.eval()
        teacher_logits_full, _ = teacher(x_teacher, targets=None)
        teacher_logits = teacher_logits_full[:, hint_len:hint_len + context_len, :]

    kd_loss, kd_stats = distill_kl_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        temperature=config["distill_temperature"],
        entropy_filter=config["entropy_filter"],
        teacher_entropy_margin=config["teacher_entropy_margin"],
    )

    aux_loss, aux_stats = uncertainty_aux_loss(student_logits, y_student, aux)

    loss = ce_loss + config["distill_weight"] * kd_loss

    if config["use_uncertainty_loss"]:
        loss = loss + config["uncertainty_weight"] * aux_loss

    synthetic_stats = {}
    if config["use_gated_synthetic_ce"]:
        synthetic_loss, synthetic_stats = gated_synthetic_ce_loss(model, teacher)
        loss = loss + config["synthetic_ce_weight"] * synthetic_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    optimizer.step()

    update_ema_teacher(teacher=teacher, student=model, decay=config["ema_decay"])

    diag = model.get_attention_diagnostics()

    return dict(
        loss=loss.item(),
        ce=ce_loss.item(),
        kd=kd_loss.item(),
        aux=aux_loss.item() if isinstance(aux_loss, torch.Tensor) else float(aux_loss),
        grad=float(grad_norm),
        diag=diag,
        kd_stats=kd_stats,
        aux_stats=aux_stats,
        synthetic_stats=synthetic_stats,
    )


def gated_synthetic_ce_loss(model, teacher=None):
    was_training = model.training
    model.eval()

    prefix_len = min(config["self_surprise_prefix_len"], config["block_size"] - 2)
    prefix, _ = get_batch("train", batch_size=config["batch_size"], block_size=prefix_len)

    with torch.no_grad():
        source_model = teacher if teacher is not None else model
        source_model.eval()
        generated = source_model.generate(
            prefix.clone(),
            max_new_tokens=config["self_surprise_rollout_tokens"],
            temperature=config["self_surprise_temperature"],
            top_k_val=config["self_surprise_top_k"],
        )

    if was_training:
        model.train()

    seq = generated[:, -min(generated.size(1), model.block_size):]
    x = seq[:, :-1]
    y = seq[:, 1:]

    logits, _, _ = model(x, y, return_aux=True)

    token_nll = token_nll_from_logits(logits, y)
    with torch.no_grad():
        gate = torch.exp(-config["synthetic_gate_alpha"] * token_nll.detach()).clamp(0.0, 1.0)

    loss = (gate * token_nll).sum() / gate.sum().clamp_min(1.0)

    stats = {
        "synthetic_ce": float(loss.detach().cpu()),
        "synthetic_gate": float(gate.mean().detach().cpu()),
    }

    return loss, stats


# ============================
# Logging
# ============================


def format_diag(diag):
    if not diag:
        return ""
    return " ".join([f"{key}={diag[key]:.3f}" for key in sorted(diag.keys())])


def format_kd_stats(kd_stats):
    if not kd_stats:
        return ""
    return (
        f"Ht={kd_stats['teacher_entropy']:.3f} "
        f"Hs={kd_stats['student_entropy']:.3f} "
        f"keep={kd_stats['kd_keep']:.2f}"
    )


def format_aux_stats(aux_stats):
    if not aux_stats:
        return ""
    return (
        f"predS={aux_stats.get('aux_pred_surprise', 0.0):.3f} "
        f"tokNLL={aux_stats.get('aux_token_nll', 0.0):.3f} "
        f"Serr={aux_stats.get('aux_surprise_error', 0.0):.3f} "
        f"chg={aux_stats.get('aux_change_prob', 0.0):.2f}"
    )


def format_synth_stats(stats):
    if not stats:
        return ""
    return f"synCE={stats.get('synthetic_ce', 0.0):.3f} gate={stats.get('synthetic_gate', 0.0):.2f}"


def format_train_cell(log):
    if log is None:
        return "".ljust(140)

    if "ce" in log:
        base = (
            f"loss={log['loss']:.4f} ce={log['ce']:.4f} kd={log['kd']:.4f} "
            f"aux={log['aux']:.4f} grad={log['grad']:.2f}"
        )
        kd_stats = format_kd_stats(log.get("kd_stats", {}))
        if kd_stats:
            base += " | " + kd_stats
    else:
        base = (
            f"loss={log['loss']:.4f} lm={log['lm']:.4f} "
            f"aux={log['aux']:.4f} grad={log['grad']:.2f}"
        )

    aux_stats = format_aux_stats(log.get("aux_stats", {}))
    if aux_stats:
        base += " | " + aux_stats

    synth_stats = format_synth_stats(log.get("synthetic_stats", {}))
    if synth_stats:
        base += " | " + synth_stats

    diag = format_diag(log.get("diag", {}))
    if diag:
        base += " | " + diag

    return base.ljust(140)


def print_side_by_side_train(step, step_logs, models):
    cells = []
    for name in models.keys():
        cells.append(f"{name.upper()}: {format_train_cell(step_logs.get(name))}")
    print(f"[step {step:5d}] " + " | ".join(cells))


def print_side_by_side_eval(step, stats_by_model, best_stats, gen_stats_by_model, models):
    print(f"\n===== eval step {step} =====")
    print("model       | train_lm | val_lm | gap    | val_H  | cal_gap | gen_S  | gen_H  | gen_amp | best_val | best_step")
    print("------------|----------|--------|--------|--------|---------|--------|--------|---------|----------|----------")

    for name in models.keys():
        s = stats_by_model[name]
        best = best_stats[name]
        gen = gen_stats_by_model.get(name, {})
        gap = s["val"]["lm"] - s["train"]["lm"]
        gen_s = gen.get("gen_self_surprise")
        gen_h = gen.get("gen_entropy")
        gen_amp = (gen_s - s["val"]["entropy"]) if gen_s is not None else None

        def fmt(v):
            return f"{v:.4f}" if v is not None else "  n/a "

        print(
            f"{name.upper():<11} | "
            f"{s['train']['lm']:.4f}   | "
            f"{s['val']['lm']:.4f} | "
            f"{gap:.4f} | "
            f"{s['val']['entropy']:.4f} | "
            f"{s['val']['calibration_gap']:.4f}  | "
            f"{fmt(gen_s)} | "
            f"{fmt(gen_h)} | "
            f"{fmt(gen_amp)}  | "
            f"{best['val_lm']:.4f}   | "
            f"{best['step']:>8d}"
        )

    print("")


def checkpoint_payload(model, step, best):
    return {
        "model": model.state_dict(),
        "config": config,
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
        "step": step,
        "best_stats": best,
    }


def save_best_checkpoint(name, model, step, best):
    os.makedirs(config["ckpt_dir"], exist_ok=True)
    ckpt_path = os.path.join(config["ckpt_dir"], f"{name}_best.pt")
    torch.save(checkpoint_payload(model=model, step=step, best=best), ckpt_path)
    print(f"saved best checkpoint: {ckpt_path}")


def csv_diag_fields():
    fields = [
        "alpha_mean",
        "alpha_min",
        "alpha_max",
        "mean_attn_dist",
        "long_attn_mass",
    ]

    for layer_idx in range(config["n_layer"]):
        fields.extend([
            f"L{layer_idx}_alpha_mean",
            f"L{layer_idx}_mean_attn_dist",
            f"L{layer_idx}_long_attn_mass",
        ])

    return fields


def csv_fieldnames():
    base = [
        "step",
        "model",
        "train_lm",
        "val_lm",
        "generalization_gap",
        "train_entropy",
        "val_entropy",
        "train_calibration_gap",
        "val_calibration_gap",
        "train_pred_surprise",
        "val_pred_surprise",
        "train_surprise_error",
        "val_surprise_error",
        "train_change_prob",
        "val_change_prob",
        "gen_self_surprise",
        "gen_entropy",
        "gen_pred_surprise",
        "gen_surprise_error",
        "gen_change_prob",
        "gen_entropy_amplification",
        "gen_ema_kl",
        "best_val_lm",
    ]

    lag_fields = []
    for lag in config["lag_corr_windows"]:
        lag_fields.append(f"lagcorr_genS_gap_plus_{lag}")

    return base + lag_fields + csv_diag_fields()


def init_csv_logger():
    csv_dir = os.path.dirname(config["csv_log_path"])
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(config["csv_log_path"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writeheader()


def append_csv_row(step, name, stats, best, diag, gen_stats, lag_corrs):
    train = stats["train"]
    val = stats["val"]
    gen_s = gen_stats.get("gen_self_surprise")

    row = {
        "step": step,
        "model": name,
        "train_lm": train["lm"],
        "val_lm": val["lm"],
        "generalization_gap": val["lm"] - train["lm"],
        "train_entropy": train["entropy"],
        "val_entropy": val["entropy"],
        "train_calibration_gap": train["calibration_gap"],
        "val_calibration_gap": val["calibration_gap"],
        "train_pred_surprise": train.get("pred_surprise"),
        "val_pred_surprise": val.get("pred_surprise"),
        "train_surprise_error": train.get("surprise_error"),
        "val_surprise_error": val.get("surprise_error"),
        "train_change_prob": train.get("change_prob"),
        "val_change_prob": val.get("change_prob"),
        "gen_self_surprise": gen_s,
        "gen_entropy": gen_stats.get("gen_entropy"),
        "gen_pred_surprise": gen_stats.get("gen_pred_surprise"),
        "gen_surprise_error": gen_stats.get("gen_surprise_error"),
        "gen_change_prob": gen_stats.get("gen_change_prob"),
        "gen_entropy_amplification": (gen_s - val["entropy"]) if gen_s is not None else None,
        "gen_ema_kl": gen_stats.get("gen_ema_kl"),
        "best_val_lm": best["val_lm"],
    }

    for lag in config["lag_corr_windows"]:
        row[f"lagcorr_genS_gap_plus_{lag}"] = lag_corrs.get(lag)

    for key in csv_diag_fields():
        row[key] = diag.get(key, "")

    # Turn None into blank cells.
    clean = {k: ("" if v is None else v) for k, v in row.items()}

    with open(config["csv_log_path"], "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writerow(clean)


def print_samples(step, models):
    print("----- samples -----")

    for name, model in models.items():
        print(f"\n[{name.upper()} sample @ step {step}]")
        sample_text(model, prefix=config["sample_prefix"], steps=config["sample_tokens"])

    print("-------------------\n")


def print_config_summary(models):
    print("\n===== config =====")
    print(f"input_path: {config['input_path']}")
    print(f"vocab_size: {vocab_size}")
    print(f"train_tokens: {len(train_data):,}")
    print(f"val_tokens: {len(val_data):,}")
    print(f"attention_method: {config['attention_method']}")
    print(f"block_size: {config['block_size']}")
    print(f"hint_len: {config['hint_len']}")
    print(f"model_block_size: {model_block_size()}")
    print(f"batch_size: {config['batch_size']}")
    print(f"n_layer: {config['n_layer']}")
    print(f"n_head: {config['n_head']}")
    print(f"n_embd: {config['n_embd']}")
    print(f"dropout: {config['dropout']}")
    print(f"learning_rate: {config['learning_rate']}")
    print(f"weight_decay: {config['weight_decay']}")
    print(f"grad_clip: {config['grad_clip']}")

    print("\n===== log decay =====")
    print(f"decay_alpha_init: {config['decay_alpha_init']}")
    print(f"decay_alpha_scale: {config['decay_alpha_scale']}")
    print(f"use_alpha_clamp: {config['use_alpha_clamp']}")
    print(f"alpha_min: {config['alpha_min']}")
    print(f"alpha_max: {config['alpha_max']}")
    print(f"long_range_fraction: {config['long_range_fraction']}")

    print("\n===== self correction / uncertainty =====")
    print(f"use_uncertainty_heads: {config['use_uncertainty_heads']}")
    print(f"use_uncertainty_loss: {config['use_uncertainty_loss']}")
    print(f"uncertainty_weight: {config['uncertainty_weight']}")
    print(f"change_surprise_margin: {config['change_surprise_margin']}")
    print(f"enable_self_surprise_eval: {config['enable_self_surprise_eval']}")
    print(f"self_surprise_prefix_len: {config['self_surprise_prefix_len']}")
    print(f"self_surprise_rollout_tokens: {config['self_surprise_rollout_tokens']}")
    print(f"use_gated_synthetic_ce: {config['use_gated_synthetic_ce']}")

    print("\n===== self distill =====")
    print(f"use_self_distill: {config['use_self_distill']}")
    print(f"distill_weight: {config['distill_weight']}")
    print(f"distill_temperature: {config['distill_temperature']}")
    print(f"ema_decay: {config['ema_decay']}")
    print(f"entropy_filter: {config['entropy_filter']}")
    print(f"teacher_entropy_margin: {config['teacher_entropy_margin']}")

    print("\n===== persistence =====")
    print(f"ckpt_dir: {config['ckpt_dir']}")
    print(f"csv_log_path: {config['csv_log_path']}")
    print(f"plot_dir: {config['plot_dir']}")
    print(f"attention_video_path: {config['attention_video_path']}")
    print(f"enable_post_training_plots: {config['enable_post_training_plots']}")
    print(f"enable_attention_video: {config['enable_attention_video']}")

    print("\n===== models =====")
    for name, model in models.items():
        print(f"{name.upper():<11} params: {count_parameters(model):,}")
    print("")


# ============================
# Lag correlation tracker
# ============================


class LagCorrelationTracker:
    def __init__(self, lags):
        self.lags = list(lags)
        self.by_model = {}

    def update(self, model_name, gen_self_surprise, gap):
        if model_name not in self.by_model:
            self.by_model[model_name] = []
        self.by_model[model_name].append({
            "gen_self_surprise": gen_self_surprise,
            "gap": gap,
        })

    def lag_corrs(self, model_name):
        rows = self.by_model.get(model_name, [])
        out = {}

        for lag in self.lags:
            xs = []
            ys = []

            for i in range(len(rows) - lag):
                s = rows[i]["gen_self_surprise"]
                g_future = rows[i + lag]["gap"]
                if s is not None and g_future is not None:
                    xs.append(s)
                    ys.append(g_future)

            out[lag] = pearson_corr(xs, ys)

        return out


# ============================
# Post-Training Artifacts
# ============================


def parse_float(value):
    if value is None or value == "":
        return None
    return float(value)


def load_metric_rows():
    with open(config["csv_log_path"], "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "step":
                    parsed[key] = int(value)
                elif key == "model":
                    parsed[key] = value
                else:
                    parsed[key] = parse_float(value)
            rows.append(parsed)
    return rows


def rows_for_model(rows, model_name):
    return sorted([row for row in rows if row["model"] == model_name], key=lambda row: row["step"])


def series(model_rows, field):
    xs = []
    ys = []
    for row in model_rows:
        value = row.get(field)
        if value is not None:
            xs.append(row["step"])
            ys.append(value)
    return xs, ys


def save_plot(fig, filename):
    os.makedirs(config["plot_dir"], exist_ok=True)
    path = os.path.join(config["plot_dir"], filename)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path


def plot_training_artifacts():
    if not config["enable_post_training_plots"]:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib is not installed; skipping post-training plots.")
        return []

    rows = load_metric_rows()
    if not rows:
        print("warning: no metric rows found; skipping post-training plots.")
        return []

    model_names = sorted(set(row["model"] for row in rows))
    by_model = {name: rows_for_model(rows, name) for name in model_names}
    written = []

    # 1. Train/validation language-modeling loss.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "train_lm")
        ax.plot(xs, ys, linestyle="--", label=f"{name} train")
        xs, ys = series(model_rows, "val_lm")
        ax.plot(xs, ys, label=f"{name} val")
    ax.set_title("Train and validation LM loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "01_train_val_lm_loss.png"))
    plt.close(fig)

    # 2. Generalization gap vs generated self-surprise.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "generalization_gap")
        ax.plot(xs, ys, label=f"{name} gap")
        xs, ys = series(model_rows, "gen_self_surprise")
        ax.plot(xs, ys, linestyle="--", label=f"{name} gen self-surprise")
    ax.set_title("Generalization gap vs generated self-surprise")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "02_gap_vs_self_surprise.png"))
    plt.close(fig)

    # 3. Calibration gap.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "val_calibration_gap")
        ax.plot(xs, ys, label=f"{name} val cal gap")
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Validation calibration gap: CE - entropy")
    ax.set_xlabel("step")
    ax.set_ylabel("CE - entropy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "03_val_calibration_gap.png"))
    plt.close(fig)

    # 4. Generated entropy amplification.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "gen_entropy_amplification")
        ax.plot(xs, ys, label=name)
    ax.axhline(0.0, linewidth=1)
    ax.set_title("Generated entropy amplification: gen self-surprise - val entropy")
    ax.set_xlabel("step")
    ax.set_ylabel("amplification")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "04_generated_entropy_amplification.png"))
    plt.close(fig)

    # 5. Predicted surprise vs actual generated surprise.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "gen_self_surprise")
        ax.plot(xs, ys, label=f"{name} actual gen S")
        xs, ys = series(model_rows, "gen_pred_surprise")
        ax.plot(xs, ys, linestyle="--", label=f"{name} predicted gen S")
    ax.set_title("Generated surprise: actual vs predicted")
    ax.set_xlabel("step")
    ax.set_ylabel("surprise")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "05_gen_surprise_actual_vs_predicted.png"))
    plt.close(fig)

    # 6. EMA KL on generated contexts.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "gen_ema_kl")
        if xs and ys:
            ax.plot(xs, ys, label=name)
    ax.set_title("Current-vs-EMA KL on generated contexts")
    ax.set_xlabel("step")
    ax.set_ylabel("KL")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "06_generated_ema_kl.png"))
    plt.close(fig)

    # 7. Alpha mean with min/max band.
    if "log_decay" in by_model:
        log_rows = by_model["log_decay"]
        xs, alpha_mean = series(log_rows, "alpha_mean")
        _, alpha_min = series(log_rows, "alpha_min")
        _, alpha_max = series(log_rows, "alpha_max")

        if xs and alpha_mean:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(xs, alpha_mean, label="alpha mean")
            if len(alpha_min) == len(xs) and len(alpha_max) == len(xs):
                ax.fill_between(xs, alpha_min, alpha_max, alpha=0.20, label="min/max")
            ax.set_title("Log-decay alpha evolution")
            ax.set_xlabel("step")
            ax.set_ylabel("alpha")
            ax.grid(True, alpha=0.25)
            ax.legend()
            written.append(save_plot(fig, "07_alpha_evolution.png"))
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for layer_idx in range(config["n_layer"]):
            xs, ys = series(log_rows, f"L{layer_idx}_alpha_mean")
            if xs and ys:
                plotted = True
                ax.plot(xs, ys, label=f"L{layer_idx}")
        if plotted:
            ax.set_title("Layer-wise log-decay alpha")
            ax.set_xlabel("step")
            ax.set_ylabel("alpha")
            ax.grid(True, alpha=0.25)
            ax.legend(ncol=2)
            written.append(save_plot(fig, "08_layerwise_alpha.png"))
        plt.close(fig)

    # 8. Long-range attention mass.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "long_attn_mass")
        if xs and ys:
            ax.plot(xs, ys, label=name)
    ax.set_title("Long-range attention mass")
    ax.set_xlabel("step")
    ax.set_ylabel("attention mass")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "09_long_range_attention_mass.png"))
    plt.close(fig)

    # 9. Mean attention distance.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, model_rows in by_model.items():
        xs, ys = series(model_rows, "mean_attn_dist")
        if xs and ys:
            ax.plot(xs, ys, label=name)
    ax.set_title("Mean attention distance")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens")
    ax.grid(True, alpha=0.25)
    ax.legend()
    written.append(save_plot(fig, "10_mean_attention_distance.png"))
    plt.close(fig)

    # 10. Best validation comparison.
    labels = []
    values = []
    for name, model_rows in by_model.items():
        best_values = [row["best_val_lm"] for row in model_rows if row.get("best_val_lm") is not None]
        if best_values:
            labels.append(name)
            values.append(min(best_values))

    if labels and values:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(labels, values)
        ax.set_title("Best validation LM loss by model")
        ax.set_xlabel("model")
        ax.set_ylabel("best val LM loss")
        ax.grid(True, axis="y", alpha=0.25)
        written.append(save_plot(fig, "11_best_val_lm_bar.png"))
        plt.close(fig)

    print(f"saved {len(written)} plot artifacts to {config['plot_dir']}")
    return written


# ============================
# Attention video
# ============================


def set_attention_capture(model, enabled):
    for block in model.transformer.h:
        block.attn.capture_attention_map = enabled
        if enabled:
            block.attn.last_attention_map = None


@torch.no_grad()
def collect_attention_snapshot(step, models, probe_x):
    snapshot = {"step": step, "models": {}}

    for name, model in models.items():
        was_training = model.training
        model.eval()
        set_attention_capture(model, True)
        model(probe_x, targets=None)
        set_attention_capture(model, False)

        layers = []
        for block in model.transformer.h:
            attention_map = block.attn.last_attention_map
            if attention_map is not None:
                layers.append(attention_map.clone())
        snapshot["models"][name] = layers

        if was_training:
            model.train()

    return snapshot


def attention_video_layer_grid(n_layer):
    max_rows = max(1, int(config["attention_video_max_rows"]))
    layer_cols = max(1, math.ceil(n_layer / max_rows))
    layer_rows = math.ceil(n_layer / layer_cols)
    return layer_rows, layer_cols


def render_attention_map(attention_map):
    max_size = int(config["attention_video_max_heatmap_size"])

    if max_size <= 0 or max(attention_map.shape) <= max_size:
        return attention_map.numpy()

    height, width = attention_map.shape
    out_h = min(height, max_size)
    out_w = min(width, max_size)

    pooled = F.adaptive_avg_pool2d(
        attention_map.view(1, 1, height, width),
        (out_h, out_w),
    )

    return pooled.squeeze(0).squeeze(0).numpy()


def build_attention_video(attention_snapshots, model_names):
    if not config["enable_attention_video"]:
        return None

    if not attention_snapshots:
        print("warning: no attention snapshots captured; skipping attention video.")
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        print("warning: matplotlib is not installed; skipping attention video.")
        return None

    os.makedirs(os.path.dirname(config["attention_video_path"]), exist_ok=True)

    n_layer = config["n_layer"]
    layer_rows, layer_cols = attention_video_layer_grid(n_layer)
    rows = layer_rows
    cols = len(model_names) * layer_cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.2 * cols, 3.0 * rows),
        squeeze=False,
    )

    images = []
    vmax = 0.0

    for snapshot in attention_snapshots:
        for name in model_names:
            for layer_map in snapshot["models"].get(name, []):
                vmax = max(vmax, float(layer_map.max()))

    vmax = max(vmax, 1e-6)

    for layer_idx in range(n_layer):
        layer_row = layer_idx // layer_cols
        layer_col = layer_idx % layer_cols

        for col_idx, name in enumerate(model_names):
            ax = axes[layer_row][col_idx * layer_cols + layer_col]
            first_map = render_attention_map(attention_snapshots[0]["models"][name][layer_idx])
            image = ax.imshow(
                first_map,
                vmin=0.0,
                vmax=vmax,
                cmap="magma",
                interpolation="nearest",
                animated=True,
            )
            ax.set_title(f"{name} L{layer_idx}")
            ax.set_xlabel("key position")
            ax.set_ylabel("query position")
            ax.set_xticks([])
            ax.set_yticks([])
            images.append((layer_idx, col_idx, image))

    for col_idx in range(len(model_names)):
        for slot_idx in range(n_layer, layer_rows * layer_cols):
            layer_row = slot_idx // layer_cols
            layer_col = slot_idx % layer_cols
            axes[layer_row][col_idx * layer_cols + layer_col].axis("off")

    title = fig.suptitle("")
    fig.tight_layout()

    def update(frame_idx):
        snapshot = attention_snapshots[frame_idx]
        title.set_text(f"attention evolution - step {snapshot['step']}")
        artists = [title]
        for layer_idx, col_idx, image in images:
            name = model_names[col_idx]
            image.set_array(render_attention_map(snapshot["models"][name][layer_idx]))
            artists.append(image)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(attention_snapshots),
        interval=1000 / max(config["attention_video_fps"], 1),
        blit=False,
    )

    video_path = config["attention_video_path"]

    if shutil.which("ffmpeg"):
        writer = animation.FFMpegWriter(fps=config["attention_video_fps"])
        anim.save(video_path, writer=writer, dpi=140)
    else:
        root, _ = os.path.splitext(video_path)
        video_path = root + ".gif"
        writer = animation.PillowWriter(fps=config["attention_video_fps"])
        anim.save(video_path, writer=writer, dpi=120)
        print("warning: ffmpeg not found; saved attention evolution as GIF instead of MP4.")

    plt.close(fig)
    print(f"saved attention evolution video: {video_path}")
    return video_path


# ============================
# Main
# ============================


models, optimizers = build_experiment_models()

# We build EMA teachers whenever needed for self-distillation, generated KL diagnostics,
# or gated synthetic CE. The teacher is not mystical. It is just yesterday's model with better manners.
teachers = build_ema_teachers(models)

init_csv_logger()
print_config_summary(models)

for model in models.values():
    model.train()

best_stats = {
    name: {
        "val_lm": float("inf"),
        "train_lm": float("inf"),
        "step": 0,
    }
    for name in models.keys()
}

lag_tracker = LagCorrelationTracker(config["lag_corr_windows"])

attention_probe_x = None
attention_snapshots = []

if config["enable_attention_video"]:
    attention_probe_x, _ = get_batch("val")
    attention_probe_x = attention_probe_x[:config["attention_video_probe_batch"]]


for step in range(1, config["max_steps"] + 1):
    xb, yb = get_batch("train")
    step_logs = {}

    for name, model in models.items():
        teacher = teachers.get(name)

        if config["use_self_distill"]:
            step_logs[name] = train_one_model_step_self_distill(
                model=model,
                teacher=teacher,
                optimizer=optimizers[name],
            )
        else:
            step_logs[name] = train_one_model_step(
                model=model,
                optimizer=optimizers[name],
                xb=xb,
                yb=yb,
                teacher=teacher,
            )

    if step % config["eval_interval"] == 0 or step == 1:
        print_side_by_side_train(step=step, step_logs=step_logs, models=models)

        stats = {name: estimate_loss(model) for name, model in models.items()}

        gen_stats_by_model = {}
        for name, model in models.items():
            gen_stats_by_model[name] = generated_self_surprise_diagnostics(
                model=model,
                teacher=teachers.get(name),
            )

        # Update lag tracker first with current observation.
        for name, s in stats.items():
            gap = s["val"]["lm"] - s["train"]["lm"]
            gen_s = gen_stats_by_model.get(name, {}).get("gen_self_surprise")
            lag_tracker.update(name, gen_s, gap)

        for name, s in stats.items():
            val_lm = s["val"]["lm"]

            if val_lm < best_stats[name]["val_lm"]:
                best_stats[name] = {
                    "val_lm": val_lm,
                    "train_lm": s["train"]["lm"],
                    "step": step,
                }

                save_best_checkpoint(
                    name=name,
                    model=models[name],
                    step=step,
                    best=best_stats[name],
                )

            append_csv_row(
                step=step,
                name=name,
                stats=s,
                best=best_stats[name],
                diag=models[name].get_attention_diagnostics(),
                gen_stats=gen_stats_by_model.get(name, {}),
                lag_corrs=lag_tracker.lag_corrs(name),
            )

        if config["enable_attention_video"] and attention_probe_x is not None:
            attention_snapshots.append(
                collect_attention_snapshot(
                    step=step,
                    models=models,
                    probe_x=attention_probe_x,
                )
            )

        print_side_by_side_eval(
            step=step,
            stats_by_model=stats,
            best_stats=best_stats,
            gen_stats_by_model=gen_stats_by_model,
            models=models,
        )

        print_samples(step=step, models=models)


print("\n===== best checkpoints =====")
print("model       | best_train_lm | best_val_lm | best_step")
print("------------|---------------|-------------|----------")

for name in models.keys():
    best = best_stats[name]
    print(
        f"{name.upper():<11} | "
        f"{best['train_lm']:.4f}        | "
        f"{best['val_lm']:.4f}      | "
        f"{best['step']:>8d}"
    )

plot_training_artifacts()
build_attention_video(
    attention_snapshots=attention_snapshots,
    model_names=list(models.keys()),
)
