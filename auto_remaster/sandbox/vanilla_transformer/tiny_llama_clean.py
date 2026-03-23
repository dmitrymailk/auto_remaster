"""
Standalone TinyLlama (LLaMA architecture) — без зависимости от transformers.

Математически идентично оригинальному LlamaForCausalLM + greedy decoding.
Токенизатор использует AutoTokenizer из transformers (разрешено условием задачи).

Загрузка весов: через safetensors напрямую.
"""

import math
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    mlp_bias: bool = False
    head_dim: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, d: dict) -> "LlamaConfig":
        # Extract rope_theta from rope_parameters if present (new-style config)
        rope_theta = 10000.0
        if "rope_parameters" in d and d["rope_parameters"] is not None:
            rope_theta = d["rope_parameters"].get("rope_theta", 10000.0)
        elif "rope_theta" in d:
            rope_theta = d["rope_theta"]

        return cls(
            vocab_size=d.get("vocab_size", 32000),
            hidden_size=d.get("hidden_size", 4096),
            intermediate_size=d.get("intermediate_size", 11008),
            num_hidden_layers=d.get("num_hidden_layers", 32),
            num_attention_heads=d.get("num_attention_heads", 32),
            num_key_value_heads=d.get("num_key_value_heads", None),
            hidden_act=d.get("hidden_act", "silu"),
            max_position_embeddings=d.get("max_position_embeddings", 2048),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            attention_bias=d.get("attention_bias", False),
            mlp_bias=d.get("mlp_bias", False),
            head_dim=d.get("head_dim", None),
            bos_token_id=d.get("bos_token_id", 1),
            eos_token_id=d.get("eos_token_id", 2),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
        )


# ---------------------------------------------------------------------------
# KV Cache — простая реализация (list of tensors per layer)
# ---------------------------------------------------------------------------


class DynamicKVCache:
    """Простой KV cache: список пар (keys, values) по слоям."""

    def __init__(self):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == len(self.key_cache):
            # Первый раз для этого слоя
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Конкатенируем по seq_len (dim=2)
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[2]


# ---------------------------------------------------------------------------
# RMSNorm — скопировано дословно из LlamaRMSNorm
# ---------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # обучаемый scale: (hidden_size,) = (2048,)
        self.variance_epsilon = eps  # 1e-5 для числовой стабильности
        # print(f"[RMSNorm init] weight.shape={self.weight.shape}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (bsz, seq_len, hidden_size)  prefill: (1, 32, 2048)  decode: (1, 1, 2048)
        # print(f"[RMSNorm.in] hidden_states={hidden_states.shape} dtype={hidden_states.dtype}")
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)  # upcast для стабильности
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # (bsz, seq_len, 1)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # self.weight: (2048,) — broadcast по последнему dim
        # print(f"[RMSNorm.out] hidden_states={hidden_states.shape}")
        return self.weight * hidden_states.to(input_dtype)  # (bsz, seq_len, 2048)


# ---------------------------------------------------------------------------
# Rotary Positional Embedding — скопировано из LlamaRotaryEmbedding
# ---------------------------------------------------------------------------


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings  # 2048
        dim = config.head_dim   # 64 — RoPE применяется на уровне head_dim
        base = config.rope_theta  # 10000.0 — база для частот
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        # inv_freq: (head_dim/2,) = (32,) — частоты для каждой пары dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0  # default rope type
        # print(f"[RoPE init] dim={dim} base={base} inv_freq.shape={inv_freq.shape}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Вычисляет cos и sin таблицы для RoPE.

        ЗАЧЕМ RoPE:
          Обычный attention не знает о порядке токенов — он работает с
          множествами. RoPE вращает Q и K в комплексной плоскости на угол,
          пропорциональный позиции токена. Тогда dot(Q_i, K_j) автоматически
          зависит от (i - j) — относительного расстояния, а не абсолютных позиций.
          Это даёт хорошую экстраполяцию на длины, не видевшиеся при обучении.

        КАК РАБОТАЕТ:
          1. inv_freq[k] = 1 / (base ** (2k / dim))  k=0..31
             — это убывающие частоты: низкие частоты закодируют крупный масштаб,
               высокие — мелкий.

          2. freq[pos, k] = pos * inv_freq[k]
             — angle для каждой пары (позиция, dim).
             Считается через матмул: (bsz,32,1) @ (bsz,1,seq) = (bsz,32,seq)
             затем .transpose → (bsz, seq, 32)

          3. emb = cat(freqs, freqs) → (bsz, seq, 64=head_dim)
             — дублируем, чтобы покрыть все 64 dim головы:
               первые 32 dim получают cos/sin с частотами inv_freq,
               вторые 32 — те же частоты (применяются сдвигом rotate_half).

          4. Возвращаем cos(emb) и sin(emb).
             apply_rotary_pos_emb потом делает:
               q_rot = q * cos + rotate_half(q) * sin
             Это эквивалентно умножению комплексного числа на e^{iθ}.

        ПОЧЕМУ autocast disabled:
          Матмул inv_freq @ position_ids должен быть в float32.
          В bfloat16 накапливается ошибка в частотах — позиции теряют точность.
        """
        # x: (bsz, seq_len, hidden_size) — нужен только для dtype/device
        # position_ids: (1, seq_len)  prefill: (1, 32)  decode: (1, 1)
        # print(f"[RoPE.in] position_ids={position_ids.shape}")
        inv_freq_expanded = (
            self.inv_freq[None, :, None]  # (1, 32, 1)
            .float()
            .expand(position_ids.shape[0], -1, 1)  # (bsz, 32, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()  # (bsz, 1, seq_len)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
                # (bsz,32,1) @ (bsz,1,seq) → (bsz,32,seq)
            ).transpose(1, 2)  # (bsz, seq_len, 32)
            emb = torch.cat((freqs, freqs), dim=-1)  # (bsz, seq_len, 64=head_dim)
            cos = emb.cos() * self.attention_scaling   # (bsz, seq_len, 64)
            sin = emb.sin() * self.attention_scaling   # (bsz, seq_len, 64)
        # print(f"[RoPE.out] cos={cos.shape}  sin={sin.shape}")
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Сдвигает половину dim — реализует "мнимую часть" комплексного вращения.

    ЗАЧЕМ:
      Если представить пару (x1, x2) как комплексное число x1 + i*x2,
      то умножение на i (поворот на 90°) даёт: -x2 + i*x1.
      Т.е. вещественная часть была x1 → становится -x2,
           мнимая    часть была x2 → становится  x1.
      Именно это и делает функция: [x1, x2] → [-x2, x1].

    В RoPE пары (dim_0, dim_32), (dim_1, dim_33), ..., (dim_31, dim_63)
    интерпретируются как комплексные числа. Мы вращаем каждую пару
    на свой угол θ_k = pos * inv_freq[k].
    """
    x1 = x[..., : x.shape[-1] // 2]  # первые 32 dim: вещественная часть
    x2 = x[..., x.shape[-1] // 2 :]  # вторые 32 dim: мнимая часть
    return torch.cat((-x2, x1), dim=-1)  # [-x2, x1] — сдвиг на 90°


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Применяет RoPE к Q и K.

    МАТЕМАТИКА (вращение комплексного числа):
      Для пары (q1, q2) при угле θ:
        q1' = q1*cos(θ) - q2*sin(θ)
        q2' = q2*cos(θ) + q1*sin(θ)

      В матричном виде: [q1, q2]·[[cos, sin], [-sin, cos]]

      В нашем коде это записано через concat-трюк:
        q * cos               →  [q1*cos, q2*cos]
        rotate_half(q) * sin  →  [-q2*sin, q1*sin]
        сумма                 →  [q1*cos - q2*sin, q2*cos + q1*sin] ✓

      Именно это и есть умножение комплексного числа (q1+i·q2) на e^{iθ}.

    ЗАЧЕМ unsqueeze_dim=1:
      cos/sin приходят формой (bsz, seq_len, head_dim).
      q/k имеют форму (bsz, num_heads, seq_len, head_dim).
      unsqueeze(1) добавляет dim heads: (bsz, 1, seq_len, head_dim)
      → broadcast по всем головам автоматически.
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # (bsz, 1, seq_len, 64) — broadcast по heads
    sin = sin.unsqueeze(unsqueeze_dim)  # (bsz, 1, seq_len, 64)
    # q: (bsz, num_heads, seq_len, 64)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # (bsz, num_heads, seq_len, 64)
    k_embed = (k * cos) + (rotate_half(k) * sin)  # (bsz, num_kv_heads, seq_len, 64)
    return q_embed, k_embed



# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Эквивалент torch.repeat_interleave(x, dim=1, repeats=n_rep).
    (batch, num_kv_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    GQA: 4 kv heads → 32 q heads  (повторяем каждую голову 8 раз)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    # print(f"[repeat_kv.in]  {hidden_states.shape}  n_rep={n_rep}")
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    # print(f"[repeat_kv.out] {(batch, num_key_value_heads * n_rep, slen, head_dim)}")
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention с поддержкой GQA и causal mask.

    ФОРМУЛА (Vaswani et al. 2017):
      Attention(Q, K, V) = softmax(Q·Kᵀ / √d_head) · V

    ЗАЧЕМ scaling = 1/√d_head (= 1/√64 ≈ 0.125):
      Без деления dot-product Q·Kᵀ растёт O(d_head) по величине.
      При больших значениях softmax насыщается → градиенты исчезают.
      Деление на √d_head стабилизирует дисперсию (если Q,K ~ N(0,1),
      то Q·Kᵀ ~ N(0, d_head), деление даёт N(0,1)).

    GQA (Grouped Query Attention):
      Q имеет 32 головы, K/V — только 4. Каждая KV-голова общая для 8 Q-голов.
      repeat_kv расширяет K/V: (bsz, 4, total, 64) → (bsz, 32, total, 64).
      Это экономит память KV-кеша в 8x по сравнению с MHA.

    CAUSAL MASK:
      attention_mask имеет форму (1, 1, seq_len, total_len).
      Верхний треугольник заполнен -inf, нижний — 0.
      После прибавления к attn_weights future-токены получают -inf
      → softmax превращает их в 0 → модель не смотрит в будущее.

    SOFTMAX в fp32:
      exp() чувствителен к overflow в bfloat16, поэтому softmax считается
      в fp32 и кастуется обратно к query.dtype.
    """
    # query: (bsz, num_heads, seq_len, head_dim)      prefill: (1, 32, 32, 64)  decode: (1, 32, 1, 64)
    # key:   (bsz, num_kv_heads, total_len, head_dim)  prefill: (1, 4, 32, 64)   decode: (1, 4, 33+N, 64)
    # print(f"[eager_attn.in] q={query.shape}  k={key.shape}  v={value.shape}")
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)
    # GQA expand: (1, 4, total, 64) → (1, 32, total, 64)
    # print(f"[eager_attn] k_expanded={key_states.shape}")

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    # matmul: (bsz,32,seq,64) @ (bsz,32,64,total) → (bsz,32,seq,total)
    # attn_weights: (bsz, num_heads, seq_len, total_len)  prefill: (1, 32, 32, 32)  decode: (1, 32, 1, 33+N)
    # print(f"[eager_attn] attn_weights={attn_weights.shape}")

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        # mask: (1, 1, seq, total) broadcast → маскирует future-токены через -inf

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # softmax по последнему dim (total_len) → веса суммируются в 1
    attn_output = torch.matmul(attn_weights, value_states)
    # (bsz,32,seq,total) @ (bsz,32,total,64) → (bsz,32,seq,64)
    # attn_output before transpose: (bsz, num_heads, seq_len, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output after transpose: (bsz, seq_len, num_heads, head_dim)
    # NOTE: .contiguous() нужен перед .reshape() в вызывающем коде
    # print(f"[eager_attn.out] attn_output={attn_output.shape}")
    return attn_output



# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------


def make_causal_mask(
    seq_len: int,
    past_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Создаёт float causal mask формы (1, 1, seq_len, past_len + seq_len).
    0 — токен участвует в attention, -inf — нет.
    """
    total_len = past_len + seq_len
    # (seq_len, total_len) — True там где можно смотреть
    mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=device)
    # Используем torch.tril для создания маски без медленных python-циклов по GPU
    mask = torch.tril(mask, diagonal=past_len)
    min_val = torch.finfo(dtype).min
    float_mask = torch.zeros(seq_len, total_len, dtype=dtype, device=device)
    float_mask = float_mask.masked_fill(~mask, min_val)
    return float_mask[None, None, :, :]  # (1, 1, seq_len, total_len)


# ---------------------------------------------------------------------------
# LlamaMLP — дословно из источника
# ---------------------------------------------------------------------------


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # gate_proj: (hidden_size → intermediate_size) = (2048 → 5632), SwiGLU gate
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        # up_proj:   (hidden_size → intermediate_size) = (2048 → 5632), SwiGLU value
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.mlp_bias
        )
        # down_proj: (intermediate_size → hidden_size) = (5632 → 2048), обратная проекция
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )
        # print(f"[MLP init] hidden={config.hidden_size} intermediate={config.intermediate_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU MLP (Noam Shazeer, 2020): замена обычного FFN с лучшей практикой.

        ОБЫЧНЫЙ FFN:
          out = W2 · ReLU(W1 · x)
          — один линейный слой с нелинейностью

        SwiGLU:
          gate = SiLU(W_gate · x)      # "ворота" — что пропустить
          val  = W_up   · x            # "значения" — что несём
          out  = W_down · (gate * val)  # поэлементное произведение + проекция

          Идея: gate динамически масштабирует каждый нейрон — можно
          "заглушить" ненужные признаки. Это мощнее ReLU (который просто
          обнуляет < 0), т.к. SiLU(x) = x·σ(x) непрерывно и гладко.

        ЗАЧЕМ SiLU вместо ReLU:
          ReLU имеет "мёртвые нейроны" (градиент=0 при x<0 навсегда).
          SiLU: x·σ(x) = x/(1+e^{-x}) — гладкая, ненулевые градиенты
          даже при x<0. Практически всегда работает лучше.

        ПОЧЕМУ intermediate_size=5632:
          Обычно FFN расширяет в 4× (2048×4=8192), но SwiGLU добавляет
          третий (gate) слой — параметров становится втрое.
          Чтобы сохранить число параметров ≈ как у обычного 2-слойного FFN,
          ширину берут ≈ 8/3 от hidden: 2048 × 8/3 ≈ 5461 → округлили до 5632.
        """
        # x: (bsz, seq_len, 2048)
        # gate_proj(x): (bsz, seq_len, 5632)  — SiLU gate
        # up_proj(x):   (bsz, seq_len, 5632)  — value
        # произведение: (bsz, seq_len, 5632)
        # down_proj:    (bsz, seq_len, 2048)
        # print(f"[MLP.in] x={x.shape}")
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        # print(f"[MLP.out] out={out.shape}")
        return out



# ---------------------------------------------------------------------------
# LlamaAttention — дословно из источника, без декораторов
# ---------------------------------------------------------------------------


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim           # 64   (hidden_size / num_heads = 2048/32)
        self.num_heads = config.num_attention_heads          # 32  (Q heads)
        self.num_key_value_heads = config.num_key_value_heads  # 4   (KV heads, GQA)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 8 (Q/KV group ratio)
        self.scaling = self.head_dim**-0.5        # 1/sqrt(64) ≈ 0.125

        # q_proj: (hidden_size → num_heads * head_dim) = (2048 → 32*64=2048)
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        # k_proj: (hidden_size → num_kv_heads * head_dim) = (2048 → 4*64=256)  ← GQA: меньше Q
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        # v_proj: (hidden_size → num_kv_heads * head_dim) = (2048 → 4*64=256)  ← GQA: меньше Q
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        # o_proj: (num_heads * head_dim → hidden_size) = (2048 → 2048)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        # print(f"[LlamaAttention init] head_dim={self.head_dim} num_heads={self.num_heads} "
        #       f"num_kv_heads={self.num_key_value_heads} kv_groups={self.num_key_value_groups}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[DynamicKVCache] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        # hidden_states: (bsz, seq_len, hidden_size)
        #   prefill: (1, 32, 2048)   decode: (1, 1, 2048)
        # print(f"[Attn.in] hidden_states={hidden_states.shape}")
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # query_states: (bsz, num_heads, seq_len, head_dim)
        #   prefill: (1, 32, 32, 64)  decode: (1, 32, 1, 64)
        # print(f"[Attn] q={query_states.shape}")
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # key_states before cache: (bsz, num_kv_heads, seq_len, head_dim)
        #   prefill: (1, 4, 32, 64)   decode: (1, 4, 1, 64)
        # print(f"[Attn] k_before_cache={key_states.shape}")
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # value_states: same shape as key_states
        # print(f"[Attn] v_before_cache={value_states.shape}")

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )
            # key_states after cache: (bsz, num_kv_heads, total_seq_len, head_dim)
            #   prefill: (1, 4, 32, 64)   decode step N: (1, 4, 32+N, 64)
            # print(f"[Attn] k_after_cache={key_states.shape}")

        attn_output = eager_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            self.num_key_value_groups,
        )
        # attn_output: (bsz, num_heads, seq_len, head_dim)
        #   prefill: (1, 32, 32, 64)  decode: (1, 32, 1, 64)
        # print(f"[Attn] attn_out_before_reshape={attn_output.shape}")

        attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()
        # attn_output: (bsz, seq_len, num_heads*head_dim) = (bsz, seq_len, hidden_size)
        #   prefill: (1, 32, 2048)   decode: (1, 1, 2048)
        attn_output = self.o_proj(attn_output)
        # attn_output: (bsz, seq_len, hidden_size)  prefill: (1, 32, 2048)  decode: (1, 1, 2048)
        # print(f"[Attn.out] attn_output={attn_output.shape}")
        return attn_output


# ---------------------------------------------------------------------------
# LlamaDecoderLayer — дословно из источника
# ---------------------------------------------------------------------------


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        # layernorm перед attention: нормализует по hidden_size=2048
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # layernorm перед MLP: нормализует по hidden_size=2048
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # print(f"[DecoderLayer {layer_idx} init] hidden={config.hidden_size}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[DynamicKVCache] = None,
    ) -> torch.Tensor:
        """
        Один декодер-слой: Pre-LN + Attention + Residual → Pre-LN + MLP + Residual.

        СТРУКТУРА Pre-LN (используется в LLaMA, в отличие от оригинального Transformer):

          ┌─────────────────────────────────────────────┐
          │  x → RMSNorm → Attention → + residual → x' │
          │  x'→ RMSNorm → MLP      → + residual → out │
          └─────────────────────────────────────────────┘

        ЗАЧЕМ RESIDUAL CONNECTIONS (He et al. 2016):
          Без них градиент затухает через глубокие сети (vanishing gradient).
          Residual даёт градиенту "прямой путь" к ранним слоям:
            dL/dx = dL/dx' · (1 + dAttn/dx)
          Слагаемое "1" гарантирует, что градиент всегда ≥ 1 по норме.

        ЗАЧЕМ Pre-LN, а не Post-LN (оригинальный Transformer):
          Post-LN: x' = LayerNorm(x + Sublayer(x))
            — нестабильно при большом lr, нужен warmup.
          Pre-LN:  x' = x + Sublayer(LayerNorm(x))
            — нормируем ВХОД в sublayer, а не выход.
            — градиенты стабильнее, можно train без warmup.
            — но чуть хуже для очень глубоких сетей (накапливается масштаб).

        ЗАЧЕМ RMSNorm вместо LayerNorm:
          LayerNorm: вычитает mean и делит на std → 2 прохода по данным.
          RMSNorm:   только делит на RMS = √(Σx²/d) → быстрее ≈ на 10-15%.
          В LLM практика показала, что mean-centering почти не нужен.
        """
        # hidden_states: (bsz, seq_len, hidden_size)
        #   prefill: (1, 32, 2048)   decode: (1, 1, 2048)
        # print(f"[Layer {self.self_attn.layer_idx}] hidden_states.in={hidden_states.shape}")

        # --- блок 1: Self-Attention ---
        residual = hidden_states                                      # сохраняем для residual
        hidden_states = self.input_layernorm(hidden_states)           # Pre-LN: (bsz, seq, 2048)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )                                                             # (bsz, seq_len, 2048)
        hidden_states = residual + hidden_states                      # residual connection
        # print(f"[Layer {self.self_attn.layer_idx}] after_attn+residual={hidden_states.shape}")

        # --- блок 2: MLP ---
        residual = hidden_states                                      # сохраняем для residual
        hidden_states = self.post_attention_layernorm(hidden_states)  # Pre-LN: (bsz, seq, 2048)
        hidden_states = self.mlp(hidden_states)
        # mlp: gate_proj → (bsz, seq_len, 5632), down_proj → (bsz, seq_len, 2048)
        hidden_states = residual + hidden_states                      # residual connection
        # print(f"[Layer {self.self_attn.layer_idx}] after_mlp+residual={hidden_states.shape}")
        return hidden_states  # (bsz, seq_len, 2048) — форма не меняется



# ---------------------------------------------------------------------------
# LlamaModel — дословно из источника
# ---------------------------------------------------------------------------


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        # embed_tokens: (vocab_size, hidden_size) = (32000, 2048)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 22 одинаковых decoder слоя, hidden_states проходят через все неизменной формы
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # финальный RMSNorm после всех слоёв: (hidden_size,) = (2048,)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.config = config
        # print(f"[LlamaModel init] vocab={config.vocab_size} hidden={config.hidden_size} "
        #       f"layers={config.num_hidden_layers}")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicKVCache] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        # input_ids:     (bsz, seq_len)              prefill: (1, 32)   decode: (1, 1)
        # inputs_embeds: (bsz, seq_len, hidden_size)  prefill: (1, 32, 2048)  decode: (1, 1, 2048)
        # print(f"[Model] input_ids={input_ids.shape}  inputs_embeds={inputs_embeds.shape}")

        # Позиции
        past_len = past_key_value.get_seq_length() if past_key_value is not None else 0
        seq_len = inputs_embeds.shape[1]

        if position_ids is not None:
            # Явные position_ids переопределяют автоматическое вычисление.
            # Используется generate_greedy при batched decode чтобы передать
            # правильные позиции для каждого сиквенса в батче независимо.
            pass  # position_ids уже задан
        elif attention_mask is not None and attention_mask.ndim == 2:
            # Для батча с left-padding нужно вычислять position_ids из маски,
            # иначе pad-токены сдвигают позиции реальных токенов.
            # Пример: mask=[0,0,1,1,1,1] → cumsum=[0,0,1,2,3,4] → -1 → [-1,-1,0,1,2,3]
            # masked_fill pad-позиций на 1 (arbitrary, они всё равно будут замаскированы).
            total_len = past_len + seq_len
            mask_for_pos = attention_mask[:, :total_len]  # (bsz, total_len)
            position_ids = mask_for_pos.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(mask_for_pos == 0, 1)
            # берём только позиции текущих seq_len токенов
            position_ids = position_ids[:, -seq_len:]  # (bsz, seq_len)
        else:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=input_ids.device
            ).unsqueeze(0)  # (1, seq_len)

        # position_ids: (bsz, seq_len)  prefill: (bsz, 32)  decode: (bsz, 1)
        # print(f"[Model] position_ids={position_ids}  past_len={past_len}")

        # Causal mask
        causal_mask = make_causal_mask(
            seq_len=seq_len,
            past_len=past_len,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        # causal_mask: (1, 1, seq_len, total_len)  prefill: (1, 1, 32, 32)  decode: (1, 1, 1, 33+N)
        # print(f"[Model] causal_mask={causal_mask.shape}")

        # Применяем padding mask если есть (2D mask -> добавляем -inf на pad токенах)
        if attention_mask is not None and attention_mask.ndim == 2:
            # attention_mask: (bsz, total_len), 1=реальный токен, 0=pad
            bsz = inputs_embeds.shape[0]
            total_len = past_len + seq_len
            pad_mask = attention_mask[:, :total_len].to(
                dtype=torch.bool
            )  # (bsz, total_len)
            # (bsz, 1, 1, total_len)
            pad_mask = pad_mask[:, None, None, :]
            min_val = torch.finfo(inputs_embeds.dtype).min
            causal_mask = causal_mask.expand(bsz, 1, seq_len, total_len).clone()
            causal_mask = causal_mask.masked_fill(~pad_mask, min_val)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        # cos/sin: (1, seq_len, head_dim)  prefill: (1, 32, 64)  decode: (1, 1, 64)
        # print(f"[Model] cos={position_embeddings[0].shape}")

        hidden_states = inputs_embeds  # (bsz, seq_len, 2048)
        for decoder_layer in self.layers:  # 22 слоя — hidden_states неизменной формы
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )

        hidden_states = self.norm(hidden_states)
        # hidden_states: (bsz, seq_len, 2048)  prefill: (1, 32, 2048)  decode: (1, 1, 2048)
        # print(f"[Model.out] hidden_states={hidden_states.shape}")
        return hidden_states


# ---------------------------------------------------------------------------
# LlamaForCausalLM
# ---------------------------------------------------------------------------


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicKVCache] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        logits = self.lm_head(hidden_states)
        return logits  # (bsz, seq_len, vocab_size)

    @classmethod
    def from_pretrained(
        cls, model_id_or_path: str, dtype: torch.dtype = torch.bfloat16
    ) -> "LlamaForCausalLM":
        """
        Загружает модель из HuggingFace Hub или локального пути.
        Требует: safetensors, huggingface_hub (для скачивания).
        """
        from huggingface_hub import snapshot_download

        # Если это путь — используем как есть, иначе скачиваем
        if os.path.isdir(model_id_or_path):
            model_dir = model_id_or_path
        else:
            model_dir = snapshot_download(model_id_or_path)

        # Читаем конфиг
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = LlamaConfig.from_dict(config_dict)

        model = cls(config)

        # Загружаем веса
        import glob

        weight_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if weight_files:
            from safetensors.torch import load_file

            state_dict = {}
            for wf in weight_files:
                state_dict.update(load_file(wf))
        else:
            # Fallback: pytorch .bin
            bin_files = sorted(glob.glob(os.path.join(model_dir, "*.bin")))
            state_dict = {}
            for bf in bin_files:
                state_dict.update(torch.load(bf, map_location="cpu", weights_only=True))

        # Преобразуем ключи: в нашей модели нет префикса 'model.' у lm_head,
        # а у transformers нет — структура совпадает.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(
                f"[WARN] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if unexpected:
            print(
                f"[WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
            )

        model = model.to(dtype=dtype)
        return model

    @torch.inference_mode()
    def generate_greedy(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Greedy decoding с KV cache. Поддерживает bsz > 1.

        Параметры:
          attention_mask    — (bsz, prompt_len), 1=реальный токен 0=pad.
                             Нужен при bsz>1 с left-padding чтобы pad-токены
                             не участвовали в attention во время prefill.
                             Передаётся ТОЛЬКО на prefill-шаге — на decode
                             каждый шаг уже 1 реальный токен без padding.
          eos_token_id=None → используем config.eos_token_id
          eos_token_id=-1   → не останавливаться по EOS (для бенчмарков)
          pad_token_id=None → используем eos_token_id как pad

        Батчевая логика:
          Каждый сиквенс в батче заканчивается независимо.
          После того как сиквенс i попал на EOS, все последующие
          токены для него заменяются на pad_token_id.
          Генерация заканчивается когда ВСЕ сиквенсы завершены.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        # -1 — sentinel "никогда не останавливаться" (для бенчмарков)
        stop_on_eos = eos_token_id != -1

        if pad_token_id is None:
            pad_token_id = eos_token_id if stop_on_eos else 0

        device = input_ids.device
        bsz = input_ids.shape[0]
        generated = input_ids.clone()  # (bsz, prompt_len)
        cache = DynamicKVCache()

        # done[i] = True если сиквенс i уже встретил EOS
        done = torch.zeros(bsz, dtype=torch.bool, device=device)

        # Вычисляем длину реального промпта для каждого сиквенса (без pad),
        # чтобы правильно считать позиции при decode.
        if attention_mask is not None:
            real_prompt_len = attention_mask.sum(dim=-1)  # (bsz,) — число реальных токенов
        else:
            real_prompt_len = torch.full((bsz,), input_ids.shape[1], device=device)

        # Prefill: прогоняем весь prompt сразу
        # attention_mask нужен чтобы pad-токены не учитывались в prefill attention
        logits = self.forward(generated, attention_mask=attention_mask, past_key_value=cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (bsz, 1)

        if stop_on_eos:
            just_done = (next_token.squeeze(-1) == eos_token_id)  # (bsz,)
            # заменяем токены для уже завершённых (на случай если EOS пришёл на prefill)
            next_token[done] = pad_token_id
            done = done | just_done

        generated = torch.cat([generated, next_token], dim=1)

        # next_decode_pos[i] = позиция следующего decode-токена для сиквенса i
        # Для неpadded батча: = real_prompt_len. После каждого decode-шага +1.
        # Именно это передаём в position_ids чтобы избежать смещения из-за pad в KV-cache.
        next_decode_pos = real_prompt_len.clone()  # (bsz,)

        # decode_attn_mask используется чтобы масковать pad-позиции в KV-cache при decode.
        # Начинаем с исходной маски промпта, и добавляем 1 для каждого нового токена.
        # Без этого decode-шаги будут аттендить на pad-ключи и получат другие результаты
        # по сравнению с одиночной генерацией (где падов нет).
        if attention_mask is not None:
            decode_attn_mask = attention_mask.clone()  # (bsz, prompt_len)
            ones_col = torch.ones(bsz, 1, device=device, dtype=decode_attn_mask.dtype)
            decode_attn_mask = torch.cat([decode_attn_mask, ones_col], dim=1)  # первый decode-токен
        else:
            decode_attn_mask = None

        # Декодируем токен за токеном
        for _ in range(max_new_tokens - 1):
            if stop_on_eos and done.all():
                break

            # Явно передаём position_ids для каждого сиквенса чтобы учесть padding
            decode_pos_ids = next_decode_pos.unsqueeze(1)  # (bsz, 1)
            # decode_attn_mask маскирует pad-ключи которые остались в KV-cache от prefill
            logits = self.forward(
                next_token,
                attention_mask=decode_attn_mask,
                position_ids=decode_pos_ids,
                past_key_value=cache,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (bsz, 1)

            if stop_on_eos:
                just_done = (next_token.squeeze(-1) == eos_token_id)  # (bsz,)
                # сиквенсы которые уже закончились — заменяем на pad
                next_token[done] = pad_token_id
                done = done | just_done

            generated = torch.cat([generated, next_token], dim=1)
            next_decode_pos += 1  # следующий токен будет на позицию дальше

            # Расширяем маску: новый токен реальный для всех сиквенсов
            if decode_attn_mask is not None:
                ones_col = torch.ones(bsz, 1, device=device, dtype=decode_attn_mask.dtype)
                decode_attn_mask = torch.cat([decode_attn_mask, ones_col], dim=1)

        return generated







# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_demo():
    from transformers import AutoTokenizer

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Загружаем токенизатор {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Загружаем модель {MODEL_ID}...")
    model = LlamaForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = model.to(DEVICE)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\nPrompt:\n{prompt}\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    output_ids = model.generate_greedy(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("Сгенерированный текст:")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))


# ---------------------------------------------------------------------------
# Unit test: проверяем что вывод совпадает с библиотечной transformers версией
# ---------------------------------------------------------------------------


def test_matches_transformers(max_new_tokens: int = 256):
    """
    Прогоняет одинаковый промпт через:
      1. нашу LlamaForCausalLM (greedy)
      2. transformers LlamaForCausalLM (greedy, do_sample=False)

    Сравнивает сгенерированные токены побитно. Тест падает если хоть один токен отличается.
    """
    import transformers as hf
    from transformers import AutoTokenizer

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32

    print("=" * 60)
    print("TEST: сравнение с transformers LlamaForCausalLM (greedy)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    # --- Наша модель ---
    print("\n[1/2] Загружаем нашу модель...")
    our_model = LlamaForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
    our_model = our_model.to(DEVICE)
    our_model.eval()

    with torch.inference_mode():
        our_output = our_model.generate_greedy(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
    our_new_tokens = our_output[0, prompt_len:]
    print(f"   Наша модель сгенерировала {our_new_tokens.shape[0]} токенов")
    print(f"   Текст: {tokenizer.decode(our_new_tokens, skip_special_tokens=False)!r}")

    # Освобождаем память
    del our_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --- Transformers модель ---
    print("\n[2/2] Загружаем transformers модель...")
    hf_model = hf.LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation="eager",  # наша реализация тоже eager — нужно совпадение
    )
    hf_model.eval()

    with torch.inference_mode():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    hf_new_tokens = hf_output[0, prompt_len:]
    print(f"   HF модель сгенерировала {hf_new_tokens.shape[0]} токенов")
    print(f"   Текст: {tokenizer.decode(hf_new_tokens, skip_special_tokens=False)!r}")

    del hf_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --- Сравнение ---
    print("\n--- Сравнение ---")
    min_len = min(our_new_tokens.shape[0], hf_new_tokens.shape[0])
    our_cmp = our_new_tokens[:min_len].cpu()
    hf_cmp = hf_new_tokens[:min_len].cpu()

    match = (our_cmp == hf_cmp).all().item()
    same_len = our_new_tokens.shape[0] == hf_new_tokens.shape[0]

    if not same_len:
        print(
            f"[WARN] Разная длина: наша={our_new_tokens.shape[0]}, hf={hf_new_tokens.shape[0]}"
        )

    if match and same_len:
        print("✅ PASS — все токены совпадают побитно!")
    else:
        # Найдём первое расхождение
        for i in range(min_len):
            if our_cmp[i] != hf_cmp[i]:
                print(f"❌ FAIL — первое расхождение на позиции {i}:")
                print(
                    f"   наша  = {our_cmp[i].item()} ({tokenizer.decode([our_cmp[i].item()])!r})"
                )
                print(
                    f"   hf    = {hf_cmp[i].item()} ({tokenizer.decode([hf_cmp[i].item()])!r})"
                )
                break
        assert False, "Токены не совпадают — см. вывод выше"

    return match and same_len


# ---------------------------------------------------------------------------
# Тест батчевой генерации
# ---------------------------------------------------------------------------


def test_batched_generation():
    """
    Проверяет батчевую генерацию для bsz=1,2,3,4,5.
    Для каждого размера батча:
      1. Берём первые bsz промптов (разной длины → left-padding)
      2. Генерируем батчем
      3. Сравниваем каждый сиквенс с одиночной генерацией
      4. Проверяем что после EOS идут только pad-токены
    """
    from transformers import AutoTokenizer

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32  # fp32 для детерминизма

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = LlamaForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE)
    model = model.to(DEVICE).eval()

    eos_id = tokenizer.eos_token_id
    pad_id = eos_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    # 5 промптов явно разной длины чтобы разные степени padding
    all_prompts = [
        "The capital of France is",                              # 6 токенов
        "Once upon a time",                                      # 4 токена
        "In the beginning",                                      # 3 токена
        "To be or not to be, that is the question",             # 10 токенов
        "Hi",                                                    # 1 токен
    ]

    MAX_NEW = 20
    grand_ok = True

    for bsz in range(1, 6):
        prompts = all_prompts[:bsz]
        print(f"\n{'='*60}")
        print(f"  BSZ={bsz}  промптов: {bsz}")
        print(f"{'='*60}")

        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        print(f"  input_ids shape: {input_ids.shape}")

        with torch.inference_mode():
            # батчевая генерация
            batch_out = model.generate_greedy(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )

            # одиночные генерации для сравнения
            single_outs = []
            for i in range(bsz):
                real_len = int(attention_mask[i].sum().item())
                prompt_i = input_ids[i, -real_len:].unsqueeze(0)
                out_i = model.generate_greedy(
                    prompt_i,
                    max_new_tokens=MAX_NEW,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )
                single_outs.append(out_i[0])

        bsz_ok = True
        for i in range(bsz):
            real_len = int(attention_mask[i].sum().item())
            batch_new = batch_out[i, input_ids.shape[1]:].cpu()
            single_new = single_outs[i][real_len:].cpu()

            min_len = min(batch_new.shape[0], single_new.shape[0])
            match = (batch_new[:min_len] == single_new[:min_len]).all().item()

            status = "✅" if match else "❌"
            print(f"  [seq {i}] {status} prompt={prompts[i]!r:30s}  "
                  f"batch={tokenizer.decode(batch_new[:8], skip_special_tokens=True)!r}...")
            if not match:
                bsz_ok = False
                grand_ok = False
                print(f"         single={tokenizer.decode(single_new[:8], skip_special_tokens=True)!r}...")

        # EOS padding check
        for i in range(bsz):
            new_tokens = batch_out[i, input_ids.shape[1]:].cpu().tolist()
            if eos_id in new_tokens:
                eos_pos = new_tokens.index(eos_id)
                after = new_tokens[eos_pos + 1:]
                if not all(t == pad_id for t in after):
                    print(f"  [seq {i}] ❌ после EOS есть не-pad токены!")
                    bsz_ok = False
                    grand_ok = False

        print(f"  → {'✅ PASS' if bsz_ok else '❌ FAIL'}")

    print(f"\n{'='*60}")
    print("  ИТОГ:", "✅ ВСЕ ТЕСТЫ ПРОШЛИ" if grand_ok else "❌ ЕСТЬ ОШИБКИ")
    print(f"{'='*60}")
    return grand_ok

# ---------------------------------------------------------------------------
# Speed benchmark: проверяем что KV cache реально ускоряет генерацию
# ---------------------------------------------------------------------------


def benchmark_speed(max_new_tokens: int = 128):
    """
    Замеряет tokens/sec для:
      1. Нашей модели с KV cache (generate_greedy)
      2. Нашей модели БЕЗ кеша (полный пересчёт каждый шаг)
      3. HF модели (для справки)
    """
    import time
    import transformers as hf
    from transformers import AutoTokenizer

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16

    print("=" * 60)
    print(f"BENCHMARK: {max_new_tokens} токенов, dtype={DTYPE}, device={DEVICE}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]

    model = LlamaForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE)
    model.eval()

    # --- 1. Наша модель WITH KV cache ---
    # warmup
    with torch.inference_mode():
        model.generate_greedy(input_ids, max_new_tokens=10)
    torch.cuda.synchronize() if DEVICE == "cuda" else None

    t0 = time.perf_counter()
    with torch.inference_mode():
        # eos_token_id=-1 — sentinel "не останавливаться по EOS"
        out = model.generate_greedy(input_ids, max_new_tokens=max_new_tokens, eos_token_id=-1)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t1 = time.perf_counter()
    n_gen = out.shape[1] - input_ids.shape[1]
    speed_cache = n_gen / (t1 - t0)
    print(
        f"\n[1] Наша модель  WITH cache: {speed_cache:.1f} tok/s  ({n_gen} токенов за {t1-t0:.2f}s)"
    )

    # --- 2. Наша модель WITHOUT KV cache (наивно: полный forward на всём контексте каждый шаг) ---
    @torch.inference_mode()
    def generate_no_cache(model, input_ids, max_new_tokens):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = model.forward(generated)  # NO cache — всегда полный контекст
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    # warmup
    generate_no_cache(model, input_ids, max_new_tokens=5)
    torch.cuda.synchronize() if DEVICE == "cuda" else None

    t0 = time.perf_counter()
    out_nc = generate_no_cache(model, input_ids, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t1 = time.perf_counter()
    n_gen_nc = out_nc.shape[1] - input_ids.shape[1]
    speed_no_cache = n_gen_nc / (t1 - t0)
    print(
        f"[2] Наша модель WITHOUT cache: {speed_no_cache:.1f} tok/s  ({n_gen_nc} токенов за {t1-t0:.2f}s)"
    )
    print(f"    Ускорение от кеша: {speed_cache / speed_no_cache:.2f}x")

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # --- 3. HF модель для справки ---
    hf_model = hf.LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE, attn_implementation="eager"
    )
    hf_model.eval()

    with torch.inference_mode():
        hf_model.generate(input_ids, max_new_tokens=10, do_sample=False)
    torch.cuda.synchronize() if DEVICE == "cuda" else None

    t0 = time.perf_counter()
    with torch.inference_mode():
        hf_out = hf_model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=None,  # отключаем EOS для честного замера
        )
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    t1 = time.perf_counter()
    n_gen_hf = hf_out.shape[1] - input_ids.shape[1]
    speed_hf = n_gen_hf / (t1 - t0)
    print(
        f"[3] HF модель   (для справки): {speed_hf:.1f} tok/s  ({n_gen_hf} токенов за {t1-t0:.2f}s)"
    )

    del hf_model

    # Результаты на RTX 5090 / TinyLlama-1.1B / bfloat16 / eager / 1024 токена:
    #   [1] WITH cache:    191.8 tok/s  (3.39x быстрее без кеша)
    #   [2] WITHOUT cache:  56.5 tok/s
    #   [3] HF eager:      171.1 tok/s  (наша реализация на 12% быстрее HF)

# ---------------------------------------------------------------------------
# Shape tracer: выводит размерности тензоров для первых N токенов
# ---------------------------------------------------------------------------


def trace_shapes(max_steps: int = 3):
    """
    Запускает генерацию и логирует shape каждого тензора через forward-хуки.
    Печатает только первые max_steps шагов (prefill + decode), затем молчит.
    Запуск: python tiny_llama_clean.py --trace
    """
    from transformers import AutoTokenizer

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = LlamaForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi!"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)

    # --- Состояние трейсера ---
    step_counter = {"n": 0}  # шаг (prefill=0, decode1=1, decode2=2, ...)

    def fmt(t):
        return f"{tuple(t.shape)}"

    def make_hook(label):
        def hook(module, inp, out):
            n = step_counter["n"]
            if n >= max_steps:
                return
            # inp — tuple тензоров
            in_shapes = ", ".join(fmt(x) for x in inp if isinstance(x, torch.Tensor))
            # out может быть тензором или tuple
            if isinstance(out, torch.Tensor):
                out_shape = fmt(out)
            elif isinstance(out, (tuple, list)):
                out_shape = "(" + ", ".join(
                    fmt(x) for x in out if isinstance(x, torch.Tensor)
                ) + ")"
            else:
                out_shape = str(type(out))
            print(f"  [{label}]  in=({in_shapes})  out={out_shape}")
        return hook

    # Регистрируем хуки на ключевых модулях первого слоя + внешних модулях
    hooks = []
    layer0 = model.model.layers[0]

    modules_to_trace = {
        "embed_tokens":                   model.model.embed_tokens,
        "L0.input_layernorm":             layer0.input_layernorm,
        "L0.attn.q_proj":                 layer0.self_attn.q_proj,
        "L0.attn.k_proj":                 layer0.self_attn.k_proj,
        "L0.attn.v_proj":                 layer0.self_attn.v_proj,
        "L0.attn.o_proj":                 layer0.self_attn.o_proj,
        "L0.post_attn_norm":              layer0.post_attention_layernorm,
        "L0.mlp.gate":                    layer0.mlp.gate_proj,
        "L0.mlp.down":                    layer0.mlp.down_proj,
        "final_norm":                     model.model.norm,
        "lm_head":                        model.lm_head,
    }

    for label, mod in modules_to_trace.items():
        hooks.append(mod.register_forward_hook(make_hook(label)))

    # Кастомизируем generate_greedy для трейса — перехватываем шаги
    eos_id = tokenizer.eos_token_id
    stop_on_eos = True
    generated = input_ids.clone()
    cache = DynamicKVCache()

    with torch.inference_mode():
        for step in range(max_steps):
            step_counter["n"] = step
            is_prefill = step == 0
            cur_input = generated if is_prefill else generated[:, -1:]
            label_step = "PREFILL" if is_prefill else f"DECODE step {step}"
            print(f"\n{'='*55}")
            print(f"  {label_step}  |  input_ids shape: {tuple(cur_input.shape)}")
            print(f"{'='*55}")

            logits = model.forward(cur_input, past_key_value=cache)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            print(f"  [next_token] shape={tuple(next_token.shape)}, "
                  f"token={next_token.item()!r} ({tokenizer.decode([next_token.item()])!r})")
            generated = torch.cat([generated, next_token], dim=1)
            if stop_on_eos and (next_token == eos_id).all():
                break

    # Показываем финальный KV cache
    print(f"\n{'='*55}")
    print("  KV CACHE после 3 шагов (слой 0):")
    if cache.key_cache:
        k = cache.key_cache[0]
        v = cache.value_cache[0]
        print(f"  key_cache[0]  : {tuple(k.shape)}  # (bsz, num_kv_heads, total_seq, head_dim)")
        print(f"  value_cache[0]: {tuple(v.shape)}")

    for h in hooks:
        h.remove()


if __name__ == "__main__":
    import sys

    if "--bench" in sys.argv:
        benchmark_speed(max_new_tokens=1024)
    elif "--trace" in sys.argv:
        trace_shapes(max_steps=3)
    elif "--batch" in sys.argv:
        test_batched_generation()
    else:
        test_matches_transformers()
