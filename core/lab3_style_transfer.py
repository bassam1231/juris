"""
lab3_style_transfer.py
======================
AI Legal Assistant & Case Simulator — Lab 3
Legal Document Style Transfer

Components:
    Part A — Style & Content Encoders
    Part B — Style Transfer Model & Training Pipeline
    Part C — LLM-Assisted Style Transfer (production-grade)
    Part D — Demo Runner

Transfer directions supported:
    US → UK legal style
    Formal → Plain English
    Aggressive → Neutral tone
    Short-form → Long-form clause expansion
    Passive → Active voice

Architecture:
    StyleEncoder   : Embedding → CNN → Global Pool → style vector
    ContentEncoder : Embedding → BiLSTM → attention → content vector
    Decoder        : content_vec + style_vec → LSTM → token logits
    Discriminator  : Classify style of generated text (adversarial style loss)

Requirements:
    pip install torch transformers langchain langchain-groq python-dotenv

Usage:
    python lab3_style_transfer.py
"""

import os
import re
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR   = Path("models/style_transfer")
DATA_DIR     = Path("data/style_pairs")
VOCAB_SIZE   = 5000
EMBED_DIM    = 128
HIDDEN_DIM   = 256
STYLE_DIM    = 64       # style vector dimension
CONTENT_DIM  = 192      # content vector dimension
SEQ_LEN      = 80
BATCH_SIZE   = 8
NUM_EPOCHS   = 40
LR           = 0.0003
CLIP_VALUE   = 0.5
TEMPERATURE  = 0.7

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Style categories
STYLE_IDS = {
    "us_formal":     0,
    "uk_formal":     1,
    "plain_english": 2,
    "neutral":       3,
    "active_voice":  4,
}

NUM_STYLES = len(STYLE_IDS)

print(f"[Lab 3] Device: {DEVICE}")
print(f"[Lab 3] Style dimensions: {NUM_STYLES} styles × {STYLE_DIM}d vectors")


# ══════════════════════════════════════════════════════════════════════════════
# STYLE CORPUS — Parallel text pairs for training
# ══════════════════════════════════════════════════════════════════════════════

STYLE_CORPUS = {

    # ── US Formal ↔ UK Formal ─────────────────────────────────────────────────
    "us_to_uk": [
        {
            "source": "The parties hereby agree that the contract shall be governed by the laws of the State of California.",
            "target": "The parties hereto agree that this agreement shall be governed by and construed in accordance with English law.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
        {
            "source": "Either party may terminate this agreement with thirty days written notice.",
            "target": "Either party may determine this agreement by giving not less than thirty days written notice to the other.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
        {
            "source": "The employee shall receive vacation time of two weeks per year.",
            "target": "The employee shall be entitled to annual leave of fourteen days in each holiday year.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
        {
            "source": "Company shall indemnify and hold harmless the contractor from all claims.",
            "target": "The Company shall indemnify the Contractor and keep the Contractor indemnified against all claims.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
        {
            "source": "This agreement constitutes the entire understanding between the parties.",
            "target": "This agreement constitutes the whole agreement between the parties relating to its subject matter.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
        {
            "source": "The contractor warrants that all work will be performed in a workmanlike manner.",
            "target": "The contractor warrants that all works shall be executed in a good and workmanlike manner.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
        {
            "source": "Any disputes arising under this contract shall be resolved by arbitration.",
            "target": "Any dispute arising out of or in connection with this contract shall be referred to arbitration.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
        {
            "source": "The landlord shall provide reasonable notice before entering the premises.",
            "target": "The landlord shall give reasonable prior written notice before entering upon the demised premises.",
            "source_style": "us_formal",
            "target_style": "uk_formal",
        },
    ],

    # ── Formal Legal ↔ Plain English ──────────────────────────────────────────
    "formal_to_plain": [
        {
            "source": "The party of the first part hereby covenants and agrees to indemnify, defend, and hold harmless the party of the second part from and against any and all claims, damages, losses, costs, and expenses.",
            "target": "The first party agrees to protect the second party from any claims, losses, or costs.",
            "source_style": "us_formal",
            "target_style": "plain_english",
        },
        {
            "source": "Notwithstanding any provision to the contrary contained herein, the obligations of the parties shall survive the termination or expiration of this agreement.",
            "target": "Even after this agreement ends, both parties still have to follow these rules.",
            "source_style": "us_formal",
            "target_style": "plain_english",
        },
        {
            "source": "The Licensee shall not, without the prior written consent of the Licensor, assign, transfer, charge, subcontract or deal in any other manner with all or any of its rights or obligations under this agreement.",
            "target": "You cannot give your rights or duties under this agreement to someone else without written permission.",
            "source_style": "us_formal",
            "target_style": "plain_english",
        },
        {
            "source": "In the event that any provision of this agreement is held to be invalid or unenforceable, the remaining provisions shall continue in full force and effect.",
            "target": "If any part of this agreement is found to be invalid, the rest of the agreement still applies.",
            "source_style": "us_formal",
            "target_style": "plain_english",
        },
        {
            "source": "The Tenant shall forthwith upon the determination of the tenancy yield up possession of the premises to the Landlord in good repair and condition.",
            "target": "When the rental period ends, you must return the property to the landlord in good condition.",
            "source_style": "uk_formal",
            "target_style": "plain_english",
        },
        {
            "source": "The parties hereto agree that time is of the essence with respect to all dates and deadlines set forth in this agreement.",
            "target": "All deadlines in this agreement must be met on time.",
            "source_style": "us_formal",
            "target_style": "plain_english",
        },
        {
            "source": "Either party may terminate this agreement forthwith by written notice to the other if the other commits a material breach of any of its obligations under this agreement.",
            "target": "Either party can end this agreement immediately if the other side seriously breaks the rules.",
            "source_style": "uk_formal",
            "target_style": "plain_english",
        },
        {
            "source": "The confidential information shall remain the sole and exclusive property of the disclosing party and nothing in this agreement shall be construed to grant any rights therein.",
            "target": "All confidential information belongs to the party that shared it. This agreement does not give you any ownership rights to it.",
            "source_style": "us_formal",
            "target_style": "plain_english",
        },
    ],

    # ── Aggressive ↔ Neutral tone ─────────────────────────────────────────────
    "aggressive_to_neutral": [
        {
            "source": "You are in clear violation of the contract and must immediately cease your illegal activities or face severe legal consequences.",
            "target": "This letter notifies you of a potential contract breach and requests your compliance within the specified timeframe.",
            "source_style": "us_formal",
            "target_style": "neutral",
        },
        {
            "source": "Your blatant disregard for our agreement is unacceptable and we will pursue every legal remedy available to us.",
            "target": "We are concerned about the current status of our agreement and wish to discuss a resolution.",
            "source_style": "us_formal",
            "target_style": "neutral",
        },
        {
            "source": "Failure to pay will result in immediate legal action and you will be held personally liable for all damages.",
            "target": "Payment is overdue and we request settlement within fourteen days to avoid further action.",
            "source_style": "us_formal",
            "target_style": "neutral",
        },
        {
            "source": "Your conduct constitutes a deliberate and willful breach that we will not tolerate.",
            "target": "The current situation appears inconsistent with the terms of our agreement.",
            "source_style": "us_formal",
            "target_style": "neutral",
        },
    ],

    # ── Passive ↔ Active voice ─────────────────────────────────────────────────
    "passive_to_active": [
        {
            "source": "The payment shall be made by the Client to the Service Provider within thirty days.",
            "target": "The Client shall pay the Service Provider within thirty days.",
            "source_style": "us_formal",
            "target_style": "active_voice",
        },
        {
            "source": "The report shall be submitted by the Contractor to the Company by the deadline.",
            "target": "The Contractor shall submit the report to the Company by the deadline.",
            "source_style": "us_formal",
            "target_style": "active_voice",
        },
        {
            "source": "Notice shall be given by the Landlord to the Tenant before the premises are entered.",
            "target": "The Landlord shall give the Tenant notice before entering the premises.",
            "source_style": "us_formal",
            "target_style": "active_voice",
        },
        {
            "source": "All materials shall be provided by the Company to the Employee for the completion of assigned tasks.",
            "target": "The Company shall provide the Employee with all materials needed to complete assigned tasks.",
            "source_style": "us_formal",
            "target_style": "active_voice",
        },
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# PART A — VOCABULARY & ENCODERS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PART A — Style & Content Encoders")
print("═"*60)


class StyleVocabulary:
    """Compact vocabulary for style transfer sequences."""

    SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self):
        self.token2idx = dict(self.SPECIAL)
        self.idx2token = {v: k for k, v in self.SPECIAL.items()}

    def build(self, texts: list[str], max_size: int = VOCAB_SIZE):
        freq = {}
        for text in texts:
            for token in self._tokenize(text):
                freq[token] = freq.get(token, 0) + 1

        sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        for token, _ in sorted_tokens[:max_size - len(self.SPECIAL)]:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx]   = token

        print(f"  [Vocab] {len(self.token2idx)} tokens built")

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"([.,;:!?()])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def encode(self, text: str, max_len: int = SEQ_LEN) -> list[int]:
        tokens = [2] + [self.token2idx.get(t, 1) for t in self._tokenize(text)] + [3]
        tokens = tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        skip = set(self.SPECIAL.values()) if skip_special else set()
        tokens = [self.idx2token.get(i, "<UNK>") for i in ids if i not in skip]
        text   = " ".join(tokens)
        # Fix spacing around punctuation
        text   = re.sub(r" ([.,;:!?])", r"\1", text)
        return text

    @property
    def size(self) -> int:
        return len(self.token2idx)


class StyleEncoder(nn.Module):
    """
    Encodes the STYLE of a text sequence — capturing
    tone, formality, phrasing patterns — using CNN
    over character/word n-grams.

    CNN is well-suited for style because style is often
    captured in local patterns (phrase choices, word
    combinations) rather than long-range dependencies.

    Output: style_vector (batch, style_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int = EMBED_DIM,
        style_dim:  int = STYLE_DIM,
        kernel_sizes: list[int] = [2, 3, 4, 5],
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale CNN — captures patterns at different n-gram lengths
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, style_dim, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),   # global max pooling
            )
            for k in kernel_sizes
        ])

        # Project concatenated CNN outputs to style_dim
        self.proj = nn.Sequential(
            nn.Linear(style_dim * len(kernel_sizes), style_dim * 2),
            nn.LayerNorm(style_dim * 2),
            nn.ReLU(),
            nn.Linear(style_dim * 2, style_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
        Returns:
            style_vec: (batch, style_dim)
        """
        emb = self.embedding(token_ids)        # (batch, seq, embed)
        emb = emb.transpose(1, 2)              # (batch, embed, seq) for Conv1d

        # Apply each CNN filter and pool
        pooled = [conv(emb).squeeze(-1) for conv in self.convs]  # list of (batch, style_dim)
        concat = torch.cat(pooled, dim=1)       # (batch, style_dim * n_kernels)

        return self.proj(concat)               # (batch, style_dim)


class ContentEncoder(nn.Module):
    """
    Encodes the CONTENT (semantics/meaning) of a text sequence
    using a bidirectional LSTM with self-attention.

    Separating content from style is the key challenge in
    style transfer. The content encoder focuses on WHAT is
    being said, the style encoder on HOW it is being said.

    Output: content_vector (batch, content_dim)
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = EMBED_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        content_dim: int = CONTENT_DIM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size    = embed_dim,
            hidden_size   = hidden_dim,
            num_layers    = 2,
            batch_first   = True,
            bidirectional = True,
            dropout       = 0.2,
        )

        # Self-attention over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Project to content_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, content_dim),
            nn.LayerNorm(content_dim),
            nn.ReLU(),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len)
        Returns:
            content_vec: (batch, content_dim)
        """
        emb           = self.embedding(token_ids)      # (batch, seq, embed)
        lstm_out, _   = self.lstm(emb)                 # (batch, seq, hidden*2)

        # Self-attention: weight each position by importance
        attn_scores = self.attention(lstm_out)         # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attended     = (attn_weights * lstm_out).sum(dim=1)  # (batch, hidden*2)

        return self.proj(attended)                     # (batch, content_dim)


# ══════════════════════════════════════════════════════════════════════════════
# PART B — STYLE TRANSFER MODEL & TRAINING
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PART B — Style Transfer Model & Training")
print("═"*60)


class StyleTransferDecoder(nn.Module):
    """
    Decodes a content vector + target style vector into a
    new token sequence in the target style.

    At each decoding step:
        input = [token_embedding; content_vec; style_vec]
        → LSTM → output projection → token logits
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int = EMBED_DIM,
        hidden_dim:  int = HIDDEN_DIM,
        content_dim: int = CONTENT_DIM,
        style_dim:   int = STYLE_DIM,
        seq_len:     int = SEQ_LEN,
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.hidden_dim  = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Initial hidden state from content + style
        self.init_hidden = nn.Sequential(
            nn.Linear(content_dim + style_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

        # LSTM input = token_embed + content_vec + style_vec
        self.lstm = nn.LSTM(
            input_size  = embed_dim + content_dim + style_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, vocab_size),
        )

    def forward(
        self,
        content_vec: torch.Tensor,   # (batch, content_dim)
        style_vec:   torch.Tensor,   # (batch, style_dim)
        target_ids:  torch.Tensor,   # (batch, seq_len) for teacher forcing
    ) -> torch.Tensor:
        """
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = content_vec.size(0)

        # Init hidden from content + style
        combined = torch.cat([content_vec, style_vec], dim=1)
        h0 = self.init_hidden(combined).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)

        # Expand content and style to match sequence length
        content_exp = content_vec.unsqueeze(1).expand(-1, self.seq_len, -1)
        style_exp   = style_vec.unsqueeze(1).expand(-1, self.seq_len, -1)

        # Token embeddings
        token_embs = self.token_embedding(target_ids)  # (batch, seq, embed)

        # Concatenate token + content + style at each step
        lstm_input = torch.cat([token_embs, content_exp, style_exp], dim=2)

        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        logits      = self.output_proj(lstm_out)

        return logits

    def generate(
        self,
        content_vec:  torch.Tensor,
        style_vec:    torch.Tensor,
        temperature:  float = TEMPERATURE,
        max_len:      int   = SEQ_LEN,
    ) -> torch.Tensor:
        """Autoregressive decoding."""
        self.eval()
        batch_size = content_vec.size(0)

        combined   = torch.cat([content_vec, style_vec], dim=1)
        h0         = self.init_hidden(combined).unsqueeze(0).repeat(2, 1, 1)
        c0         = torch.zeros_like(h0)
        hidden     = (h0, c0)

        current = torch.full((batch_size, 1), 2, dtype=torch.long, device=content_vec.device)
        generated = [current]

        with torch.no_grad():
            for _ in range(max_len - 1):
                emb    = self.token_embedding(current)   # (batch, 1, embed)
                c_step = content_vec.unsqueeze(1)
                s_step = style_vec.unsqueeze(1)
                inp    = torch.cat([emb, c_step, s_step], dim=2)

                out, hidden = self.lstm(inp, hidden)
                logits      = self.output_proj(out.squeeze(1)) / temperature
                probs       = torch.softmax(logits, dim=-1)
                current     = torch.multinomial(probs, 1)
                generated.append(current)

                # Stop if all sequences have generated EOS
                if (current == 3).all():
                    break

        return torch.cat(generated, dim=1)


class StyleTransferModel(nn.Module):
    """
    Full style transfer model combining:
        StyleEncoder   → style_vec
        ContentEncoder → content_vec
        Decoder        → output sequence in target style

    Training objective:
        reconstruction_loss + style_loss + content_preservation_loss
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.style_encoder   = StyleEncoder(vocab_size)
        self.content_encoder = ContentEncoder(vocab_size)
        self.decoder         = StyleTransferDecoder(vocab_size)

    def forward(
        self,
        source_ids:  torch.Tensor,   # (batch, seq_len) — source text
        target_ids:  torch.Tensor,   # (batch, seq_len) — target text (teacher forcing)
        target_style_ids: torch.Tensor,  # (batch, seq_len) — target style exemplar
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:      (batch, seq_len, vocab_size)
            content_vec: (batch, content_dim)
            style_vec:   (batch, style_dim)
        """
        content_vec     = self.content_encoder(source_ids)
        target_style_vec = self.style_encoder(target_style_ids)
        logits          = self.decoder(content_vec, target_style_vec, target_ids)
        return logits, content_vec, target_style_vec

    def transfer(
        self,
        source_ids:       torch.Tensor,
        target_style_ids: torch.Tensor,
        temperature:      float = TEMPERATURE,
    ) -> torch.Tensor:
        """Inference — transfer source to target style."""
        self.eval()
        with torch.no_grad():
            content_vec      = self.content_encoder(source_ids)
            target_style_vec = self.style_encoder(target_style_ids)
            generated        = self.decoder.generate(content_vec, target_style_vec, temperature)
        return generated


class StyleTransferDataset(Dataset):
    """
    Dataset of (source, target, target_style_exemplar) triples.
    """

    def __init__(self, vocab: StyleVocabulary, seq_len: int = SEQ_LEN):
        self.vocab   = vocab
        self.seq_len = seq_len
        self.samples = self._load()

    def _load(self) -> list[dict]:
        samples = []
        for direction, pairs in STYLE_CORPUS.items():
            for pair in pairs:
                # Encode source and target
                src  = self.vocab.encode(pair["source"], self.seq_len)
                tgt  = self.vocab.encode(pair["target"], self.seq_len)

                # Use another example of the target style as the style exemplar
                target_style = pair["target_style"]
                style_examples = [
                    p["target"] for p in STYLE_CORPUS.get(direction, [])
                    if p["target_style"] == target_style
                       and p["target"] != pair["target"]
                ]
                exemplar_text = random.choice(style_examples) if style_examples else pair["target"]
                exemplar      = self.vocab.encode(exemplar_text, self.seq_len)

                samples.append({
                    "source":   torch.tensor(src,      dtype=torch.long),
                    "target":   torch.tensor(tgt,      dtype=torch.long),
                    "exemplar": torch.tensor(exemplar, dtype=torch.long),
                    "direction": direction,
                })

        print(f"  [Dataset] {len(samples)} style transfer pairs loaded")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


@dataclass
class StyleTrainingMetrics:
    epoch:     int   = 0
    total_loss: float = 0.0
    recon_loss: float = 0.0
    style_loss: float = 0.0
    duration:  float = 0.0


class StyleTransferTrainer:
    """
    Trains the StyleTransferModel with a composite loss:

    total_loss = λ_recon * reconstruction_loss
               + λ_style * style_consistency_loss
               + λ_content * content_preservation_loss

    reconstruction_loss   : Cross-entropy on target token prediction
    style_consistency_loss: MSE between generated style vec and target style vec
    content_preservation  : MSE between source and output content vectors
    """

    LAMBDA_RECON   = 1.0
    LAMBDA_STYLE   = 0.3
    LAMBDA_CONTENT = 0.2

    def __init__(self, model: StyleTransferModel):
        self.model    = model.to(DEVICE)
        self.opt      = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        self.sched    = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=NUM_EPOCHS)
        self.recon_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.mse_fn   = nn.MSELoss()
        self.metrics: list[StyleTrainingMetrics] = []

    def _compute_loss(
        self,
        source_ids:  torch.Tensor,
        target_ids:  torch.Tensor,
        exemplar_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Compute composite loss."""
        logits, src_content, tgt_style = self.model(
            source_ids, target_ids, exemplar_ids
        )

        # Reconstruction loss — predict target tokens
        logits_flat  = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        targets_flat = target_ids[:, 1:].contiguous().view(-1)
        recon_loss   = self.recon_fn(logits_flat, targets_flat)

        # Style consistency — generated style should match target style
        gen_ids        = logits.argmax(dim=-1)
        gen_style      = self.model.style_encoder(gen_ids)
        style_loss     = self.mse_fn(gen_style, tgt_style.detach())

        # Content preservation — source content ≈ output content
        gen_content    = self.model.content_encoder(gen_ids)
        content_loss   = self.mse_fn(gen_content, src_content.detach())

        total = (
            self.LAMBDA_RECON   * recon_loss +
            self.LAMBDA_STYLE   * style_loss +
            self.LAMBDA_CONTENT * content_loss
        )

        return total, recon_loss.item(), style_loss.item()

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
        log_every:  int = 5,
        save_every: int = 10,
    ) -> list[StyleTrainingMetrics]:
        """Full training loop."""
        print(f"\n  [Trainer] Training style transfer — {num_epochs} epochs on {DEVICE}")
        print(f"  [Trainer] Dataset: {len(dataloader.dataset)} pairs | Batch: {BATCH_SIZE}")

        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            self.model.train()

            total_losses, recon_losses, style_losses = [], [], []

            for batch in dataloader:
                src  = batch["source"].to(DEVICE)
                tgt  = batch["target"].to(DEVICE)
                exmp = batch["exemplar"].to(DEVICE)

                self.opt.zero_grad()
                loss, recon, style = self._compute_loss(src, tgt, exmp)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_VALUE)
                self.opt.step()

                total_losses.append(loss.item())
                recon_losses.append(recon)
                style_losses.append(style)

            self.sched.step()

            m = StyleTrainingMetrics(
                epoch      = epoch,
                total_loss = sum(total_losses) / len(total_losses),
                recon_loss = sum(recon_losses) / len(recon_losses),
                style_loss = sum(style_losses) / len(style_losses),
                duration   = round(time.time() - t_start, 2),
            )
            self.metrics.append(m)

            if epoch % log_every == 0:
                print(
                    f"  Epoch [{epoch:3d}/{num_epochs}] | "
                    f"Total: {m.total_loss:.4f} | "
                    f"Recon: {m.recon_loss:.4f} | "
                    f"Style: {m.style_loss:.4f} | "
                    f"{m.duration:.1f}s"
                )

            if epoch % save_every == 0:
                path = MODELS_DIR / f"style_transfer_epoch_{epoch:03d}.pt"
                torch.save(self.model.state_dict(), path)
                print(f"  [Trainer] Checkpoint saved: {path.name}")

        # Save final
        torch.save(self.model.state_dict(), MODELS_DIR / "style_transfer_final.pt")
        print(f"\n  [Trainer] ✅ Training complete")
        return self.metrics


# ══════════════════════════════════════════════════════════════════════════════
# PART C — LLM-ASSISTED STYLE TRANSFER (Production-Grade)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PART C — LLM-Assisted Style Transfer")
print("═"*60)


STYLE_TRANSFER_PROMPTS = {

    "us_to_uk": {
        "instruction": """
You are a legal document specialist who converts US legal drafting style to UK legal drafting style.

Key differences to apply:
- Replace "shall" with "shall" (kept) but remove redundant "hereby"
- Replace "vacation" → "annual leave / holiday"
- Replace "attorney" → "solicitor" or "barrister"
- Replace "real estate" → "real property" or "land"
- Replace "lawsuit" → "claim" or "action"
- Replace "statute of limitations" → "limitation period"
- Use "whilst" instead of "while", "amongst" instead of "among"
- Replace "gotten" → "got"
- Add "hereto" and "herein" in appropriate places
- Replace "in the event that" → "if" (UK prefers brevity)
- Keep formal register throughout
""".strip(),
    },

    "formal_to_plain": {
        "instruction": """
You are a plain language legal editor. Convert formal legal text into clear, accessible English
that a non-lawyer can understand.

Rules:
- Replace Latin phrases with English equivalents (inter alia → among other things)
- Replace "hereinafter" → "from now on" or remove
- Replace "notwithstanding" → "despite" or "even if"
- Replace "pursuant to" → "under" or "following"
- Replace "shall" → "must" or "will"
- Replace "in the event that" → "if"
- Break long sentences into shorter ones (max 25 words each)
- Define legal terms in parentheses if unavoidable
- Use active voice where possible
- Keep all the legal substance — just make it readable
""".strip(),
    },

    "plain_to_formal": {
        "instruction": """
You are a legal document drafter. Convert plain English text into formal legal drafting style.

Rules:
- Use "shall" for obligations and "may" for permissions
- Add appropriate recitals and definitions
- Replace informal phrasing with legal equivalents
- Use passive constructions where appropriate
- Add precision: "reasonable" → "commercially reasonable", "soon" → "within 30 days"
- Use defined terms with capital letters (the "Company", the "Agreement")
- Add standard legal qualifiers where appropriate
- Maintain formal register throughout
""".strip(),
    },

    "aggressive_to_neutral": {
        "instruction": """
You are a professional legal editor specializing in dispute communications.
Convert aggressive or threatening legal language into professional, neutral tone.

Rules:
- Remove accusatory language ("you violated", "your illegal actions")
- Replace with factual descriptions ("the terms were not met", "the deadline passed")
- Remove emotional amplifiers ("blatant", "egregious", "deliberately")
- Replace threats with requests ("we will sue you" → "we may pursue available remedies")
- Keep all factual content and legal substance
- Maintain firmness without aggression
- Use "we understand" and "we request" framing
- Keep the deadline and demands, just tone them down
""".strip(),
    },

    "passive_to_active": {
        "instruction": """
You are a legal writing editor converting passive voice to active voice in legal documents.

Rules:
- Identify the actor/subject performing each action
- Restructure: "shall be paid by X" → "X shall pay"
- Restructure: "shall be submitted by Y" → "Y shall submit"
- Keep "shall" for obligations
- Maintain all legal substance and meaning
- Only change voice, not terminology or formality level
""".strip(),
    },
}


class LLMStyleTransfer:
    """
    Production-grade style transfer using Groq LLM.
    Used alongside the neural model for high-quality output.
    """

    def __init__(self):
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage
        self.HumanMessage  = HumanMessage
        self.SystemMessage = SystemMessage
        self.llm = ChatGroq(
            model       = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature = 0.15,
            api_key     = os.getenv("GROQ_API_KEY"),
        )

    def transfer(
        self,
        source_text:     str,
        transfer_type:   str,
        preserve_structure: bool = True,
    ) -> dict:
        """
        Apply style transfer to a legal text using the LLM.

        Args:
            source_text:    Original legal text
            transfer_type:  Key from STYLE_TRANSFER_PROMPTS
            preserve_structure: Whether to keep original structure

        Returns:
            dict with transferred text, explanation, and quality notes
        """
        if transfer_type not in STYLE_TRANSFER_PROMPTS:
            raise ValueError(f"Unknown transfer type: {transfer_type}")

        prompt_config = STYLE_TRANSFER_PROMPTS[transfer_type]
        structure_note = (
            "\nIMPORTANT: Preserve the original document structure, "
            "paragraph breaks, and clause numbering."
            if preserve_structure else ""
        )

        system_prompt = prompt_config["instruction"] + structure_note + """

Return your response in this exact format:
TRANSFERRED TEXT:
<the transferred text>

CHANGES MADE:
<bullet list of key changes applied>

QUALITY NOTES:
<any issues or limitations in the transfer>
"""

        response = self.llm.invoke([
            self.SystemMessage(content=system_prompt),
            self.HumanMessage(content=f"Convert this legal text:\n\n{source_text}"),
        ])

        raw = response.content.strip()

        # Parse response sections
        transferred = self._extract_section(raw, "TRANSFERRED TEXT:")
        changes     = self._extract_section(raw, "CHANGES MADE:")
        notes       = self._extract_section(raw, "QUALITY NOTES:")

        return {
            "source":        source_text,
            "transferred":   transferred or raw,
            "transfer_type": transfer_type,
            "changes":       changes,
            "quality_notes": notes,
        }

    def _extract_section(self, text: str, header: str) -> str:
        """Extract a named section from the response."""
        pattern = rf"{re.escape(header)}\s*\n(.+?)(?=\n[A-Z ]+:|\Z)"
        match   = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def batch_transfer(
        self,
        texts:         list[str],
        transfer_type: str,
    ) -> list[dict]:
        """Transfer multiple texts with the same style."""
        results = []
        for i, text in enumerate(texts):
            print(f"  [LLM Transfer] Processing {i+1}/{len(texts)}...")
            result = self.transfer(text, transfer_type)
            results.append(result)
            time.sleep(0.3)  # Rate limiting
        return results

    def evaluate_transfer(
        self,
        source:      str,
        transferred: str,
        transfer_type: str,
    ) -> dict:
        """Evaluate the quality of a style transfer."""
        response = self.llm.invoke([
            self.SystemMessage(content="""
You are a legal document quality evaluator. Score a style transfer on:
1. Content preservation (0-10): Is all original meaning retained?
2. Style accuracy (0-10): Does the output match the target style?
3. Fluency (0-10): Is the output grammatically correct and natural?
4. Legal validity (0-10): Does the output remain legally sound?

Return JSON only: {"content": N, "style": N, "fluency": N, "legal": N, "overall": N, "notes": "..."}
""".strip()),
            self.HumanMessage(content=f"""
Transfer type: {transfer_type}

ORIGINAL:
{source}

TRANSFERRED:
{transferred}
"""),
        ])

        try:
            raw    = re.sub(r"```json|```", "", response.content).strip()
            scores = json.loads(raw)
        except Exception:
            scores = {"content": 0, "style": 0, "fluency": 0, "legal": 0, "overall": 0, "notes": "Evaluation failed"}

        return scores


# ══════════════════════════════════════════════════════════════════════════════
# PART D — DEMO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    sep = "─" * 60
    print("\n\n" + "═"*60)
    print("  LAB 3 DEMO — Document Style Transfer")
    print("═"*60)

    # ── Build vocabulary ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 1: Building Style Vocabulary")
    print(sep)

    all_texts = []
    for pairs in STYLE_CORPUS.values():
        for pair in pairs:
            all_texts.append(pair["source"])
            all_texts.append(pair["target"])

    vocab = StyleVocabulary()
    vocab.build(all_texts, max_size=VOCAB_SIZE)

    # ── Build dataset & model ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 2: Initializing Style Transfer Model")
    print(sep)

    dataset    = StyleTransferDataset(vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = StyleTransferModel(vocab_size=vocab.size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters : {total_params:,}")
    print(f"  StyleEncoder     : CNN with kernel sizes [2,3,4,5]")
    print(f"  ContentEncoder   : BiLSTM + self-attention")
    print(f"  Decoder          : LSTM with content+style conditioning")

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  Step 3: Training Style Transfer Model ({NUM_EPOCHS} epochs)")
    print(sep)

    trainer = StyleTransferTrainer(model)
    metrics = trainer.train(dataloader, num_epochs=NUM_EPOCHS, log_every=10)

    # ── Neural style transfer demo ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 4: Neural Style Transfer Demo")
    print(sep)

    model.eval()
    test_pairs = STYLE_CORPUS["formal_to_plain"][:2]

    for pair in test_pairs:
        print(f"\n  SOURCE  : {pair['source'][:150]}")
        src_ids  = torch.tensor([vocab.encode(pair["source"])], dtype=torch.long).to(DEVICE)
        tgt_ids  = torch.tensor([vocab.encode(pair["target"])], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            generated = model.transfer(src_ids, tgt_ids, temperature=TEMPERATURE)

        output = vocab.decode(generated[0].cpu().tolist())
        print(f"  NEURAL  : {output[:150]}")
        print(f"  TARGET  : {pair['target'][:150]}")

    # ── LLM style transfer demo ────────────────────────────────────────────────
    if not os.getenv("GROQ_API_KEY"):
        print(f"\n{sep}")
        print("  Step 5: LLM Style Transfer Demo — SKIPPED (no GROQ_API_KEY)")
        print(sep)
    else:
        print(f"\n{sep}")
        print("  Step 5: LLM-Assisted Style Transfer Demo")
        print(sep)

        llm_transfer = LLMStyleTransfer()

        demo_texts = {
            "us_to_uk": "The parties hereby agree that the contract shall be governed by the laws of the State of New York. Either party may terminate this agreement with thirty days written notice. The employee shall receive two weeks vacation per year.",
            "formal_to_plain": "Notwithstanding any provision to the contrary contained herein, the indemnifying party shall defend, indemnify, and hold harmless the indemnified party from and against any and all claims, damages, losses, costs, and expenses, including reasonable attorneys fees.",
            "aggressive_to_neutral": "Your blatant disregard for our contractual obligations is completely unacceptable. You are in clear violation of the agreement and we will pursue every legal remedy available if you do not immediately cease your illegal conduct.",
            "passive_to_active": "Payment shall be made by the Client to the Service Provider within thirty days of invoice receipt. All deliverables shall be reviewed and approved by the Project Manager before acceptance.",
        }

        for transfer_type, text in demo_texts.items():
            print(f"\n  [{transfer_type.upper().replace('_', ' → ')}]")
            print(f"  SOURCE     : {text[:120]}...")

            result = llm_transfer.transfer(text, transfer_type)
            print(f"  TRANSFERRED: {result['transferred'][:200]}...")

            if result["changes"]:
                print(f"  CHANGES    : {result['changes'][:150]}...")

        # Evaluate one transfer
        print(f"\n  Evaluating quality of formal→plain transfer...")
        source_text = demo_texts["formal_to_plain"]
        result      = llm_transfer.transfer(source_text, "formal_to_plain")
        scores      = llm_transfer.evaluate_transfer(
            source_text, result["transferred"], "formal_to_plain"
        )

        print(f"\n  Quality Scores:")
        for metric, score in scores.items():
            if metric != "notes":
                print(f"    {metric:<20}: {score}/10")
        print(f"    notes               : {scores.get('notes', '')}")

    print("\n\n" + "═"*60)
    print("  LAB 3 COMPLETE ✅")
    print("═"*60)
    print("\n  Components built:")
    print("    ✅  StyleVocabulary     — legal text tokenizer")
    print("    ✅  StyleEncoder        — CNN multi-scale style extractor")
    print("    ✅  ContentEncoder      — BiLSTM + attention content encoder")
    print("    ✅  StyleTransferDecoder— conditioned LSTM decoder")
    print("    ✅  StyleTransferModel  — full encode-transfer-decode pipeline")
    print("    ✅  StyleTransferTrainer— composite loss training")
    print("    ✅  LLMStyleTransfer    — Groq LLM production transfer")
    print(f"\n  Transfer directions:")
    for k in STYLE_TRANSFER_PROMPTS:
        print(f"    → {k.replace('_', ' ')}")
    print("\n  Next: Run lab4_integrated_system.py")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_demo()
