"""
lab2_gan_document_generator.py
===============================
AI Legal Assistant & Case Simulator — Lab 2
GAN-based Legal Document Generation

Components:
    Part A — GAN Architecture for Legal Text Generation
    Part B — Synthetic Legal Dataset Generation
    Part C — Conditional Contract Type Generation
    Part D — Training Pipeline & Demo Runner

Architecture:
    Generator  : Embedding → LSTM → Linear → token logits
    Discriminator: Embedding → LSTM → Linear → real/fake score
    Training   : Alternating G/D updates with gradient clipping
    Conditioning: Contract type label embeddings injected into Generator

Requirements:
    pip install torch transformers python-dotenv

Usage:
    python lab2_gan_document_generator.py
"""

import os
import re
import json
import time
import random
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR      = Path("data/contracts")
SYNTH_DIR     = Path("data/synthetic")
MODELS_DIR    = Path("models/gan_checkpoints")
VOCAB_SIZE    = 5000       # vocabulary size
EMBED_DIM     = 128        # embedding dimension
HIDDEN_DIM    = 256        # LSTM hidden size
SEQ_LEN       = 64         # sequence length for training
BATCH_SIZE    = 16
LATENT_DIM    = 128        # noise vector size for generator
NUM_EPOCHS    = 50         # training epochs (increase for better quality)
LR_G          = 0.0002     # generator learning rate
LR_D          = 0.0001     # discriminator learning rate
CLIP_VALUE    = 0.5        # gradient clipping
TEMPERATURE   = 0.8        # generation temperature (higher = more creative)
NUM_CONTRACT_TYPES = 5     # number of conditional contract classes

# Contract type labels
CONTRACT_TYPES = {
    0: "employment",
    1: "nda",
    2: "service_agreement",
    3: "lease",
    4: "purchase",
}

CONTRACT_TYPE_IDS = {v: k for k, v in CONTRACT_TYPES.items()}

for d in [DATA_DIR, SYNTH_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[Lab 2] Device: {DEVICE}")
print(f"[Lab 2] Config: vocab={VOCAB_SIZE}, embed={EMBED_DIM}, hidden={HIDDEN_DIM}")


# ══════════════════════════════════════════════════════════════════════════════
# PART A — VOCABULARY & TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PART A — Vocabulary & Tokenizer")
print("═"*60)

# ── Sample legal contract corpus ───────────────────────────────────────────────
# Used when no real contract data is available in data/contracts/

SAMPLE_CONTRACTS = {

    "employment": [
        """EMPLOYMENT AGREEMENT This Employment Agreement is entered into as of the date
        signed below between the Company and the Employee. The Employee agrees to perform
        all duties assigned by the Company in a professional and timely manner. The Company
        agrees to compensate the Employee at the agreed upon salary rate. Either party may
        terminate this agreement with thirty days written notice. The Employee agrees to
        maintain confidentiality of all proprietary information. This agreement shall be
        governed by the laws of the applicable jurisdiction. The Employee acknowledges
        receipt of the company handbook and agrees to comply with all policies.""",

        """OFFER OF EMPLOYMENT The Company hereby offers employment to the candidate subject
        to the following terms and conditions. The position offered is full time and at will.
        Compensation shall include base salary and benefits as outlined in the attached
        schedule. The employee shall be entitled to paid vacation and sick leave pursuant
        to company policy. Performance reviews shall be conducted annually. The company
        reserves the right to modify job duties and compensation with reasonable notice.
        All work product created during employment shall be the exclusive property of
        the company under applicable work for hire provisions.""",
    ],

    "nda": [
        """NON DISCLOSURE AGREEMENT This Non Disclosure Agreement is entered into between
        the Disclosing Party and the Receiving Party. The Receiving Party agrees to hold
        in strict confidence all confidential information disclosed by the Disclosing Party.
        Confidential information means any information that is marked confidential or that
        reasonably should be understood to be confidential given the nature of the information.
        The Receiving Party shall not disclose confidential information to any third party
        without prior written consent. This obligation shall survive termination of this
        agreement for a period of three years. The Receiving Party shall use confidential
        information solely for the purpose of evaluating the potential business relationship.""",

        """MUTUAL NON DISCLOSURE AGREEMENT Both parties acknowledge that each may disclose
        to the other certain proprietary and confidential information for the purpose of
        exploring a potential business relationship. Each party agrees to protect the
        confidential information of the other party using the same degree of care it uses
        to protect its own confidential information but no less than reasonable care.
        Neither party shall reverse engineer analyze or decompile any confidential materials.
        All confidential information shall remain the property of the disclosing party.
        Upon request each party shall return or destroy all confidential materials.""",
    ],

    "service_agreement": [
        """SERVICE AGREEMENT This Service Agreement is made between the Service Provider
        and the Client. The Service Provider agrees to perform the services described in
        the attached Statement of Work. The Client agrees to pay the fees specified in
        the payment schedule. Services shall be performed in a professional and workmanlike
        manner consistent with industry standards. The Service Provider shall maintain
        adequate insurance coverage throughout the term of this agreement. Either party
        may terminate this agreement for cause upon written notice if the other party
        materially breaches any provision and fails to cure within thirty days of notice.
        The Service Provider shall indemnify and hold harmless the Client from claims
        arising from the negligence of the Service Provider.""",

        """CONSULTING SERVICES AGREEMENT This agreement sets forth the terms under which
        the Consultant will provide consulting services to the Company. The Consultant
        shall perform services as an independent contractor and not as an employee.
        The Consultant shall invoice the Company monthly for services rendered. Payment
        is due within thirty days of receipt of invoice. The Consultant retains the right
        to perform services for other clients provided there is no conflict of interest.
        All work product delivered under this agreement shall be owned by the Company upon
        full payment. The Consultant warrants that services will be performed competently.""",
    ],

    "lease": [
        """RESIDENTIAL LEASE AGREEMENT This Lease Agreement is entered into between the
        Landlord and the Tenant for the rental of the premises described herein. The Tenant
        agrees to pay monthly rent on the first day of each month. A security deposit equal
        to one month rent is due upon signing and shall be held in trust. The Tenant shall
        maintain the premises in clean and sanitary condition. The Landlord shall be
        responsible for major repairs and maintenance. The Tenant shall not sublet or
        assign this lease without written consent. Upon termination the Tenant shall
        restore the premises to its original condition less normal wear and tear.
        The Landlord shall provide twenty four hours notice before entering the premises.""",

        """COMMERCIAL LEASE AGREEMENT This Commercial Lease is made between Landlord and
        Tenant for the commercial premises. Tenant shall use the premises solely for
        lawful business purposes. Tenant shall pay base rent plus a proportionate share
        of operating expenses and property taxes as calculated annually. Tenant shall
        maintain the interior of the premises in good condition. Landlord shall maintain
        the structural elements roof and common areas. Tenant shall obtain and maintain
        commercial general liability insurance naming Landlord as additional insured.
        Any alterations or improvements shall require prior written approval from Landlord.""",
    ],

    "purchase": [
        """PURCHASE AGREEMENT This Purchase Agreement is made between the Seller and the
        Buyer for the sale of the goods or property described herein. The Buyer agrees to
        pay the purchase price in the manner and at the time specified. The Seller warrants
        that they have clear title to the property and the right to sell it. Risk of loss
        passes to the Buyer upon delivery. The Seller makes no warranties express or implied
        beyond those stated in this agreement. Any disputes arising from this agreement
        shall be resolved through binding arbitration. This agreement constitutes the entire
        understanding between the parties and supersedes all prior negotiations.""",

        """ASSET PURCHASE AGREEMENT This Agreement governs the purchase and sale of certain
        assets. Seller agrees to sell and Buyer agrees to purchase the assets free and
        clear of all liens and encumbrances except as specified herein. The purchase price
        shall be paid in full at closing. Seller represents and warrants that the assets
        are in good working condition and that Seller has the authority to sell them.
        Buyer shall conduct due diligence prior to closing and may terminate this agreement
        if due diligence reveals material undisclosed issues. The parties shall cooperate
        to complete the transfer of all necessary titles and registrations at closing.""",
    ],
}


class LegalVocabulary:
    """
    Word-level vocabulary built from legal contract text.
    Maps tokens ↔ integer indices.
    """

    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,   # beginning of sequence
        "<EOS>": 3,   # end of sequence
        "<SEP>": 4,   # separator between clauses
    }

    def __init__(self):
        self.token2idx: dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.idx2token: dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.freq:      dict[str, int] = {}
        self.built      = False

    def tokenize(self, text: str) -> list[str]:
        """Simple word-level tokenizer with legal text cleaning."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\.\,\;\:\'\-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def build(self, texts: list[str], max_vocab: int = VOCAB_SIZE):
        """Build vocabulary from a list of texts."""
        for text in texts:
            for token in self.tokenize(text):
                self.freq[token] = self.freq.get(token, 0) + 1

        # Sort by frequency, keep top max_vocab - len(special_tokens)
        sorted_tokens = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        n_special     = len(self.SPECIAL_TOKENS)

        for token, _ in sorted_tokens[:max_vocab - n_special]:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx]   = token

        self.built = True
        print(f"  [Vocab] Built vocabulary: {len(self.token2idx)} tokens")

    def encode(self, text: str, max_len: int = SEQ_LEN) -> list[int]:
        """Encode text to integer indices, padded to max_len."""
        tokens  = [self.SPECIAL_TOKENS["<BOS>"]] + \
                  [self.token2idx.get(t, self.SPECIAL_TOKENS["<UNK>"]) for t in self.tokenize(text)] + \
                  [self.SPECIAL_TOKENS["<EOS>"]]
        tokens  = tokens[:max_len]
        padding = [self.SPECIAL_TOKENS["<PAD>"]] * (max_len - len(tokens))
        return tokens + padding

    def decode(self, indices: list[int], skip_special: bool = True) -> str:
        """Decode integer indices back to text."""
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special else set()
        tokens      = [
            self.idx2token.get(i, "<UNK>")
            for i in indices
            if i not in special_ids
        ]
        return " ".join(tokens)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"token2idx": self.token2idx, "idx2token": self.idx2token}, f)
        print(f"  [Vocab] Saved to {path}")

    def load(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.token2idx = data["token2idx"]
        self.idx2token = data["idx2token"]
        self.built     = True
        print(f"  [Vocab] Loaded from {path} ({len(self.token2idx)} tokens)")


# ══════════════════════════════════════════════════════════════════════════════
# PART A — GAN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PART A — GAN Architecture")
print("═"*60)


class ContractGenerator(nn.Module):
    """
    Generator network for legal contract text generation.

    Architecture:
        noise_vector (latent_dim) + type_embedding
            → Linear projection → hidden state
            → LSTM (unrolled for seq_len steps)
            → Linear → vocab logits per timestep

    Conditioning:
        Contract type label is embedded and concatenated with
        the noise vector, allowing conditional generation of
        specific contract types (employment, NDA, lease, etc.)
    """

    def __init__(
        self,
        vocab_size:   int = VOCAB_SIZE,
        embed_dim:    int = EMBED_DIM,
        hidden_dim:   int = HIDDEN_DIM,
        latent_dim:   int = LATENT_DIM,
        seq_len:      int = SEQ_LEN,
        num_types:    int = NUM_CONTRACT_TYPES,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim

        # Contract type conditioning embedding
        self.type_embedding = nn.Embedding(num_types, embed_dim)

        # Project noise + type embedding → initial LSTM hidden state
        self.noise_proj = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
        )

        # LSTM for sequential token generation
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
        )

        # Token embedding for teacher-forced training input
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Output projection to vocabulary
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, vocab_size),
        )

    def forward(
        self,
        noise:       torch.Tensor,   # (batch, latent_dim)
        type_labels: torch.Tensor,   # (batch,) integer type ids
        input_ids:   Optional[torch.Tensor] = None,  # (batch, seq_len) for teacher forcing
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = noise.size(0)

        # Condition on contract type
        type_emb   = self.type_embedding(type_labels)          # (batch, embed_dim)
        combined   = torch.cat([noise, type_emb], dim=1)       # (batch, latent+embed)

        # Project to initial hidden state
        h0 = self.noise_proj(combined)                         # (batch, hidden)
        h0 = h0.unsqueeze(0).repeat(2, 1, 1)                  # (2, batch, hidden) — 2 LSTM layers
        c0 = torch.zeros_like(h0)

        # Token inputs — BOS token repeated for seq_len if no teacher forcing
        if input_ids is None:
            bos_id   = 2  # <BOS> token
            input_ids = torch.full(
                (batch_size, self.seq_len), bos_id,
                dtype=torch.long, device=noise.device
            )

        token_embs = self.token_embedding(input_ids)           # (batch, seq, embed)
        lstm_out, _ = self.lstm(token_embs, (h0, c0))         # (batch, seq, hidden)
        logits      = self.output_proj(lstm_out)               # (batch, seq, vocab)

        return logits

    def generate(
        self,
        noise:        torch.Tensor,
        type_labels:  torch.Tensor,
        temperature:  float = TEMPERATURE,
        top_k:        int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive generation — each token feeds back as next input.

        Returns:
            generated_ids: (batch, seq_len) integer token ids
        """
        self.eval()
        batch_size = noise.size(0)

        # Initialize hidden state from noise + type
        type_emb  = self.type_embedding(type_labels)
        combined  = torch.cat([noise, type_emb], dim=1)
        h0        = self.noise_proj(combined).unsqueeze(0).repeat(2, 1, 1)
        c0        = torch.zeros_like(h0)

        # Start with BOS token
        current_token = torch.full(
            (batch_size, 1), 2, dtype=torch.long, device=noise.device
        )
        generated = [current_token]
        hidden    = (h0, c0)

        with torch.no_grad():
            for _ in range(self.seq_len - 1):
                emb          = self.token_embedding(current_token)   # (batch, 1, embed)
                out, hidden  = self.lstm(emb, hidden)                # (batch, 1, hidden)
                logits       = self.output_proj(out.squeeze(1))      # (batch, vocab)

                # Temperature scaling
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
                    min_val        = top_k_vals[:, -1].unsqueeze(-1)
                    logits         = logits.masked_fill(logits < min_val, float('-inf'))

                probs         = torch.softmax(logits, dim=-1)
                current_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
                generated.append(current_token)

        return torch.cat(generated, dim=1)  # (batch, seq_len)


class ContractDiscriminator(nn.Module):
    """
    Discriminator network — classifies sequences as real or generated.

    Architecture:
        token_ids → Embedding → LSTM → pool → Linear → scalar score

    Also conditioned on contract type to distinguish real contracts
    of a given type from fake contracts of that same type.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim:  int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_types:  int = NUM_CONTRACT_TYPES,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.type_embedding  = nn.Embedding(num_types, embed_dim)

        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
            bidirectional = True,    # bidirectional for better context
        )

        # hidden_dim * 2 (bidirectional) + embed_dim (type conditioning)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        token_ids:   torch.Tensor,   # (batch, seq_len)
        type_labels: torch.Tensor,   # (batch,)
    ) -> torch.Tensor:
        """
        Returns:
            scores: (batch, 1) — higher = more likely real
        """
        token_embs = self.token_embedding(token_ids)         # (batch, seq, embed)
        type_emb   = self.type_embedding(type_labels)        # (batch, embed)

        lstm_out, (h_n, _) = self.lstm(token_embs)          # (batch, seq, hidden*2)

        # Mean pool over sequence
        pooled = lstm_out.mean(dim=1)                        # (batch, hidden*2)

        # Concatenate type embedding for conditioning
        combined = torch.cat([pooled, type_emb], dim=1)     # (batch, hidden*2 + embed)
        scores   = self.classifier(combined)                 # (batch, 1)

        return scores


# ══════════════════════════════════════════════════════════════════════════════
# PART B — DATASET & SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PART B — Dataset & Synthetic Data Generation")
print("═"*60)


class LegalContractDataset(Dataset):
    """
    PyTorch Dataset for legal contract training sequences.
    Each item is a (sequence, contract_type_id) pair.
    """

    def __init__(
        self,
        vocab:    LegalVocabulary,
        seq_len:  int = SEQ_LEN,
        augment:  bool = True,
    ):
        self.vocab   = vocab
        self.seq_len = seq_len
        self.augment = augment
        self.samples: list[tuple[list[int], int]] = []
        self._load_samples()

    def _load_samples(self):
        """Load and encode all training samples."""
        print("  [Dataset] Loading training samples...")

        # Try loading from disk first
        if DATA_DIR.exists():
            for file_path in DATA_DIR.glob("*.txt"):
                try:
                    text      = file_path.read_text(encoding="utf-8")
                    type_name = file_path.stem.split("_")[0]
                    type_id   = CONTRACT_TYPE_IDS.get(type_name, 0)
                    encoded   = self.vocab.encode(text, self.seq_len)
                    self.samples.append((encoded, type_id))
                except Exception as e:
                    print(f"  [Dataset] Warning: could not load {file_path.name}: {e}")

        # Fall back to sample contracts
        if not self.samples:
            print("  [Dataset] Using sample contract corpus...")
            for type_name, contracts in SAMPLE_CONTRACTS.items():
                type_id = CONTRACT_TYPE_IDS.get(type_name, 0)
                for contract in contracts:
                    # Sliding window augmentation — multiple overlapping sequences
                    tokens = self.vocab.tokenize(contract)
                    if self.augment:
                        step = self.seq_len // 2
                        for start in range(0, max(1, len(tokens) - self.seq_len), step):
                            chunk   = " ".join(tokens[start:start + self.seq_len])
                            encoded = self.vocab.encode(chunk, self.seq_len)
                            self.samples.append((encoded, type_id))
                    else:
                        encoded = self.vocab.encode(contract, self.seq_len)
                        self.samples.append((encoded, type_id))

        print(f"  [Dataset] Total samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, type_id = self.samples[idx]
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(type_id, dtype=torch.long),
        )


class SyntheticDatasetGenerator:
    """
    Generates synthetic legal contract datasets using the trained GAN.
    Outputs text files organized by contract type.
    """

    def __init__(self, generator: ContractGenerator, vocab: LegalVocabulary):
        self.generator = generator.to(DEVICE)
        self.vocab     = vocab

    def generate_batch(
        self,
        contract_type: str,
        batch_size:    int = 8,
        temperature:   float = TEMPERATURE,
    ) -> list[str]:
        """Generate a batch of synthetic contracts of a given type."""
        type_id = CONTRACT_TYPE_IDS.get(contract_type, 0)

        noise       = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        type_labels = torch.full((batch_size,), type_id, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            generated_ids = self.generator.generate(
                noise, type_labels, temperature=temperature
            )

        documents = []
        for i in range(batch_size):
            ids  = generated_ids[i].cpu().tolist()
            text = self.vocab.decode(ids)
            text = self._post_process(text, contract_type)
            documents.append(text)

        return documents

    def _post_process(self, text: str, contract_type: str) -> str:
        """Clean and format generated text into a document structure."""
        # Capitalize first letter of sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.capitalize() for s in sentences if len(s.strip()) > 3]
        text      = " ".join(sentences)

        # Add contract header based on type
        headers = {
            "employment":       "EMPLOYMENT AGREEMENT",
            "nda":              "NON-DISCLOSURE AGREEMENT",
            "service_agreement":"SERVICE AGREEMENT",
            "lease":            "LEASE AGREEMENT",
            "purchase":         "PURCHASE AGREEMENT",
        }
        header = headers.get(contract_type, "LEGAL AGREEMENT")
        return f"{header}\n\n{text}\n\n[SYNTHETIC DOCUMENT - Generated by Juris GAN]"

    def generate_dataset(
        self,
        samples_per_type: int = 10,
        temperature:      float = TEMPERATURE,
    ) -> dict[str, list[str]]:
        """Generate a full synthetic dataset across all contract types."""
        print("\n  [SyntheticGen] Generating synthetic legal dataset...")
        dataset = {}

        for type_name in CONTRACT_TYPES.values():
            print(f"  [SyntheticGen] Generating {samples_per_type} × {type_name}...")
            documents = []
            batches   = (samples_per_type + 7) // 8  # ceil(n/8)

            for _ in range(batches):
                batch_size = min(8, samples_per_type - len(documents))
                batch      = self.generate_batch(type_name, batch_size, temperature)
                documents.extend(batch)

            dataset[type_name] = documents[:samples_per_type]

            # Save to disk
            type_dir = SYNTH_DIR / type_name
            type_dir.mkdir(exist_ok=True)
            for i, doc in enumerate(dataset[type_name]):
                (type_dir / f"synthetic_{i+1:03d}.txt").write_text(doc, encoding="utf-8")

        total = sum(len(v) for v in dataset.values())
        print(f"  [SyntheticGen] ✅ Dataset complete — {total} documents across {len(dataset)} types")
        print(f"  [SyntheticGen] Saved to: {SYNTH_DIR}")
        return dataset

    def evaluate_quality(self, documents: list[str]) -> dict:
        """Basic quality metrics for generated documents."""
        if not documents:
            return {}

        word_counts    = [len(d.split()) for d in documents]
        unique_counts  = [len(set(d.lower().split())) for d in documents]
        legal_terms    = ["agreement", "party", "shall", "hereby", "whereas",
                          "pursuant", "notwithstanding", "indemnify", "warrant",
                          "terminate", "covenant", "obligation", "liability"]

        legal_term_counts = [
            sum(1 for t in legal_terms if t in d.lower())
            for d in documents
        ]

        return {
            "count":               len(documents),
            "avg_word_count":      round(sum(word_counts) / len(word_counts), 1),
            "avg_unique_words":    round(sum(unique_counts) / len(unique_counts), 1),
            "avg_legal_terms":     round(sum(legal_term_counts) / len(legal_term_counts), 1),
            "vocabulary_richness": round(
                sum(u/w for u,w in zip(unique_counts, word_counts)) / len(documents), 3
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PART C — TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  PART C — GAN Training Pipeline")
print("═"*60)


@dataclass
class TrainingMetrics:
    """Tracks GAN training metrics per epoch."""
    epoch:      int   = 0
    g_loss:     float = 0.0
    d_loss:     float = 0.0
    d_real_acc: float = 0.0
    d_fake_acc: float = 0.0
    duration:   float = 0.0


class GANTrainer:
    """
    Alternating GAN training loop with:
    - Binary cross-entropy adversarial loss
    - Gradient clipping for stability
    - Label smoothing for discriminator
    - Checkpoint saving every N epochs
    """

    def __init__(
        self,
        generator:     ContractGenerator,
        discriminator: ContractDiscriminator,
        vocab:         LegalVocabulary,
    ):
        self.G       = generator.to(DEVICE)
        self.D       = discriminator.to(DEVICE)
        self.vocab   = vocab
        self.metrics: list[TrainingMetrics] = []

        self.opt_G = optim.Adam(self.G.parameters(), lr=LR_G, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=LR_D, betas=(0.5, 0.999))

        self.criterion = nn.BCEWithLogitsLoss()

        # Learning rate schedulers
        self.sched_G = optim.lr_scheduler.StepLR(self.opt_G, step_size=20, gamma=0.5)
        self.sched_D = optim.lr_scheduler.StepLR(self.opt_D, step_size=20, gamma=0.5)

    def _train_discriminator(
        self,
        real_seqs:   torch.Tensor,
        type_labels: torch.Tensor,
    ) -> tuple[float, float, float]:
        """One discriminator update step."""
        self.D.train()
        self.opt_D.zero_grad()
        batch_size = real_seqs.size(0)

        # Label smoothing — real labels = 0.9 instead of 1.0
        real_labels = torch.full((batch_size, 1), 0.9, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)

        # Real samples
        d_real  = self.D(real_seqs, type_labels)
        loss_real = self.criterion(d_real, real_labels)

        # Fake samples — generate with random noise
        noise        = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_seqs    = self.G.generate(noise, type_labels)
        d_fake       = self.D(fake_seqs.detach(), type_labels)
        loss_fake    = self.criterion(d_fake, fake_labels)

        # Total discriminator loss
        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        nn.utils.clip_grad_norm_(self.D.parameters(), CLIP_VALUE)
        self.opt_D.step()

        # Accuracy metrics
        real_acc = (torch.sigmoid(d_real) > 0.5).float().mean().item()
        fake_acc = (torch.sigmoid(d_fake) < 0.5).float().mean().item()

        return d_loss.item(), real_acc, fake_acc

    def _train_generator(
        self,
        batch_size:  int,
        type_labels: torch.Tensor,
    ) -> float:
        """One generator update step."""
        self.G.train()
        self.opt_G.zero_grad()

        # Generate fake sequences
        noise     = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_seqs = self.G.generate(noise, type_labels)

        # Generator wants discriminator to output 1 (real) for fake sequences
        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        d_fake      = self.D(fake_seqs, type_labels)
        g_loss      = self.criterion(d_fake, real_labels)

        g_loss.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), CLIP_VALUE)
        self.opt_G.step()

        return g_loss.item()

    def train(
        self,
        dataloader:       DataLoader,
        num_epochs:       int = NUM_EPOCHS,
        save_every:       int = 10,
        n_critic:         int = 2,   # D steps per G step
        log_every:        int = 5,
    ) -> list[TrainingMetrics]:
        """
        Full GAN training loop.

        Args:
            dataloader:  PyTorch DataLoader with (sequence, type_id) pairs
            num_epochs:  Number of training epochs
            save_every:  Save checkpoints every N epochs
            n_critic:    Number of discriminator updates per generator update
            log_every:   Print progress every N epochs
        """
        print(f"\n  [Trainer] Starting GAN training — {num_epochs} epochs on {DEVICE}")
        print(f"  [Trainer] Dataset size: {len(dataloader.dataset)} samples")
        print(f"  [Trainer] Batch size: {BATCH_SIZE} | D steps per G step: {n_critic}")

        for epoch in range(1, num_epochs + 1):
            t_start = time.time()
            g_losses, d_losses, real_accs, fake_accs = [], [], [], []

            for batch_seqs, batch_types in dataloader:
                batch_seqs  = batch_seqs.to(DEVICE)
                batch_types = batch_types.to(DEVICE)
                batch_size  = batch_seqs.size(0)

                # Train discriminator n_critic times
                for _ in range(n_critic):
                    d_loss, real_acc, fake_acc = self._train_discriminator(
                        batch_seqs, batch_types
                    )
                    d_losses.append(d_loss)
                    real_accs.append(real_acc)
                    fake_accs.append(fake_acc)

                # Train generator once
                g_loss = self._train_generator(batch_size, batch_types)
                g_losses.append(g_loss)

            # Step schedulers
            self.sched_G.step()
            self.sched_D.step()

            # Record metrics
            metrics = TrainingMetrics(
                epoch      = epoch,
                g_loss     = sum(g_losses) / len(g_losses),
                d_loss     = sum(d_losses) / len(d_losses),
                d_real_acc = sum(real_accs) / len(real_accs),
                d_fake_acc = sum(fake_accs) / len(fake_accs),
                duration   = round(time.time() - t_start, 2),
            )
            self.metrics.append(metrics)

            if epoch % log_every == 0:
                print(
                    f"  Epoch [{epoch:3d}/{num_epochs}] | "
                    f"G: {metrics.g_loss:.4f} | "
                    f"D: {metrics.d_loss:.4f} | "
                    f"D_real: {metrics.d_real_acc:.3f} | "
                    f"D_fake: {metrics.d_fake_acc:.3f} | "
                    f"{metrics.duration:.1f}s"
                )

            # Save checkpoints
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

        print(f"\n  [Trainer] ✅ Training complete — {num_epochs} epochs")
        self.save_checkpoint(num_epochs, final=True)
        return self.metrics

    def save_checkpoint(self, epoch: int, final: bool = False):
        """Save model checkpoints."""
        suffix = "final" if final else f"epoch_{epoch:03d}"
        torch.save(self.G.state_dict(), MODELS_DIR / f"generator_{suffix}.pt")
        torch.save(self.D.state_dict(), MODELS_DIR / f"discriminator_{suffix}.pt")
        print(f"  [Trainer] Checkpoint saved: {suffix}")

    def load_checkpoint(self, path_g: Path, path_d: Path):
        """Load model weights from checkpoints."""
        self.G.load_state_dict(torch.load(path_g, map_location=DEVICE))
        self.D.load_state_dict(torch.load(path_d, map_location=DEVICE))
        print(f"  [Trainer] Loaded: {path_g.name} + {path_d.name}")

    def plot_metrics(self):
        """Print training metrics summary."""
        if not self.metrics:
            print("  [Trainer] No metrics to display.")
            return

        print("\n  Training Metrics Summary:")
        print(f"  {'Epoch':>6} {'G Loss':>8} {'D Loss':>8} {'D Real':>8} {'D Fake':>8}")
        print("  " + "─"*44)
        for m in self.metrics[::max(1, len(self.metrics)//10)]:
            print(f"  {m.epoch:>6} {m.g_loss:>8.4f} {m.d_loss:>8.4f} "
                  f"{m.d_real_acc:>8.3f} {m.d_fake_acc:>8.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# PART D — DEMO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    sep = "─" * 60
    print("\n\n" + "═"*60)
    print("  LAB 2 DEMO — GAN Document Generator")
    print("═"*60)

    # ── Build vocabulary ───────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 1: Building Vocabulary")
    print(sep)

    vocab = LegalVocabulary()
    all_texts = []
    for contracts in SAMPLE_CONTRACTS.values():
        all_texts.extend(contracts)
    vocab.build(all_texts, max_vocab=VOCAB_SIZE)

    vocab_path = MODELS_DIR / "vocabulary.pkl"
    vocab.save(vocab_path)

    # ── Build dataset ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 2: Building Dataset")
    print(sep)

    dataset    = LegalContractDataset(vocab, seq_len=SEQ_LEN, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"  [Dataset] {len(dataset)} samples | {len(dataloader)} batches")

    # ── Initialize models ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 3: Initializing GAN Models")
    print(sep)

    G = ContractGenerator(
        vocab_size = len(vocab.token2idx),
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        latent_dim = LATENT_DIM,
        seq_len    = SEQ_LEN,
        num_types  = NUM_CONTRACT_TYPES,
    )

    D = ContractDiscriminator(
        vocab_size = len(vocab.token2idx),
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        num_types  = NUM_CONTRACT_TYPES,
    )

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"  Generator     : {g_params:,} parameters")
    print(f"  Discriminator : {d_params:,} parameters")
    print(f"  Total         : {g_params + d_params:,} parameters")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  Step 4: Training GAN ({NUM_EPOCHS} epochs)")
    print(sep)

    trainer = GANTrainer(G, D, vocab)
    metrics = trainer.train(dataloader, num_epochs=NUM_EPOCHS, log_every=10)
    trainer.plot_metrics()

    # ── Generate synthetic documents ───────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 5: Generating Synthetic Contracts")
    print(sep)

    synth_gen = SyntheticDatasetGenerator(G, vocab)
    dataset   = synth_gen.generate_dataset(samples_per_type=4)

    print("\n  Sample generated documents:")
    for type_name, docs in dataset.items():
        print(f"\n  [{type_name.upper()}]")
        if docs:
            print(f"  {docs[0][:300]}...")

    # ── Quality evaluation ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 6: Quality Evaluation")
    print(sep)

    for type_name, docs in dataset.items():
        metrics_q = synth_gen.evaluate_quality(docs)
        print(f"\n  {type_name}:")
        for k, v in metrics_q.items():
            print(f"    {k:<25}: {v}")

    # ── Conditional generation demo ────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 7: Conditional Generation Demo")
    print(sep)

    print("\n  Generating one document per contract type (conditional):")
    for type_name in CONTRACT_TYPES.values():
        docs = synth_gen.generate_batch(type_name, batch_size=1, temperature=0.7)
        print(f"\n  Type: {type_name.upper()}")
        print(f"  {docs[0][:200]}...")

    print("\n\n" + "═"*60)
    print("  LAB 2 COMPLETE ✅")
    print("═"*60)
    print("\n  Components built:")
    print("    ✅  LegalVocabulary  — word-level tokenizer")
    print("    ✅  ContractGenerator — conditional LSTM GAN generator")
    print("    ✅  ContractDiscriminator — bidirectional LSTM discriminator")
    print("    ✅  LegalContractDataset — PyTorch dataset with augmentation")
    print("    ✅  GANTrainer — alternating training loop + checkpoints")
    print("    ✅  SyntheticDatasetGenerator — batch + dataset generation")
    print(f"\n  Outputs saved to:")
    print(f"    models/gan_checkpoints/ — GAN weights")
    print(f"    data/synthetic/         — synthetic contract dataset")
    print("\n  Next: Run lab3_style_transfer.py")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_demo()
