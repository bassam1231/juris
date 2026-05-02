# Juris — AI Legal Assistant & Case Simulator
### LLM + GAN Integration · 4 Labs

An end-to-end AI legal research system combining large language models with
generative adversarial networks for legal document generation and style transfer.

---

## Lab Structure

| Lab | File | Core Tech | What It Builds |
|-----|------|-----------|----------------|
| **Lab 1** | `lab1_legal_foundation.py` | Groq LLM, LangChain, FAISS | Legal chatbot · Prompt engineering · RAG pipeline · Bias evaluation |
| **Lab 2** | `lab2_gan_document_generator.py` | PyTorch GAN | Contract Generator · Discriminator · Synthetic dataset generation |
| **Lab 3** | `lab3_style_transfer.py` | PyTorch, CNN, BiLSTM | Style encoder · Content encoder · US↔UK · Formal↔Plain · Tone transfer |
| **Lab 4** | `lab4_integrated_system.py` | All + FastAPI | Unified pipeline · 5 REST endpoints · Frontend integration |

---

## Architecture

```
Lab 1 — Legal Foundation
├── LegalChatbot         (4 personas: general_counsel, contract_specialist, rights_advisor, case_simulator)
├── PromptEngineer       (zero-shot, few-shot, chain-of-thought, role-play)
├── LegalRAGPipeline     (FAISS vector store over legal documents)
└── BiasEvaluator        (gender, socioeconomic, racial, nationality bias testing)

Lab 2 — GAN Document Generator
├── ContractGenerator    (Conditional LSTM GAN — 5 contract types)
├── ContractDiscriminator(Bidirectional LSTM + type conditioning)
├── LegalContractDataset (PyTorch Dataset with sliding window augmentation)
└── SyntheticDatasetGen  (Batch generation + quality evaluation)

Lab 3 — Style Transfer
├── StyleEncoder         (Multi-scale CNN — captures phrasing patterns)
├── ContentEncoder       (BiLSTM + self-attention — captures meaning)
├── StyleTransferDecoder (Content + style conditioned LSTM)
└── LLMStyleTransfer     (Groq LLM production transfer + evaluation)

Lab 4 — Integrated System
├── LegalAdvisorModule   (Lab 1 integration)
├── ContractGeneratorModule (Lab 2 + LLM fallback)
├── StyleTransferModule  (Lab 3 integration)
├── JurisIntegratedPipeline (Orchestrator)
└── FastAPI Server       (5 endpoints)
```

---

## Project Structure

```
juris/
├── core/
│   ├── lab1_legal_foundation.py        ← Run standalone
│   ├── lab2_gan_document_generator.py  ← Run standalone
│   ├── lab3_style_transfer.py          ← Run standalone
│   ├── lab4_integrated_system.py       ← Run as API server
│   │
│   ├── data/
│   │   ├── contracts/                  ← Add contracts for GAN training
│   │   ├── style_pairs/                ← Style transfer pairs
│   │   ├── synthetic/                  ← GAN outputs saved here
│   │   └── vector_store/               ← Embedded legal knowledge base
│   │
│   └── models/
│       ├── gan_checkpoints/            ← Lab 2 GAN weights
│       └── style_transfer/             ← Lab 3 model weights
│
├── requirements.txt
├── .env.example
├── vercel.json
│
├── api/                                ← Vercel serverless (Lab 4 only) *not implemented yet*
│   ├── index.py                    
│   └── requirements.txt
│
└── frontend/
    └── index.html                      ← UI (3-column workbench)
```

---

## Quick Start

### Run each lab independently

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
cp .env.example .env
# Add your GROQ_API_KEY to .env

# Run labs in order
python lab1_legal_foundation.py
python lab2_gan_document_generator.py
python lab3_style_transfer.py
python lab4_integrated_system.py
```

### Run as API server (Lab 4)

```bash
uvicorn core.lab4_integrated_system:app --host 0.0.0.0 --port 8000 --reload
*http://localhost:8000/*
```

Then open `frontend/index.html` in your browser.

## API Endpoints (Lab 4)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/advise` | Legal Q&A + case simulation (Lab 1) |
| `POST` | `/api/generate` | Contract generation + style transfer (Labs 2+3) |
| `POST` | `/api/transfer` | Document style transfer only (Lab 3) |
| `POST` | `/api/full` | Full integrated pipeline (all labs) |
| `GET`  | `/api/health` | System health check |

---

## Frontend

Three-column legal workbench (`frontend/index.html`):
- **Left sidebar** — Pipeline mode selector (Lab 1–4) + contract type + transfer type
- **Center workspace** — Query input, examples, summary metrics cards
- **Right panel** — Tabbed output: Rights · Contract · Transfer · Pipeline Log + PDF export

Design: Off-white parchment + near-black ink + crimson accent. `Cormorant Garant` serif + `Courier Prime` monospace. Deliberately static — no distracting animations.

---

## GAN Architecture (Lab 2)

```
Generator:
  noise(128) + type_embedding(128) → Linear → h0
  h0 → 2-layer LSTM (256 hidden) → Linear → vocab_logits(5000)
  Inference: autoregressive with temperature + top-k sampling

Discriminator:
  token_ids → Embedding → 2-layer BiLSTM → mean_pool
  + type_embedding → Linear layers → real/fake score

Training:
  n_critic=2 (D updates per G update)
  Gradient clipping: 0.5
  Label smoothing: 0.9 for real labels
  StepLR: halve LR every 20 epochs
```

---

## Style Transfer Architecture (Lab 3)

```
StyleEncoder:  Embedding → CNN [k=2,3,4,5] → GlobalMaxPool → style_vec(64)
ContentEncoder: Embedding → BiLSTM → SelfAttention → content_vec(192)
Decoder:        [content_vec; style_vec] → LSTM → vocab_logits

Loss = λ_recon * CrossEntropy
     + λ_style * MSE(gen_style, target_style)
     + λ_content * MSE(gen_content, src_content)

Transfer modes: US↔UK · Formal↔Plain · Aggressive→Neutral · Passive→Active
```

---

## ⚠️ Disclaimer

Juris is an AI research system for **educational purposes only**. It does not constitute legal advice. Always consult a licensed attorney before taking legal action.

---

## Authors

Built by **Bassam Hassan**, **Mai Farahat**, **Mahmoud Maher**, **Mohand Sabry**, and **Islam Fouad** — Agentic AI Engineering portfolio project.
