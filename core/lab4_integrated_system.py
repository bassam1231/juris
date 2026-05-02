"""
lab4_integrated_system.py
=========================
AI Legal Assistant & Case Simulator — Lab 4
Integrated Legal AI System

Integrates all previous labs into a unified pipeline:
    Lab 1 → Legal chatbot + RAG + bias evaluation
    Lab 2 → GAN-based contract generation
    Lab 3 → Document style transfer

Pipeline modes:
    ADVISE      — Legal Q&A with RAG (Lab 1)
    SIMULATE    — Case outcome simulation (Lab 1)
    GENERATE    — GAN contract generation (Lab 2)
    TRANSFER    — Document style transfer (Lab 3)
    FULL        — All modes combined into one report

FastAPI Endpoints:
    POST /api/advise      — Legal Q&A + case simulation
    POST /api/generate    — GAN contract generation
    POST /api/transfer    — Style transfer
    POST /api/full        — Full integrated pipeline
    GET  /api/health      — System health check

Requirements:
    pip install torch transformers langchain langchain-groq
                langchain-community faiss-cpu sentence-transformers
                fastapi uvicorn python-dotenv

Usage:
    # Run standalone demo
    python lab4_integrated_system.py

    # Run as API server
    uvicorn lab4_integrated_system:app --host 0.0.0.0 --port 8000 --reload
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
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MODELS_DIR   = Path("models")
DATA_DIR     = Path("data")

VOCAB_SIZE   = 5000
EMBED_DIM    = 128
HIDDEN_DIM   = 256
STYLE_DIM    = 64
CONTENT_DIM  = 192
LATENT_DIM   = 128
SEQ_LEN      = 64
NUM_CONTRACT_TYPES = 5
NUM_STYLES   = 5
TEMPERATURE  = 0.8

CONTRACT_TYPES = {
    0: "employment", 1: "nda", 2: "service_agreement", 3: "lease", 4: "purchase"
}
CONTRACT_TYPE_IDS = {v: k for k, v in CONTRACT_TYPES.items()}

STYLE_TRANSFER_TYPES = [
    "us_to_uk", "formal_to_plain", "plain_to_formal",
    "aggressive_to_neutral", "passive_to_active"
]

print(f"[Lab 4] Integrated System starting on {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE — Legal case context passed through pipeline
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegalCase:
    """
    Shared state object passed through the integrated pipeline.
    Each module reads from and writes to specific fields.
    """
    # Input
    user_query:        str = ""
    jurisdiction:      str = ""
    contract_type:     str = "employment"
    transfer_type:     str = "formal_to_plain"
    source_document:   str = ""

    # Lab 1 outputs
    legal_domain:      str = ""
    key_facts:         list[str] = field(default_factory=list)
    rights_analysis:   str = ""
    case_outcome:      str = ""
    confidence:        int = 0
    rag_sources:       list[str] = field(default_factory=list)
    chat_history:      list[dict] = field(default_factory=list)

    # Lab 2 outputs
    generated_contract: str = ""
    generation_quality: dict = field(default_factory=dict)

    # Lab 3 outputs
    transferred_document: str = ""
    transfer_changes:     str = ""
    transfer_scores:      dict = field(default_factory=dict)

    # Pipeline metadata
    pipeline_log:  list[str] = field(default_factory=list)
    errors:        list[str] = field(default_factory=list)
    duration:      float = 0.0

    def log(self, module: str, msg: str):
        entry = f"[{module}] {msg}"
        self.pipeline_log.append(entry)
        print(f"  {entry}")

    def error(self, module: str, msg: str):
        entry = f"[{module} ERROR] {msg}"
        self.errors.append(entry)
        print(f"  ⚠  {entry}")


# ══════════════════════════════════════════════════════════════════════════════
# LEGAL KNOWLEDGE BASE (from Lab 1)
# ══════════════════════════════════════════════════════════════════════════════

LEGAL_KNOWLEDGE = [
    {
        "source": "employment_law",
        "tags":   ["employment", "termination", "discrimination", "wages", "harassment"],
        "content": (
            "Employment law governs employer-employee relationships. Key federal protections "
            "include Title VII (prohibits discrimination based on race, sex, religion, national "
            "origin), ADA (disability accommodations), ADEA (age 40+), and FLSA (minimum wage, "
            "overtime). Most US states follow at-will employment but wrongful termination occurs "
            "when firing violates protected rights, whistleblower laws, or employment contracts. "
            "File EEOC charges within 180-300 days of discrimination. The FLSA requires 1.5x pay "
            "for hours over 40/week. Workers can recover back wages plus liquidated damages."
        ),
    },
    {
        "source": "contract_law",
        "tags":   ["contract", "breach", "agreement", "damages", "remedy"],
        "content": (
            "A valid contract requires offer, acceptance, and consideration. Breach occurs when "
            "a party fails obligations without legal excuse. Remedies include compensatory damages, "
            "specific performance, and rescission. The statute of limitations for written contracts "
            "is typically 4-6 years. Common clauses: indemnification (shifting liability), "
            "limitation of liability (capping damages), force majeure (excusing performance for "
            "unforeseeable events), and arbitration clauses. Demand letters create paper trails "
            "and often resolve disputes before litigation."
        ),
    },
    {
        "source": "tenant_rights",
        "tags":   ["tenant", "landlord", "lease", "eviction", "deposit", "housing"],
        "content": (
            "Landlords must maintain habitable conditions (working plumbing, heating, structural "
            "safety). Security deposits must be returned within 14-30 days of move-out with "
            "itemized deductions — normal wear cannot be deducted. Eviction requires proper "
            "legal process: written notice, court filing, hearing, and order. Self-help evictions "
            "(lockouts, utility shutoffs) are illegal in all US states. Retaliation after tenant "
            "complaints is prohibited. Tenants may withhold rent or repair-and-deduct after "
            "proper notice of habitability failures."
        ),
    },
    {
        "source": "civil_rights",
        "tags":   ["police", "search", "miranda", "constitutional", "civil rights"],
        "content": (
            "Fourth Amendment protects against unreasonable searches — police generally need "
            "warrants for homes. You can refuse consent to vehicle searches. Fifth Amendment "
            "right to remain silent must be actively invoked: say 'I invoke my right to remain "
            "silent and want an attorney.' Sixth Amendment guarantees counsel — public defenders "
            "provided if needed. Section 1983 allows federal lawsuits against officials violating "
            "constitutional rights. Title VII, ADA, and Equal Protection Clause prohibit "
            "government discrimination. Document all incidents with dates and witnesses."
        ),
    },
    {
        "source": "consumer_protection",
        "tags":   ["consumer", "refund", "fraud", "debt collector", "credit"],
        "content": (
            "FTC Act prohibits deceptive trade practices. Credit card chargebacks available "
            "within 60 days under Fair Credit Billing Act. FDCPA prohibits debt collector "
            "harassment — collectors cannot call before 8am/after 9pm or use abusive language. "
            "FCRA gives right to dispute inaccurate credit information. Small claims court "
            "handles disputes up to $5,000-$20,000 without a lawyer. File complaints with "
            "FTC at reportfraud.ftc.gov, CFPB, and state attorney general. Many states allow "
            "treble damages and attorney fees for consumer protection violations."
        ),
    },
    {
        "source": "family_law",
        "tags":   ["divorce", "custody", "child", "domestic", "restraining order"],
        "content": (
            "Divorce property division uses community property (50/50 split — CA, TX, AZ) or "
            "equitable distribution (fair split — most states). Alimony based on marriage length, "
            "income disparity, and standard of living. Child custody: legal (decisions) and "
            "physical (residence) — courts apply best interests standard. Protective orders "
            "can be issued same-day for domestic violence; violation is criminal. National DV "
            "Hotline: 1-800-799-7233. Document abuse with dates, photos, witnesses."
        ),
    },
]


def retrieve_legal_knowledge(query: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Keyword-based retrieval from legal knowledge base."""
    words  = set(re.findall(r'\w+', query.lower()))
    scored = []

    for chunk in LEGAL_KNOWLEDGE:
        score = sum(
            len(set(re.findall(r'\w+', tag)) & words) * 2 +
            (3 if tag in query.lower() else 0)
            for tag in chunk["tags"]
        )
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:top_k]] or LEGAL_KNOWLEDGE[:2]

    context = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['content']}" for c in top
    )
    sources = [c["source"] for c in top]
    return context, sources


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — LEGAL ADVISOR (Lab 1 integration)
# ══════════════════════════════════════════════════════════════════════════════

class LegalAdvisorModule:
    """
    Integrates Lab 1 components:
    - Legal chatbot with RAG
    - Case outcome simulation
    - Conversation memory
    """

    SYSTEM_PROMPT = """
You are Juris, an integrated AI legal assistant combining legal advisory,
case simulation, and document generation capabilities.

For each query you must provide:

LEGAL DOMAIN: <identified area of law>
JURISDICTION NOTES: <relevant jurisdiction considerations>

YOUR RIGHTS:
<3-4 paragraphs explaining the user's legal rights in plain English,
grounded in the retrieved legal documents>

CASE SIMULATION:
Outcome: <Favorable / Unfavorable / Mixed / Uncertain>
Confidence: <0-100>%
Reasoning: <3-4 bullet points explaining prediction>
Key Factors: <2-3 factors most influencing outcome>
Timeline: <estimated duration if litigated>

RECOMMENDED ACTIONS:
<3-5 numbered immediate action steps>

Always end with: "Consult a licensed attorney before taking legal action."
""".strip()

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment.")
        self.llm = ChatGroq(
            model=LLM_MODEL, temperature=0.15, api_key=GROQ_API_KEY
        )
        self.conversation_history: list = []

    def analyze(self, case: LegalCase) -> LegalCase:
        """Run legal analysis on the case."""
        case.log("LegalAdvisor", f"Analyzing: {case.user_query[:60]}...")

        # Retrieve relevant knowledge
        context, sources  = retrieve_legal_knowledge(case.user_query)
        case.rag_sources  = sources
        case.log("LegalAdvisor", f"RAG retrieved {len(sources)} sources: {sources}")

        # Build messages with conversation history
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]
        for h in self.conversation_history[-4:]:
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

        user_msg = f"""
Legal Question: {case.user_query}
Jurisdiction: {case.jurisdiction or 'Not specified'}

Retrieved Legal Context:
{context}

Provide full legal analysis following your instructions.
""".strip()
        messages.append(HumanMessage(content=user_msg))

        response = self.llm.invoke(messages)
        full     = response.content.strip()

        # Update conversation history
        self.conversation_history.append({"role": "user",      "content": case.user_query})
        self.conversation_history.append({"role": "assistant", "content": full})
        case.chat_history = self.conversation_history[-6:]

        # Extract structured fields
        domain_match = re.search(r"LEGAL DOMAIN:\s*(.+?)(?:\n|$)", full, re.IGNORECASE)
        case.legal_domain = domain_match.group(1).strip() if domain_match else "General Legal"

        outcome_match = re.search(r"Outcome:\s*(.+?)(?:\n|$)", full, re.IGNORECASE)
        case.case_outcome = outcome_match.group(1).strip() if outcome_match else "Uncertain"

        conf_match = re.search(r"Confidence:\s*(\d+)", full, re.IGNORECASE)
        case.confidence = int(conf_match.group(1)) if conf_match else 50

        rights_match = re.search(
            r"YOUR RIGHTS:\s*\n(.+?)(?=CASE SIMULATION:)", full, re.DOTALL | re.IGNORECASE
        )
        case.rights_analysis = rights_match.group(1).strip() if rights_match else full

        case.log("LegalAdvisor",
            f"Domain: {case.legal_domain} | "
            f"Outcome: {case.case_outcome} | "
            f"Confidence: {case.confidence}%"
        )
        return case


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — CONTRACT GENERATOR (Lab 2 integration)
# ══════════════════════════════════════════════════════════════════════════════

class ContractGeneratorModule:
    """
    Integrates Lab 2 GAN contract generation.
    Falls back to LLM generation if GAN checkpoint not found.
    """

    GAN_CHECKPOINT = MODELS_DIR / "gan_checkpoints" / "generator_final.pt"
    VOCAB_PATH     = MODELS_DIR / "gan_checkpoints" / "vocabulary.pkl"

    SAMPLE_CONTRACTS = {
        "employment": [
            "EMPLOYMENT AGREEMENT This Employment Agreement is entered into between the Company and the Employee. "
            "The Employee agrees to perform all duties assigned by the Company in a professional manner. "
            "Compensation shall be paid at the agreed rate. Either party may terminate with thirty days notice. "
            "The Employee agrees to maintain confidentiality of all proprietary information.",
        ],
        "nda": [
            "NON DISCLOSURE AGREEMENT This Agreement is entered into between the Disclosing Party and Receiving Party. "
            "The Receiving Party agrees to hold in strict confidence all confidential information. "
            "This obligation shall survive termination for three years. "
            "Confidential information shall not be disclosed to third parties without written consent.",
        ],
        "service_agreement": [
            "SERVICE AGREEMENT This Agreement is made between the Service Provider and the Client. "
            "The Provider agrees to perform services described in the Statement of Work. "
            "The Client agrees to pay fees as specified. Services shall be performed professionally. "
            "Either party may terminate for cause with thirty days written notice.",
        ],
        "lease": [
            "LEASE AGREEMENT This Lease is entered into between the Landlord and Tenant. "
            "Tenant agrees to pay monthly rent on the first day of each month. "
            "The Landlord shall maintain the premises in habitable condition. "
            "A security deposit equal to one month rent is required upon signing.",
        ],
        "purchase": [
            "PURCHASE AGREEMENT This Agreement is made between the Seller and Buyer. "
            "The Buyer agrees to pay the purchase price as specified. "
            "The Seller warrants clear title to the property being sold. "
            "Risk of loss passes to the Buyer upon delivery and acceptance.",
        ],
    }

    def __init__(self):
        self.llm = ChatGroq(
            model=LLM_MODEL, temperature=0.2, api_key=GROQ_API_KEY
        ) if GROQ_API_KEY else None
        self.gan_available = self.GAN_CHECKPOINT.exists()
        print(f"  [ContractGen] GAN checkpoint: {'found' if self.gan_available else 'not found — using LLM fallback'}")

    def generate(self, case: LegalCase) -> LegalCase:
        """Generate a contract using GAN or LLM fallback."""
        case.log("ContractGen", f"Generating {case.contract_type} contract...")

        if self.gan_available:
            case = self._generate_with_gan(case)
        else:
            case = self._generate_with_llm(case)

        # Quality evaluation
        case.generation_quality = self._evaluate_quality(case.generated_contract)
        case.log("ContractGen",
            f"Generated {len(case.generated_contract.split())} words | "
            f"Quality: {case.generation_quality.get('overall', 'N/A')}"
        )
        return case

    def _generate_with_gan(self, case: LegalCase) -> LegalCase:
        """Load GAN from checkpoint and generate."""
        try:
            # Import generator from lab2
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "lab2", Path("lab2_gan_document_generator.py")
            )
            if spec and spec.loader:
                lab2 = importlib.util.load_module_from_spec(spec)
                spec.loader.exec_module(lab2)

                vocab = lab2.LegalVocabulary()
                vocab.load(self.VOCAB_PATH)

                G = lab2.ContractGenerator(
                    vocab_size=len(vocab.token2idx),
                    embed_dim=EMBED_DIM,
                    hidden_dim=HIDDEN_DIM,
                    latent_dim=LATENT_DIM,
                    seq_len=SEQ_LEN,
                    num_types=NUM_CONTRACT_TYPES,
                )
                G.load_state_dict(torch.load(self.GAN_CHECKPOINT, map_location=DEVICE))

                gen = lab2.SyntheticDatasetGenerator(G, vocab)
                docs = gen.generate_batch(case.contract_type, batch_size=1, temperature=TEMPERATURE)
                case.generated_contract = docs[0] if docs else ""
            else:
                raise ImportError("Could not load lab2 module")

        except Exception as e:
            case.log("ContractGen", f"GAN generation failed ({e}) — falling back to LLM")
            case = self._generate_with_llm(case)

        return case

    def _generate_with_llm(self, case: LegalCase) -> LegalCase:
        """Generate contract using Groq LLM as fallback."""
        if not self.llm:
            # Use template fallback
            templates = self.SAMPLE_CONTRACTS.get(case.contract_type, self.SAMPLE_CONTRACTS["employment"])
            base_text = random.choice(templates)

            # Customize with case context
            if case.user_query:
                context_note = f"\n\nContext note: {case.user_query[:200]}"
                base_text += context_note

            case.generated_contract = (
                f"{case.contract_type.upper().replace('_', ' ')} AGREEMENT\n\n"
                f"{base_text}\n\n"
                f"[TEMPLATE DOCUMENT - Customize before use]\n\n"
                f"Both parties acknowledge receipt of this agreement and agree to its terms.\n\n"
                f"________________________          ________________________\n"
                f"Party A Signature                   Party B Signature\n\n"
                f"Date: ___________________          Date: ___________________"
            )
            return case

        CONTRACT_PROMPTS = {
            "employment": "Generate a professional employment agreement including: position duties, compensation, termination clauses, confidentiality, and intellectual property ownership.",
            "nda":        "Generate a mutual non-disclosure agreement including: definition of confidential information, obligations of receiving party, term, exclusions, and return of materials.",
            "service_agreement": "Generate a service agreement including: scope of services, payment terms, independent contractor status, IP ownership, termination, and liability limitations.",
            "lease":      "Generate a residential lease agreement including: rent, security deposit, maintenance obligations, entry notice, subletting restrictions, and termination procedures.",
            "purchase":   "Generate a purchase agreement including: description of goods/property, purchase price, payment terms, warranties, risk of loss, and dispute resolution.",
        }

        prompt = CONTRACT_PROMPTS.get(
            case.contract_type,
            f"Generate a professional {case.contract_type.replace('_', ' ')} agreement."
        )

        context_clause = ""
        if case.user_query:
            context_clause = f"\n\nAdditional context from user's situation: {case.user_query[:300]}"

        response = self.llm.invoke([
            SystemMessage(content="""
You are a professional legal document drafter. Generate complete, professional legal contracts.
Use appropriate legal language with [PARTY NAME] and [DATE] placeholders.
Structure with clear numbered sections. Include all standard clauses for the contract type.
Add a signature block at the end. Mark the document as a template requiring attorney review.
""".strip()),
            HumanMessage(content=prompt + context_clause),
        ])

        case.generated_contract = response.content.strip()
        return case

    def _evaluate_quality(self, document: str) -> dict:
        """Evaluate generated contract quality."""
        if not document:
            return {"overall": "N/A"}

        words         = document.lower().split()
        word_count    = len(words)
        unique_words  = len(set(words))

        legal_terms   = ["shall", "party", "agreement", "herein", "pursuant",
                         "indemnify", "terminate", "covenant", "warrant", "liability"]
        legal_density = sum(1 for w in words if w in legal_terms) / max(word_count, 1)

        required_sections = ["agreement", "term", "payment", "termination", "signature"]
        sections_found    = sum(1 for s in required_sections if s in document.lower())

        score = min(10, round(
            (min(word_count, 400) / 400) * 3 +
            (unique_words / max(word_count, 1)) * 3 +
            legal_density * 20 +
            sections_found * 0.4,
            1
        ))

        return {
            "word_count":    word_count,
            "unique_words":  unique_words,
            "legal_density": round(legal_density, 3),
            "sections":      f"{sections_found}/{len(required_sections)}",
            "overall":       f"{score}/10",
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — STYLE TRANSFER (Lab 3 integration)
# ══════════════════════════════════════════════════════════════════════════════

class StyleTransferModule:
    """
    Integrates Lab 3 document style transfer.
    Uses LLM-based transfer for production quality.
    """

    TRANSFER_PROMPTS = {
        "us_to_uk": "Convert this US legal drafting to UK legal style. Replace: vacation→annual leave, attorney→solicitor, statute of limitations→limitation period. Use 'whilst', 'amongst', 'hereto'.",
        "formal_to_plain": "Convert this formal legal text to plain English. Replace: notwithstanding→despite, pursuant to→under, shall→must. Break long sentences. Max 25 words per sentence.",
        "plain_to_formal": "Convert this plain English to formal legal drafting. Add: shall/may for obligations/permissions, defined terms in capitals, standard qualifiers and precision.",
        "aggressive_to_neutral": "Convert this aggressive legal language to professional neutral tone. Remove accusatory language, replace threats with requests, maintain legal substance.",
        "passive_to_active": "Convert passive voice to active voice. Identify the actor and restructure: 'shall be paid by X' → 'X shall pay'. Keep all legal substance.",
    }

    def __init__(self):
        self.llm = ChatGroq(
            model=LLM_MODEL, temperature=0.1, api_key=GROQ_API_KEY
        ) if GROQ_API_KEY else None

    def transfer(self, case: LegalCase) -> LegalCase:
        """Apply style transfer to the source document."""
        if not case.source_document:
            case.log("StyleTransfer", "No source document provided — skipping")
            return case

        case.log("StyleTransfer", f"Applying {case.transfer_type} transfer...")

        if not self.llm:
            case.transferred_document = (
                f"[STYLE TRANSFER: {case.transfer_type}]\n\n"
                f"Original:\n{case.source_document}\n\n"
                f"[LLM not available — GROQ_API_KEY required for transfer]"
            )
            return case

        prompt = self.TRANSFER_PROMPTS.get(
            case.transfer_type,
            f"Apply {case.transfer_type.replace('_', ' ')} style transfer."
        )

        response = self.llm.invoke([
            SystemMessage(content=f"""
You are a legal document style editor. {prompt}

Preserve all legal substance and meaning.
Return:
TRANSFERRED:
<transferred text>

CHANGES:
<bullet list of key changes>
""".strip()),
            HumanMessage(content=f"Document to transfer:\n\n{case.source_document}"),
        ])

        raw = response.content.strip()

        transferred_match = re.search(
            r"TRANSFERRED:\s*\n(.+?)(?=\nCHANGES:|\Z)", raw, re.DOTALL
        )
        changes_match = re.search(
            r"CHANGES:\s*\n(.+?)(?=\Z)", raw, re.DOTALL
        )

        case.transferred_document = (
            transferred_match.group(1).strip() if transferred_match else raw
        )
        case.transfer_changes = (
            changes_match.group(1).strip() if changes_match else ""
        )

        case.log("StyleTransfer", f"Transfer complete — {len(case.transferred_document.split())} words")
        return case


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class JurisIntegratedPipeline:
    """
    Orchestrates all 3 modules in sequence.
    Supports running individual modules or the full pipeline.
    """

    def __init__(self):
        print("\n  [Pipeline] Initializing Juris Integrated System...")
        self.advisor  = LegalAdvisorModule()
        self.generator = ContractGeneratorModule()
        self.transfer  = StyleTransferModule()
        print("  [Pipeline] ✅ All modules loaded")

    def run_advise(self, case: LegalCase) -> LegalCase:
        """Run legal advisory only."""
        start = time.time()
        case  = self.advisor.analyze(case)
        case.duration = round(time.time() - start, 2)
        return case

    def run_generate(self, case: LegalCase) -> LegalCase:
        """Run contract generation only."""
        start = time.time()
        case  = self.generator.generate(case)
        case.duration = round(time.time() - start, 2)
        return case

    def run_transfer(self, case: LegalCase) -> LegalCase:
        """Run style transfer only."""
        start = time.time()
        case  = self.transfer.transfer(case)
        case.duration = round(time.time() - start, 2)
        return case

    def run_full(self, case: LegalCase) -> LegalCase:
        """
        Run the complete integrated pipeline:
        1. Legal advisory + RAG (Lab 1)
        2. Contract generation (Lab 2)
        3. Style transfer on generated contract (Lab 3)
        """
        start = time.time()
        case.log("Pipeline", "Starting full integrated pipeline...")

        # Module 1: Legal advisory
        case.log("Pipeline", "Step 1/3 — Legal Advisory + RAG")
        case = self.advisor.analyze(case)

        # Module 2: Contract generation
        case.log("Pipeline", "Step 2/3 — Contract Generation")
        case = self.generator.generate(case)

        # Module 3: Style transfer (apply to generated contract)
        if not case.source_document and case.generated_contract:
            case.source_document = case.generated_contract
        case.log("Pipeline", "Step 3/3 — Style Transfer")
        case = self.transfer.transfer(case)

        case.duration = round(time.time() - start, 2)
        case.log("Pipeline",
            f"✅ Full pipeline complete in {case.duration}s | "
            f"Errors: {len(case.errors)}"
        )
        return case

    def to_response(self, case: LegalCase) -> dict:
        """Serialize LegalCase to API response dict."""
        return {
            "legal_analysis": {
                "domain":     case.legal_domain,
                "rights":     case.rights_analysis,
                "outcome":    case.case_outcome,
                "confidence": case.confidence,
                "sources":    case.rag_sources,
            },
            "contract": {
                "type":     case.contract_type,
                "content":  case.generated_contract,
                "quality":  case.generation_quality,
            },
            "style_transfer": {
                "type":        case.transfer_type,
                "original":    case.source_document,
                "transferred": case.transferred_document,
                "changes":     case.transfer_changes,
            },
            "metadata": {
                "duration":  case.duration,
                "errors":    case.errors,
                "log":       case.pipeline_log,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "Juris — AI Legal Assistant & Case Simulator",
    description = "Integrated LLM + GAN legal AI system",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Lazy-load pipeline on first request
_pipeline: Optional[JurisIntegratedPipeline] = None

def get_pipeline() -> JurisIntegratedPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = JurisIntegratedPipeline()
    return _pipeline


# ── Request / Response Models ──────────────────────────────────────────────────

class AdviseRequest(BaseModel):
    query:        str = Field(..., min_length=10, max_length=3000)
    jurisdiction: str = Field(default="", max_length=100)

class GenerateRequest(BaseModel):
    contract_type: str = Field(default="employment")
    context:       str = Field(default="", max_length=1000)
    transfer_type: str = Field(default="formal_to_plain")

class TransferRequest(BaseModel):
    document:      str = Field(..., min_length=10, max_length=5000)
    transfer_type: str = Field(default="formal_to_plain")

class FullRequest(BaseModel):
    query:         str = Field(..., min_length=10, max_length=3000)
    jurisdiction:  str = Field(default="")
    contract_type: str = Field(default="employment")
    transfer_type: str = Field(default="formal_to_plain")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/api/advise")
async def advise(req: AdviseRequest):
    """Legal advisory with RAG — Lab 1."""
    try:
        pipeline = get_pipeline()
        case     = LegalCase(user_query=req.query, jurisdiction=req.jurisdiction)
        case     = pipeline.run_advise(case)
        return {
            "domain":     case.legal_domain,
            "rights":     case.rights_analysis,
            "outcome":    case.case_outcome,
            "confidence": case.confidence,
            "sources":    case.rag_sources,
            "duration":   case.duration,
            "errors":     case.errors,
            "log":        case.pipeline_log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Contract generation — Lab 2."""
    if req.contract_type not in CONTRACT_TYPE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid contract type. Choose from: {list(CONTRACT_TYPE_IDS.keys())}"
        )
    try:
        pipeline = get_pipeline()
        case     = LegalCase(
            contract_type  = req.contract_type,
            user_query     = req.context,
            transfer_type  = req.transfer_type,
        )
        case = pipeline.run_generate(case)

        # Optionally apply style transfer
        if req.transfer_type and case.generated_contract:
            case.source_document = case.generated_contract
            case = pipeline.run_transfer(case)

        return {
            "contract_type":   case.contract_type,
            "content":         case.generated_contract,
            "transferred":     case.transferred_document,
            "transfer_type":   case.transfer_type,
            "quality":         case.generation_quality,
            "duration":        case.duration,
            "errors":          case.errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transfer")
async def transfer(req: TransferRequest):
    """Document style transfer — Lab 3."""
    if req.transfer_type not in STYLE_TRANSFER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transfer type. Choose from: {STYLE_TRANSFER_TYPES}"
        )
    try:
        pipeline = get_pipeline()
        case     = LegalCase(
            source_document = req.document,
            transfer_type   = req.transfer_type,
        )
        case = pipeline.run_transfer(case)
        return {
            "original":    case.source_document,
            "transferred": case.transferred_document,
            "changes":     case.transfer_changes,
            "type":        case.transfer_type,
            "duration":    case.duration,
            "errors":      case.errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/full")
async def full_pipeline(req: FullRequest):
    """Full integrated pipeline — all labs."""
    try:
        pipeline = get_pipeline()
        case     = LegalCase(
            user_query    = req.query,
            jurisdiction  = req.jurisdiction,
            contract_type = req.contract_type,
            transfer_type = req.transfer_type,
        )
        case = pipeline.run_full(case)
        return pipeline.to_response(case)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {
        "status":       "ok",
        "model":        LLM_MODEL,
        "device":       str(DEVICE),
        "labs":         ["lab1_rag_chatbot", "lab2_gan_generator", "lab3_style_transfer"],
        "api_version":  "1.0.0",
        "gan_available": (MODELS_DIR / "gan_checkpoints" / "generator_final.pt").exists(),
    }


@app.get("/")
async def root():
    return {
        "name":      "Juris API",
        "version":   "1.0.0",
        "docs":      "/docs",
        "endpoints": ["/api/advise", "/api/generate", "/api/transfer", "/api/full", "/api/health"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_demo():
    sep = "─" * 60
    print("\n\n" + "═"*60)
    print("  LAB 4 DEMO — Integrated Legal AI System")
    print("═"*60)

    pipeline = JurisIntegratedPipeline()

    # ── Demo 1: Legal Advisory ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Demo 1: Legal Advisory + Case Simulation")
    print(sep)

    case1 = LegalCase(
        user_query   = "I was fired two days after I filed a workers compensation claim for a back injury at work. My employer says it was for performance reasons but I had positive reviews. What are my rights?",
        jurisdiction = "California",
    )
    case1 = pipeline.run_advise(case1)

    print(f"\n  Domain     : {case1.legal_domain}")
    print(f"  Outcome    : {case1.case_outcome} ({case1.confidence}%)")
    print(f"  Sources    : {case1.rag_sources}")
    print(f"  Rights     : {case1.rights_analysis[:400]}...")

    # ── Demo 2: Contract Generation ────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Demo 2: Contract Generation (GAN + LLM Fallback)")
    print(sep)

    for contract_type in ["employment", "nda", "service_agreement"]:
        case2 = LegalCase(
            contract_type = contract_type,
            user_query    = "Software development services agreement for a 6-month project.",
        )
        case2 = pipeline.run_generate(case2)
        print(f"\n  [{contract_type.upper()}]")
        print(f"  Quality  : {case2.generation_quality}")
        print(f"  Preview  : {case2.generated_contract[:200]}...")

    # ── Demo 3: Style Transfer ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Demo 3: Document Style Transfer")
    print(sep)

    test_doc = (
        "Notwithstanding any provision to the contrary contained herein, "
        "the party of the first part shall indemnify, defend, and hold harmless "
        "the party of the second part from and against any and all claims, "
        "damages, losses, costs, and expenses arising out of or related to "
        "any breach of this agreement by the indemnifying party."
    )

    for transfer_type in ["formal_to_plain", "passive_to_active"]:
        case3 = LegalCase(
            source_document = test_doc,
            transfer_type   = transfer_type,
        )
        case3 = pipeline.run_transfer(case3)
        print(f"\n  [{transfer_type.upper()}]")
        print(f"  Source     : {test_doc[:120]}...")
        print(f"  Transferred: {case3.transferred_document[:200]}...")

    # ── Demo 4: Full Pipeline ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Demo 4: Full Integrated Pipeline")
    print(sep)

    case4 = LegalCase(
        user_query    = "My landlord refuses to return my security deposit of $3,200 after I moved out 60 days ago. He claims I caused damage but won't show any receipts or itemized list.",
        jurisdiction  = "New York",
        contract_type = "lease",
        transfer_type = "formal_to_plain",
    )
    case4 = pipeline.run_full(case4)

    print(f"\n  Full Pipeline Results:")
    print(f"  Legal domain     : {case4.legal_domain}")
    print(f"  Outcome          : {case4.case_outcome} ({case4.confidence}%)")
    print(f"  Contract preview : {case4.generated_contract[:200]}...")
    print(f"  Transfer preview : {case4.transferred_document[:200]}...")
    print(f"  Total duration   : {case4.duration}s")
    print(f"  Pipeline steps   : {len(case4.pipeline_log)}")
    print(f"  Errors           : {len(case4.errors)}")

    print("\n\n" + "═"*60)
    print("  LAB 4 COMPLETE ✅")
    print("═"*60)
    print("\n  Integrated system components:")
    print("    ✅  LegalAdvisorModule   — RAG chatbot + case simulation")
    print("    ✅  ContractGeneratorModule — GAN + LLM contract generation")
    print("    ✅  StyleTransferModule  — Document style transfer")
    print("    ✅  JurisIntegratedPipeline — Unified orchestrator")
    print("    ✅  FastAPI server        — 5 production endpoints")
    print("\n  To start the API server:")
    print("    uvicorn core.lab4_integrated_system:app --host 0.0.0.0 --port 8000 --reload")
    print("\n  Next: Run the frontend (frontend/index.html)")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_demo()
