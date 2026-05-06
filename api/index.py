"""
api/index.py
------------
Vercel serverless FastAPI handler for Juris — AI Legal Assistant & Case Simulator.

Replicates all Lab 4 pipeline modes without heavy ML dependencies:
    - PyTorch / FAISS / sentence-transformers replaced with in-memory equivalents
    - GAN generation replaced with LLM-based contract drafting
    - Neural style transfer replaced with LLM-based style transfer
    - RAG replaced with keyword-scored in-memory legal knowledge base

Endpoints:
    POST /api/advise      — Legal advisory + case simulation (Lab 1)
    POST /api/generate    — Contract generation + style transfer (Labs 2+3)
    POST /api/transfer    — Document style transfer (Lab 3)
    POST /api/full        — Full integrated pipeline (Lab 4)
    GET  /api/health      — System health check

Usage (local):
    uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import json
import time
import random
from mangum import Mangum
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

CONTRACT_TYPES = ["employment", "nda", "service_agreement", "lease", "purchase"]
TRANSFER_TYPES = [
    "us_to_uk", "formal_to_plain", "plain_to_formal",
    "aggressive_to_neutral", "passive_to_active",
]

app = FastAPI(
    title       = "Juris — AI Legal Assistant & Case Simulator",
    description = "LLM + GAN integrated legal AI system (Vercel serverless edition)",
    version     = "1.0.0",
)

handler = Mangum(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

def get_llm(temperature: float = 0.1) -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables.")
    return ChatGroq(model=LLM_MODEL, temperature=temperature, api_key=GROQ_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY LEGAL KNOWLEDGE BASE (replaces FAISS)
# ══════════════════════════════════════════════════════════════════════════════

LEGAL_KNOWLEDGE = [
    {
        "source": "employment_law",
        "tags":   ["employment", "fired", "termination", "discrimination",
                   "wages", "harassment", "overtime", "fmla", "workplace"],
        "content": (
            "Employment law governs employer-employee relationships. Key federal statutes: "
            "Title VII (prohibits discrimination by race, sex, religion, national origin), "
            "ADA (disability accommodations), ADEA (age 40+ protection), FLSA (minimum wage "
            "and 1.5x overtime for hours over 40/week). Most US states follow at-will "
            "employment — termination is lawful unless it violates a protected right, "
            "whistleblower laws, or an employment contract. Wrongful termination claims "
            "must be filed with the EEOC within 180-300 days. Successful wage claims "
            "recover back wages plus equal liquidated damages. FMLA entitles eligible "
            "employees to 12 weeks unpaid job-protected leave per year."
        ),
    },
    {
        "source": "contract_law",
        "tags":   ["contract", "breach", "agreement", "damages", "remedy",
                   "arbitration", "indemnification", "service", "payment"],
        "content": (
            "A valid contract requires offer, acceptance, and consideration. Breach occurs "
            "when a party fails obligations without legal excuse. Remedies: compensatory "
            "damages, specific performance, or rescission. Statute of limitations for "
            "written contracts is typically 4-6 years. Key clauses include: indemnification "
            "(shifting liability), limitation of liability (capping damages), force majeure "
            "(excusing performance for unforeseeable events), and arbitration clauses "
            "(resolving disputes outside court). Demand letters create paper trails and "
            "often resolve disputes before litigation. Small claims courts handle disputes "
            "up to $5,000-$20,000 without a lawyer."
        ),
    },
    {
        "source": "tenant_rights",
        "tags":   ["tenant", "landlord", "lease", "eviction", "deposit",
                   "rent", "housing", "habitability", "repair"],
        "content": (
            "Landlords must maintain habitable conditions — working plumbing, heating, "
            "and structural safety. Security deposits must be returned within 14-30 days "
            "of move-out with an itemized deduction statement; normal wear cannot be "
            "deducted. Wrongful withholding allows recovery of 2-3x deposit in many states. "
            "Eviction requires written notice, court filing, hearing, and order — self-help "
            "evictions (lockouts, utility shutoffs) are illegal everywhere. Landlords must "
            "give 24 hours notice before entry. Retaliation against tenants who report "
            "violations is prohibited. Tenants may repair-and-deduct after proper notice."
        ),
    },
    {
        "source": "civil_rights",
        "tags":   ["police", "search", "warrant", "miranda", "arrest",
                   "fourth amendment", "fifth amendment", "rights", "constitutional"],
        "content": (
            "Fourth Amendment protects against unreasonable searches — police generally "
            "need warrants for homes; exceptions include consent, plain view, and exigent "
            "circumstances. You can clearly refuse vehicle search consent. Fifth Amendment: "
            "invoke silence explicitly ('I invoke my right to remain silent and want an "
            "attorney'). Miranda rights must be read before custodial interrogation. "
            "Sixth Amendment guarantees counsel — public defenders appointed if needed. "
            "Section 1983 permits federal lawsuits against officials violating constitutional "
            "rights. Document all incidents: date, time, officer badge numbers, witnesses."
        ),
    },
    {
        "source": "consumer_protection",
        "tags":   ["consumer", "refund", "fraud", "scam", "debt collector",
                   "credit", "chargeback", "product", "ftc", "fdcpa"],
        "content": (
            "FTC Act prohibits deceptive trade practices. Credit card chargebacks available "
            "within 60 days under Fair Credit Billing Act for goods/services not received. "
            "FDCPA strictly regulates debt collectors — no calls before 8am or after 9pm, "
            "no abusive language, must cease contact on written request. FCRA gives right "
            "to dispute inaccurate credit information. Report fraud at reportfraud.ftc.gov "
            "and to your state attorney general. Many state consumer protection statutes "
            "allow recovery of treble damages and attorney fees."
        ),
    },
    {
        "source": "family_law",
        "tags":   ["divorce", "custody", "child", "domestic violence",
                   "restraining order", "alimony", "marriage", "separation"],
        "content": (
            "Divorce property division: community property states (CA, TX, AZ) split "
            "marital assets 50/50; equitable distribution states divide fairly. Alimony "
            "based on marriage length, income disparity, and standard of living. Child "
            "custody: legal (decisions) and physical (residence) — both can be sole or "
            "joint; courts apply best interests of the child standard. Protective orders "
            "for domestic violence can be issued same-day; violation is criminal. "
            "National DV Hotline: 1-800-799-7233. Document all incidents thoroughly."
        ),
    },
    {
        "source": "legal_process",
        "tags":   ["lawyer", "attorney", "legal aid", "court", "filing",
                   "evidence", "documentation", "small claims", "pro se"],
        "content": (
            "Free legal resources: legal aid organizations serve low-income individuals, "
            "law school clinics offer free help, many attorneys offer free consultations. "
            "Employment and civil rights attorneys often work on contingency. Find legal "
            "aid at lawhelp.org. Always document everything: keep all written communications, "
            "photograph evidence with timestamps, write contemporaneous notes after incidents "
            "with dates, times, and witness names, send important notices via certified mail. "
            "Strong documentation dramatically improves your legal position."
        ),
    },
]

DOMAIN_KEYWORDS = {
    "Employment Law":       ["fired", "employer", "employee", "job", "workplace",
                             "wages", "harassment", "discrimination", "termination"],
    "Contract Law":         ["contract", "agreement", "breach", "payment", "services",
                             "clause", "damages", "owe", "invoice"],
    "Tenant & Housing Law": ["landlord", "tenant", "rent", "lease", "eviction",
                             "deposit", "apartment", "habitability"],
    "Civil Rights Law":     ["police", "arrested", "search", "warrant", "rights",
                             "constitutional", "discrimination", "race", "miranda"],
    "Consumer Protection":  ["product", "refund", "scam", "fraud", "consumer",
                             "debt collector", "credit card", "chargeback"],
    "Family Law":           ["divorce", "custody", "child", "spouse", "domestic",
                             "restraining order", "alimony", "marriage"],
}


def detect_domain(text: str) -> str:
    words  = set(re.findall(r'\w+', text.lower()))
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(
            len(set(re.findall(r'\w+', kw.lower())) & words)
            for kw in keywords
        )
        scores[domain] = score
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "General Legal Matter"


def retrieve_knowledge(query: str, top_k: int = 3) -> tuple[str, list[str]]:
    words  = set(re.findall(r'\w+', query.lower()))
    scored = []
    for chunk in LEGAL_KNOWLEDGE:
        score = sum(
            len(set(re.findall(r'\w+', tag.lower())) & words) * 2 +
            (3 if tag.lower() in query.lower() else 0)
            for tag in chunk["tags"]
        )
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:top_k]] or LEGAL_KNOWLEDGE[:2]
    context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['content']}" for c in top)
    sources = [c["source"] for c in top]
    return context, sources


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegalCase:
    user_query:           str = ""
    jurisdiction:         str = ""
    contract_type:        str = "employment"
    transfer_type:        str = "formal_to_plain"
    source_document:      str = ""
    legal_domain:         str = ""
    rights_analysis:      str = ""
    full_analysis:        str = ""
    case_outcome:         str = ""
    confidence:           int = 0
    rag_sources:          list[str] = field(default_factory=list)
    generated_contract:   str = ""
    generation_quality:   dict = field(default_factory=dict)
    transferred_document: str = ""
    transfer_changes:     str = ""
    pipeline_log:         list[str] = field(default_factory=list)
    errors:               list[str] = field(default_factory=list)
    duration:             float = 0.0

    def log(self, module: str, msg: str):
        self.pipeline_log.append(f"[{module}] {msg}")

    def err(self, module: str, msg: str):
        entry = f"[{module} ERROR] {msg}"
        self.errors.append(entry)
        self.pipeline_log.append(entry)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — LEGAL ADVISOR (Lab 1)
# ══════════════════════════════════════════════════════════════════════════════

ADVISOR_SYSTEM = """
You are Juris, an integrated AI legal research assistant combining legal advisory,
case simulation, and document generation. Analyze the user's legal situation
using the retrieved legal knowledge provided.

Respond using this EXACT structure:

LEGAL DOMAIN: <identified area of law>
JURISDICTION NOTES: <relevant notes for the stated jurisdiction>

YOUR RIGHTS:
<3-4 paragraphs explaining the user's legal rights in plain English,
grounded in the retrieved legal knowledge. Be specific to their situation.>

CASE SIMULATION:
Outcome: <Favorable / Unfavorable / Mixed / Uncertain>
Confidence: <integer 0-100>%
Reasoning:
<3 bullet points explaining the prediction>
Key Factors: <2-3 factors most influencing the outcome>
Timeline: <estimated duration if litigated>

RECOMMENDED ACTIONS:
<3-5 numbered immediate action steps>

End with: "Consult a licensed attorney before taking legal action."
""".strip()


def run_advisor(case: LegalCase) -> LegalCase:
    case.log("LegalAdvisor", f"Analyzing: {case.user_query[:60]}...")

    context, sources  = retrieve_knowledge(case.user_query)
    case.rag_sources  = sources
    case.legal_domain = detect_domain(case.user_query)
    case.log("LegalAdvisor", f"Domain: {case.legal_domain} | Sources: {sources}")

    llm = get_llm(temperature=0.15)
    response = llm.invoke([
        SystemMessage(content=ADVISOR_SYSTEM),
        HumanMessage(content=f"""
Legal question: {case.user_query}
Jurisdiction: {case.jurisdiction or 'Not specified'}

Retrieved legal knowledge:
{context}

Provide full legal analysis following the structure in your instructions.
""".strip()),
    ])

    full = response.content.strip()
    case.full_analysis = full

    # Extract domain
    dm = re.search(r"LEGAL DOMAIN:\s*(.+?)(?:\n|$)", full, re.IGNORECASE)
    if dm:
        case.legal_domain = dm.group(1).strip()

    # Extract outcome and confidence
    om = re.search(r"Outcome:\s*(.+?)(?:\n|$)", full, re.IGNORECASE)
    cm = re.search(r"Confidence:\s*(\d+)", full, re.IGNORECASE)
    case.case_outcome = om.group(1).strip() if om else "Uncertain"
    case.confidence   = int(cm.group(1))    if cm else 50

    # Extract rights section
    rm = re.search(
        r"YOUR RIGHTS:\s*\n(.+?)(?=CASE SIMULATION:)", full, re.DOTALL | re.IGNORECASE
    )
    case.rights_analysis = rm.group(1).strip() if rm else full

    case.log("LegalAdvisor",
        f"Outcome: {case.case_outcome} | Confidence: {case.confidence}%"
    )
    return case


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — CONTRACT GENERATOR (Lab 2 — LLM fallback for serverless)
# ══════════════════════════════════════════════════════════════════════════════

CONTRACT_PROMPTS = {
    "employment": (
        "Generate a complete, professional employment agreement. Include: "
        "job title and duties, compensation and benefits, start date and term, "
        "at-will or fixed-term provisions, confidentiality and IP ownership, "
        "termination procedures (notice periods, cause), non-compete if applicable, "
        "governing law, and a signature block."
    ),
    "nda": (
        "Generate a mutual non-disclosure agreement. Include: "
        "definition of confidential information, obligations of each receiving party, "
        "permitted disclosures (employees, advisors on need-to-know basis), "
        "exclusions (publicly known, independently developed, lawfully received), "
        "term and survival period (typically 3 years), "
        "return or destruction of materials, and remedies for breach."
    ),
    "service_agreement": (
        "Generate a professional services agreement. Include: "
        "scope of services and deliverables, payment terms and invoicing schedule, "
        "independent contractor status and tax obligations, "
        "IP ownership of work product, confidentiality obligations, "
        "limitation of liability, indemnification, "
        "termination (for cause and for convenience with 30-day notice), "
        "dispute resolution, and governing law."
    ),
    "lease": (
        "Generate a residential lease agreement. Include: "
        "property description and permitted use, lease term and renewal, "
        "monthly rent amount and due date, security deposit terms, "
        "landlord maintenance obligations and habitability warranty, "
        "tenant obligations (condition, no subletting without consent), "
        "entry notice requirements (24 hours), "
        "termination procedures and move-out conditions, "
        "late fees and grace period, and applicable law."
    ),
    "purchase": (
        "Generate a purchase agreement. Include: "
        "description of goods or property being sold, "
        "purchase price and payment method, "
        "seller's warranty of title and right to sell, "
        "condition of goods (as-is or with warranties), "
        "delivery terms and risk of loss transfer, "
        "inspection period and acceptance, "
        "dispute resolution and governing law, "
        "and a signature block."
    ),
}

CONTRACT_GENERATOR_SYSTEM = """
You are a professional legal document drafter specializing in clear, complete contracts.

Generate a well-structured legal contract following these standards:
- Use proper legal headings (numbered sections)
- Use [PARTY NAME], [DATE], [AMOUNT], [CITY, STATE] as placeholders
- Use "shall" for obligations, "may" for permissions
- Include all standard clauses for the contract type requested
- Add a signature block at the end
- Mark it as: "TEMPLATE — Review with a licensed attorney before use"

Produce only the contract document — no preamble or commentary.
""".strip()


def run_generator(case: LegalCase) -> LegalCase:
    case.log("ContractGen", f"Generating {case.contract_type} contract...")

    base_prompt = CONTRACT_PROMPTS.get(
        case.contract_type,
        f"Generate a professional {case.contract_type.replace('_', ' ')} agreement."
    )

    context_note = ""
    if case.user_query:
        context_note = f"\n\nAdditional context from user: {case.user_query[:400]}"
    if case.jurisdiction:
        context_note += f"\nJurisdiction: {case.jurisdiction}"

    llm = get_llm(temperature=0.15)
    response = llm.invoke([
        SystemMessage(content=CONTRACT_GENERATOR_SYSTEM),
        HumanMessage(content=base_prompt + context_note),
    ])

    case.generated_contract = response.content.strip()

    # Quality metrics
    words        = case.generated_contract.lower().split()
    legal_terms  = ["shall", "party", "agreement", "herein", "pursuant",
                    "terminate", "warrant", "liability", "indemnify", "covenant"]
    legal_density = round(sum(1 for w in words if w in legal_terms) / max(len(words), 1), 3)
    sections      = len(re.findall(r'^\d+\.', case.generated_contract, re.MULTILINE))

    case.generation_quality = {
        "word_count":    len(words),
        "unique_words":  len(set(words)),
        "legal_density": legal_density,
        "sections":      sections,
        "source":        "LLM (Groq llama-3.3-70b)",
    }

    case.log("ContractGen",
        f"{len(words)} words | {sections} sections | "
        f"legal density: {legal_density}"
    )
    return case


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — STYLE TRANSFER (Lab 3 — LLM-based for serverless)
# ══════════════════════════════════════════════════════════════════════════════

TRANSFER_INSTRUCTIONS = {
    "us_to_uk": (
        "Convert US legal drafting style to UK legal style. Apply: "
        "vacation → annual leave, attorney → solicitor, "
        "statute of limitations → limitation period, "
        "real estate → real property, lawsuit → claim or action. "
        "Use: whilst, amongst, hereto, determine (for terminate). "
        "UK style prefers brevity — replace 'in the event that' with 'if'."
    ),
    "formal_to_plain": (
        "Convert formal legal text to plain English for non-lawyers. Rules: "
        "notwithstanding → despite, pursuant to → under, shall → must, "
        "hereinafter → from now on (or remove), inter alia → among other things. "
        "Break sentences to max 25 words. Use active voice. "
        "Keep all legal substance — just make it readable."
    ),
    "plain_to_formal": (
        "Convert plain English to formal legal drafting. Apply: "
        "use 'shall' for obligations, 'may' for permissions. "
        "Add defined terms in capitals (the 'Company', the 'Agreement'). "
        "Add precision: 'reasonable' → 'commercially reasonable', "
        "'soon' → 'within 30 days'. Use passive voice where appropriate. "
        "Add standard legal qualifiers."
    ),
    "aggressive_to_neutral": (
        "Convert aggressive or threatening legal language to professional neutral tone. "
        "Remove: accusatory phrases, emotional amplifiers (blatant, egregious, deliberately). "
        "Replace threats with requests: 'we will sue you' → 'we may pursue available remedies'. "
        "Replace 'you violated' with 'the terms were not met'. "
        "Keep all factual content, deadlines, and demands — only change the tone."
    ),
    "passive_to_active": (
        "Convert passive voice constructions to active voice throughout. "
        "Identify the performing party and restructure: "
        "'shall be paid by X' → 'X shall pay', "
        "'shall be submitted by Y' → 'Y shall submit'. "
        "Keep 'shall' for obligations. Maintain all legal substance — only change voice."
    ),
}

STYLE_TRANSFER_SYSTEM = """
You are a specialist legal document editor performing style transfer.
Apply the requested transformation precisely and completely.

Return your response in this EXACT format:

TRANSFERRED:
<the fully transferred document>

CHANGES:
<bullet list of the key changes applied>
""".strip()


def run_transfer(case: LegalCase) -> LegalCase:
    if not case.source_document:
        case.log("StyleTransfer", "No source document — skipping")
        return case

    case.log("StyleTransfer", f"Applying {case.transfer_type}...")

    instruction = TRANSFER_INSTRUCTIONS.get(
        case.transfer_type,
        f"Apply {case.transfer_type.replace('_', ' ')} style transformation."
    )

    llm = get_llm(temperature=0.1)
    response = llm.invoke([
        SystemMessage(content=STYLE_TRANSFER_SYSTEM),
        HumanMessage(content=f"""
Transformation: {instruction}

Document to transfer:
{case.source_document}
""".strip()),
    ])

    raw = response.content.strip()

    t_match = re.search(r"TRANSFERRED:\s*\n(.+?)(?=\nCHANGES:|\Z)", raw, re.DOTALL)
    c_match = re.search(r"CHANGES:\s*\n(.+?)(?=\Z)", raw, re.DOTALL)

    case.transferred_document = t_match.group(1).strip() if t_match else raw
    case.transfer_changes     = c_match.group(1).strip() if c_match else ""

    case.log("StyleTransfer",
        f"Complete — {len(case.transferred_document.split())} words output"
    )
    return case


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR (Lab 4)
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(case: LegalCase) -> LegalCase:
    """Run all 3 modules sequentially."""
    case.log("Pipeline", "Starting full integrated pipeline...")

    case.log("Pipeline", "Step 1/3 — Legal Advisory + RAG")
    case = run_advisor(case)

    case.log("Pipeline", "Step 2/3 — Contract Generation")
    case = run_generator(case)

    # Feed generated contract into style transfer if no explicit source doc
    if not case.source_document and case.generated_contract:
        case.source_document = case.generated_contract

    case.log("Pipeline", "Step 3/3 — Style Transfer")
    case = run_transfer(case)

    case.log("Pipeline",
        f"✅ Complete | {len(case.pipeline_log)} steps | {len(case.errors)} errors"
    )
    return case


def to_response(case: LegalCase) -> dict:
    return {
        "legal_analysis": {
            "domain":     case.legal_domain,
            "rights":     case.rights_analysis,
            "full":       case.full_analysis,
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
            "duration": case.duration,
            "errors":   case.errors,
            "log":      case.pipeline_log,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class AdviseRequest(BaseModel):
    query:        str = Field(..., min_length=10, max_length=3000,
        description="Legal question or situation in plain English",
        example="I was fired two days after filing a workers compensation claim.")
    jurisdiction: str = Field(default="", max_length=100,
        example="California")


class GenerateRequest(BaseModel):
    contract_type: str = Field(default="employment",
        description=f"One of: {CONTRACT_TYPES}")
    context:       str = Field(default="", max_length=1000,
        description="Additional context or requirements for the contract")
    transfer_type: str = Field(default="formal_to_plain",
        description=f"Style transfer to apply after generation. One of: {TRANSFER_TYPES}")


class TransferRequest(BaseModel):
    document:      str = Field(..., min_length=10, max_length=5000,
        description="Legal document or clause to transfer")
    transfer_type: str = Field(default="formal_to_plain",
        description=f"One of: {TRANSFER_TYPES}")


class FullRequest(BaseModel):
    query:         str = Field(..., min_length=10, max_length=3000)
    jurisdiction:  str = Field(default="")
    contract_type: str = Field(default="employment")
    transfer_type: str = Field(default="formal_to_plain")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/advise")
async def advise(req: AdviseRequest):
    """
    Legal advisory + case simulation — Lab 1.
    Returns rights analysis, outcome prediction, and action steps.
    """
    start = time.time()
    try:
        case = LegalCase(user_query=req.query, jurisdiction=req.jurisdiction)
        case = run_advisor(case)
        return {
            "domain":     case.legal_domain,
            "rights":     case.rights_analysis,
            "full":       case.full_analysis,
            "outcome":    case.case_outcome,
            "confidence": case.confidence,
            "sources":    case.rag_sources,
            "jurisdiction": case.jurisdiction,
            "duration":   round(time.time() - start, 2),
            "errors":     case.errors,
            "log":        case.pipeline_log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advisor failed: {str(e)}")


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Contract generation (Lab 2 GAN — LLM fallback on serverless)
    + optional style transfer (Lab 3).
    """
    if req.contract_type not in CONTRACT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid contract_type. Choose from: {CONTRACT_TYPES}"
        )
    if req.transfer_type not in TRANSFER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transfer_type. Choose from: {TRANSFER_TYPES}"
        )
    start = time.time()
    try:
        case = LegalCase(
            contract_type = req.contract_type,
            user_query    = req.context,
            transfer_type = req.transfer_type,
        )
        case = run_generator(case)

        # Apply style transfer to generated contract
        if req.transfer_type:
            case.source_document = case.generated_contract
            case = run_transfer(case)

        return {
            "contract_type":   case.contract_type,
            "content":         case.generated_contract,
            "transferred":     case.transferred_document,
            "transfer_changes":case.transfer_changes,
            "transfer_type":   case.transfer_type,
            "quality":         case.generation_quality,
            "duration":        round(time.time() - start, 2),
            "errors":          case.errors,
            "log":             case.pipeline_log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/transfer")
async def transfer(req: TransferRequest):
    """
    Document style transfer — Lab 3.
    Transfers a legal document to the target style.
    """
    if req.transfer_type not in TRANSFER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transfer_type. Choose from: {TRANSFER_TYPES}"
        )
    start = time.time()
    try:
        case = LegalCase(
            source_document = req.document,
            transfer_type   = req.transfer_type,
        )
        case = run_transfer(case)
        return {
            "original":    case.source_document,
            "transferred": case.transferred_document,
            "changes":     case.transfer_changes,
            "type":        case.transfer_type,
            "duration":    round(time.time() - start, 2),
            "errors":      case.errors,
            "log":         case.pipeline_log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer failed: {str(e)}")


@app.post("/full")
async def full_pipeline(req: FullRequest):
    """
    Full integrated pipeline — Lab 4.
    Runs all 3 modules: advisory → generation → style transfer.
    """
    if req.contract_type not in CONTRACT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid contract_type. Choose from: {CONTRACT_TYPES}"
        )
    if req.transfer_type not in TRANSFER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transfer_type. Choose from: {TRANSFER_TYPES}"
        )
    start = time.time()
    try:
        case = LegalCase(
            user_query    = req.query,
            jurisdiction  = req.jurisdiction,
            contract_type = req.contract_type,
            transfer_type = req.transfer_type,
        )
        case = run_full_pipeline(case)
        case.duration = round(time.time() - start, 2)
        return to_response(case)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")


@app.get("/health")
async def health():
    """System health check."""
    return {
        "status":       "ok",
        "model":        LLM_MODEL,
        "api_version":  "1.0.0",
        "deployment":   "vercel",
        "labs":         ["lab1_rag_advisory", "lab2_contract_gen", "lab3_style_transfer"],
        "contract_types": CONTRACT_TYPES,
        "transfer_types": TRANSFER_TYPES,
    }


@app.get("/")
async def root():
    return {
        "name":      "Juris API",
        "version":   "1.0.0",
        "docs":      "/docs",
        "endpoints": [
            "/advise",
            "/generate",
            "/transfer",
            "/full",
            "/health",
        ],
    }
