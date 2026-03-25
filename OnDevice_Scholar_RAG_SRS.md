# Software Requirements Specification (SRS)

**Project:** Secure On-Device RAG Pipeline for Academic Research  
**Version:** 1.1.0  
**Lead Architect:** Edwin R. Cho (Hyun Heum Cho)  
**Target Platform:** Local Edge Devices (Optimized for Apple Silicon MPS)  
**Last Updated:** 2026-03-24  

---

## 1. Introduction

### 1.1 Purpose
The purpose of this system is to provide a privacy-first, fully offline Retrieval-Augmented Generation (RAG) pipeline for academic research labs. The system enables researchers to query confidential, unreleased, or proprietary academic documents without exposing sensitive intellectual property to external cloud APIs, third-party inference services, or remote storage systems.

### 1.2 Scope
This project focuses on a compact, deployable on-device RAG architecture optimized for local execution on Apple Silicon hardware.

The scope includes:
- **Core LLM Engine:** Qwen2.5-3B-Instruct, 4-bit NF4 quantized
- **Backend Framework:** FastAPI for asynchronous API serving
- **Embedding Model:** Lightweight local sentence embedding model
- **Vector Store:** Persistent local FAISS index
- **Document Types:** PDF, Markdown, TXT
- **Primary Users:** Academic researchers, lab PIs, and local administrators

The scope excludes cloud inference, distributed deployment, and large-scale enterprise infrastructure.

### 1.3 Business Context
This project serves as an initial small-scale product prototype for a privacy-preserving AI research assistant targeted at academic labs and IP-sensitive research environments.

Its business value is based on the following:
- Research labs increasingly require local AI systems due to unpublished paper confidentiality
- Many existing RAG platforms assume cloud connectivity or external model APIs
- A lightweight offline RAG product can serve as a wedge product for broader edge AI infrastructure offerings
- Academic adoption can become an early validation channel before expansion into enterprise research, legal, biomedical, and defense-adjacent use cases

### 1.4 Product Vision
The product vision is to become a secure local research copilot that can answer questions over internal academic documents with traceable citations, predictable latency, and minimal hardware requirements.

### 1.5 Stakeholders

| Role | Description | Permissions |
|------|-------------|-------------|
| Researcher | End user querying indexed documents | Query, view cited responses |
| Lab PI | Research supervisor managing lab documents | Upload, delete, query |
| System Administrator | Maintains local deployment and configuration | Full access |
| Evaluation Committee | Reviews technical validity and feasibility | Read-only review of specification and architecture |

### 1.6 Definitions and Acronyms

| Term | Meaning |
|------|---------|
| RAG | Retrieval-Augmented Generation |
| SLM | Small Language Model |
| LLM | Large Language Model |
| FAISS | Facebook AI Similarity Search |
| MPS | Metal Performance Shaders |
| TTFT | Time To First Token |
| RBAC | Role-Based Access Control |
| NF4 | 4-bit NormalFloat quantization |
| Top-K | Number of most relevant retrieved chunks |

---

## 2. Operating Environment

### 2.1 Hardware Environment
The system shall be optimized for Apple Silicon devices, especially M-series chips such as M2 Pro, M3 Pro, or higher.

Target deployment assumptions:
- Apple Silicon M-series processor
- Unified memory in the range of 8GB to 18GB
- Preferred operating envelope for the complete pipeline: 4GB to 8GB effective shared memory usage
- SSD-based local storage for model files and vector index persistence

### 2.2 Software Environment
The system shall operate in the following software environment:
- **Operating System:** macOS Ventura 13.0 or later
- **Programming Language:** Python 3.11+
- **Backend Runtime:** FastAPI + Uvicorn
- **ML Framework:** PyTorch with MPS backend
- **Vector Search:** FAISS (CPU mode)
- **Document Processing:** PyPDF2 or equivalent parser
- **Orchestration/Chunking:** LangChain text splitter utilities
- **Model Loading:** Hugging Face Transformers + BitsAndBytes-compatible quantized loading where feasible

### 2.3 Deployment Model
The system shall be deployed locally on a single edge device. No dependency on public cloud inference or external database services shall exist during runtime.

---

## 3. System Overview

### 3.1 High-Level Goal
The system ingests academic documents, transforms them into semantic chunks, embeds the chunks into vector representations, stores them in a local FAISS index, retrieves relevant chunks for a user query, and generates an answer grounded only in retrieved evidence.

### 3.2 Design Principles
The system shall follow these core design principles:
- **Privacy-first:** No external transmission of document content
- **Grounded generation:** Answers must be tied to retrieved evidence
- **Resource efficiency:** Must operate on constrained local hardware
- **Auditability:** Outputs must include traceable citations
- **Fail-safe behavior:** The system must refuse unsupported claims

### 3.3 Assumptions
- Users have local access to the deployment environment
- Indexed documents are primarily English academic texts, with future extensibility for multilingual support
- The local administrator pre-downloads required model weights before offline use
- The number of concurrently active users remains limited due to device-level memory constraints

### 3.4 Constraints
- Strict offline execution
- Memory-constrained hardware
- Single-device inference
- Limited concurrent inference due to MPS memory contention
- Need for predictable behavior suitable for academic evaluation settings

---

## 4. Functional Requirements

## 4.1 Document Ingestion

### FR-1 Document Upload
The system shall allow authorized users to upload documents in PDF, Markdown, and TXT formats through an ingestion endpoint or equivalent local workflow.

### FR-2 Document Parsing
The system shall parse uploaded documents and extract readable textual content.

For PDF documents, the system shall preserve, when available:
- title
- abstract
- author metadata
- page numbers
- section headers

### FR-3 Metadata Preservation
Each ingested document shall retain metadata sufficient for source grounding. At minimum, the system shall preserve:
- source filename
- page number, if applicable
- detected or assigned section label
- chunk identifier
- ingestion timestamp

### FR-4 Unsupported File Handling
If a file is corrupted, unsupported, encrypted in an unreadable way, or parsing fails, the system shall reject the file and return a structured error message identifying the filename and failure reason.

---

## 4.2 Chunking and Preprocessing

### FR-5 Semantic Chunking
The system shall split parsed text into semantically coherent chunks using Recursive Character Text Splitting.

### FR-6 Chunk Size Configuration
The default chunking policy shall be:
- `chunk_size = 2048 characters`
- `chunk_overlap = 200 to 400 characters`

This character-based setting is intended to approximate roughly 512 tokens with 10% to 20% overlap for typical academic English text.

### FR-7 Chunk Metadata Binding
Every chunk shall retain a metadata payload including:
- source filename
- page number
- section header or inferred section label
- chunk index within the document

### FR-8 Preprocessing Consistency
All preprocessing applied during indexing shall be consistent with preprocessing applied during retrieval so that embedding behavior remains stable across ingestion and query workflows.

---

## 4.3 Embedding and Vector Storage

### FR-9 Local Embedding
The system shall generate dense vector embeddings using a lightweight local embedding model stored on-device.

### FR-10 No External Embedding Calls
The system shall not call any outbound embedding API or remote model service during indexing or retrieval.

### FR-11 Vector Normalization
The system shall L2-normalize embeddings before indexing and retrieval so that inner product search can approximate cosine similarity correctly.

### FR-12 Vector Index Type
The system shall use a persistent FAISS `IndexFlatIP` index or functionally equivalent cosine-compatible local index.

### FR-13 Persistent Storage
The FAISS index and associated metadata store shall persist across restarts.

### FR-14 Incremental Indexing
The system shall support appending newly uploaded documents to the existing vector store without requiring a full rebuild for every ingestion event.

---

## 4.4 Retrieval

### FR-15 Query Processing
The system shall accept a natural language query from an authenticated user and convert it into an embedding using the same embedding model family used for indexing.

### FR-16 Similarity Search
The system shall search the persistent local vector index using cosine-compatible similarity search.

### FR-17 Top-K Retrieval
The system shall retrieve the Top-K most relevant chunks for a given user query.

Default values:
- default `K = 3`
- configurable range `K = 1 to 10`

### FR-18 Retrieval Score Access
The system shall retain similarity scores for retrieved chunks for use in thresholding, debugging, and response validation.

### FR-19 Threshold-Based Relevance Filtering
If retrieval confidence is below a configured threshold, the system shall not generate a speculative answer and shall instead return an ignorance fallback response.

---

## 4.5 Generation and Grounding

### FR-20 Context Injection
The system shall inject retrieved chunks directly into the model prompt before generation.

### FR-21 Prompt Grounding Policy
The generation prompt shall explicitly instruct the model to:
- answer only from provided context
- avoid unsupported speculation
- refuse to answer when evidence is insufficient
- cite the source of each factual claim

### FR-22 Response Generation
The system shall generate an answer using the local quantized language model only after retrieval has completed successfully.

### FR-23 Mandatory Citation
Every factual answer returned by the system shall include source grounding, at minimum:
- source filename
- page number when available
- section name when available

### FR-24 Citation Validator
The system shall validate that generated responses include citations in the required format before returning them to the user. If citations are missing, the response shall be rejected and replaced with a safe fallback.

### FR-25 Ignorance Fallback
If relevant information is not found, the system shall return one of the following safe responses:
- `I don't know based on the provided documents.`
- `No relevant information found in the provided documents.`

The system shall not fabricate an answer under insufficient evidence conditions.

---

## 4.6 Document Management

### FR-26 Document Deletion
Authorized users shall be able to delete an indexed document. Deletion shall remove:
- corresponding metadata entries
- associated chunk records
- vector references from the retrieval index

### FR-27 Full Index Rebuild
An administrator shall be able to trigger a complete rebuild of the local vector index using persisted parsed/chunked data.

### FR-28 Rebuild Without Re-parsing
If cached intermediate chunk metadata exists, the system should rebuild the FAISS index without re-parsing source files whenever possible.

---

## 4.7 Authentication and Authorization

### FR-29 Authentication
All non-public endpoints shall require authentication.

### FR-30 Authorization
The system shall enforce role-based access control such that:
- Researchers may query
- Lab PIs may upload, delete, and query
- System Administrators may configure, rebuild, and fully manage the system

### FR-31 Unauthorized Access Handling
Unauthorized requests shall receive an appropriate error response and shall not expose internal state, model information, or document metadata.

---

## 4.8 Error Handling

### FR-32 Parsing Errors
If parsing fails for a document, the system shall skip that document and return a structured error object containing the filename and error reason.

### FR-33 Model Load Failure
If the quantized model fails to load due to insufficient memory or incompatible runtime conditions, the system shall return a service-unavailable response and log the failure condition.

### FR-34 Index Corruption Recovery
If the FAISS index fails to load or is detected as corrupted, the system shall attempt recovery through rebuild from persisted metadata or cached embeddings.

### FR-35 Structured Error Responses
All operational failures shall return structured machine-readable errors suitable for frontend or CLI consumption.

---

## 5. Non-Functional Requirements

## 5.1 Privacy and Security

### NFR-1 Full Offline Operation
The system shall execute fully offline during ingestion, retrieval, and inference after required models are preinstalled.

### NFR-2 Zero Outbound Requests
The runtime system shall initiate zero outbound network requests for document processing, embedding generation, vector retrieval, or model inference.

### NFR-3 Encryption at Rest
Sensitive document files, metadata, and vector index files shall be stored with encryption at rest or within an encrypted local storage volume.

### NFR-4 Access Control
Protected endpoints shall require authenticated access using bearer token authentication or an equivalent local secure authentication mechanism.

### NFR-5 Least Privilege
Permissions shall be restricted by role, and administrative operations shall not be available to ordinary researchers.

### NFR-6 Auditability
The system shall log administrative and ingestion events with timestamps and request identifiers for traceability.

---

## 5.2 Performance

### NFR-7 TTFT
The system shall target a Time-To-First-Token of under 2 seconds under the following baseline conditions:
- Apple M3 Pro class device
- 500 indexed chunks
- single active inference request
- query length less than or equal to 50 tokens

### NFR-8 Retrieval Latency
Top-K retrieval from the local vector index should complete with low enough latency to preserve an interactive question-answering workflow.

### NFR-9 Embedding Latency
Single-chunk embedding latency should remain low enough for practical local ingestion of academic papers on edge hardware.

### NFR-10 Throughput Constraint
The system may serialize model inference to preserve stability under limited unified memory conditions.

---

## 5.3 Efficiency

### NFR-11 Memory Envelope
The full runtime pipeline shall operate within a practical 4GB to 8GB memory envelope on target devices.

### NFR-12 Storage Footprint
The combined storage used by the quantized model, embedding model, vector index, and metadata store should remain suitable for local laptop deployment.

### NFR-13 Quantized Inference
The generation model shall use a quantized representation to reduce memory consumption while maintaining acceptable answer quality.

---

## 5.4 Reliability

### NFR-14 Stable Degradation
When system resources are insufficient, the system shall fail safely and return an explicit error or fallback rather than producing unstable or misleading output.

### NFR-15 Request Traceability
Every API response shall include a request identifier or equivalent correlation handle for debugging and evaluation.

### NFR-16 Recoverability
The system shall support restart and recovery without requiring complete re-ingestion of all documents whenever persisted index artifacts remain valid.

---

## 5.5 Usability

### NFR-17 Citation Readability
Returned citations shall be human-readable and sufficiently specific for academic verification.

### NFR-18 Predictable Output Format
The system shall maintain a stable response format suitable for later frontend, CLI, or evaluation integration.

### NFR-19 Minimal Operator Burden
A technically capable graduate-level user should be able to deploy and operate the system locally without enterprise infrastructure.

---

## 6. Architecture

### 6.1 Logical Architecture

1. **Ingestion Layer**  
   Raw document input  
   → parser  
   → metadata extraction  
   → recursive chunking

2. **Embedding Layer**  
   chunk text  
   → local embedding model  
   → normalized vectors

3. **Storage Layer**  
   normalized vectors  
   → FAISS index  
   → metadata store

4. **Execution Layer**  
   user query  
   → query embedding  
   → Top-K retrieval  
   → threshold filtering  
   → context-augmented prompt  
   → local LLM inference  
   → citation validation  
   → final response

### 6.2 Text Architecture Diagram

```text
┌──────────────────────────────────────────────┐
│                 Client Layer                 │
│        Local CLI / Script / REST Client      │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│                API Layer                     │
│   FastAPI Endpoints + Auth + RBAC + Logs     │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│              Ingestion Layer                 │
│   PDF/MD/TXT Parser → Metadata Extraction    │
│   → Recursive Character Chunking             │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│              Embedding Layer                 │
│   Local Embedding Model → L2 Normalization   │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│               Storage Layer                  │
│      FAISS Index + Metadata Persistence      │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│              Retrieval Layer                 │
│   Query Embedding → Top-K Search → Filter    │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│              Generation Layer                │
│   Prompt Assembly → Qwen2.5-3B NF4 → Output  │
│   → Citation Validator → Final Response      │
└──────────────────────────────────────────────┘
```

---

## 7. API Requirements

### 7.1 Endpoint Summary

| Method | Endpoint | Access Role | Description |
|--------|----------|-------------|-------------|
| POST | `/query` | Researcher+ | Submit natural language query |
| POST | `/ingest` | Lab PI+ | Upload and index a document |
| DELETE | `/document/{id}` | Lab PI+ | Delete a document and its chunks |
| POST | `/admin/rebuild-index` | Admin | Rebuild vector index |
| GET | `/health` | Public or Local Only | Service health check |

### 7.2 Query Endpoint Behavior
The `/query` endpoint shall:
1. authenticate the user
2. embed the query locally
3. retrieve Top-K relevant chunks
4. apply relevance threshold logic
5. generate a cited answer or fallback
6. return a structured response payload

### 7.3 Query Response Format
A successful response should include:
- answer text
- citations
- retrieved source metadata
- request identifier
- optional relevance diagnostics

Example structure:

```json
{
  "request_id": "uuid-string",
  "answer": "The proposed method uses contrastive pretraining [paper1.pdf, Methods, p.7].",
  "citations": [
    {
      "source_filename": "paper1.pdf",
      "section_header": "Methods",
      "page_number": 7
    }
  ],
  "status": "ok"
}
```

### 7.4 Error Response Format
Operational errors should use a structured payload.

Example:

```json
{
  "request_id": "uuid-string",
  "status": "error",
  "error": "Parsing failed",
  "filename": "paper.pdf",
  "reason": "Unsupported or corrupted PDF structure"
}
```

---

## 8. Prompting Policy

### 8.1 System Prompt Requirements
The system prompt shall instruct the model to:
- behave as a precise academic assistant
- rely only on supplied context
- avoid unsupported claims
- cite every factual statement
- return a fallback if evidence is missing

### 8.2 Reference Prompt Template

```text
[SYSTEM]
You are a precise academic research assistant.
Answer ONLY using the provided context.
Do not infer facts not explicitly supported.
Cite the source filename, section, and page for each factual claim.
If the answer is not contained in the context, respond:
"No relevant information found in the provided documents."

[CONTEXT]
{chunk_1} [Source: filename.pdf, Section: Introduction, Page: 3]
{chunk_2} [Source: filename.pdf, Section: Methods, Page: 7]
{chunk_3} [Source: filename.pdf, Section: Results, Page: 10]

[QUERY]
{user_query}
```

---

## 9. Data Requirements

### 9.1 Supported Input Types
The system shall support:
- PDF
- Markdown (`.md`)
- Plain text (`.txt`)

### 9.2 Metadata Schema
Each chunk shall maintain metadata with at least the following fields:

```json
{
  "source_filename": "paper.pdf",
  "page_number": 3,
  "section_header": "Introduction",
  "chunk_id": "paper_pdf_chunk_001",
  "ingested_at": "2026-03-24T12:00:00"
}
```

### 9.3 Data Retention
Documents and indexes shall remain on the local device unless explicitly deleted by an authorized user.

---

## 10. Acceptance Criteria

| Category | Requirement | Pass Condition |
|----------|-------------|----------------|
| Privacy | Offline execution | No runtime outbound requests |
| Grounding | Citation coverage | 100% of factual responses include citations |
| Safety | Fallback behavior | Irrelevant or low-confidence queries return fallback |
| Performance | TTFT | Under 2 seconds on baseline hardware |
| Efficiency | Peak memory | Stays within target local memory envelope |
| Reliability | Recovery | Index corruption can be rebuilt from persisted state |
| Security | Authentication | Protected endpoints reject unauthenticated requests |
| Manageability | Document lifecycle | Upload, delete, and rebuild operations work correctly |

---

## 11. Risks and Mitigations

| Risk | Description | Mitigation |
|------|-------------|------------|
| Memory overflow | MPS inference may exceed available unified memory | Quantized model, serialized inference, smaller batch sizes |
| Retrieval failure | Relevant chunk may not be surfaced in Top-K | Tune chunking, thresholding, K, and metadata quality |
| Parsing noise | PDF extraction quality may be inconsistent | Add parser fallback strategies and structured error handling |
| Hallucination risk | Model may answer beyond context | Strict prompt grounding, citation validator, fallback logic |
| Operational misuse | Unauthorized local usage | Authentication and RBAC |
| Scaling limitation | Edge device cannot support many concurrent users | Limit concurrency and queue requests |

---

## 12. Out of Scope

The following items are explicitly out of scope for version 1.1.0:
- LoRA fine-tuning or adapter training pipeline
- multimodal retrieval over figures, tables, and images
- OCR-heavy scanned PDF recovery pipeline
- cloud synchronization
- distributed multi-node deployment
- browser-based full GUI frontend
- mobile deployment
- enterprise IAM integration

---

## 13. Future Extensions

Possible future versions may include:
- multilingual retrieval support
- OCR integration for scanned documents
- table-aware parsing
- experiment tracking dashboard
- local web interface
- document-level access control policies
- domain-specific re-ranking
- lab knowledge graph integration

---

## 14. Conclusion

This SRS defines a privacy-preserving, offline, edge-deployable academic RAG system designed for confidential research environments. The system emphasizes grounded generation, strict citation, safe fallback behavior, and practical operation within Apple Silicon memory constraints. As an initial business-oriented prototype, it is intentionally scoped to validate both technical feasibility and market relevance in academic research settings.