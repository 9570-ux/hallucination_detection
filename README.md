# Enhanced LLM Hallucination Detection System using GNN

A comprehensive, production-ready system for detecting hallucinations in Large Language Model (LLM) outputs using **Graph Attention Networks (GAT)**, ensemble methods, multi-source evidence validation, and confidence-based trust scoring.

---

## Contributors

* **Vivek Jaiswal**
* **Abhishek Bapna**

---

## Project Overview

Large Language Models (LLMs) can generate factually incorrect or unsupported information, commonly known as **hallucinations**.

This system automatically evaluates claims by:

1. Taking a claim as input
2. Extracting key entities from the claim
3. Fetching evidence from multiple knowledge sources
4. Constructing a **Claim-Evidence Graph**
5. Processing the graph using a **Graph Attention Network (GAT)**
6. Performing rule-based and ensemble validation
7. Applying confidence/trust-score calibration
8. Returning a final classification with supporting evidence

### Supported Classifications

| Label                  |   Trust Score | Meaning                                                      |
| ---------------------- | ------------: | ------------------------------------------------------------ |
| ✅ **SUPPORTS**         | High (70–95%) | Claim is factually supported by available evidence           |
| ❌ **REFUTES**          |   Low (5–30%) | Claim contradicts available evidence; hallucination detected |
| ⚠️ **NOT_ENOUGH_INFO** | Medium (~50%) | Available evidence is insufficient to verify the claim       |

---

# 🚀 Features

## Core Capabilities

* **Graph Neural Network Analysis** — GAT-based modeling of claim-evidence relationships
* **Ensemble Analysis** — Combines GNN and rule-based reasoning, with optional transformer-based analysis
* **Multi-source Evidence** — Wikipedia REST API, Wikipedia Summary/Search, and Wikidata
* **Real-time Processing** — Single-claim and batch processing
* **Interactive Visualization** — Dynamic dark-themed claim-evidence graph
* **Trust Score** — Indicates how factually trustworthy the analyzed claim is
* **Evidence Display** — Shows the evidence used for the final prediction

## Advanced Detection

* **Adversarial Detection** — Identifies prompt injection and suspicious inputs
* **Numerical Fact Checking** — Detects incorrect statistics, measurements, counts, and numerical claims
* **Temporal Consistency** — Checks dates and chronological relationships
* **Year Contradiction Detection** — Identifies incorrect or conflicting years
* **Known Myth Detection** — Detects predefined/common hallucination patterns
* **Negation Pattern Detection** — Handles contradictory or negated statements
* **Entity Relationship Validation** — Uses entity extraction and knowledge-graph information
* **Unsupported Claim Detection** — Identifies claims without sufficient evidence

## Analytics & Monitoring

* Real-time accuracy, precision, recall, and F1-score tracking
* Prediction distribution analysis
* Confidence distribution and histogram
* Usage statistics
* Historical analysis tracking
* Source effectiveness analysis
* CSV/JSON result export
* Audit trail through session-based history

---

# 🏗️ System Architecture

```text
Input Claim
     ↓
Adversarial Detection
     ↓
Key Entity Extraction
     ↓
Multi-Source Evidence Fetching
     ↓
Claim-Evidence Graph Construction
     ↓
Graph Attention Network (GAT)
     ↓
Ensemble Analysis
     ├── GNN Reasoning
     ├── Rule-Based Analysis
     │      ├── Year Contradiction
     │      ├── Number Contradiction
     │      ├── Known Myth Detection
     │      └── Negation Detection
     └── Transformer Analysis (Optional)
     ↓
Confidence / Trust Calibration
     ↓
Final Prediction + Evidence + Trust Score
```

### Evidence Retrieval Pipeline

```text
Claim
  ↓
Wikipedia REST API (Primary)
  ↓
Wikipedia Summary (Fallback 1)
  ↓
Wikipedia Search API (Fallback 2)
  ↓
Wikidata Facts (Fallback 3)
  ↓
Evidence Ranking / Selection
  ↓
Claim-Evidence Graph
```

---

# 🧠 Model Architecture

## Graph Neural Network

The core model uses a **Graph Attention Network (GAT)** to model relationships between claims and evidence.

| Component        | Configuration                 |
| ---------------- | ----------------------------- |
| Architecture     | Graph Attention Network (GAT) |
| GAT Layers       | 3                             |
| Layer 1          | 4 attention heads             |
| Layer 2          | 2 attention heads             |
| Layer 3          | 1 attention head              |
| Input Dimension  | 384                           |
| Hidden Dimension | 128                           |
| Output Classes   | 3                             |
| Pooling          | Global Mean Pooling           |

### Output Classes

```text
SUPPORTS
REFUTES
NOT_ENOUGH_INFO
```

---

# ⚖️ Ensemble Analysis

The system combines multiple reasoning approaches.

### 1. GNN Model

Graph-based reasoning over claim-evidence relationships.

**Configured weight: 20%**

### 2. Rule-Based Checker

Handles deterministic patterns such as:

* Numerical contradictions
* Temporal/year contradictions
* Known myths
* Negation patterns

**Configured weight: 80%**

### 3. Transformer Model

Optional contextual semantic analysis using **Sentence Transformers**.

---

# 🎯 Confidence & Trust Score

The system separates model confidence from the final **Trust Score**.

```python
if prediction == "REFUTES":
    trust_score = 100 - confidence

elif prediction == "SUPPORTS":
    trust_score = confidence * evidence_quality

else:
    trust_score = 50.0
```

The trust score is designed so that:

* **High trust** → claim is likely factual
* **Medium trust** → evidence is inconclusive
* **Low trust** → claim is likely hallucinated

---

# 📚 Evidence Sources

Evidence is retrieved in the following priority order:

1. **Wikipedia REST API**
2. **Wikipedia Python Package / Summary**
3. **Wikipedia Search API**
4. **Wikidata API**

The architecture can additionally be extended to other sources such as:

* Google Search
* PubMed
* Custom domain-specific knowledge bases

Example APIs:

```text
Wikipedia REST:
https://en.wikipedia.org/api/rest_v1/page/summary/{topic}

Wikipedia Search:
https://en.wikipedia.org/w/api.php

Wikidata:
https://www.wikidata.org/w/api.php
```

---

# 🔍 Hallucination Types Detected

### Factual Inaccuracies

Incorrect facts that are contradicted by available evidence.

### Temporal Inconsistencies

Incorrect dates, chronology, or historical ordering.

### Numerical Errors

Incorrect statistics, measurements, quantities, or counts.

### Entity Misattribution

Incorrect relationships between people, organizations, places, or other entities.

### Unsupported Claims

Claims for which sufficient evidence cannot be found.

### Known Myths

Commonly repeated false claims, such as the misconception that the Great Wall of China is visible from the Moon.

---

# 🎨 User Interface

The Streamlit application contains four primary sections.

### 🔍 Single Analysis

* Individual claim analysis
* Evidence cards
* Claim-evidence graph
* Prediction
* Confidence
* Trust score
* Class probabilities

### 📊 Batch Processing

* Analyze multiple claims
* Batch predictions
* CSV export
* JSON export

### 📈 Analytics

* Accuracy
* Precision
* Recall
* F1-score
* Prediction distribution
* Confidence histogram
* Historical trends

### ⚙️ Settings

* Model configuration
* Confidence threshold
* Evidence limits
* Data-source configuration
* Ensemble settings

---

# 📊 Visualizations

The application provides:

* **Interactive Claim-Evidence Graph** — Displays relationships between claims and evidence
* **Trust Score Display** — High / Medium / Low trust indication
* **Class Probability Bars** — SUPPORTS / REFUTES / NOT_ENOUGH_INFO
* **Evidence Cards** — Displays retrieved evidence
* **Distribution Charts** — Prediction distribution and confidence histogram
* **Timeline / Historical Views** — Tracks previous analyses

---

# 📦 Installation & Setup

## Prerequisites

* **Python 3.9 or 3.10** recommended
* Internet connection for external evidence retrieval
* Git
* Python virtual environment

> Python 3.11+ may cause compatibility issues with some PyTorch Geometric configurations.

---

## 1. Clone Repository

```bash
git clone <repository-url>
cd hallucination_gnn
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

After activation, `(.venv)` should appear in the terminal.

---

## 3. Upgrade pip

```bash
pip install --upgrade pip
```

---

## 4. Install PyTorch

### CPU Version

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

For GPU installations, use the appropriate PyTorch installation command for your CUDA version.

---

## 5. Install PyTorch Geometric

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```

If installation fails:

```bash
pip install torch-geometric torch-scatter torch-sparse --no-build-isolation
```

---

## 6. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 7. Install spaCy Model

Optional, for NER/entity validation:

```bash
python -m spacy download en_core_web_sm
```

---

# 📦 Dependencies

| Package               | Version  | Purpose                   |
| --------------------- | -------- | ------------------------- |
| streamlit             | >=1.28.0 | Web UI                    |
| torch                 | 2.0.1    | Deep learning             |
| torch-geometric       | latest   | Graph Neural Network      |
| plotly                | >=5.0.0  | Interactive visualization |
| networkx              | >=3.0    | Graph construction        |
| pandas                | >=2.0.0  | Data handling/export      |
| numpy                 | >=1.24.0 | Numerical operations      |
| wikipedia             | 1.4.0    | Wikipedia evidence        |
| requests              | latest   | API requests              |
| sentence-transformers | latest   | Semantic embeddings       |
| scipy                 | latest   | Scientific computing      |
| scikit-learn          | latest   | ML utilities              |
| spacy                 | latest   | NER/entity validation     |

The primary sentence embedding model is:

```text
all-MiniLM-L6-v2
```

---

# 🚀 Running the Application

## Streamlit

```bash
streamlit run clean_enhanced_app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🐍 Basic Python Usage

```python
from clean_enhanced_app import (
    analyze_single_sentence,
    fetch_multi_source_evidence
)

claim = "The Eiffel Tower was built in 1889."

evidence = fetch_multi_source_evidence(claim)

prediction, confidence = analyze_single_sentence(
    claim,
    evidence
)

print(
    f"Prediction: {prediction}, "
    f"Confidence: {confidence:.1f}%"
)
```

---

# 🔌 API

If `enhanced_api.py` is used, the system provides REST endpoints.

## Single Analysis

```http
POST /analyze
Content-Type: application/json
```

```json
{
  "claim": "The Eiffel Tower was built in 1889.",
  "enable_multi_source": true,
  "enable_ensemble": true,
  "enable_calibration": true,
  "enable_adversarial_check": true
}
```

## Batch Processing

```http
POST /analyze/batch
Content-Type: application/json
```

```json
{
  "claims": [
    "Claim 1",
    "Claim 2",
    "Claim 3"
  ],
  "enable_multi_source": true,
  "enable_ensemble": true
}
```

## System Metrics

```http
GET /metrics
```

## Usage Statistics

```http
GET /stats
```

---

# ⚙️ Configuration

Example model configuration:

```python
{
    "model_type": "ensemble",
    "confidence_threshold": 0.7,
    "max_evidence_sentences": 5,
    "enable_calibration": True,
    "ensemble_weights": {
        "gnn": 0.7,
        "rules": 0.3
    }
}
```

Example evidence-source configuration:

```python
{
    "wikipedia": {
        "enabled": True,
        "max_sentences": 5
    },
    "google": {
        "enabled": False,
        "api_key": "..."
    },
    "pubmed": {
        "enabled": False,
        "max_results": 3
    }
}
```

> The exact ensemble weights should match the implementation being used. The model architecture section above documents the current GNN/rule-based configuration.

---

# 📈 Model Performance

The documented evaluation results are:

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **94.2%** |
| Precision | **91.8%** |
| Recall    | **89.3%** |
| F1-Score  | **90.5%** |

These figures should be treated as project-reported results and should be re-evaluated on the target deployment dataset before making production performance claims.

---

# 🧪 Testing & Evaluation

## Example Test Cases

| Claim                                                       | Expected Result    |
| ----------------------------------------------------------- | ------------------ |
| The Eiffel Tower is located in Paris, France                | ✅ SUPPORTS         |
| The Eiffel Tower was built in 1999                          | ❌ REFUTES          |
| Python was created in 1985 for NASA                         | ❌ REFUTES          |
| The Great Wall of China is visible from the Moon            | ❌ REFUTES          |
| Linux was invented by Microsoft in 1992                     | ❌ REFUTES          |
| Water boils at 100°C at sea level                           | ✅ SUPPORTS         |
| The Taj Mahal was completed in 1662 with a blue marble dome | ❌ REFUTES          |
| Google was originally named BackTrack                       | ❌ REFUTES          |
| Earth has two natural moons named Selene                    | ❌ REFUTES          |
| John Smith likes pizza                                      | ⚠️ NOT_ENOUGH_INFO |

## Evaluation Datasets

* **FEVER Dataset** — Fact Extraction and VERification benchmark
* **Custom Test Suite** — Domain-specific evaluation
* **Adversarial Examples** — Robustness and security testing

---

# 🔒 Security Features

## Adversarial Detection

The system checks for:

* Prompt injection attempts
* Suspicious character sequences
* Unusual input patterns
* Malicious instructions
* Invalid or malformed inputs

## Data Privacy

* No permanent claim storage in the current application
* Session-based analysis history
* Personal information can be anonymized/redacted
* External API communication uses secure HTTPS connections

---

# 📊 Monitoring & Analytics

The system can track:

* Processing speed
* Throughput
* Accuracy
* Precision
* Recall
* F1-score
* Prediction distribution
* Confidence patterns
* Evidence-source effectiveness
* Error/failure rates
* Historical analysis

For production deployments, monitoring can be extended using:

* Prometheus
* Grafana
* Redis
* PostgreSQL

---

# 📁 Project Structure

```text
hallucination_gnn/
│
├── clean_enhanced_app.py       # Main Streamlit application
├── enhanced_api.py             # API server
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── 01_download_fever.py        # Download FEVER dataset
├── 02_preprocess_fever.py      # Preprocess FEVER data
├── 03_build_graphs.py          # Build graph structures
├── 04_train_gnn.py             # Train GNN model
│
├── data/                       # Dataset files
├── graphs/                     # Generated graph files
├── models/                     # Saved model weights
└── outputs/                    # Analysis/evaluation outputs
```

---

# Deployment

## Docker

```dockerfile
FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN pip install torch==2.0.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install torch-geometric
RUN pip install -r requirements.txt

EXPOSE 8501

CMD [
    "streamlit",
    "run",
    "clean_enhanced_app.py",
    "--server.port=8501"
]
```

---

# Cloud Deployment

The application can be deployed using:

### AWS

* ECS
* Lambda
* EC2

### Google Cloud

* Cloud Run
* Compute Engine

### Azure

* Container Instances
* App Service

---

# 📈 Scaling Considerations

For large-scale deployments:

* **Load Balancing** — Run multiple API/application instances
* **Caching** — Redis for frequently requested evidence
* **Database** — PostgreSQL for persistent analytics/audit data
* **Monitoring** — Prometheus + Grafana
* **Batch Processing** — Process multiple claims efficiently
* **Evidence Caching** — Reduce repeated external API requests

---

# ❗ Common Errors & Fixes

### `ModuleNotFoundError: torch_geometric`

```bash
pip install torch-geometric --no-build-isolation
```

### `ModuleNotFoundError: sentence_transformers`

```bash
pip install sentence-transformers
```

### `ModuleNotFoundError: wikipedia`

```bash
pip install wikipedia
```

### `ModuleNotFoundError: torch_scatter`

```bash
pip install torch-scatter \
-f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```

### Claim-Evidence Graph Not Showing

```bash
pip install networkx plotly
```

Also verify that the graph construction step is receiving both the claim node and evidence nodes.

### `KeyError: claim_history`

Make sure the latest `clean_enhanced_app.py` is being used.

### No Evidence Found

Check:

1. Internet connection
2. Wikipedia API availability
3. Wikidata API availability
4. The application's **Test Wikipedia Connection** option

### PyTorch Version Warning

If the application reports a newer PyTorch requirement while using the documented `torch==2.0.1` environment, verify the installed package versions against the project's requirements.

### Windows PyTorch Runtime Error

Try reinstalling the documented CPU build:

```bash
pip install torch==2.0.1 \
--index-url https://download.pytorch.org/whl/cpu
```

---

# 🤝 Contributing

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/amazing-feature
```

3. Commit your changes

```bash
git commit -m "Add amazing feature"
```

4. Push the branch

```bash
git push origin feature/amazing-feature
```

5. Open a Pull Request

---

# Acknowledgments

This project uses and builds upon:

* **FEVER Dataset** — Fact Extraction and VERification benchmark
* **PyTorch Geometric** — Graph Neural Network framework
* **Sentence Transformers** — Semantic text embeddings
* **Streamlit** — Interactive web application framework
* **Wikipedia REST API** — Knowledge-base evidence
* **Wikidata API** — Structured factual knowledge
* **NetworkX** — Graph construction and manipulation
* **Plotly** — Interactive visualization

---

# License

This project is licensed under the **MIT License**.

---

# Support

For any questions, issues, contributions, or project-related inquiries:

* **Email:** [jaiswalvivek421@gmail.com](mailto:jaiswalvivek421@gmail.com)
* **GitHub:** github.com/vivekjais03/hallucination_detection
* **GitHub Issues:** Create an issue in the project repository
* **Pull Requests:** Contributions are welcome


---

## Research Focus

**Ensuring LLM Safety, Reliability, and Factual Consistency through Graph-Based Evidence Validation.**

Built with ❤️ using **Python, Streamlit, PyTorch, PyTorch Geometric, Sentence Transformers, Wikipedia, and Wikidata APIs**.
