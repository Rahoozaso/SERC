```markdown
# SERC: Semantic Error-Reduction and Correction Framework

<div align="center">
</div>

> **Abstract:** This repository contains the official implementation of **SERC** (Semantic Error-Reduction and Correction). SERC addresses LLM hallucinations by re-conceptualizing text generation as a transmission process over a *Semantic Noisy Channel*. Inspired by Error Correcting Codes (ECC) and LDPC codes, it employs a sparse verification strategy to detect and correct errors using Retrieval-Augmented Generation (RAG).

---

## 📂 Project Structure

```text
.
├── .env                    # API Keys (OpenAI, Tavily, Google, etc.)
├── config.yaml             # Main configuration file (Models, Data paths)
├── environment.yml         # Conda environment setup
├── baselines/              # Baseline implementations
│   ├── run_cove.py         # CoVe (No-RAG)
│   ├── run_cove_rag.py     # CoVe + RAG
│   ├── run_rarr.py         # RARR
│   └── run_re_ex.py        # Re-Ex
├── experiments/            # Main SERC experiments & Ablations
│   ├── run_SERC.py         # Main Framework (Proposed)
│   ├── run_dense.py        # Ablation: High-Density Verification
│   ├── run_no_firewall.py  # Ablation: Without Entity Firewall
│   └── run_no_rag.py       # Ablation: Self-Correction (No RAG)
├── src/                    # Source code
│   ├── rag_retriever.py    # RAG Logic (Tavily/Google)
│   ├── model_wrappers.py   # LLM Generation Wrappers
│   └── prompts.py          # Prompt Templates
├── data/                   # Dataset directory
└── notebooks/              # Analysis notebooks

```

---

## ⚡ Quick Start

### 1. Installation

**Prerequisites:** [Anaconda](https://www.anaconda.com/) or Miniconda.

```bash
# 1. Create Conda Environment
conda env create -f environment.yml

# 2. Activate Environment
conda activate serc_env

# 3. Download NLTK Data (Required for sentence splitting)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

```

### 2. Configuration (`.env`)

Create a `.env` file in the root directory and add your API keys. SERC utilizes **Tavily** for search and **OpenAI/Anthropic/HuggingFace** for generation.

```bash
# .env file
OPENAI_API_KEY="your key"
TAVILY_API_KEY="your key"
GOOGLE_API_KEY="your key" 
OPENAI_API_KEY="your key"

```

---

## 🚀 Usage

All scripts utilize `argparse` and can be configured via command line arguments.

**Common arguments:**

* `--model`: Model identifier (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`, `gpt-4o`)
* `--dataset`: Dataset key defined in `config.yaml` (e.g., `longform_bio`, `truthfulqa`)
* `--output_dir`: Directory to save results.

### 1. Run Main Method (SERC)

To run the proposed SERC framework:

```bash
python experiments/run_SERC.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset "longform_bio" \
    --output_dir "results/serc"

```

### 2. Run Baselines

We provide implementations for key baselines used in the paper.

**CoVe (Chain-of-Verification) + RAG:**

```bash
python baselines/run_cove_rag.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset "longform_bio"

```

**RARR (Retrofit Attribution):**

```bash
python baselines/run_rarr.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset "longform_bio"

```

**Re-Ex (Re-Extraction):**

```bash
python baselines/run_re_ex.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset "longform_bio"

### 3. Run Ablation Studies

Scripts to analyze the contribution of specific components:

* **No-RAG (Self-Correction only):**
```bash
python experiments/run_no_rag.py --model "..." --dataset "..."

* **No-Firewall (Skip Entity Consistency):**
```bash
python experiments/run_no_firewall.py --model "..." --dataset "..."

* **High-Density (Atomic) Verification:**
```bash
python experiments/run_dense.py --model "..." --dataset "..."


## 📊 Evaluation

Results are saved as `.jsonl` files in the `results/` directory. You can analyze them using the provided notebooks in `notebooks/` or parse the logs directly.

```bash
# Example evaluation script usage
python experiments/evaluate.py --input_file "results/serc/your_result_file.jsonl"


##  Contact

For any questions, please contact **Gyumin Kim** via rhzs1208@hufs.ac.kr or open an issue.

```

```