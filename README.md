# RAG with PubMed – Biomedical Question Answering

This repository contains the code and data used to explore retrieval‑augmented generation (RAG) for biomedical question answering on the PubMedQA dataset. The goal is to combine knowledge retrieval, a biomedical knowledge graph and large language models to produce accurate, explainable answers to clinical questions.

## Repository layout

- **`dataset/`** – Folder containing the PubMedQA dataset splits (e.g. `train.jsonl`, `dev.jsonl`, `test.jsonl`). These files include questions, candidate answers and ground‑truth labels.
- **`graph_construction.ipynb`** – Jupyter notebook used to build and visualise the biomedical knowledge graph. It reads the dataset, extracts biomedical entities and relations and constructs a medical graph using functions from `build_medical_graph.py`. This notebook was previously named `NLP_(1).ipynb` and has been renamed to better reflect its purpose.
- **`PubMedBERT_SOTA.ipynb`** – Notebook implementing a baseline question‑answering model using the PubMedBERT transformer. It serves as a non‑RAG benchmark for comparison.
- **`PubMedQA_CBR_RAG_Gemma.ipynb`** – Notebook that implements a case‑based reasoning (CBR) retrieval‑augmented generation pipeline using the Gemma large language model. It retrieves similar Q&A pairs from the case base (`cbr_retrieval.py`) and uses them, along with retrieved documents, to condition the LLM.
- **`build_medical_graph.py`** – Python script with helper functions to build a Neo4j‑compatible medical knowledge graph from PubMed abstracts. It is imported by `graph_construction.ipynb` and can be run as a stand‑alone script.
- **`cbr_retrieval.py`** – Helper functions for retrieving similar cases and documents for the CBR‑RAG pipeline. This script encapsulates retrieval logic used by the RAG notebooks.
- **`requirements.txt`** – List of Python dependencies required to run the notebooks and scripts. See the “Setup” section below for installation instructions.

## Setup

1. **Install dependencies.** Create a virtual environment if desired and install the required packages:

   ```bash
   pip install -r requirements.txt

2. **Prepare the dataset.** Download the PubMedQA dataset and place the JSONL files into the dataset/ folder.  The notebooks assume paths such as `dataset/dev.jsonl` and `dataset/test.jsonl` exist.
3. **Configure external services.**  Some notebooks access APIs (e.g. Gemma LLM) or a Neo4j instance.  Create a .env file with the necessary credentials (e.g. `API_KEY=<your_key>` or `NEO4J_URI=bolt://localhost:7687`) or modify the notebooks to set these variables directly.

## Running Experiments

1. **Building the knowledge graph**
The knowledge‑graph construction code lives in `graph_construction.ipynb` and `build_medical_graph.py`.  To build and inspect the graph:

  1. Start a Jupyter notebook server and open `graph_construction.ipynb`.
  2. Run all cells.  The notebook loads the dataset, extracts entities and relations and uses `build_medical_graph.py` to populate a Neo4j database. Adjust the connection URI and credentials at the top of the notebook if your Neo4j instance is not running on the default port.

2. **Training and evaluating a baseline QA model**

The `PubMedBERT_SOTA.ipynb` notebook trains a PubMedBERT model on the PubMedQA dataset and evaluates its performance.  To reproduce the baseline results:

  1. Open the notebook in Jupyter.
  2. Update the dataset paths.
  3. Execute the cells sequentially.  The notebook will download the PubMedBERT model, fine‑tune it on the training set and report accuracy on the dev/test sets.

3. **CBR‑RAG pipeline with Gemma**

The `PubMedQA_CBR_RAG_Gemma.ipynb` notebook implements the full retrieval‑augmented pipeline:

1. **Retrieve candidate documents.**  The notebook uses functions from `cbr_retrieval.py` to fetch the top‑k relevant abstracts for each question (keyword and/or semantic retrieval).  It also retrieves similar Q&A examples to support case‑based reasoning.
2. **Generate answers.**  The retrieved context and case examples are fed into the Gemma LLM via API.  Ensure your Gemma API key is available in the environment or set in the notebook.

To reproduce these results, open the notebook and run all cells.  Make sure the necessary external services (e.g. Gemma API) are configured.

   
## Notes
* Depending on your hardware and API quota, running the RAG pipeline may take several hours.  You can reduce the number of questions or documents retrieved to speed up experimentation.
