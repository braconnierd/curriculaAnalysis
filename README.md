# curriculaAnalysis
Analytical tool developed for topic based curricular analysis to enable systematic examination of curricular content and implementation of data-driven decisions regarding curricula. It uses a sentence embedding model to identify whether topics are covered within a curriculum. doi: (tba) ####-###-#####

# Capabilities of the Repository
- Process large collections of PDF documents and create corresponding txt files
- Create topic lists from given PDFs
- Compare topic lists and course content

# Requirements
- Python 3.11+
- Ollama
- Additional python libraries as specified within the requirements file
This script was originally designed for MacOS, and there has been limited testing on Windows.

Install Ollama at https://ollama.com/download. The default models used within the project are qwen3:4b-instruct-2507-q4_k_M and qwen2.5vl:3b.

# Project Workflow
Load Topics
      │
      ▼
Generate Topic Embeddings
      │
      ▼
Load Documents
      │
      ▼
Chunk Documents
      │
      ▼
Compare Chunks to Topics
      │
      ▼
Extract Evidence
      │
      ▼
Write Output Files
