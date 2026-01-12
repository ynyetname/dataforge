# DataForge

DataForge is a modular Python-based framework for document understanding, semantic retrieval, entity extraction, and knowledge graph construction.
It is designed as a lightweight foundation for RAG (Retrieval-Augmented Generation) and knowledge-driven AI systems.

* Features

1) Document Processing
- Extracts and preprocesses text from documents for downstream tasks.

2? Entity Extraction
- Identifies key entities from text to enable structured understanding.

3) Knowledge Graph Construction
- Builds a directed knowledge graph from extracted entities and relationships.

4) Semantic Retrieval
- Retrieves relevant document chunks using embedding-based similarity search.

5) Answer Generation
- Generates answers based on retrieved context and structured knowledge.

6) Modular Design
- Each component is independent and easy to extend or replace.

* Project Structure
  
dataforge/
│
├── main.py               # Entry point of the application
├── doc_processor.py      # Document loading and preprocessing
├── entity_extractor.py   # Named entity extraction logic
├── knowledge_graph.py    # Knowledge graph creation and visualization
├── retrieval.py          # Semantic retrieval and similarity search
├── answer_gen.py         # Answer generation module
├── explainer.py          # Explanation / reasoning utilities
│
├── README.md             # Project documentation
├── LICENSE               # MIT License
└── .gitignore            # Git ignore rules

* Installation

- Clone the repository:

git clone https://github.com/ynyetname/dataforge.git
cd dataforge


- Install required dependencies:

pip install spacy networkx pyvis PyPDF2 sentence-transformers


- Download a spaCy model:

python -m spacy download en_core_web_sm

* Usage

- Run the main pipeline:

python main.py

* Architecture Overview
  
Document
   ↓
Document Processor
   ↓
Entity Extractor ───► Knowledge Graph
   ↓
Retriever
   ↓
Answer Generator


- This design allows you to plug in:

1) Different embedding models

2) Custom entity extractors

3) Alternative retrieval strategies

4) Any LLM or rule-based answer generator

* Use Cases

1) Retrieval-Augmented Generation (RAG)

2) Knowledge Graph–based QA systems

3) Research document analysis

4) Educational AI tools

5) Semantic search engines

* Future Improvements (Ideas)

- Add FAISS for scalable vector search

- Add configuration via YAML/JSON

- Add unit tests

- Add API interface (FastAPI)


* License

- This project is licensed under the MIT License — feel free to use, modify, and distribute.

* Author

Ayyan Aftab
GitHub: https://github.com/ynyetname
