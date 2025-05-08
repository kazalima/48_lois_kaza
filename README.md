# Les 48 lois du pouvoir A

Un assistant LLM pour répondre à des questions sur *Les 48 lois du pouvoir*.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Placez le PDF dans `data/raw/48_laws_of_power.pdf`.
2. Extrayez le texte : `python src/data_processing/extract_text.py`.
3. Générez les embeddings : `python src/data_processing/generate_embeddings.py`.
