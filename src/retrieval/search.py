import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import yaml
import torch

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_embeddings(embeddings_path):
    with open(embeddings_path, "r", encoding="utf-8") as file:
        embeddings_data = json.load(file)
    return embeddings_data

def search_laws(question, config, top_k=3):
    model_name = config["model"]["embedding_model"]
    model = SentenceTransformer(model_name)
    
    # Charger les embeddings des lois
    embeddings_data = load_embeddings(config["data"]["embeddings_path"])
    law_texts = [item["law"] for item in embeddings_data]
    law_embeddings = np.array([item["embedding"] for item in embeddings_data], dtype=np.float32)
    
    # Pré-filtrage par mots-clés
    keywords = {
        "manipuler": ["manipulation", "séduire", "intention", "contrôler"],
        "pouvoir": ["pouvoir", "autorité", "influence", "domination"],
        "imprévisible": ["imprévisible", "fluide", "adaptable", "surprise"],
        "éviter manipuler": ["protection", "anticipation", "espion", "méfiance"]
    }
    relevant_indices = list(range(len(law_texts)))
    for key, words in keywords.items():
        if key in question.lower():
            relevant_indices = [
                i for i in range(len(law_texts))
                if any(word in law_texts[i].lower() for word in words)
            ]
            break
    
    if not relevant_indices:
        relevant_indices = list(range(len(law_texts)))  # Pas de filtre si aucun mot-clé
    
    filtered_texts = [law_texts[i] for i in relevant_indices]
    filtered_embeddings = law_embeddings[relevant_indices]
    
    # Générer l'embedding de la question
    question_embedding = model.encode(question, show_progress_bar=False, convert_to_numpy=True, dtype=np.float32)
    question_embedding = torch.tensor(question_embedding, dtype=torch.float32)
    filtered_embeddings = torch.tensor(filtered_embeddings, dtype=torch.float32)
    
    # Calculer la similarité cosinus
    similarities = util.cos_sim(question_embedding, filtered_embeddings)[0]
    
    # Trouver les top-k lois les plus similaires
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    results = [
        {"law": filtered_texts[idx], "score": similarities[idx].item()}
        for idx in top_k_indices
    ]
    
    return results

if __name__ == "__main__":
    config = load_config()
    question = "Comment manipuler quelqu’un selon les lois du pouvoir ?"
    results = search_laws(question, config)
    print(f"Question : {question}")
    print("Lois pertinentes :")
    for result in results:
        print(f"Score : {result['score']:.4f}")
        print(f"{result['law'][:100]} ...")
        print("-" * 50)
