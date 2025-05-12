
import sys
import os

# Obtenir le chemin absolu du fichier actuel
current_file_path = os.path.abspath(__file__)

# Obtenir le répertoire contenant le fichier actuel
current_dir = os.path.dirname(current_file_path)

# Obtenir le répertoire parent (racine du projet)
project_root = os.path.dirname(current_dir)

# Ajouter la racine du projet au chemin Python
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.retrieval.search import search_laws
import yaml
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def clean_explanation(explanation, prompt):
    # Supprime les répétitions du prompt ou des fragments inutiles
    for fragment in [prompt, "Question :", "Loi :", "En 3 phrases maximum", "Réponse :"]:
        explanation = explanation.replace(fragment, "")
    return explanation.strip()

def generate_explanations(question, config, top_k=3):
    results = search_laws(question, config, top_k=top_k)
    
    llm_config = config["model"]["llm_params"]
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["llm_model"],
        torch_dtype=torch.__dict__[llm_config["torch_dtype"]],
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_model"])
    tokenizer.pad_token = tokenizer.eos_token
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.__dict__[llm_config["torch_dtype"]],
        device_map="auto"
    )
    
    max_context_length = 1024  # DistilGPT2 supporte jusqu'à 1024 tokens
    
    for result in results:
        prompt = (
            f"Tu es un expert en stratégie, spécialisé dans les *48 Lois du Pouvoir* de Robert Greene.\n"
            f"Question : {question}\n"
            f"Loi : {result['law'][:300]}...\n"
            f"En 3 phrases maximum, explique comment cette loi s'applique à la question, dans un style stratégique et sophistiqué. "
            f"Évite les répétitions ou les informations non pertinentes.\n"
            f"Réponse : "
        )
        tokens = tokenizer(prompt, return_tensors="pt")
        prompt_length = tokens.input_ids.shape[1]
        if prompt_length > max_context_length - llm_config["max_new_tokens"]:
            raise ValueError(f"Prompt trop long ({prompt_length} tokens). Réduisez la longueur de la loi ou augmentez max_context_length.")
        
        response = llm(
            prompt,
            max_new_tokens=llm_config["max_new_tokens"],
            num_return_sequences=1,
            do_sample=True,
            temperature=llm_config["temperature"],
            top_p=llm_config["top_p"],
            truncation=True,
            pad_token_id=tokenizer.eos_token_id
        )[0]["generated_text"]
        explanation = response[len(prompt):].strip()
        result["explanation"] = clean_explanation(explanation, prompt)
    
    return results

if __name__ == "__main__":
    config = load_config()
    question = "Comment manipuler quelqu’un selon les lois du pouvoir ?"
    results = generate_explanations(question, config)
    print(f"Question : {question}")
    print("Lois pertinentes :")
    for result in results:
        print(f"Score : {result['score']:.4f}")
        print(f"{result['law'][:100]} ...")
        print(f"Explication : {result['explanation']}")
        print("-" * 50)
