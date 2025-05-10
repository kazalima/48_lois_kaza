
import yaml
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.retrieval.search import search_laws

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def generate_explanations(question, config, top_k=3):
    # Récupérer les lois pertinentes avec search.py
    results = search_laws(question, config, top_k=top_k)
    
    # Initialiser Phi-3-mini avec quantification 4-bit
    llm_config = config["model"]["llm_params"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["llm_model"],
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_model"])
    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.__dict__[llm_config["torch_dtype"]],
        device_map="auto"
    )
    
    # Générer des explications
    for result in results:
        prompt = (
            f"Tu es un expert en stratégie, spécialisé dans les *48 Lois du Pouvoir* de Robert Greene.\\n"
            f"Question : {question}\\n"
            f"Loi : {result['law'][:500]}...\\n"
            f"Explique en 3 phrases maximum comment cette loi répond à la question, dans le style de Robert Greene."
        )
        response = llm(
            prompt,
            max_length=llm_config["max_length"],
            num_return_sequences=1,
            do_sample=True,
            temperature=llm_config["temperature"]
        )[0]["generated_text"]
        result["explanation"] = response[len(prompt):].strip()
    
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
