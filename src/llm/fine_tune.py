import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Charger la configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Charger le modèle et le tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config["model"]["llm_model"],
    torch_dtype=torch.__dict__[config["model"]["llm_params"]["torch_dtype"]],
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(config["model"]["llm_model"])
tokenizer.pad_token = tokenizer.eos_token

# Configurer LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True  # Ajout pour éviter l'avertissement
)
model = get_peft_model(model, lora_config)

# Charger les ensembles de données
train_dataset = load_dataset("json", data_files=config["data"]["train_data"], split="train")
val_dataset = load_dataset("json", data_files=config["data"]["val_data"], split="train")
test_dataset = load_dataset("json", data_files=config["data"]["test_data"], split="train")

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir=config["model"]["model_output_dir"],
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=50,
    eval_strategy="steps",  # Changé de evaluation_strategy à eval_strategy
    eval_steps=50,  # Évaluer toutes les 50 étapes
    learning_rate=2e-5,
    fp16=True,
    save_strategy="steps",  # Doit correspondre à eval_strategy
    save_steps=50,  # Aligner avec eval_steps
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Configurer le trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    max_seq_length=1024,
)

# Lancer le fine-tuning
trainer.train()

# Sauvegarder le modèle fine-tuné
trainer.save_model(f"{config['model']['model_output_dir']}/fine_tuned_distilgpt2")

# Évaluer sur l'ensemble de test
test_results = trainer.evaluate(test_dataset)
print("Résultats sur l'ensemble de test :")
print(test_results)
