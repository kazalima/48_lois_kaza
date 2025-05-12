import json
import random

# Charger les embeddings existants
with open("data/processed/embeddings.json", "r", encoding="utf-8") as f:
    embeddings_data = json.load(f)

# Extraire les lois (on suppose que embeddings_data est une liste de dicts avec une clé 'law')
laws = [entry["law"] for entry in embeddings_data]

# Définir une liste de questions pertinentes
questions = [
    "Comment manipuler quelqu’un selon les lois du pouvoir ?",
    "Comment être imprévisible selon Robert Greene ?",
    "Comment éviter d’être manipulé selon les lois du pouvoir ?",
    "Comment gagner le respect selon les lois du pouvoir ?",
    "Comment contrôler une situation selon Robert Greene ?",
    "Comment influencer les autres selon les lois du pouvoir ?",
    "Comment dominer ses ennemis selon Robert Greene ?",
    "Comment se protéger des trahisons selon les lois du pouvoir ?",
    "Comment paraître puissant selon Robert Greene ?",
    "Comment créer une aura de mystère selon les lois du pouvoir ?",
    "Comment exploiter les faiblesses des autres selon Robert Greene ?",
    "Comment rester maître de ses émotions selon les lois du pouvoir ?",
    "Comment anticiper les actions des autres selon Robert Greene ?",
    "Comment se faire des alliés selon les lois du pouvoir ?",
    "Comment gérer les conflits selon Robert Greene ?",
    "Comment se réinventer selon les lois du pouvoir ?",
    "Comment maintenir le contrôle dans un groupe selon Robert Greene ?",
    "Comment éviter les erreurs stratégiques selon les lois du pouvoir ?",
    "Comment utiliser le silence comme arme selon Robert Greene ?",
    "Comment transformer ses faiblesses en atouts selon les lois du pouvoir ?"
]

# Modèles de réponses synthétiques dans le style de Robert Greene
response_templates = [
    "Pour {action}, appliquez la {law_name} en {strategy_1}. {strategy_2}, vous permettant de {outcome_1}. Ainsi, vous {outcome_2} avec une maîtrise stratégique.",
    "La {law_name} vous enseigne à {action} par {strategy_1}. En {strategy_2}, vous {outcome_1} sans résistance. Cette approche garantit que vous {outcome_2}.",
    "En suivant la {law_name}, {action} en utilisant {strategy_1}. {strategy_2}, ce qui vous permet de {outcome_1}. Votre pouvoir s’affirme alors par {outcome_2}.",
    "Avec la {law_name}, vous pouvez {action} en {strategy_1}. Cette stratégie repose sur {strategy_2}, assurant que vous {outcome_1}. Vous {outcome_2} avec une élégance calculée.",
    "La {law_name} recommande de {action} par {strategy_1}. En {strategy_2}, vous créez {outcome_1}, renforçant votre position. Ainsi, vous {outcome_2} dans l’ombre."
]

actions = {
    "Comment manipuler quelqu’un selon les lois du pouvoir ?": "manipuler autrui",
    "Comment être imprévisible selon Robert Greene ?": "être imprévisible",
    "Comment éviter d’être manipulé selon les lois du pouvoir ?": "éviter d’être manipulé",
    "Comment gagner le respect selon les lois du pouvoir ?": "gagner le respect",
    "Comment contrôler une situation selon Robert Greene ?": "contrôler une situation",
    "Comment influencer les autres selon les lois du pouvoir ?": "influencer les autres",
    "Comment dominer ses ennemis selon Robert Greene ?": "dominer vos ennemis",
    "Comment se protéger des trahisons selon les lois du pouvoir ?": "vous protéger des trahisons",
    "Comment paraître puissant selon Robert Greene ?": "paraître puissant",
    "Comment créer une aura de mystère selon les lois du pouvoir ?": "créer une aura de mystère",
    "Comment exploiter les faiblesses des autres selon Robert Greene ?": "exploiter les faiblesses des autres",
    "Comment rester maître de ses émotions selon les lois du pouvoir ?": "rester maître de vos émotions",
    "Comment anticiper les actions des autres selon Robert Greene ?": "anticiper les actions des autres",
    "Comment se faire des alliés selon les lois du pouvoir ?": "vous faire des alliés",
    "Comment gérer les conflits selon Robert Greene ?": "gérer les conflits",
    "Comment se réinventer selon les lois du pouvoir ?": "vous réinventer",
    "Comment maintenir le contrôle dans un groupe selon Robert Greene ?": "maintenir le contrôle dans un groupe",
    "Comment éviter les erreurs stratégiques selon les lois du pouvoir ?": "éviter les erreurs stratégiques",
    "Comment utiliser le silence comme arme selon Robert Greene ?": "utiliser le silence comme arme",
    "Comment transformer ses faiblesses en atouts selon les lois du pouvoir ?": "transformer vos faiblesses en atouts"
}

strategies = [
    ["séduisant les cœurs", "comprenant leurs désirs", "manipuler subtilement", "contrôler les émotions"],
    ["restant fluide et adaptable", "changeant vos plans", "déstabiliser vos adversaires", "préserver votre avantage"],
    ["contrôlant votre réputation", "montrant une force mesurée", "éviter les pièges", "maintenir votre crédibilité"],
    ["jouant sur les perceptions", "flattant discrètement", "gagner la confiance", "influencer sans résistance"],
    ["dominant le terrain", "attirant vos ennemis", "imposer vos règles", "contrôler l’environnement"]
]

outcomes = [
    ["vous imposez sans effort", "dirigez leurs actions", "gagnez leur loyauté", "renforcez votre emprise"],
    ["vous restez insaisissable", "maintenez l’avantage", "déstabilisez l’ennemi", "assurez votre survie"],
    ["vous protégez des attaques", "affirmez votre pouvoir", "gagnez le respect", "éliminez les menaces"],
    ["vous contrôlez les esprits", "orientez les décisions", "dominez subtilement", "devenez indispensable"],
    ["vous manipulez les perceptions", "prenez l’ascendant", "neutralisez l’opposition", "assurez votre domination"]
]

# Générer 500 exemples
with open("data/fine_tuning_data.jsonl", "w", encoding="utf-8") as f:
    example_count = 0
    while example_count < 500:
        for question in questions:
            if example_count >= 500:
                break
            # Sélectionner une loi au hasard
            law = random.choice(laws)
            law_name = law.split(" PRINCIPE ")[0]  # Ex. "LOI 43 PARLEZ AUX CURS ET AUX ESPRITS"
            
            # Créer le prompt
            prompt = (
                f"Tu es un expert en stratégie, spécialisé dans les *48 Lois du Pouvoir* de Robert Greene.\n"
                f"Question : {question}\n"
                f"Loi : {law[:300]}...\n"
                f"En 3 phrases maximum, explique comment cette loi s'applique à la question, dans un style stratégique et sophistiqué. "
                f"Évite les répétitions ou les informations non pertinentes.\n"
                f"Réponse : "
            )
            
            # Générer une réponse synthétique
            template = random.choice(response_templates)
            action = actions[question]
            strategy = random.choice(strategies)
            outcome = random.choice(outcomes)
            completion = template.format(
                law_name=law_name,
                action=action,
                strategy_1=strategy[0],
                strategy_2=strategy[1],
                outcome_1=outcome[0],
                outcome_2=outcome[1]
            )
            
            # Écrire l'exemple dans le fichier
            entry = {"prompt": prompt, "completion": completion}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            example_count += 1

# Vérifier le nombre d'exemples
with open("data/fine_tuning_data.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print(f"Nombre total d'exemples générés : {len(lines)}")
