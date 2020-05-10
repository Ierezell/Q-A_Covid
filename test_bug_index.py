# import glob
# import json
# import os
# import re

# import numpy as np
import torch
# from unicodedata import normalize
# from rich.progress import track
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer, CamembertForQuestionAnswering,
                          pipeline)


model = "bert-large-uncased-whole-word-masking-finetuned-squad"
# # model = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model)
# bertizer = AutoModel.from_pretrained(model)
# bertizer = CamembertForQuestionAnswering.from_pretrained(model)
bertizer = AutoModelForQuestionAnswering.from_pretrained(model)
q_a = pipeline("question-answering", model=bertizer, tokenizer=tokenizer)


# with open("./demo_q_a/covid_raw.json", "r") as file:
# dic = json.load(file)


# c = normalize("NFC", dic[list(dic.keys())[0]]["content_fr"])
# q = normalize("NFC", "Combien y en a t-il ?")

# c = "Au cours des prochaines semaines, certaines activités reprendront"
# c = "Au cours des jours"
# q = "Combien y en a t-il ?"
# q = "Combien ?"


c = "Quelques jours après leur avoir dit de rester à la maison, le gouvernement Legault tente maintenant de rassurer les sexagénaires en affirmant qu'il est sécuritaire pour eux de retourner travailler.Plusieurs enseignants et éducateurs sont inquiets pour leur santé et ça se comprend, a indiqué la vice-première ministre Geneviève Guilbault, mercredi après-midi, alors qu'elle remplaçait François Legault pour la conférence de presse quotidienne du gouvernement du Québec. C'est compréhensible que des gens puissent avoir des inquiétudes.Tout indique pourtant que les travailleurs âgés de 60 à 69 ans ne pourront pas invoquer leur âge pour éviter de rentrer au travail, selon le plan de déconfinement des écoles primaires et des services de garde présenté la semaine dernière, qui prévoit une réouverture graduelle des établissements à compter de lundi.La santé publique a établi le facteur de risque à 70 ans et non à 60 ans, a indiqué Mme Guilbault. À partir de 70 ans, le risque de développer des complications est plus important, mais en bas de 70 ans, les gens peuvent retourner travailler , à condition de respecter les consignes de la santé publique en matière de distanciation et d'hygiène.Plus spécifiquement, la vice-première ministre a affirmé que ce sera possible pour les éducateurs et les enseignants âgés entre 60 et 69 ans de reprendre le travail dès la semaine prochaine s'ils respectent ces mesures."
q = "À quel âge pouvous-nous développer des complications ?"
# print(c)
# print(q)
res = q_a({"question": q, "context": c})
print(res)
# print("OKOKOKOKOKOKOKOKOKOK")


# inputs = tokenizer.encode_plus(
#     q, add_special_tokens=True, return_tensors="pt")
# input_ids = inputs["input_ids"].tolist()[0]
# text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
# answer_start_scores, answer_end_scores = bertizer(**inputs)


# ctx = """Quelques jours après leur avoir dit de rester à la maison,\xa0 le gouvernement Legault tente maintenant de rassurer les sexagénaires en affirmant qu'il est sécuritaire pour eux de retourner travailler.Plusieurs enseignants et éducateurs sont inquiets pour leur santé et ça se comprend, a indiqué la vice-première ministre Geneviève Guilbault, mercredi après-midi, alors qu'elle remplaçait François Legault pour la conférence de presse quotidienne du gouvernement du Québec. C'est compréhensible que des gens puissent avoir des inquiétudes.Tout indique pourtant que les travailleurs âgés de 60 à 69 ans ne pourront pas invoquer leur âge pour éviter de rentrer au travail, selon le plan de déconfinement des écoles primaires et des services de garde présenté la semaine dernière, qui prévoit une réouverture graduelle des établissements à compter de lundi.La santé publique a établi le facteur de risque à 70 ans et non à 60 ans, a indiqué Mme Guilbault. À partir de 70 ans, le risque de développer des complications est plus important, mais en bas de 70 ans, les gens peuvent retourner travailler , à condition de respecter les consignes de la santé publique en matière de distanciation et d'hygiène.Plus spécifiquement, la vice-première ministre a affirmé que ce sera possible pour les éducateurs et les enseignants âgés entre 60 et 69 ans de reprendre le travail dès la semaine prochaine s'ils respectent ces mesures."""
# res = q_a(
# {"question": "À quel âge pouvous-nous développer des complications ?", "context": ctx})
# print(res)


# from transformers import AutoTokenizer, CamembertForQuestionAnswering
# import torch

# tokenizer = AutoTokenizer.from_pretrained("camembert-base")
# model = CamembertForQuestionAnswering.from_pretrained("camembert-base")

# text = r"""Some text in french"""
# questions = ["Just one question in french"]

for question in [q]:
    inputs = tokenizer.encode_plus(
        question, c, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = bertizer(**inputs)

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")
