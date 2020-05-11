
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer, CamembertForQuestionAnswering,
                          pipeline)
import json
from q_a_sdk import get_answer, print_results
# model = "fmikaelian/camembert-base-fquad"
# model = "illuin/camembert-base-fquad"
CamQA_Model = "illuin/camembert-large-fquad"
CamTokQA = AutoTokenizer.from_pretrained(CamQA_Model)
CamQA = CamembertForQuestionAnswering.from_pretrained(CamQA_Model)
q_a_pipeline = pipeline('question-answering', model=CamQA, tokenizer=CamTokQA)

Emb_model = "camembert-base"
CamTok = AutoTokenizer.from_pretrained(Emb_model)
Cam = AutoModel.from_pretrained(Emb_model)


questions = [
    # "Que faire si je presente des symptomes du Covid-19 ?",
    # "Que se passe-t-il si je dois m'absenter",
    # "Que dois-je faire avec le renvoie de mes appels ?",
    # "Existe-t-il des aides pour les gens inaptes au travail ?",
    # "Je peux aller travailler en rentrant de voyage ?",
    # "Combien de temps une personne reste infectieuse ?",
    # "Que dois-je faire avant de reintegrer mon travail ?",
    # "J'ai peur pour mes enfants dont j'ai la garde partagée",
    # "J'ai peur pour mon conjoint qui travaille dans la santé",
    # "Comment fonctionnent les services de garde d'urgence ?",
    # "Dois-je envoyer mon enfant à l'école ?",
    # "si mon employeur offre le télétravail, suis-je obligé de retourner au bureau?",
    # "ma région est-elle déconfinée?",
    # "J'ai peur pour mon conjoint qui travaille dans la santé",
    # "Mon enfant a peur de retourner a l'école",
    # "Ma conjointe est immunosupprimée doit-elle retourner au travail ? ",
    # "Est-ce que je peux me fabriquer un masque ?",
    # "Quels secteurs pourront reprendre leurs activités?",
    # "J'ai vu un client tousser à l'épicerie devrais-je lui dire de quitter ?",
    # "quand est-ce que j’vais pouvoir lancer la construction de ma maison",
    # "est-ce que je peux visiter des appartements malgré la pandemie"
]

with open("covid_data.json", "r") as file:
    dico = json.load(file)

resultats = {}
for question in questions:
    resultats[question] = get_answer(question, dico, CamTok, Cam, q_a_pipeline)

for question, list_data in resultats.items():
    print_results(question, list_data)


while True:
    question = input("Question : ")
    top_3_answer = get_answer(question, dico, CamTok, Cam, q_a_pipeline)
    print_results(question, top_3_answer)
