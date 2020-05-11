
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

Emb_model = "camembert/camembert-large"
CamTok = AutoTokenizer.from_pretrained(Emb_model)
Cam = AutoModel.from_pretrained(Emb_model)


questions = ["Est-ce qu'il existe un vaccin"]

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
