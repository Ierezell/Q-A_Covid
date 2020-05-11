from unicodedata import normalize

import numpy as np
import torch
from sanic import Sanic
from sanic.exceptions import abort
from sanic.response import json
from transformers import (BertForQuestionAnswering, BertModel, BertTokenizer,
                          CamembertForQuestionAnswering, CamembertModel,
                          CamembertTokenizer, QuestionAnsweringPipeline,
                          pipeline)

from model_loader import (get_loading_status, get_models_for_lang, load_models,
                          preload_weights)

app = Sanic("Covid_NLU")

DEVICE = torch.device(
    "cuda") if torch.cuda.is_available()else torch.device("cpu")
dev_pipeline = 0 if torch.cuda.is_available() else -1
print(f"Model will be on {DEVICE}")


TOK_EMB_EN = BertTokenizer.from_pretrained("bert-large-uncased")
EMB_EN = BertModel.from_pretrained("bert-large-uncased")
# TOK_EMB_FR = CamembertTokenizer.from_pretrained("camembert-base")
# EMB_FR = CamembertModel.from_pretrained("camembert-base")

TOK_QA_EN = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
QA_EN = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")

EMB_EN = EMB_EN.to(DEVICE)
QA_EN = QA_EN.to(DEVICE)
# TOK_QA_FR = CamembertTokenizer.from_pretrained("illuin/camembert-large-fquad")
# QA_FR = CamembertForQuestionAnswering.from_pretrained(
# "illuin/camembert-large-fquad")

# PIP_Q_A_FR = pipeline("question-answering", model=QA_FR, tokenizer=TOK_QA_FR)

PIP_Q_A_EN = QuestionAnsweringPipeline(model=QA_EN, tokenizer=TOK_QA_EN,
                                       device=dev_pipeline)

# TOK_EMB_FR = TOK_EMB_FR.to(DEVICE)
# EMB_FR = EMB_FR.to(DEVICE)
# PIP_Q_A_FR = PIP_Q_A_FR.to(DEVICE)


@app.post("/embeddings")
async def get_embedding(request):
    lang = request.json.get('lang')
    text = request.json.get('text')

    if lang == "fr":
        embedder = EMB_FR
        tokenizer = TOK_EMB_FR
    elif lang == "en":
        embedder = EMB_EN
        tokenizer = TOK_EMB_EN
    splited_text = np.array(text.split(" "))
    splitted_chunk_text = np.array_split(splited_text,
                                         (len(splited_text)//200)+1)
    chunk_text = [" ".join(s) for s in splitted_chunk_text]
    try:
        with torch.no_grad():
            input_tensor = tokenizer.batch_encode_plus(chunk_text,
                                                       pad_to_max_length=True,
                                                       return_tensors="pt")
            last_hidden, pool = embedder(input_tensor["input_ids"].to(DEVICE),
                                         input_tensor["attention_mask"].to(DEVICE))
            emb_text = torch.mean(torch.mean(last_hidden, axis=1), axis=0)
            emb_text = emb_text.squeeze().detach().cpu().data.numpy().tolist()
    except RuntimeError as e:
        return json({"error": f"Be careful, special strings will be tokenized in many pieces and the model will not be able to fit : {e}"})
    return json({"embeddings": emb_text})


@app.post("/answers")
async def get_answer(request):
    lang = request.json.get('lang')
    if lang == "fr":
        q_a_pipeline = PIP_Q_A_FR
    elif lang == "en":
        q_a_pipeline = PIP_Q_A_EN
    question = normalize("NFC", request.json.get('question'))
    documents = [normalize("NFC", d) for d in request.json.get('docs')]

    # question = question.to(DEVICE)
    # doc = doc.to(DEVICE)
    results = [q_a_pipeline({'question': question, 'context': doc})
               for doc in documents]

    return json({"answers": results})

# load_models()

if __name__ == "__main__":
    # TODO load models in an async coro
    app.run(host="0.0.0.0", port=8000, workers=1)
