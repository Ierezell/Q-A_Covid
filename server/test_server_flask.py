
from unicodedata import normalize

import numpy as np
import torch
from flask import Flask, jsonify, request
from transformers import (BertForQuestionAnswering, BertModel, BertTokenizer,
                          CamembertForQuestionAnswering, CamembertModel,
                          CamembertTokenizer, pipeline)

app = Flask("Covid_nlu")


# TOK_EMB_EN = BertTokenizer.from_pretrained("bert-large-uncased")
# EMB_EN = BertModel.from_pretrained("bert-large-uncased")
TOK_EMB_FR = CamembertTokenizer.from_pretrained("camembert-base")
EMB_FR = CamembertModel.from_pretrained("camembert-base")

# TOK_QA_EN = BertTokenizer.from_pretrained(
# "bert-large-uncased-whole-word-masking-finetuned-squad")
# QA_EN = BertForQuestionAnswering.from_pretrained(
# "bert-large-uncased-whole-word-masking-finetuned-squad")

TOK_QA_FR = CamembertTokenizer.from_pretrained("illuin/camembert-large-fquad")
QA_FR = CamembertForQuestionAnswering.from_pretrained(
    "illuin/camembert-large-fquad")

PIP_Q_A_FR = pipeline("question-answering", model=QA_FR, tokenizer=TOK_QA_FR)
# PIP_Q_A_EN = pipeline("question-answering", model=QA_EN, tokenizer=TOK_QA_EN)


@app.route("/embeddings", methods=['POST'])
def get_embedding():
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
            last_hidden, pool = embedder(input_tensor["input_ids"],
                                         input_tensor["attention_mask"])
            emb_text = torch.mean(torch.mean(last_hidden, axis=1), axis=0)
            emb_text = emb_text.squeeze().detach().cpu().data.numpy().tolist()
    except RuntimeError as e:
        return jsonify({"error": f"Be careful, special strings will be tokenized in many pieces and the model will not be able to fit : {e}"})
    return jsonify({"embeddings": emb_text})


@app.route("/answers", methods=['POST'])
def get_answer():
    lang = request.json.get('lang')
    if lang == "fr":
        q_a_pipeline = PIP_Q_A_FR
    elif lang == "en":
        q_a_pipeline = PIP_Q_A_EN
    question = normalize("NFC", request.json.get('question'))
    documents = [normalize("NFC", d) for d in request.json.get('docs')]

    results = [q_a_pipeline({'question': question, 'context': doc})
               for doc in documents]

    return jsonify({"answers": results})


if __name__ == "__main__":
    # TODO load models in an async coro
    app.run(port=8000)
