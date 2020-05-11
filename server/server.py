from sanic import Sanic
from sanic.exceptions import abort
from sanic.response import json
import torch
import numpy as np
from unicodedata import normalize
from model_loader import (
    preload_weights, get_loading_status, get_models_for_lang, load_models)

app = Sanic("Covid_NLU")
preload_weights()


DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_DOCS  = 25


@app.get('status')
async def get_info(req):
    return json({'status': get_loading_status()})


@app.post("/embeddings")
async def get_embedding(request):
    lang = request.json.get('lang')
    documents = request.json.get('documents')

    if len(documents) > MAX_DOCS :
        abort(400, f'Too many documents to embed, please send no more than {MAX_DOCS }')

    try:
        tokenizer, embedder, _ = get_models_for_lang(lang)
    except RuntimeError:
        abort(400, 'Model not loaded')

    embeddings = []
    for doc in documents:
        splited_text = np.array(doc.split(" "))
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
                embeddings.append(emb_text)
        except RuntimeError as e:
            return json({"error": f"Be careful, special strings will be tokenized in many pieces and the model will not be able to fit : {e}"})
    return json({"embeddings": embeddings})


@app.post("/answers")
async def get_answer(request):
    lang = request.json.get('lang')
    question = normalize("NFC", request.json.get('question'))
    documents = [normalize("NFC", d) for d in request.json.get('documents')]
    try:
        _, __, q_a_pipeline = get_models_for_lang(lang)
    except RuntimeError:
        abort(400, 'Model not loaded')
    else:
        results = [q_a_pipeline({'question': question, 'context': doc})
                   for doc in documents]

        return json({"answers": results})

load_models()

if __name__ == "__main__":
    # TODO load models in an async coro
    app.run(host="0.0.0.0", port=8000, workers=1)
