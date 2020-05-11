from transformers import (AutoModel, CamembertForQuestionAnswering,
                          pipeline, AutoModelForQuestionAnswering, AutoTokenizer)
import numpy as np
import json
import os
import glob
import torch
import re
from rich.progress import track
LANG = "fr"
# LANG = "en"

try:
    os.remove(f"./all_{LANG}/content_{LANG}.txt")
except FileNotFoundError:
    pass

for file in glob.iglob(f"./{LANG}/*"):
    os.remove(file)

# model = "illuin/camembert-large-fquad"
model = "camembert/camembert-large"
# model = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model)
# bertizer = AutoModelForQuestionAnswering.from_pretrained(model)
bertizer = AutoModel.from_pretrained(model)

with open("covid_splitted_raw.json", "r") as file:
    dico = json.load(file)

for source, sub_dic in track(dico.items()):
    try:
        raw_text = sub_dic["content_fr"]
        raw_text = sub_dic["title_fr"]
    except KeyError:
        continue

    try:
        non_html_text = re.sub(
            r'(((http|ftp|https):\/\/)|(www\.))([\wàâçéèêëîïôûùüÿñæœ.,@?^=%&:\\\/~+#-]*[\w@?^=%&\/~+#-])?',
            " ",
            raw_text)

        splited_text = np.array(non_html_text.split(" "))
        chunk_text = np.array_split(splited_text, (len(splited_text)//200)+1)
        utterances = [" ".join(s) for s in chunk_text]
        # Computing context embedding
        with torch.no_grad():
            input_tensor = tokenizer.batch_encode_plus(utterances,
                                                       pad_to_max_length=True,
                                                       return_tensors="pt")
            last, pool = bertizer(input_tensor["input_ids"],
                                  input_tensor["attention_mask"])
            embed_content = torch.mean(torch.mean(last, axis=1), axis=0)

        dico[source]["embedding_content_fr"] = embed_content.detach(
        ).cpu().data.numpy().tolist()

        # Computing title embedding
        title = sub_dic["title_fr"]
        with torch.no_grad():
            input_tensor = tokenizer.batch_encode_plus([title],
                                                       pad_to_max_length=True,
                                                       return_tensors="pt")
            last_hidden_title, pool = bertizer(input_tensor["input_ids"],
                                               input_tensor["attention_mask"])
            embed_title = torch.mean(last_hidden_title, axis=1).squeeze()
        dico[source]["embedding_title_fr"] = embed_title.detach(
        ).cpu().data.numpy().tolist()

    except RuntimeError as e:
        print(input_tensor["input_ids"].size(0))
        print(e)
        input()

with open(f"covid_data.json", "w+") as file:
    json.dump(dico, file)
