from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer, CamembertForQuestionAnswering,
                          QuestionAnsweringPipeline)
import os

models = {
    "Camembert_Q_A": "illuin/camembert-large-fquad",
    # "Camembert": "camembert-base",
    "Camembert": "camembert/camembert-large",
    "Bert": "bert-large-uncased",
    "Bert_Q_A": "bert-large-uncased-whole-word-masking-finetuned-squad"
}

for folder, model in models.items():
    if not os.path.exists(f"./weights/{folder}"):
        os.mkdir(f"./weights/{folder}")

if not os.path.exists("./weights/Camembert_Q_A/pytorch_model.bin"):
    QA_MODEL_NAME_FR = "illuin/camembert-large-fquad"
    QA_TOK_FR = AutoTokenizer.from_pretrained(QA_MODEL_NAME_FR)
    QA_MODEL_FR = CamembertForQuestionAnswering.from_pretrained(
        QA_MODEL_NAME_FR)
    QA_FR = QuestionAnsweringPipeline(model=QA_MODEL_FR, tokenizer=QA_TOK_FR)
    QA_FR.save_pretrained("./weights/Camembert_Q_A")
    del QA_FR
    del QA_TOK_FR
    del QA_MODEL_FR

if not os.path.exists("./weights/Camembert/pytorch_model.bin"):
    EMB_MODEL_NAME_FR = "camembert-base"
    EMB_TOK_FR = AutoTokenizer.from_pretrained(EMB_MODEL_NAME_FR)
    EMB_FR = AutoModel.from_pretrained(EMB_MODEL_NAME_FR)
    EMB_FR.save_pretrained("./weights/Camembert")
    EMB_TOK_FR.save_pretrained("./weights/Camembert")
    del EMB_TOK_FR
    del EMB_FR

if not os.path.exists("./weights/Bert_Q_A/pytorch_model.bin"):
    QA_MODEL_NAME_EN = "bert-large-uncased-whole-word-masking-finetuned-squad"
    QA_TOK_EN = AutoTokenizer.from_pretrained(QA_MODEL_NAME_EN)
    QA_MODEL_EN = AutoModelForQuestionAnswering.from_pretrained(
        QA_MODEL_NAME_EN)
    QA_EN = QuestionAnsweringPipeline(model=QA_MODEL_EN, tokenizer=QA_TOK_EN)
    QA_EN.save_pretrained("./weights/Bert_Q_A")
    del QA_EN
    del QA_MODEL_EN
    del QA_TOK_EN

if not os.path.exists("./weights/Bert/pytorch_model.bin"):
    EMB_MODEL_NAME_EN = "bert-large-uncased"
    EMB_TOK_EN = AutoTokenizer.from_pretrained(EMB_MODEL_NAME_EN)
    EMB_EN = AutoModel.from_pretrained(EMB_MODEL_NAME_EN)
    EMB_EN.save_pretrained("./weights/Bert")
    EMB_TOK_EN.save_pretrained("./weights/Bert")
    del EMB_TOK_EN
    del EMB_EN
