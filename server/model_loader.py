import os

from rich import print
from transformers import (AutoModel, AutoModelForQuestionAnswering,
                          AutoTokenizer, CamembertForQuestionAnswering,
                          pipeline)


WEIGHTS_PATH = './weights'
QA_MODEL_NAME_FR = f'{WEIGHTS_PATH}/Camembert_Q_A'
EMB_MODEL_NAME_FR = f'{WEIGHTS_PATH}/Camembert'
QA_MODEL_NAME_EN = f'{WEIGHTS_PATH}/Bert_Q_A'
EMB_MODEL_NAME_EN = f'{WEIGHTS_PATH}/Bert'

_LOADED_MODELS = {}
def load_models():
    print(":floppy_disk: [yellow]Loading FR model ...[/yellow]")
    QA_TOK_FR = AutoTokenizer.from_pretrained(QA_MODEL_NAME_FR)
    QA_MODEL_FR = CamembertForQuestionAnswering.from_pretrained(QA_MODEL_NAME_FR)
    _LOADED_MODELS['FR'] = {
        'QNA': pipeline('question-answering', model=QA_MODEL_FR, tokenizer=QA_TOK_FR),
        'TOK': AutoTokenizer.from_pretrained(EMB_MODEL_NAME_FR),
        'EMB': AutoModel.from_pretrained(EMB_MODEL_NAME_FR)
    }
    print(":floppy_disk: [green]Loaded FR models[/green]")

    print(":floppy_disk: [yellow]Loading EN model[/yellow] ...")
    QA_TOK_EN = AutoTokenizer.from_pretrained(QA_MODEL_NAME_EN)
    QA_MODEL_EN = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME_EN)
    _LOADED_MODELS['EN'] = {
        'QNA': pipeline('question-answering', model=QA_MODEL_EN, tokenizer=QA_TOK_EN),
        'TOK': AutoTokenizer.from_pretrained(EMB_MODEL_NAME_EN),
        'EMB': AutoModel.from_pretrained(EMB_MODEL_NAME_EN)
    }
    print(":floppy_disk: [green]Loaded EN models[/green]")

def get_loading_status():
    return "loaded" if _LOADED_MODELS is not {} else "loading"
    

def get_models_for_lang(lang:str):
    lang_models = _LOADED_MODELS.get(lang.upper())
    if lang_models is None:
        raise RuntimeError()
    else:
        return (lang_models['TOK'], lang_models['EMB'], lang_models['QNA'])

def preload_weights():
    models = {
        "Camembert_Q_A": "illuin/camembert-large-fquad",
        "Camembert": "camembert-base",
        "Bert": "bert-large-uncased",
        "Bert_Q_A": "bert-large-uncased-whole-word-masking-finetuned-squad"
    }

    for folder in models.keys():
        p = f'{WEIGHTS_PATH}/{folder}'
        if not os.path.exists(p):
            os.makedirs(p)

    if not os.path.exists(f'{WEIGHTS_PATH}/Camembert_Q_A/pytorch_model.bin'):
        QA_MODEL_NAME_FR = "illuin/camembert-large-fquad"
        QA_TOK_FR = AutoTokenizer.from_pretrained(QA_MODEL_NAME_FR)
        QA_MODEL_FR = CamembertForQuestionAnswering.from_pretrained(
            QA_MODEL_NAME_FR)
        QA_FR = pipeline('question-answering',
                        model=QA_MODEL_FR, tokenizer=QA_TOK_FR)
        QA_FR.save_pretrained(f'{WEIGHTS_PATH}/Camembert_Q_A')
        del QA_FR
        del QA_TOK_FR
        del QA_MODEL_FR

    if not os.path.exists(f'{WEIGHTS_PATH}/Camembert/pytorch_model.bin'):
        EMB_MODEL_NAME_FR = "camembert-base"
        EMB_TOK_FR = AutoTokenizer.from_pretrained(EMB_MODEL_NAME_FR)
        EMB_FR = AutoModel.from_pretrained(EMB_MODEL_NAME_FR)
        EMB_FR.save_pretrained(f'{WEIGHTS_PATH}/Camembert')
        EMB_TOK_FR.save_pretrained(f'{WEIGHTS_PATH}/Camembert')
        del EMB_TOK_FR
        del EMB_FR

    if not os.path.exists(f'{WEIGHTS_PATH}/Bert_Q_A/pytorch_model.bin'):
        QA_MODEL_NAME_EN = "bert-large-uncased-whole-word-masking-finetuned-squad"
        QA_TOK_EN = AutoTokenizer.from_pretrained(QA_MODEL_NAME_EN)
        QA_MODEL_EN = AutoModelForQuestionAnswering.from_pretrained(
            QA_MODEL_NAME_EN)
        QA_EN = pipeline('question-answering',
                        model=QA_MODEL_EN, tokenizer=QA_TOK_EN)
        QA_EN.save_pretrained(f'{WEIGHTS_PATH}/Bert_Q_A')
        del QA_EN
        del QA_MODEL_EN
        del QA_TOK_EN

    if not os.path.exists(f'{WEIGHTS_PATH}/Bert/pytorch_model.bin'):
        EMB_MODEL_NAME_EN = "bert-large-uncased"
        EMB_TOK_EN = AutoTokenizer.from_pretrained(EMB_MODEL_NAME_EN)
        EMB_EN = AutoModel.from_pretrained(EMB_MODEL_NAME_EN)
        EMB_EN.save_pretrained(f'{WEIGHTS_PATH}/Bert')
        EMB_TOK_EN.save_pretrained(f'{WEIGHTS_PATH}/Bert')
        del EMB_TOK_EN
        del EMB_EN
