import os

import torch
from torch.utils.data import DataLoader

from utils import preprocess, tools
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast







if __name__ == "__main__":

    model = "allenai/scibert_scivocab_uncased"
    max_len = 128
    batch_size = 32
    epochs = 3
    lr = 2e-5
    max_norm = 1.0

    preprocess.set_seed(42)
    device = torch.device('cpu')

    train, train_gold = preprocess.preprocess('./data/neoplasm/train')
    validate, validate_gold = preprocess.preprocess('./data/neoplasm/validate')
    test, test_gold = preprocess.preprocess('./data/neoplasm/test')

    experiment_name = 'train_' + '_epoch_' + str(1) + '_maxlen_' + str(128)
    predict_name = experiment_name + "_test_"
    model_output_path = './output/model'
    evaluation_output_path = './output' + '/' + 'evaluation'
    os.makedirs(model_output_path, exist_ok=True)
    os.makedirs(evaluation_output_path, exist_ok=True)

    prediction_output_path = './output' + '/' + 'prediction'
    os.makedirs(prediction_output_path, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(model, add_prefix_space=True)
    model = BertForSequenceClassification.from_pretrained(model, num_labels=5)
    model.resize_token_embeddings(len(tokenizer))
    model.bert.embeddings.word_embeddings.padding_idx = 1
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    training_set = preprocess.DataSet(train, tokenizer, max_len)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    evaluation_output = {}

    for epoch in range(epochs):
        print(f"Training epoch: {epoch + 1}")
        tools.train(training_loader, model, optimizer, device, max_norm)
        print(f"Validating epoch: {epoch + 1}")
        validate_pred = tools.model_predict(device, model, max_len, tokenizer, validate)


















