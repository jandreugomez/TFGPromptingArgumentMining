import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score


class Constants:
    OUTPUT_LABELS = ['0', 'B-Claim', 'I-Claim', 'B-Premise', 'I-Premise']
    LABELS_TO_IDS = {v: k for k, v in enumerate(OUTPUT_LABELS)}

def train(training_loader, model, optimizer, device, max_norm):

    tr_loss = 0
    tr_accuracy = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.to(device)
    model.train()

    epoch_iterator = tqdm(training_loader, desc="Iteration")
    for idx, batch in enumerate(epoch_iterator):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        tr_logits = outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        flattened_targets = labels.view(-1)
        active_logits = tr_logits.view(-1, model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)

        active_accuracy = labels.view(-1) != -100

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=max_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def get_sentence_predictions(device, model, max_len, tokenizer, sentence):
    inputs = tokenizer(sentence,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=max_len,
                       return_tensors="pt")
    ids = inputs["inputs_ids"].to(device)
    mask = inputs["attention_mask"].to(device)

    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [Constants.OUTPUT_LABELS[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
        else:
            continue
    return prediction

def get_final_prediction(test_text_df, y_pred):
    final_preds = []
    for i in tqdm(range(len(test_text_df))):
        idx = test_text_df.id.values[i]
        pred = [x.replace('B-', '').replace('I-', '') for x in y_pred[i]]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == '0':
                j += 1
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1

            if cls !='0' and cls != '' and end - j > 10:
                final_preds.append((idx, cls, ' '.join(map(str, list(range(j, end))))))

            j = end
    return final_preds

def model_predict(device, model, max_len, tokenizer, dataframe):
    pred = []
    for i, t in enumerate(dataframe['text_split'].tolist()):
        o = get_sentence_predictions(device, model, max_len, tokenizer, t)
        pred.append(o)

    pred = pd.DataFrame(get_final_prediction(dataframe, pred))
    return pred







