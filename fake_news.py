import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import FakeNewsDataset

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from model_uploader import Model_Uploader

#For distilbert-base-cased
from transformers import DistilBertTokenizerFast as TokenizerClass
from transformers import DistilBertForSequenceClassification as ModelClass

from transformers import AdamW

from utils import DistilBertClassiferHyperparams


from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initalize params
hyperparams = DistilBertClassiferHyperparams()
hyperparams.epochs = 10
hyperparams.batch_size = 32
hyperparams.learning_rate = 1e-5


dataset_directory_path = 'dataset/'
saved_weights_path = 'model/distilbert-base-cased_saved_weights.pt'


def prepare_data():
  training_set = pd.read_csv(dataset_directory_path + "Constraint_English_Training_Set.csv")
  training_texts = training_set['tweet']
  training_labels = training_set['label'].map({'real': 0, 'fake': 1})

  testing_set = pd.read_csv(dataset_directory_path + "english_test_with_labels.csv")
  testing_texts = testing_set['tweet']
  testing_labels = testing_set['label'].map({'real': 0, 'fake': 1})

  validation_set = pd.read_csv(dataset_directory_path + "Constraint_English_Validation_Set.csv")
  validation_texts = validation_set['tweet']
  validation_labels = validation_set['label'].map({'real': 0, 'fake': 1})
  
  return training_texts, training_labels, testing_texts, testing_labels, validation_texts, validation_labels


def train_model():
  # prepare data
  training_texts, training_labels, testing_texts, testing_labels, validation_texts, validation_labels = prepare_data()

  # initialize model related preparation
  model_name = 'distilbert-base-cased'
  tokenizer = TokenizerClass.from_pretrained(model_name)
  tokenizer.do_lower_case = False
  pt_model = ModelClass.from_pretrained(model_name, num_labels=2)

  training_encodings = tokenizer(training_texts.tolist(), max_length=100, padding='max_length', truncation=True)
  training_encodings.do_lower_case = False
  validation_encodings = tokenizer(validation_texts.tolist(), max_length=100, padding='max_length', truncation=True)
  validation_encodings.do_lower_case = False
  testing_encodings = tokenizer(testing_texts.tolist(), max_length=100, padding='max_length', truncation=True)
  testing_encodings.do_lower_case = False

  training_dataset = FakeNewsDataset(training_encodings, training_labels)
  validation_dataset = FakeNewsDataset(validation_encodings, validation_labels)
  testing_dataset = FakeNewsDataset(testing_encodings, testing_labels)

  training_loader = DataLoader(training_dataset, batch_size=hyperparams.batch_size, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=hyperparams.batch_size, shuffle=True)
  testing_loader = DataLoader(testing_dataset, batch_size=hyperparams.batch_size, shuffle=True)

  # Need to also tune the pre-trained distilbert with the classification dataset to acheive better result
  for param in pt_model.distilbert.parameters():
    param.requires_grad = True

  optimizer = AdamW(pt_model.parameters(), lr = hyperparams.learning_rate)



  pt_model.to(device)
  pt_model.train()

  training_losses, validation_losses = [], []
  best_validation_loss = float('inf')
  for epoch in range(hyperparams.epochs):
      print('Epoch {:} / {:}:'.format(epoch + 1, hyperparams.epochs))
      training_loss = train(pt_model, training_loader, optimizer)
      validation_loss = evaluate(pt_model, validation_loader, optimizer)
      if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(pt_model.state_dict(), saved_weights_path)
      training_losses.append(training_loss)
      validation_losses.append(validation_loss)


  #load weights of best model
  pt_model.load_state_dict(torch.load(saved_weights_path))

# get predictions for test data
  with torch.no_grad():
    total_preds, true_labels = [], []
    for step, batch in enumerate(testing_loader):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = pt_model(input_ids, attention_mask=attention_mask)
      logits = outputs[0]
      logits = logits.detach().cpu().numpy()
      total_preds.append(logits)
      label_ids = labels.to('cpu').numpy()
      true_labels.append(label_ids)

    total_preds = np.concatenate(total_preds, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    preds = np.argmax(total_preds, axis = 1)
    print(classification_report(true_labels, preds))

    acc = accuracy_score(true_labels, preds)

    if acc > 0.9:
      mu = Model_Uploader(saved_weights_path)
      mu.upload_2_minio()
    else:
      print("Accuracy is not enough, reject to upload model")

def train(model, dataloader, optimizer):
    print('Training...')
    model.train()
    total_loss = 0
    pg = tqdm(dataloader, leave=False, total=len(dataloader))
    for step, batch in enumerate(pg):
      if step % 50 == 0 and not step == 0:
        print('  Batch {:>5,}  of  {:>5,}'.format(step, len(dataloader)))
      optimizer.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs[0]
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, optimizer):
    print('Evaluating...')
    model.eval()
    total_loss = 0
    pg = tqdm(dataloader, leave=False, total=len(dataloader))
    for step, batch in enumerate(pg):
      if step % 50 == 0 and not step == 0:
        print('  Batch {:>5,}  of  {:>5,}'.format(step, len(dataloader)))
      optimizer.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs[0]
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
    avg_loss = total_loss / len(dataloader)
    return avg_loss