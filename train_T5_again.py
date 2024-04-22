from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

import torch
import json
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
#import nltk
#import spacy
#import string
#import evaluate  # Bleu
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pandas as pd
import numpy as np
#import transformers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

#MODEL = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
#TOKENIZER = AutoTokenizer.from_pretrained("google-t5/t5-small")

#MODEL = AutoModelForSeq2SeqLM.from_pretrained("t5_model")
#TOKENIZER = AutoTokenizer.from_pretrained("t5_model")

TOKENIZER = T5Tokenizer.from_pretrained("t5-base")
MODEL = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
OPTIMIZER = Adam(MODEL.parameters(), lr=0.0001)
Q_LEN = 256   # Question Length
T_LEN = 100    # Target Length
BATCH_SIZE = 4
DEVICE = "cuda"


def prepare_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    articles = []
    
    for article in data:      
        inputs = {"question": article['instruction'], "answer": article['generated_output']}
        articles.append(inputs)

    return articles

data = prepare_data("output_dataset.json")
data = pd.DataFrame(data)


class QA_Dataset(Dataset):
    def __init__(self, tokenizer, dataframe, q_len, t_len):
        self.tokenizer = tokenizer
        self.q_len = q_len
        self.t_len = t_len
        self.data = dataframe
        self.questions = self.data["question"]
        self.answer = self.data['answer']
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answer[idx]
        
        question_tokenized = self.tokenizer(question, max_length=self.q_len, padding="max_length",
                                                    truncation=True, pad_to_max_length=True, add_special_tokens=True)
        answer_tokenized = self.tokenizer(answer, max_length=self.t_len, padding="max_length", 
                                          truncation=True, pad_to_max_length=True, add_special_tokens=True)
        
        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.long)
        labels[labels == 0] = -100
        
        return {
            "input_ids": torch.tensor(question_tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(question_tokenized["attention_mask"], dtype=torch.long),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(answer_tokenized["attention_mask"], dtype=torch.long)
        }


# Dataloader

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_sampler = RandomSampler(train_data.index)
val_sampler = RandomSampler(val_data.index)

qa_dataset = QA_Dataset(TOKENIZER, data, Q_LEN, T_LEN)

train_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(qa_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

train_loss = 0
val_loss = 0
train_batch_count = 0
val_batch_count = 0

train_losses = []
val_losses = []

for epoch in range(40):
    MODEL.to(DEVICE)
    MODEL.train()
    for batch in tqdm(train_loader, desc="Training batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        train_loss += outputs.loss.item()
        train_batch_count += 1

    average_train_loss = train_loss / train_batch_count
    train_losses.append(average_train_loss)
    
    #Evaluation
    MODEL.eval()
    for batch in tqdm(val_loader, desc="Validation batches"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        outputs = MODEL(
                          input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          decoder_attention_mask=decoder_attention_mask
                        )

        OPTIMIZER.zero_grad()
        outputs.loss.backward()
        OPTIMIZER.step()
        val_loss += outputs.loss.item()
        val_batch_count += 1
    
    average_val_loss = val_loss / val_batch_count
    val_losses.append(average_val_loss)

    print(f"{epoch+1}/{2} -> Train loss: {train_loss / train_batch_count}\tValidation loss: {val_loss/val_batch_count}")

    if epoch > 2:
        if abs(train_losses[-1] - train_losses[-2]) < 0.01 and abs(val_losses[-1] - val_losses[-2]) < 0.01:
            break


# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('training_validation_losses.png')
plt.show()

MODEL.save_pretrained("t5_model")
TOKENIZER.save_pretrained("t5_model")