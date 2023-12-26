import pandas as pd
from collections import Counter
import spacy
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BartForConditionalGeneration
import model
import numpy as np
from torch import nn


file_path = "./chinese_english_translation_corpus/train.en"
with open(file_path, 'r', encoding='utf-8') as f:
    eng_text = f.read()

file_path2 = "./chinese_english_translation_corpus/train.zh"
with open(file_path2, 'r', encoding='utf-8') as f2:
    chi_text = f2.read()


MAX_SENTENCES = 8000

eng_sentences = eng_text.split('\n')[:MAX_SENTENCES]
chi_sentences = chi_text.split('\n')[:MAX_SENTENCES]

eng_sentences = [sentence.lower() for sentence in eng_sentences]
chi_sentences = [sentence.lower() for sentence in chi_sentences]
print(f"eng_sentences: {eng_sentences}", '\n')
print(f"chi_sentences: {chi_sentences}", '\n')

# build vocab
START_TOKEN = 'start_token'
PADDING_TOKEN = 'pad_token'
END_TOKEN = 'end_token'
eng_vocab = [START_TOKEN, END_TOKEN, PADDING_TOKEN]
chi_vocab = [START_TOKEN, END_TOKEN, PADDING_TOKEN]

for sentence in eng_sentences:
    for word in sentence.split():
    # for word in sentence:
        if word not in eng_vocab:
            eng_vocab.append(word)
print("english vocab:", eng_vocab, '\n')

eng_to_idx = {v:k for k,v in enumerate(eng_vocab)}
idx_to_eng = {k:v for k,v in enumerate(eng_vocab)}
print("English to index:", eng_to_idx, '\n')

for sentence in chi_sentences:
    for word in sentence:
        if word not in chi_vocab:
            chi_vocab.append(word)
print("chinese vocab:", chi_vocab, '\n')

chi_to_idx = {v:k for k,v in enumerate(chi_vocab)}
idx_to_chi = {k:v for k,v in enumerate(chi_vocab)}
print("Chinese to index:", chi_to_idx, '\n')


max_eng_len = max(len(sentence.split()) for sentence in eng_sentences)
print("max english length:", max_eng_len)

max_chi_len = max(map(len, chi_sentences))
print("max chinese length:", max_chi_len)


# eng_lengths = [len(sentence.split()) for sentence in eng_sentences]
# chi_lengths = [len(sentence) for sentence in chi_sentences]

# avg_eng_len = sum(eng_lengths) / len(eng_lengths)
# avg_chi_len = sum(chi_lengths) / len(chi_lengths)

# print("Average English length:", avg_eng_len)
# print("Average Chinese length:", avg_chi_len)

# max english length: 108
# max chinese length: 33
# Average English length: 7.949
# Average Chinese length: 12.761


valid_sentences = []
for i in range(len(eng_sentences)):
    english_sent, chinese_sent = eng_sentences[i], chi_sentences[i]
    eng_len = len(english_sent.split())
    chi_len = len(list(chinese_sent))
    if 6 < eng_len < 10 and 11 < chi_len < 15:
        valid_sentences.append(i)

eng_sentences = [eng_sentences[i] for i in valid_sentences]
chi_sentences = [chi_sentences[i] for i in valid_sentences]

print(f"Number of valid sentences: {len(valid_sentences)}")

# print(eng_sentences)
# print(chi_sentences)

# max_eng_len = max(map(len, eng_sentences.split()))
max_eng_len = max(len(sentence.split()) for sentence in eng_sentences)
print("new max english length:", max_eng_len)

max_chi_len = max(map(len, chi_sentences))
print("new max chinese length:", max_chi_len)


# hyperparameters
n_embd = 64
# batch_size = len(valid_sentences) // 10
batch_size = 100
max_sequence_length = max(max_eng_len, max_chi_len) + 2
print(f"max sequence length: {max_sequence_length}")
num_epochs = 100
learning_rate = 1e-3
chi_vocab_size = len(chi_vocab)
print(f"chi vocab size: {chi_vocab_size}")



# load data into model
class TranslationDataset(Dataset):
    def __init__(self, english_sequences, chinese_sequences):
        self.english_sequences = english_sequences
        self.chinese_sequences = chinese_sequences

    def __len__(self):
        return len(self.english_sequences)

    def __getitem__(self, idx):
        return self.english_sequences[idx], self.chinese_sequences[idx]


dataset = TranslationDataset(eng_sentences, chi_sentences)

trainer = DataLoader(dataset, batch_size)
iterator = iter(trainer)



model = model.get_model(n_embd, max_sequence_length, chi_vocab_size, eng_to_idx, chi_to_idx, START_TOKEN, END_TOKEN, PADDING_TOKEN)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(trainer)
    for batch_num, batch in enumerate(iterator):
        model.train()
        eng_batch, chi_batch = batch
        # decoder self attention mask should prevent attending to future positions in the sequence
        # upper triangular set to -infinity; diagonal and lower triangular set to 0
        optimizer.zero_grad()
        # print(f"english batch: {eng_batch}")
        # print(f"chinese batch: {chi_batch}")

        chi_predictions = model(eng_batch, chi_batch, enc_start_token=False, enc_end_token=False, dec_start_token=True, dec_end_token=True)

        # print(f"chinese predictions: {chi_predictions}")
        # print(chi_predictions.shape)

        ground_truth_labels = model.decoder.sentence_embedding.tokenize_batch(chi_batch, start_token=False, end_token=True)
        # print(f"ground truth labels shape: {ground_truth_labels.shape}")

        loss = criterion(chi_predictions.view(-1, len(chi_vocab)).to(device), ground_truth_labels.view(-1).to(device)).to(device)
        valid_indicies = torch.where(ground_truth_labels.view(-1) == chi_to_idx[PADDING_TOKEN], False, True)
        print(f"loss sum: {loss.sum()}")
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optimizer.step()

        chi_sentence_predicted = torch.argmax(chi_predictions[0], axis=1)
        predicted_sentence = ""
        for idx in chi_sentence_predicted:
            if idx == chi_to_idx[END_TOKEN]:
                break
            predicted_sentence += idx_to_chi[idx.item()]

        print(f"Iteration {batch_num} : item loss: {loss.item()}")
        print(f"English: {eng_batch[0]}")
        print(f"Chinese Translation: {chi_batch[0]}")
        print(f"Chinese Prediction: {predicted_sentence}")

