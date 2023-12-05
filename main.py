import pandas as pd
from collections import Counter
import spacy
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BartForConditionalGeneration



file_path = "./chinese_english_translation_corpus/train.en"
with open(file_path, 'r', encoding='utf-8') as f:
    eng_text = f.read()

file_path2 = "./chinese_english_translation_corpus/train.zh"
with open(file_path2, 'r', encoding='utf-8') as f2:
    chi_text = f2.read()


MAX_SENTENCES = 100

eng_sentences = eng_text.split('\n')[:MAX_SENTENCES]
chi_sentences = chi_text.split('\n')[:MAX_SENTENCES]


# tokenize sentences
nlp_english = spacy.load("en_core_web_sm")
eng_sentences = [[token.text.lower() for token in nlp_english(sentence)] for sentence in eng_sentences]
print("Tokenized English Sentences:", eng_sentences)


nlp_chinese = spacy.load("zh_core_web_sm")
chi_sentences = [[token.text.lower() for token in nlp_chinese(sentence)] for sentence in chi_sentences]
print("Tokenized Chinese Sentences:", chi_sentences)




# build vocab
eng_to_idx = {"<UNK>": 0}
idx_to_eng = {0: "<UNK"}
index = 1

for sentence in eng_sentences:
    for token in sentence:
        if token not in eng_to_idx:
            eng_to_idx[token] = index
            idx_to_eng[index] = token
            index+=1
print("English vocabulary:", eng_to_idx)


chi_to_idx = {"<UNK>": 0}
idx_to_chi = {0: "<UNK"}
index = 1

for sentence in chi_sentences:
    for token in sentence:
        if token not in chi_to_idx:
            chi_to_idx[token] = index
            idx_to_chi[index] = token
            index+=1
print("Chinese vocabulary:", chi_to_idx)



# indexing and encoding
eng_indexed_sentences = [[eng_to_idx.get(token.lower(), eng_to_idx["<UNK>"]) for token in sentence] for sentence in eng_sentences]
print("English indexed sentences: ", eng_indexed_sentences)

chi_indexed_sentences = [[chi_to_idx.get(token.lower(), chi_to_idx["<UNK>"]) for token in sentence] for sentence in chi_sentences]
print("Chinese indexed sentences: ", chi_indexed_sentences)




# padding or truncating
max_eng_len = max(map(len, eng_indexed_sentences))
print("max english length:", max_eng_len)

max_chi_len = max(map(len, chi_indexed_sentences))
print("max chinese length:", max_chi_len)

max_len = max(max_eng_len, max_chi_len)


eng_padded_sequences = pad_sequence([torch.tensor(eng_seq + [0] * max(0, max_len - len(eng_seq))) for eng_seq in eng_indexed_sentences], batch_first=True, padding_value=0)
print(eng_padded_sequences)

chi_padded_sequences = pad_sequence([torch.tensor(chi_seq + [0] * max(0, max_len - len(chi_seq))) for chi_seq in chi_indexed_sentences], batch_first=True, padding_value=0)
print(chi_padded_sequences)


# load data into model

class TranslationDataset(Dataset):
    def __init__(self, english_sequences, chinese_sequences):
        self.english_sequences = english_sequences
        self.chinese_sequences = chinese_sequences

    def __len__(self):
        return len(self.english_sequences)

    def __getitem__(self, idx):
        return self.english_sequences[idx], self.chinese_sequences[idx]


dataset = TranslationDataset(eng_padded_sequences, chi_padded_sequences)

print(dataset[1])


# define model and train
batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 5

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = [torch.tensor(seq).long() for seq in batch]
        eng_seq, chi_seq = inputs

        assert eng_seq.size(1) == chi_seq.size(1), "Input sequences must have the same length."

        outputs = model(input_ids=eng_seq, labels=chi_seq)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


input_text = eng_text.split('\n')[1]

print(f"input text: {input_text}")
input_tokenized = [token.text.lower() for token in nlp_english(input_text)]
input_idx = [eng_to_idx.get(token.lower(), eng_to_idx["<UNK>"]) for token in input_tokenized]
print(f"input indexed: {input_idx}")
input_padded = input_idx[:max_len] + [0] * max(0, max_len - len(input_idx))
print(f"input padded: {input_padded}")
input_tensor = torch.tensor(input_padded)
print(f"input tensor: {input_tensor}")

model.eval()

with torch.no_grad():
    generated_ids = model.generate(input_tensor.unsqueeze(0))
    print(f"generated: {generated_ids}")


generated_chinese_text = ''.join([idx_to_chi[idx] for idx in generated_ids.squeeze().tolist() if idx in idx_to_chi])

print(f"Generated Chinese text: {generated_chinese_text}")
