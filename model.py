import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
n_embd = 64
num_heads = 6
num_layers = 4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'




class SentenceEmbedding(nn.Module):
    """ For a given sentence, create an embedding """
    def __init__(self, max_seq_len, d_model, lang_to_idx, START_TOKEN, END_TOKEN, PADDING_TOKEN, split_by_space=False):
        super().__init__()
        self.vocab_size = len(lang_to_idx)
        self.max_seq_len = max_seq_len
        self.lang_to_idx = lang_to_idx
        self.token_embedding_table = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.split_by_space = split_by_space

    def tokenize_batch(self, batch, start_token, end_token):
        def tokenize(sentence, start_token, end_token):
            tokens = sentence.split() if self.split_by_space else list(sentence)
            print(f"tokens: {tokens}")
            word_indices = [self.lang_to_idx[token] for token in tokens]
            print(f"sentence word indices: {word_indices}")
            if start_token:
                word_indices.insert(0, self.lang_to_idx[self.START_TOKEN])
            if end_token:
                word_indices.append(self.lang_to_idx[self.END_TOKEN])
            for _ in range(len(word_indices), self.max_seq_len):
                word_indices.append(self.lang_to_idx[self.PADDING_TOKEN])
            return torch.tensor(word_indices)

        tokenized = []
        for sentence_num in range(len(batch)):
            print(batch[sentence_num])
            t = tokenize(batch[sentence_num], start_token, end_token)
            print(f"[DBG] tokenized batch [i]: {t}")
            print(f"[DBG] tokenized batch [i] size: {t.size()}")
            tokenized.append(t)
        tokenized = torch.stack(tokenized)
        return tokenized

    def forward(self, x, start_token, end_token):
        x = self.tokenize_batch(x, start_token, end_token)
        print(f"after tokenize_batch: {x}")
        print(f"x shape in sentenceembed: {x.shape}")
        B, T = x.shape

        mask = torch.eq(x, 2)
        print(f"padding mask: {mask}")

        # x = self.embedding(x)
        tok_emb = self.token_embedding_table(x)
        print(f"token embedding: {tok_emb}")
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        print(f"positional embedding: {pos_emb}")

        x = tok_emb + pos_emb
        print(f"x after tok and pos embeddings: {x}")
        x = self.dropout(x)
        print(f"after dropout: {x}")
        print(x.shape)
        x[mask.unsqueeze(-1).expand_as(x)] = -5
        print(f"after mask x: {x}")
        return x

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        print(f"inputs before layernorm: {inputs.size()}")
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        print(f"out after layernorm: {out.size()}")
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layers = num_layers
        self.selfattention = MultiHeadAttention(num_heads, n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.ffwd = FeedForward(n_embd)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])

    def forward(self, x):
        x_clone = x.clone()
        x = self.selfattention(x)
        x = self.dropout1(x)
        x = self.norm1(x + x_clone)
        x = self.ffwd(x)
        x = self.dropout2(x)
        x = self.norm2(x + x_clone)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, max_seq_len, lang_to_idx, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_seq_len, d_model, lang_to_idx, START_TOKEN, END_TOKEN, PADDING_TOKEN, split_by_space=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model) for _ in range(num_layers)])

    def forward(self, x, start_token, end_token):
        print(f"x before passing through encoder: {len(x)}")
        print(f"x before sentence embedding: {x}")
        x = self.sentence_embedding(x, start_token, end_token)
        print(f"x after sentence embedding: {x}")
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.layers = num_layers
        self.selfattention = MultiHeadAttention(num_heads, n_embd)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.crossattention = MultiHeadCrossAttention(num_heads, n_embd, max_seq_len)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.ffwd = FeedForward(n_embd)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])

    def forward(self, x, y):
        y_clone = y.clone()
        y = self.selfattention(y)
        y = self.dropout1(y)
        y = self.norm1(y + y_clone)
        print(f"y in decoder just before cross attention {y.shape}")

        y = self.crossattention(x, y)
        print(f"y in decoder after crossattention {y.shape}")
        y = self.dropout2(y)
        y = self.norm2(y + y_clone)

        y = self.ffwd(y)
        y = self.dropout3(y)
        y = self.norm3(y + y_clone)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, max_seq_len, lang_to_idx, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_seq_len, d_model, lang_to_idx, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = nn.ModuleList([DecoderLayer(d_model, max_seq_len) for _ in range(num_layers)])

    def forward(self, x, y, start_token, end_token):
        print(f"x before passing through decoder: {x.size()}")
        print(f"y before passing through decoder: {len(y)}")
        y = self.sentence_embedding(y, start_token, end_token)
        for layer in self.layers:
            y = layer(x, y)
        print(f"x after passing through decoder: {x.size()}")
        print(f"y after passing through decoder: {len(y)}")
        return y

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(f"in head forward x shape: {x.shape}") # [20, 86, 64]
        print(f"x: {x}")
        B,T,C = x.shape # batch_size, max_seq_len, C
        k = self.key(x)   # (B,T,C)
        q = self.query(x)   # (B,T,C)

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        # padding_mask = (x[:, :, 0] != 2).unsqueeze(-1) # (B, T, 1)

        print(f"wei shape: {wei.shape}")
        print(f"wei: {wei}")
        # print(f"pad mask shape: {padding_mask.shape}")
        # print(f"pad mask: {padding_mask}")

        # wei = wei.masked_fill(~padding_mask, float('-inf'))  # Apply padding mask

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class CrossAttentionHead(nn.Module):
    """ one head of cross-attention """

    def __init__(self, head_size, max_seq_len):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        print(f"in cross attention head forward x shape: {x.shape}") # [20, 86, 64]
        B, T, C = x.shape
        k = self.key(y)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # padding_mask = (x[:, :, 0] != 2).unsqueeze(-1).bool() # (B, T, 1)
        # future_mask = self.tril[None, None, :].bool() # [1, 1, T, T]
        future_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()

        # print(f"pad mask shape cross: {padding_mask.shape}")
        # print(f"pad mask cross: {padding_mask}")
        print(f"future mask shape: {future_mask.shape}")
        print(f"future mask: {future_mask}")

        # combined_mask = torch.logical_and(future_mask[:, :T, :T], padding_mask)
        # combined_mask = padding_mask | future_mask
        # print(f"combined mask shape: {combined_mask.shape}") # should be [B, 1, T, T]
        # print(f"combined mask: {combined_mask}")

        print(f"wei shape cross attention: {wei.shape}") # should be [B,T,T]

        wei = wei.masked_fill(future_mask, float('-inf'))  # future mask
        # wei = wei.masked_fill(future_mask[:, :T, :T] == 0 | (padding_mask == 0), float('-inf'))  # padding and future mask
        wei = F.softmax(wei, dim=-1)
        # print(f"wei softmax shape cross attention: {wei.shape}")
        # print(f"wei softmax cross attention: {wei}")
        wei = self.dropout(wei)
        v = self.value(y)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(f"x before multiheadattention forward: {x.size()}")
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        print(f"out after multiheadattention forward: {out.size()}")
        return out

class MultiHeadCrossAttention(nn.Module):
    """ multiple heads of cross-attention in parallel """

    def __init__(self, num_heads, head_size, max_seq_len):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size, max_seq_len) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        print(f"x before passing through multiheadcrossattention: {x.size()}")
        out = torch.cat([h(x, y) for h in self.heads], dim=-1)
        out = self.dropout(out)
        out = self.proj(out)
        print(f"out after passing through multiheadcrossattention: {out.size()}")
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        print(f"x size before feedforward: {x.size()}")
        x = self.net(x)
        print(f"x shape after feedforward: {x.size()}")
        return x

class Transformer(nn.Module):

    def __init__(self, d_model, max_seq_len, chi_vocab_size,
                 eng_to_index, chi_to_index,
                 START_TOKEN, END_TOKEN, PADDING_TOKEN                 
                 ):
        super().__init__()
        self.encoder = Encoder(d_model, max_seq_len, eng_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, max_seq_len, chi_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, chi_vocab_size)

    def forward(self, x, y, enc_start_token=False, enc_end_token=False, dec_start_token=False, dec_end_token=False):
        # x and y are the english and chinese batches (batch_size number tuples of sentences)
        print("in transformer model forward")
        x = self.encoder(x, start_token=enc_start_token, end_token=enc_end_token) # padding mask used in encoder
        print(f"x after passing through encoder: {x}")
        out = self.decoder(x, y, start_token=dec_start_token, end_token=dec_end_token) # future mask and padding mask used in decoder
        print(f"out after passing through decoder: {out}")
        out = self.linear(out)
        print(f"out after passing through transformer: {out.size()}")
        return out


def get_model(d_model, max_seq_len, chi_vocab_size, eng_to_index, chi_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
    """Instantiate a transformer object and return it."""
    model = Transformer(d_model, max_seq_len, chi_vocab_size, eng_to_index, chi_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
    return model
