# Install necessary libraries
!pip install torch torchvision torchtext

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
import time

# [Data loading and preprocessing same as before]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, hidden_size, dropout),
            num_layers
        )
        self.decoder = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        output = self.transformer(src, self.src_mask)
        output = self.decoder(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Hyperparameters
embed_size = 200
hidden_size = 200
num_heads = 2
num_layers = 2
vocab_size = len(vocab)
model = TransformerModel(vocab_size, embed_size, num_heads, hidden_size, num_layers).to(DEVICE)

criterion = nn.CrossEntropyLoss()
lr = 5.0
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# Adjust get_batch function
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].t()  # Shape: (seq_len, batch_size)
    target = source[i+1:i+1+seq_len].t().reshape(-1)
    return data, target

# Training and evaluation functions (adjusted for Transformer)
def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)  # Output shape: (seq_len, batch_size, vocab_size)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'| Epoch {epoch+1} | {batch}/{len(train_data) // bptt} batches | '
                  f'lr {scheduler.get_last_lr()[0]:.2f} | ms/batch {elapsed * 1000 / log_interval:.2f} | '
                  f'loss {cur_loss:.2f} | ppl {math.exp(cur_loss):.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

# Train the model
bptt = 35
num_epochs = 5

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(valid_data)
    print('-' * 89)
    print(f'| End of epoch {epoch+1} | Time: {(time.time() - epoch_start_time):.2f}s | '
          f'Valid loss {val_loss:.2f} | Valid ppl {math.exp(val_loss):.2f}')
    print('-' * 89)
    scheduler.step()

# Generate text using the Transformer model
def generate_text(prompt, max_len=50, temperature=1.0):
    model.eval()
    words = tokenizer(prompt)
    input_ids = torch.tensor([vocab[w] for w in words], dtype=torch.long).unsqueeze(1).to(DEVICE)
    generated_words = words.copy()
    with torch.no_grad():
        for _ in range(max_len):
            output = model(input_ids)
            word_weights = output[-1].squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_ids = torch.cat([input_ids, word_idx.unsqueeze(0).unsqueeze(1).to(DEVICE)], dim=0)
            word = vocab.lookup_token(word_idx)
            generated_words.append(word)
            if word == '<eos>':
                break
    return ' '.join(generated_words)

# Few-shot prompting example
prompt = """Once upon a time in a land far, far away, there lived a"""

generated_text = generate_text(prompt)
print(generated_text)