# Install necessary libraries
!pip install torch torchvision torchtext

# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import time

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
tokenizer = get_tokenizer('basic_english')

# Build vocabulary
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

train_iter = WikiText2(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

# Prepare data
def data_process(raw_text_iter):
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(data)

train_iter, valid_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
valid_data = data_process(valid_iter)
test_data = data_process(test_iter)

# Create batches
def batchify(data, bsz):
    # Divide the dataset into bsz parts
    nbatch = data.size(0) // bsz
    # Trim off extra elements
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide data across bsz batches
    data = data.view(bsz, -1).t().contiguous()
    return data.to(DEVICE)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)
valid_data = batchify(valid_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# Build the model with attention
class AttentionRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(AttentionRNNModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.5, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        emb = self.dropout(self.encoder(input))  # Shape: (batch_size, seq_len, embed_size)
        output, hidden = self.rnn(emb, hidden)   # Output shape: (batch_size, seq_len, hidden_size)
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)  # Shape: (batch_size, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), output).squeeze(1)        # Shape: (batch_size, hidden_size)
        decoded = self.decoder(self.dropout(context))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.rnn.num_layers, bsz, self.rnn.hidden_size),
                weight.new_zeros(self.rnn.num_layers, bsz, self.rnn.hidden_size))

# Hyperparameters
embed_size = 200
hidden_size = 200
num_layers = 2
vocab_size = len(vocab)
model = AttentionRNNModel(vocab_size, embed_size, hidden_size, num_layers).to(DEVICE)

criterion = nn.CrossEntropyLoss()
lr = 20
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

# Training and evaluation functions (same as before)
# [Include the train, evaluate, get_batch, and repackage_hidden functions from previous code]

# Train the model
bptt = 35
num_epochs = 5
best_val_loss = None

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(valid_data)
    print('-' * 89)
    print(f'| End of epoch {epoch+1} | Time: {(time.time() - epoch_start_time):.2f}s | '
          f'Valid loss {val_loss:.2f} | Valid ppl {torch.exp(torch.tensor(val_loss)):.2f}')
    print('-' * 89)
    scheduler.step()

# Generate text using the trained model with attention
def generate_text(prompt, max_len=50, temperature=1.0):
    model.eval()
    words = tokenizer(prompt)
    input = torch.tensor([vocab[w] for w in words], dtype=torch.long).unsqueeze(0).to(DEVICE)
    hidden = model.init_hidden(1)
    generated_words = words.copy()
    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = vocab.lookup_token(word_idx)
            generated_words.append(word)
            if word == '<eos>':
                break
    return ' '.join(generated_words)

# Few-shot prompting example
prompt = """Once upon a time in a land far, far away, there lived a"""

generated_text = generate_text(prompt)
print(generated_text)