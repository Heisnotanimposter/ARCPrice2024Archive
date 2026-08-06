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

# Build the model
class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(vocab_size, embed_size)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.5)
        else:
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout=0.5)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.init_weights()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                    weight.new_zeros(self.num_layers, bsz, self.hidden_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.hidden_size)

# Hyperparameters
embed_size = 200
hidden_size = 200
num_layers = 2
vocab_size = len(vocab)
model = RNNModel('LSTM', vocab_size, embed_size, hidden_size, num_layers).to(DEVICE)

criterion = nn.CrossEntropyLoss()
lr = 20
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

# Training and evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, vocab_size), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'| Epoch {epoch+1} | {batch}/{len(train_data) // bptt} batches | '
                  f'lr {scheduler.get_last_lr()[0]:.2f} | ms/batch {elapsed * 1000 / log_interval:.2f} | '
                  f'loss {cur_loss:.2f} | ppl {torch.exp(torch.tensor(cur_loss)):.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, vocab_size)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

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

# Function to generate text using the trained model
def generate_text(prompt, max_len=50, temperature=1.0):
    model.eval()
    words = tokenizer(prompt)
    input = torch.tensor([vocab[w] for w in words], dtype=torch.long).unsqueeze(1).to(DEVICE)
    hidden = model.init_hidden(1)
    with torch.no_grad():
        output, hidden = model(input, hidden)
        input = input[-1:]
        generated_words = words.copy()
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
prompt = """The concept of "farduddle" means to jump up and down quickly. An example in a sentence is:
We won the game and started to farduddle in joy.

The term "whatpu" refers to a small, furry animal native to Tanzania. An example in a sentence is:
On our safari, we spotted several whatpus in the trees.

To "blithering" means to speak at length without making much sense. An example in a sentence is:
During the meeting, he kept blithering about irrelevant topics.

Now, use "snollygoster" in a sentence, where "snollygoster" means a shrewd, unprincipled person.

An example in a sentence is:"""

generated_text = generate_text(prompt)
print(generated_text)