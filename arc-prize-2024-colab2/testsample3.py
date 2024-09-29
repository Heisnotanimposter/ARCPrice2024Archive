openai_api_key = "your-api-key-here"

"""## **Setup**

This does a few things:
* Installs Python packages and sets OpenAI API key.
* Downloads the Abstract Reasoning Corpus (ARC) benchmark.

**Note:** only needs a CPU (public) runtime.
"""

!pip install openai transformers

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import openai
import pickle
from transformers import GPT2Tokenizer
# import tiktoken  # Faster than GPT2Tokenizer.

openai.api_key = openai_api_key

if not os.path.exists("ARC"):
  !git clone https://github.com/fchollet/ARC

"""## **API:** Large Language Models

Define helper functions to call large language models and the tokenizer.

**Note:** this can get expensive.
"""

model = "text-davinci-003"
token_limit = 4096

def LLM(prompt, stop=None, max_tokens=256, temperature=0):
  responses = openai.Completion.create(engine=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
  text = [response['text'] for response in responses['choices']]
  return text

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

LLM("hello world!")

"""## **Alphabet:** Token Set

Build a fixed token set by random sampling from the LLM's token vocabulary.
"""

item_delim = tokenizer.encode(",")
row_delim = tokenizer.encode("\n")
sample_delim = tokenizer.encode("---\n")

# Handpicked: comma-separated number matrices.
alphabet = [tokenizer.encode(" " + str(a))[0] for a in range(10)]
value_to_token = lambda x: {i:a for i, a in enumerate(alphabet)}[x]

# Random sampled tokens.
# seed_offset = 0
# np.random.seed(42 + seed_offset)
# alphabet = [int(i) for i in np.random.randint(tokenizer.vocab_size, size=10)]
# value_to_token = lambda x: {i:a for i, a in enumerate(alphabet)}[x]

print("Token Set:", {i:value_to_token(i) for i in np.arange(10)})

"""## **Load:** ARC Benchmark

Load tasks from the ARC benchmark.
"""

def state_to_tokens(state, value_to_token_fn):
  tokens = []
  for row in state:
    for i, value in enumerate(row):
      tokens +=[value_to_token_fn(value)]
      if i < len(row) - 1:
        tokens += item_delim
    tokens += row_delim
  return tokens


def task_json_to_tokens(task_json, value_to_token_fn):

  # Training examples.
  train_samples = []
  for sample in task_json["train"]:
    tokens = []
    tokens += tokenizer.encode("input:\n")
    tokens += state_to_tokens(sample["input"], value_to_token_fn)
    tokens += tokenizer.encode("output:\n")
    tokens += state_to_tokens(sample["output"], value_to_token_fn)
    tokens += sample_delim
    train_samples.append(tokens)

  # Testing examples.
  test_inputs = []
  test_outputs = []
  for sample in task_json["test"]:
    inputs, outputs = [], []
    inputs += tokenizer.encode("input:\n")
    inputs += state_to_tokens(sample["input"], value_to_token_fn)
    inputs += tokenizer.encode("output:\n")
    test_inputs.append(inputs)
    outputs += state_to_tokens(sample["output"], value_to_token_fn)
    test_outputs.append(outputs)
  return train_samples, test_inputs, test_outputs

tasks_jsons = []
tasks_names = []
tasks_len = []
task_dir = "ARC/data/training"
for task_file in sorted(os.listdir(task_dir)):
  with open(os.path.join(task_dir, task_file)) as fid:
    task_json = json.load(fid)
  tasks_jsons.append(task_json)
  tasks_names.append(task_file)
  tokens, _, _ = task_json_to_tokens(task_json, value_to_token)
  tasks_len.append(np.sum([len(sample) for sample in tokens]))

task_dir = "ARC/data/evaluation"
for task_file in sorted(os.listdir(task_dir)):
  with open(os.path.join(task_dir, task_file)) as fid:
    task_json = json.load(fid)
  tasks_jsons.append(task_json)
  tasks_names.append(task_file)
  tokens, _, _ = task_json_to_tokens(task_json, value_to_token)
  tasks_len.append(np.sum([len(sample) for sample in tokens]))

sorted_task_ids = np.argsort(tasks_len)

print("Total number of tasks:", len(sorted_task_ids))

"""## **Example:** ARC Problem

Show the LLM prompt for an ARC problem and visualize the grids used as inputs and outputs.
"""

colors = [(0, 0, 0),
          (0, 116, 217),
          (255, 65, 54),
          (46, 204, 6),
          (255, 220, 0),
          (170, 170, 170),
          (240, 18, 190),
          (255, 133, 27),
          (127, 219, 255),
          (135, 12, 37)]

def grid_to_img(grid):
  grid = np.int32(grid)
  scale = 10
  img = np.zeros((grid.shape[0] * scale + 1, grid.shape[1] * scale + 1, 3), dtype=np.uint8)
  for r in range(grid.shape[0]):
    for c in range(grid.shape[1]):
      img[r*scale+1:(r+1)*scale, c*scale+1:(c+1)*scale, :] = colors[grid[r, c]]
  new_img = img.copy()
  new_img[0::10, :, :] = np.uint8(np.round((0.7 * np.float32(img[0::10, :, :]) + 0.3 * 255)))
  new_img[:, 0::10, :] = np.uint8(np.round((0.7 * np.float32(img[:, 0::10, :]) + 0.3 * 255)))
  return new_img

example_json = tasks_jsons[sorted_task_ids[0]]

context = []
train_xy, test_x, test_y = task_json_to_tokens(example_json, value_to_token)
for sample in train_xy:
  context += sample
context += test_x[0]

print("PROMPT:")
print(tokenizer.decode(context, skip_special_tokens=True))
print("SOLUTION:")
print(tokenizer.decode(test_y[0], skip_special_tokens=True))

# Show problem.
print("TRAIN:")
for i, ex in enumerate(example_json["train"]):
  in_img = grid_to_img(ex["input"])
  out_img = grid_to_img(ex["output"])
  plt.subplot(1, 2, 1); plt.imshow(grid_to_img(ex["input"]))
  plt.subplot(1, 2, 2); plt.imshow(grid_to_img(ex["output"]))
  plt.show()
print("TEST:")
for i, ex in enumerate(example_json["test"]):
  in_img = grid_to_img(ex["input"])
  out_img = grid_to_img(ex["output"])
  plt.subplot(1, 2, 1); plt.imshow(grid_to_img(ex["input"]))
  plt.subplot(1, 2, 2); plt.imshow(grid_to_img(ex["output"]))
  plt.show()

"""## **Evaluate:** ARC Benchmark

Evaluate on the available 800 tasks.

**Note:** LLM temperature is set to 0 (deterministic), but your results might still vary depending on stability of the API.
"""

success = {}
for task_id in sorted_task_ids:
  task_json, task_name = tasks_jsons[task_id], tasks_names[task_id]

  # Lazy load: skip evals where we already have results.
  if task_name in success:
    continue

  # Build context and expected output labels.
  context = []
  batch_prompts = []
  batch_labels = []
  train_xy, test_x, test_y = task_json_to_tokens(task_json, value_to_token)
  test_num_tokens = np.max([len(x) + len(y) for x, y in zip(test_x, test_y)])
  for sample in train_xy:
    if len(context) + len(sample) + test_num_tokens > token_limit:  # Ensure both train and test examples can fit in the prompt.
      break
    context += sample

  # There can be multiple test examples so put them in the same batch.
  for x, y in zip(test_x, test_y):
    batch_prompts.append(context + x)
    batch_labels.append(y)

  # Run LLM.
  try:
    stop_token = tokenizer.decode(sample_delim, skip_special_tokens=True)
    max_tokens = int(np.max([len(y) for y in test_y])) + 10
    batch_responses = LLM(batch_prompts, stop=stop_token, max_tokens=max_tokens, temperature=0)
  except Exception as e:
    print(task_name, f"LLM failed. {e}")
    continue

  # Check answers and save success rates.
  success[task_name] = 0
  for response, label in zip(batch_responses, batch_labels):
    label_str = tokenizer.decode(label, skip_special_tokens=True)
    is_success = label_str.strip() in response
    success[task_name] += is_success / len(batch_labels)
  success[task_name] = int(success[task_name] > 0.99)  # All test cases need to correct.

  # Debug prints.
  total_success = np.sum(list(success.values()))
  print(task_name, "Success:", success[task_name], "Total:", f"{total_success} / {len(success)}")

  # # Save results.
  # result_file = f"arc-{model}-alphabet-{'-'.join(map(str, alphabet))}.pkl"
  # with open(result_file, 'wb') as fid:
  #   pickle.dump(success, fid, protocol=pickle.HIGHEST_PROTOCOL)

  print('Import Necessary Modules')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm # taqadum for loading ui
import json # for read json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from collections import Counter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_FOLDER = '/kaggle/input/arc-prize-2024'
CMAP = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
NORM = colors.Normalize(vmin=0, vmax=10)
BATCH_SIZE = 128
file_paths = {
    "train_file_path": {
        "data_file_path": f"{BASE_FOLDER}/arc-agi_training_challenges.json",
        "target_file_path": f"{BASE_FOLDER}/arc-agi_training_solutions.json"
    },
    "val_file_path": {
        "data_file_path": f"{BASE_FOLDER}/arc-agi_evaluation_challenges.json",
        "target_file_path": f"{BASE_FOLDER}/arc-agi_evaluation_solutions.json"
    },
    "test_file_path": {
        "data_file_path": f"{BASE_FOLDER}/arc-agi_test_challenges.json"
    },
}

print("ARC Dataset Class")
class ARCDataset:
    
    def __init__(self, train_file_path, val_file_path, test_file_path, batch_size):
        self.output = {
            "train_output":{},
            "val_output":{}
        }
        self.origin_data = {}
        self.train_data = self.extract_file(train_file_path, "train")
        self.val_data = self.extract_file(val_file_path, "val")
        self.test_data = self.extract_file(test_file_path, "test")
        self.batch_size = batch_size
        
    #   for dataset class, we just need the input and output data
    def extract_data(self, data):
        d = []
        for key, inps, targ, index in data:
            d.append([inps, targ])
        return d
        
    def train_dataset(self):
        return DataLoader(self.extract_data(self.train_data), batch_size=self.batch_size, shuffle=True)
    
    def val_dataset(self):
        return DataLoader(self.extract_data(self.val_data), batch_size=self.batch_size, shuffle=False)
    
    def test_dataset(self):
        return self.test_data

    #   extract json file
    def extract_file(self, file_path, type_data):
        data_file_path = file_path["data_file_path"]
        target_file_path = file_path["target_file_path"] if type_data != "test" else None
        if target_file_path != None:
            with open(target_file_path, 'r') as f:
                sol = json.load(f)
            for i in sol.keys():
                self.output[f"{type_data}_output"][i] = sol[i]
        return self.load_data(data_file_path, type_data)

    def load_data(self, file_path, type_data):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.origin_data[type_data] = data
        return self.parse_data(data, type_data)

    #   add '0' value for padding. each row must have 30 length
    def expand_data(self, data, data_append=0):
        return np.array([*data, *[data_append for _ in range(30 - len(data))]])

    #   add '0' or np.zeros(30) so the data shape become (30,30) (900 after flatten)
    def prep_data(self, data):
        data = np.array(data)
        
        ndata = []
        for d in data:
            ndata.append(self.expand_data(d, 0))
        return torch.tensor(self.expand_data(ndata, np.zeros(30)).flatten())

    # the input data idea is give the nn example_input + example_target + test_input so LSTM can remember what it should do 
    def parse_data(self, data, type_data):
        ndata = []
        for key in tqdm(data.keys(), desc=type_data):
            train_data = data[key]['train']
            test_data = data[key]['test']
            train_temp, test_temp = [], []
            for trd in train_data:
                input_tensor = self.prep_data(trd['input'])
                output_tensor = self.prep_data(trd['output'])
                train_temp.append([
                    input_tensor,
                    output_tensor
                ])
            for i in range(len(test_data)):
                input_tensor = self.prep_data(test_data[i]['input'])
                if type_data != 'test' and key in self.output[f"{type_data}_output"]:
                    output_tensor = self.prep_data(self.output[f"{type_data}_output"][key][i])
                else:
                    output_tensor = np.zeros(900)
                test_temp.append([
                    input_tensor,
                    output_tensor
                ])
            for i, trd_1 in enumerate(train_temp):
                for j, tsd in enumerate(test_temp):
                    ndata.append([key, torch.tensor([*[*trd_1[0], 10, *trd_1[1]], 11, *tsd[0], 10]), torch.tensor(tsd[1]), j])
            
        print(f"Data type: {type_data}. Unique Puzzle: {len(data.keys())}. Parsing Puzzle: {len(ndata)}")
        return ndata

dataset = ARCDataset(**file_paths, batch_size=BATCH_SIZE)

train_origin = dataset.origin_data["train"]
val_origin = dataset.origin_data["val"]
test_origin = dataset.origin_data["test"]

train_dataset = dataset.train_dataset()
val_dataset = dataset.val_dataset()
test_dataset = dataset.test_dataset()

print("Dimension Class")
class Dimension:
    def __init__(self, data):
        self.dim = self.extract_dim(data)
        
    def extract_dim(self, data):
        keys = list(data.keys())
        ndata = {}
        for key in tqdm(keys):
            data_row = data[key]
            ndata[key] = self.check_dim(data_row)
        return ndata
            
    def dim(self, data):
        return np.array(data).shape

    def get_dim(self, data):
        inp_dim = self.dim(data['input'])
        out_dim = self.dim(data['output']) if 'output' in data else [1,1]
        return inp_dim, out_dim
    
    #   check the habits of data. if the input and output sizes are always same, its easier to get the right output size
    def check_dim(self,data):
        train_data = data["train"]
        test_data = data["test"]
        train_dim = []
        for d in train_data:
            inp_dim, out_dim = self.get_dim(d)
            same = inp_dim == out_dim
            diff1 = out_dim[0] / inp_dim[0]
            diff2 = out_dim[1] / inp_dim[1]
            train_dim.append([
                *inp_dim,
                *out_dim,
                int(same),
                diff1,
                diff2
            ])
        out_dim_data = []
        for i in range(len(test_data)):
            inp_dim, out_dim = self.get_dim(test_data[i])
            same = all([s[4] for s in train_dim])
            if same:
                out_dim = inp_dim
            else:
                for dim in train_dim:
                    if inp_dim[0] == dim[0] and inp_dim[1] == dim[1]:
                        out_dim = (dim[2], dim[3])
                        break
                y1 = Counter([dim[5] for dim in train_dim]).most_common(1)[0][0]
                y2 = Counter([dim[6] for dim in train_dim]).most_common(1)[0][0]
                out_dim = (int(inp_dim[0] * y1), int(inp_dim[1] * y2))
            out_dim_data.append(out_dim)
        return out_dim_data

print('LSTM Class')
class LSTM(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.ReLU()
        )

    def forward(self, input_data):
        lstm_out, _ = self.lstm(input_data)
        predictions = self.fc(lstm_out)
        return predictions

print('Training Class')
class Training:
    def __init__(self, model, train_loader, criterion, optimizer, device, loss = 100):
        self.model = model 
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
    
    def _train_one(self, model, data, criterion, optimizer):
        # declare model for train mode
        model.train()
        
        # data is on cpu, transfer to gpu if gpu is available
        input_data, target = data
        input_data, target = input_data.to(self.device).float(), target.to(self.device).float()

        # get the output
        output = model(input_data)
        
        # calculate the loss
        loss = criterion(output, target)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    
    def _train_loop(self, model, train_loader, criterion, optimizer):
        model.train()
        history = {'train_loss': []}
        loss = self.loss
        epoch = 0
        patient = 0
        while True:
            epoch += 1
            train_loss = 0
            for data in train_loader:
                ls = self._train_one(model, data, criterion, optimizer)
                train_loss += ls
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            print(f'\rEpoch : {epoch}, Loss: {train_loss:.5f}, Lowest Loss: {loss:.5f}, Patient: {patient}', end='')

            # if loss is smaller than before, save the model
            if train_loss < loss:
                loss = train_loss
                torch.save(model.state_dict(), 'model.pth')
                patient = 0
            else:
                patient += 1
            # I'm being greedy here. Sorry. if you dont like it, just remove 'and epoch > 2500'
            if patient >= 20 and epoch > 2500:
                break

        self.loss = loss
        return history
    
    def train(self):
        history = self._train_loop(self.model, self.train_loader, self.criterion, self.optimizer)
        self._plot_loss(history)
        
    def _plot_loss(self, history):
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], 'o-', label='train_loss')
        plt.legend()
        plt.title('Loss Plot')
        plt.show()

IN_DIM = len(test_dataset[1][1]) # 2703
OUT_DIM = 900
LATENT_DIM = 1800

print("Start training with train data")
model = LSTM(IN_DIM, OUT_DIM, LATENT_DIM).to(DEVICE)
criterion = nn.MSELoss()
# load pre trained model
model.load_state_dict(torch.load('/kaggle/input/arc-puzzle-solver-v1/pytorch/v1/1/model.pth',map_location=DEVICE))
# Fine Tuning with smaller learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)
training = Training(model, train_dataset, criterion, optimizer, DEVICE)
training.train()

# load the best model from previous training
model.load_state_dict(torch.load('model.pth'))

print('Prediction Class')
class Prediction:
    def __init__(self, model, data, origin_data, output = {}):
        self.model = model
        self.data = data
        self.origin_data = origin_data
        self.dimension = Dimension(origin_data)
        self.parsed_data = {}
        self.output = output
        
    def score(self, data, key):
        s = []
        if key not in self.output:
            return 0
        for d in range(len(data)):
            attempt = []
            output = np.array(self.output[key][d])
            attempt_1 = np.array(data[d]['attempt_1'])
            attempt_2 = np.array(data[d]['attempt_2'])
            
            if output.shape != attempt_1.shape:
                attempt.append(0)
            else:
                attempt.append(int(all(output.flatten() == attempt_1.flatten())))    
                
            if output.shape != attempt_2.shape:
                attempt.append(0)
            else:
                attempt.append(int(all(output.flatten() == attempt_2.flatten())))
                
            s.append(max(attempt))
        return max(s)
        
    def calculate_score(self):
        score, data_count = 0, 0
        for key in self.parsed_data.keys():
            data = self.parsed_data[key]["test"]
            score += self.score(data, key)
            data_count += 1
        print(f"Total Data: {data_count}. Total Correct: {score}. Accuracy: {score/data_count}")
    
    def predict(self, model, data):
        model.eval()
        input_data, target = data
        input_data, target = torch.tensor(input_data).to(DEVICE).float(), torch.tensor(target).to(DEVICE).float()

        with torch.no_grad():
            input_data = input_data.unsqueeze(0)
            output = model(input_data)

        return output[0]
    
    def extract_dim(self, key, output, idx=0):
        origin_data = self.origin_data[key]
        dim = self.dimension.dim[key][idx]
        data = np.array(output).astype(int).reshape(30,30)
        ndata = []
        for i in range(dim[0]):
            row_data = data[i]
            ndata.append(row_data[:dim[1]])
        return np.array(ndata)
        
    def get_output(self, attempt_1, attempt_2, model, key, idx=0):
        out1 = self.predict(model, attempt_1)
        out2 = self.predict(model, attempt_2)
        out1 = self.extract_dim(key, torch.round(out1.cpu()), idx).tolist()
        out2 = self.extract_dim(key, torch.round(out2.cpu()), idx).tolist()
        return out1, out2
    
    def plot_train(self, data):
        print("Train Data")
        fig, ax = plt.subplots(2, len(data), figsize=(len(data) * 2, 2))
        ax = np.array(ax)  # Ensure ax is always a 2D array
        for i in range(len(data)):
            ax[0, i].imshow(data[i]['input'], cmap=CMAP, norm=NORM)
            ax[1, i].imshow(data[i]['output'], cmap=CMAP, norm=NORM)
        plt.show()  # Add this to display the plot

    def plot_test(self, data):
        print("Test Data")
        fig, ax = plt.subplots(3, len(data), figsize=(len(data) * 3, 3))
        ax = np.array(ax)  # Ensure ax is always a 2D array
        if len(data) > 1:
            for i in range(len(data)):
                ax[0, i].imshow(data[i]['input'], cmap=CMAP, norm=NORM)
                ax[1, i].imshow(data[i]['attempt_1'], cmap=CMAP, norm=NORM)
                ax[2, i].imshow(data[i]['attempt_2'], cmap=CMAP, norm=NORM)
        else:
            ax[0].imshow(data[0]['input'], cmap=CMAP, norm=NORM)
            ax[1].imshow(data[0]['attempt_1'], cmap=CMAP, norm=NORM)
            ax[2].imshow(data[0]['attempt_2'], cmap=CMAP, norm=NORM)
        plt.show()  # Add this to display the plot
        
    def pred_all(self):
        model = self.model
        origin_data = self.origin_data
        temp_data = {}
        submit_data = {}
        for data in tqdm(self.data):
            
            key = data[0]
            idx = data[3]
            data_input = [data[1], data[2]]
            
            if key not in temp_data:
                temp_data[key] = {}
            if idx not in temp_data[key]:
                temp_data[key][idx] = {
                    "attempt_1": data_input,
                    "attempt_2": data_input
                }
            else:
                temp_data[key][idx]["attempt_2"] = data_input
                
        for key in tqdm(temp_data.keys()):
            data_list = temp_data[key]
            data_list = {key: data_list[key] for key in sorted(data_list)}
            for data in data_list:
                data_row = data_list[data]
                at1, at2 = self.get_output(data_row["attempt_1"], data_row["attempt_2"],model,key, data)
                origin_data[key]["test"][data]["attempt_1"] = at1
                origin_data[key]["test"][data]["attempt_2"] = at2
                
        for key in origin_data.keys():
            submit_data[key] = origin_data[key]["test"]
        self.parsed_data = origin_data
        return submit_data
    
    def plot_all(self, step=1):
        count = 0
        parsed_data = self.parsed_data
        for key in parsed_data.keys():
            count+=1
            if count % step != 0:
                continue
            print(f"===== {key} =====")
            self.plot_train(parsed_data[key]["train"])
            self.plot_test(parsed_data[key]["test"])

sanity = Prediction(model, dataset.train_data, train_origin, dataset.output['train_output'])

sanity.pred_all()
sanity.calculate_score()
# sanity.plot_all(15)

pred = Prediction(model, dataset.test_dataset(), test_origin)

res = pred.pred_all()
pred.plot_all(15)

json_object = json.dumps(res, indent=4)
with open('submission.json', 'w') as f:
    f.write(json_object)