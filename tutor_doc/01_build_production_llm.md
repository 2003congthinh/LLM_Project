# Build a Production-Ready LLM: Step-by-Step Roadmap

> Read this from top to bottom. Each milestone builds on the one before it.
> Tick the boxes (`- [x]`) as you go. Don't start a milestone until you finish
> the one above it.

---

## 0. Read this first: what "production-ready" really means

There are two ways to end up with an LLM you can put in a product:

| Path | What you do | Cost | When it's used |
|---|---|---|---|
| **A. From scratch** | Write the model, train the tokenizer, pretrain on billions of tokens, then fine-tune it for chat | From tens of dollars (tiny model) to millions (frontier model) | To learn, to do research, or when you need full control |
| **B. Adapt an open-weight model** | Take a strong pretrained model (Llama, Qwen, Mistral, Gemma, ...) and fine-tune it on your data | From a few dollars to a few hundred | **Almost every real product** |

A model you pretrain alone on a small budget won't match open-weight models
that cost millions to train. That doesn't make Path A pointless: it is how you
*understand* what you ship. This roadmap takes you through both:

- **Milestones 0–8 (Path A):** you rebuild your GPT the modern way and train a real GPT-2-class chat model. This turns you from someone who followed a tutorial into someone who understands LLMs.
- **Milestones 9–10 (Path B):** you take what you learned and ship a model that's good enough for real users.

Your hardware is an **RTX 4060 with 8 GB VRAM**. That's enough for
Milestones 0–5, 9 and 10. Milestone 6, and possibly 8, need a rented cloud GPU
for a few hours.

---

## 1. Where you are now (based on your `LLM-tutorial/elliotarledge - freeCodeCamp.org` folder)

## What you have already done ✅

Below is every skill you've already practised, with the exact code you wrote for it.
The code is copied unchanged from your files in `LLM-tutorial/elliotarledge - freeCodeCamp.org/`.
Cell numbers start at 0, as Jupyter counts them.

---

## 🧩 Skill 1: PyTorch tensor basics

`stack`, `cat`, `tril`/`triu`, `masked_fill`, `softmax`, `multinomial`, `nn.Embedding`, `nn.Linear`, `view`, matmul, dtypes, CPU vs. GPU timing.

Source: `torch_examples.ipynb` (all cells)

*torch_examples.ipynb, cell 0*

```python
import torch
```

*torch_examples.ipynb, cell 1*

```python
import numpy as np
import time
```

*torch_examples.ipynb, cell 2*

```python
randint = torch.randint(-100, 100, (6,))
randint
```

*torch_examples.ipynb, cell 3*

```python
tensor = torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
tensor
```

*torch_examples.ipynb, cell 4*

```python
zeros = torch.zeros(2, 3)
zeros
```

*torch_examples.ipynb, cell 5*

```python
ones = torch.ones(3, 4)
ones
```

*torch_examples.ipynb, cell 6*

```python
input = torch.empty(2, 3)
input
```

*torch_examples.ipynb, cell 7*

```python
arange = torch.arange(5)
arange
```

*torch_examples.ipynb, cell 8*

```python
linspace = torch.linspace(3, 10, steps=5)
linspace
```

*torch_examples.ipynb, cell 9*

```python
logspace = torch.logspace(start=-10, end=10, steps=5)
logspace
```

*torch_examples.ipynb, cell 10*

```python
eye = torch.eye(5)
eye
```

*torch_examples.ipynb, cell 11*

```python
a = torch.empty((2, 3), dtype=torch.int64)
empty_like = torch.empty_like(a)
empty_like
```

> CPU vs GPU performance

*torch_examples.ipynb, cell 13*

```python
device= 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

*torch_examples.ipynb, cell 14*

```python
%%time

start_time = time.time()
zeros = torch.zeros(1, 1)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"{elapsed_time:.4f}")
```

> using the time library to record execution time %%time to record the time taken  fo the cell to execute

*torch_examples.ipynb, cell 16*

```python
torch_rand1 = torch.rand(200, 200, 200, 200).to(device)
torch_rand2 = torch.rand(200, 200, 200, 200).to(device)
np_rand1 = torch.rand(200, 200, 200, 200)
np_rand2 = torch.rand(200, 200, 200, 200)

start_time = time.time()
rand = (torch_rand1 @ torch_rand2)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"{elapsed_time:.8}")

start_time = time.time()
rand = np.multiply(np_rand1, np_rand2)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"{elapsed_time:.8f}")
```

> Pytorch functions:\
> torch.stack, torch.multinomial, torch..tril, torch.triu, input.T / input.transpose, nn.Linear, torch.cat, F.softmax (show all the examples of functions/methods with pytorch docs)

*torch_examples.ipynb, cell 18*

```python
# Define a probability tensor
probabilities = torch.tensor([0.1, 0.9])
# 10% or 0.1 => 0, 90% or 0.9 => 1. Each probability points to the index of the probability in the tensor
# Draw 5 samples from the multinomial distribution
samples = torch.multinomial(probabilities, num_samples=10, replacement=True)
print(samples)
```

*torch_examples.ipynb, cell 19*

```python
tensor = torch.tensor([1, 2, 3, 4])
out = torch.cat((tensor, torch.tensor([5])), dim=0)
out
```

*torch_examples.ipynb, cell 20*

```python
out = torch.tril(torch.ones(5, 5))  # tril = triangle lower
out
```

*torch_examples.ipynb, cell 21*

```python
out = torch.triu(torch.ones(5, 5))  # tril = triangle upper
out
```

*torch_examples.ipynb, cell 22*

```python
out = torch.zeros(5, 5).masked_fill(torch.tril(torch.ones(5, 5)) == 0, float('-inf'))
out
```

*torch_examples.ipynb, cell 23*

```python
torch.exp(out)
```

*torch_examples.ipynb, cell 24*

```python
input = torch.zeros(2, 3, 4)
out = input.transpose(0, 2)
out.shape
```

*torch_examples.ipynb, cell 25*

```python
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
tensor3 = torch.tensor([7, 8, 9])

# Stack the tensors along a new dimension
stacked_tensor = torch.stack([tensor1, tensor2, tensor3])
stacked_tensor
```

*torch_examples.ipynb, cell 26*

```python
import torch.nn as nn
sample = torch.tensor([10., 10., 10.])
linear = nn.Linear(3, 3, bias=False)
print(linear(sample))
```

*torch_examples.ipynb, cell 27*

```python
import torch.nn.functional as F

# Create a tensor
tensor1 = torch.tensor([1.0, 2.0, 3.0])

# Apply softmax using torch.nn.functional.softmax()
softmax_output = F.softmax(tensor1, dim=0)

print(softmax_output)
```

> Embedding vectors

*torch_examples.ipynb, cell 29*

```python
# Initialize an embedding layer
vocab_size = 80
embedding_dim = 6
embedding = nn.Embedding(vocab_size, embedding_dim)

# Create some input indices
input_indices = torch.LongTensor([1, 5, 3, 2])

# Apply the embedding layer
embedded_output = embedding(input_indices)

# The output will be a tensor of shape (4, 100), where 4 is the number of inputs
# and 100 is the dimensionality of the embedding vectors
print(embedded_output.shape)
print(embedded_output)
```

*torch_examples.ipynb, cell 30*

```python
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(a@b)
```

> Matmul implementation

*torch_examples.ipynb, cell 32*

```python
# or 
torch.matmul(a, b)
```

*torch_examples.ipynb, cell 33*

```python
# Need to be the same data type or it will cause error

# type int64
int_64 = torch.randint(1, (3, 2))
# type float32
float_32 = torch.rand(2, 3)
print(int_64.dtype, float_32.dtype)
result = torch.matmul(int_64, float_32)
print(result)
```

*torch_examples.ipynb, cell 34*

```python
# cast a float to int and everything is fine
int_64 = torch.randint(1, (3, 2)).float()
float_32 = torch.rand(2, 3)
print(int_64.dtype, float_32.dtype)
result = torch.matmul(int_64, float_32)
print(result)
```

> Logits and Reshaping

*torch_examples.ipynb, cell 36*

```python
# What is view?
a = torch.rand(2, 3, 5)
print(a.shape)
x, y, z = a.shape
a = a.view(x, y, z)
print(x, y, z)
print(a.shape)
```

---

## 🧩 Skill 2: Character-level tokenizer (`encode` / `decode`)

Source: `bigram.ipynb`

*bigram.ipynb, cell 4*

```python
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(text[:200])
```

*bigram.ipynb, cell 5*

```python
chars = sorted(set(text))
vocab_size = len(chars)
print(chars)
```

*bigram.ipynb, cell 6*

```python
# A type of Tokenizer
string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
```

*bigram.ipynb, cell 7*

```python
print(encode("hello"))
```

*bigram.ipynb, cell 8*

```python
encoded = encode("hello")
decoded = decode(encoded)
print(decoded)
```

*bigram.ipynb, cell 9*

```python
# Tensors instead of arrays
data = torch.tensor(encode(text), dtype=torch.long)
```

*bigram.ipynb, cell 10*

```python
print(data[:100])
```

---

## 🧩 Skill 3: Hyperparameters, train/val split, `get_batch`, and targets shifted by one

Source: `bigram.ipynb`

*bigram.ipynb, cell 3*

```python
block_size = 8  # How many char
batch_size = 4  # How many blocks doing in parallel 
max_iters = 1000
eval_interval = 2500
learning_rate = 3e-4
eval_iters = 250
dropout = 0.2
```

*bigram.ipynb, cell 11*

```python
# Explaination start: 25:42 - 29:30
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

x, y = get_batch('train')
print('inputs:')
print(x)
print('targets:')
print(y)
```

*bigram.ipynb, cell 12*

```python
# block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print('When input is', context, 'target is', target)
```

Output:

```text
When input is tensor([1]) target is tensor(1)
When input is tensor([1, 1]) target is tensor(28)
When input is tensor([ 1,  1, 28]) target is tensor(39)
When input is tensor([ 1,  1, 28, 39]) target is tensor(42)
When input is tensor([ 1,  1, 28, 39, 42]) target is tensor(39)
When input is tensor([ 1,  1, 28, 39, 42, 39]) target is tensor(44)
When input is tensor([ 1,  1, 28, 39, 42, 39, 44]) target is tensor(32)
When input is tensor([ 1,  1, 28, 39, 42, 39, 44, 32]) target is tensor(49)
```

---

## 🧩 Skill 4: Bigram language model, logits, and cross-entropy loss

Source: `bigram.ipynb`

*bigram.ipynb, cell 1*

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
```

*bigram.ipynb, cell 2*

```python
device= 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

> <h3>nnModule subclass</h3>
> <n>Logits and Reshaping

*bigram.ipynb, cell 14*

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # What is forward pass? And why should write it from scratch
    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape                                  # Batch, Time, Channels 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # Index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self.forward(index)
            # Focus only on the last time step
            logits = logits[:, -1, :]                               # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)                       # (B, C)
            # Sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)    # (B, C)
            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=-1)          # (B, T+1)
        return index

model = BigramLanguageModel(vocab_size)
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
```

---

## 🧩 Skill 5: Training loop: AdamW, `zero_grad` → `backward` → `step`, and `estimate_loss` with `model.eval()`

Source: `bigram.ipynb`, including your notes on optimizers

> <n>Training loop + Optimizer + Zerograd explanation

*bigram.ipynb, cell 16*

```python
@torch.no_grad()   # Make sure pytorch doesn't use gradient add on here
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

*bigram.ipynb, cell 17*

```python
# Loss reporting + Train vs Eval mode
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train_loss: {losses['train']:.4f}, val_loss: {losses['eval']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
```

Output:

```text
step 0: train_loss: 2.9954, val_loss: 3.0029
step 250: train_loss: 2.9535, val_loss: 2.9655
step 500: train_loss: 2.9467, val_loss: 2.9759
step 750: train_loss: 2.9174, val_loss: 2.9171
3.0164828300476074
```

*bigram.ipynb, cell 18*

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
```

> <n>2:13:58\
> ***NOTE: do research on MSE, Gradient Descent, Momentum, RMSprop, Adam, AdamW***
> 1. **Mean Squared Error (MSE)**: MSE is a common loss function used in regression problems, where the goal is to predict a continuous output. It measures the average squared difference between the predicted and the actual values, an is often used to train neural networks for regression tasks.
> 2. **Gradient Descent (GD)**: is an optimization algorithm used to minimize the loss function of a ML model. The loss function measures how well the model is able to predict the target variable based on the input features. The idea of GD is to iteratively adjust the model parameters in the direction of the steepest descent of the loss function.
> 3. **Momentum**: Momentum is an extension of SGD that adds a "momentum" term to the parameter updates. this term helps smooth out the updates and allows the optimizer to continue moving in the right direction, even if the gradient changes direction or varies in magnitude. Momentum is particularly useful for training deep neural networks.
> 4. **RMSprop**: RMSprop is an optimization algorithm that uses a moving average of the squared gradient to adapt the learning rate of each parameter. This helps to avoid oscillations in the parameter updates and can improve convergence in some cases.
> 5. **Adam**: Adam is a popular optimization algorithm that combines the ideas of Momentum and RMSprop. It uses a moving average of both the gradient and its squared value to adapt the learning rate of each parameter. Adam is often used as a default optimizer for deep learning models.
> 6. **AdamW**: AdamW is a modification of the Adam optimizer that adds weight decay to the parameter updates. This helps to regularize the model and can improve generalization performance. We will be using AdamW optimizer as it best suits the properties of the model we will train next
> \
> \
> <n>find more optimizers and details at torch.optim</n>

---

## 🧩 Skill 6: Decoder-only GPT: self-attention head, multi-head attention, feed-forward, residual + LayerNorm, positional embeddings, and weight init

Source: `gpt-v1.ipynb` (Wizard of Oz version)

*gpt-v1.ipynb, cell 4*

```python
# (block_size, batch_size) - large contributors, (n_embd, n_layer, n_head) - not as much: are big contributor on how much memory ur gonna use, so play around it
block_size = 128  # How many char
batch_size = 32  # How many blocks doing in parallel 
max_iters = 1000
# eval_interval = 500
learning_rate = 3e-4
eval_iters = 100
dropout = 0.2
n_embd = 384
n_layer = 4
n_head = 4


# Should learn auto-tunning
```

*gpt-v1.ipynb, cell 6*

```python
chars = "" # Making it empty just to be safe
with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)
```

*gpt-v1.ipynb, cell 7*

```python
string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
```

*gpt-v1.ipynb, cell 8*

```python
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # print(ix)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

*gpt-v1.ipynb, cell 9*

```python
class Head(nn.Module):
    """ one head of self_attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)                                                 # (B, T, hs)
        q = self.query(x)                                               # (B, T, hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5               # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)                                    # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)                                               # (B, T, hs)
        out = wei @ v                                                   # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
```

*gpt-v1.ipynb, cell 10*

```python
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # heads = 3 => from (B, T, F) -> to (B, T, [h1, h1, h1, h1, h2, h2, h2, h2, h3, h3, h3, h3])
        out = self.dropout(self.proj(out))
        return out
```

*gpt-v1.ipynb, cell 11*

```python
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

*gpt-v1.ipynb, cell 12*

```python
class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x
```

*gpt-v1.ipynb, cell 13*

```python
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)    # final layer norm        
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # What is forward pass? And why should write it from scratch
    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape                                              # Batch, Time, Channels 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # Index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self.forward(index)
            # Focus only on the last time step
            logits = logits[:, -1, :]                               # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)                       # (B, C)
            # Sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)    # (B, C)
            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=-1)          # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
m = model.to(device)
```

*gpt-v1.ipynb, cell 14*

```python
@torch.no_grad()   # Make sure pytorch doesn't use gradient add on here
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

*gpt-v1.ipynb, cell 15*

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train_loss: {losses['train']:.4f}, val_loss: {losses['eval']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
```

*gpt-v1.ipynb, cell 16*

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
```

---

## 🧩 Skill 7: Preparing a large dataset (OpenWebText parquet → `train_split.txt` / `val_split.txt` / `vocab.txt`)

Source: `data_extract.py` (full file)

```python
import os
import lzma
from tqdm import tqdm
import pandas as pd
import concurrent.futures
import random

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".parquet") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "openwebtext/plain_text"
output_file_train = "train_split.txt"
output_file_val = "val_split.txt"
vocab_file = "vocab.txt"

# IMPORTANT: Parquet files are tabular. You must specify which column holds the text data.
# Most OpenWebText datasets use 'text' as the column name. Change this if yours is different.
text_column = "text"

files = xz_files_in_dir(folder_path)
total_files = len(files)

# Calculate split index
split_index = int(total_files * 0.9) # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

# Process the files for training and validation separately
vocab = set()

# Process the training files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        # 1. Read the parquet file into a pandas DataFrame
        df = pd.read_parquet(file_path)
        # 2. Extract the text column, drop empty rows, and join into a single string
        text = "\n".join(df[text_column].dropna().astype(str).tolist())
        outfile.write(text + "\n")
        # 3. Update vocabulary
        vocab.update(set(text))

        # with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        #     text = infile.read()
        #     outfile.write(text)
        #     characters = set(text)
        #     vocab.update(characters)

# Process the validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_parquet(file_path)
        text = "\n".join(df[text_column].dropna().astype(str).tolist())
        outfile.write(text + "\n")
        vocab.update(set(text))
        # with lzma.open(file_path, "rt", encoding="utf-8") as infile:
        #     text = infile.read()
        #     outfile.write(text)
        #     characters = set(text)
        #     vocab.update(characters)

# Write the vocabulary to vocab.txt
with open(vocab_file, "w", encoding="utf_8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')
```

---

## 🧩 Skill 8: Reading huge files with `mmap` and training on OpenWebText

Source: `gpt-v1.ipynb` (OpenWebText version). The model classes in cells 21–25 are identical to skill 6, so they appear only once, under skill 6.

> ### Openwebtext version

*gpt-v1.ipynb, cell 18*

```python
chars = "" # Making it empty just to be safe
with open('vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)
```

*gpt-v1.ipynb, cell 19*

```python
string_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_string = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])
```

*gpt-v1.ipynb, cell 20*

```python
# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

*gpt-v1.ipynb, cell 26*

```python
@torch.no_grad()   # Make sure pytorch doesn't use gradient add on here
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

*gpt-v1.ipynb, cell 27*

```python
# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step {iter}: train_loss: {losses['train']:.4f}, val_loss: {losses['eval']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())
```

Output:

```text
step 0: train_loss: 10.4965, val_loss: 10.4987
step 100: train_loss: 2.7529, val_loss: 2.7301
step 200: train_loss: 2.6678, val_loss: 2.6745
step 300: train_loss: 2.6196, val_loss: 2.6287
step 400: train_loss: 2.5833, val_loss: 2.5841
step 500: train_loss: 2.5041, val_loss: 2.5264
step 600: train_loss: 2.3905, val_loss: 2.3593
step 700: train_loss: 2.3235, val_loss: 2.2943
step 800: train_loss: 2.2707, val_loss: 2.3006
step 900: train_loss: 2.1835, val_loss: 2.2152
2.1266348361968994
```

---

## 🧩 Skill 9: CLI arguments with `argparse`, saving and loading a model, and a prompt loop

Source: `gpt-v1.ipynb`, saving, loading, and generating

*gpt-v1.ipynb, cell 28*

```python
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved!')
```

*gpt-v1.ipynb, cell 29*

```python
# Run the training code block over and over, maybe 1000 to 2000 iterations over night
print('loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')
m = model.to(device)
```

*gpt-v1.ipynb, cell 30*

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)
```

Source: `training.py`, argument parsing (top of file)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')

# Hence we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=str, required=True, help='Please provide batch size')

args = parser.parse_args()

# Now we can use the argument value in our program
print(f'batch size: {args.batch_size}')
```

Source: `training.py`, saving the model (end of file)

```python
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Model saved!')
```

Source: `chatbot.py`, imports, argument parsing, and hyperparameters (top of file)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demonstration program')

# Hence we add an argument to the parser, specifying the expected type, a help message, etc.
parser.add_argument('-batch_size', type=str, required=True, help='Please provide batch size')

args = parser.parse_args()

# Now we can use the argument value in our program
print(f'batch size: {args.batch_size}')

device= 'cuda' if torch.cuda.is_available() else 'cpu'

# MAKE SURE THIS PARAMS ARE THE SAME AS TRAINING.PY WHEN LOAD THE MODEL UP
block_size = 128  # How many char
batch_size = int(args.batch_size)
max_iters = 1000
# eval_interval = 500
learning_rate = 3e-4
eval_iters = 100
dropout = 0.2
n_embd = 384
n_layer = 4
n_head = 4
```

Source: `chatbot.py`, the `generate` method inside `GPTLanguageModel`, with context cropping added (the rest of the model classes are identical to skill 6)

```python
    def generate(self, index, max_new_tokens):
        # Index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # Get the predictions
            logits, loss = self.forward(index_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]                               # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)                       # (B, C)
            # Sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)    # (B, C)
            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=-1)          # (B, T+1)
        return index
```

Source: `chatbot.py`, loading the model and running the prompt loop (end of file)

```python
model = GPTLanguageModel(vocab_size)
print('loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')
m = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens = 150)[0].tolist())  # max_new_tokens = block_size or it will yield error, or use the updated idx crop on line 168 to not need to worry about this params
    print(f'Completion:\n{generated_chars}')
```

---

## 🧩 Skill 10: Drawing the Transformer architecture

Source: `transformer_architecture.drawio`

![Your Transformer architecture diagram](<../LLM-tutorial/elliotarledge‬ - freeCodeCamp.org/transformer_architecture_image.png>)

That covers the core of an LLM. Everything after this point is about making it
**bigger, faster, smarter, and usable**.

---

## Gaps and bugs in your current code (you'll fix these in Milestone 0)

| # | Issue | Where | Why it matters |
|---|---|---|---|
| 1 | Your diagram shows an **encoder-decoder** Transformer, but your code is **decoder-only**: it has no encoder and no cross-attention. | `transformer_architecture.drawio` | GPT models are decoder-only. Get this mental model right before you go further. |
| 2 | The character vocabulary has ~32,000 symbols, mostly rare Unicode characters. The embedding table plus `lm_head` hold **~25M of your ~32M parameters**. | `vocab.txt`, `training.py` | Most of the model is spent on memorizing rare characters. A BPE tokenizer fixes this (Milestone 2). |
| 3 | `-batch_size` is parsed but never used: `batch_size = 32` is hard-coded. | `training.py` | You think you're changing batch size, but you aren't. |
| 4 | `get_random_chunk` reads one ~4 KB window and samples all 32 sequences from it, so every batch overlaps heavily. | `training.py` | Correlated batches mean noisy gradients and a worse model. |
| 5 | Training ran for only 1,000 steps × 32 × 128 ≈ **4M characters out of ~36 billion** (~0.01% of your data). | `gpt-v1.ipynb` output | The model has barely started learning (loss 2.1). |
| 6 | The model is saved with `pickle.dump(model)`. | `training.py`, `chatbot.py` | It breaks if you rename a class, and it's a **security risk**: loading a pickle can run arbitrary code. Use `state_dict` / `safetensors`. |
| 7 | Post-LayerNorm (`ln(x + sublayer(x))`). | `Block.forward` | Modern GPTs use **Pre-LN** (`x + sublayer(ln(x))`), which trains more stably when the model is deep. |
| 8 | Attention runs one head at a time in a Python loop with manual softmax. | `Head`, `MultiHeadAttention` | Slow and memory-hungry. `F.scaled_dot_product_attention` (FlashAttention) is several times faster. |
| 9 | The notebook's `generate` doesn't crop the context to `block_size`. | `gpt-v1.ipynb` | It crashes after 128 tokens (you already fixed this in `chatbot.py`). |
| 10 | There's no LR schedule, gradient clipping, mixed precision, mid-training checkpoints, or logging. | `training.py` | Real training runs need all of these (Milestone 5). |
| 11 | `chatbot.py` *continues* text; it doesn't *answer*. | `chatbot.py` | It's a **base model**. A chat model needs instruction tuning (Milestone 8). |

---

## 2. Milestone overview

Tick a milestone only when every small goal inside it is done.

- [ ] **M0:** Fix and clean up your current GPT
- [ ] **M1:** Fill the math and deep learning gaps
- [ ] **M2:** Build a real tokenizer (BPE)
- [ ] **M3:** Build a production data pipeline
- [ ] **M4:** Modernize the architecture
- [ ] **M5:** Train like a professional (training loop engineering)
- [ ] **M6:** Scale up: reproduce GPT-2 (124M) on cloud GPUs
- [ ] **M7:** Evaluate your model properly
- [ ] **M8:** Turn the base model into a chat model (post-training)
- [ ] **M9:** The production path: fine-tune an open-weight model
- [ ] **M10:** Make the model ship-ready (safety, packaging, release)

Then go on to `02_host_llm.md` to put it online.

---

## M0: Fix and clean up your current GPT

**Goal:** Turn your tutorial code into a clean, correct project before you build on it.

**Knowledge to learn**
- Decoder-only vs. encoder-decoder Transformers (GPT vs. the original "Attention Is All You Need" model vs. BERT)
- `model.state_dict()`, `torch.save` / `torch.load(weights_only=True)`, and why pickle is unsafe
- Organizing a Python project: modules, a `config` object, one source of truth for hyperparameters

**Steps**
1. Create a new folder (for example `my-gpt/`) and split your code into `model.py`, `data.py`, `train.py`, `sample.py`, and `config.py`. Define the model **once** and import it everywhere. Right now `chatbot.py` and `training.py` each carry their own copy.
2. Put the hyperparameters in a dataclass and save it next to the weights:
   ```python
   torch.save({"config": asdict(cfg), "model": model.state_dict()}, "ckpt.pt")
   # load
   ckpt = torch.load("ckpt.pt", map_location=device, weights_only=True)
   model = GPT(GPTConfig(**ckpt["config"])); model.load_state_dict(ckpt["model"])
   ```
3. Fix bugs #3 (use `args.batch_size`, with `type=int`), #4 (sample each sequence from its own random offset in the file), and #9 (crop the context).
4. Switch the Block to Pre-LN:
   ```python
   x = x + self.sa(self.ln1(x))
   x = x + self.ffwd(self.ln2(x))
   ```
5. Write a function that counts parameters and prints where they are (embedding vs. blocks vs. head). See bug #2 for yourself.
6. Redraw your diagram as a **decoder-only** stack: Embedding + Position → N × [LN → Masked MHA → +residual → LN → FFN → +residual] → LN → Linear → Softmax.
7. Put the project under git and commit.

**Small goals**
- [ ] Code is split into modules, with the model defined in exactly one place
- [ ] Checkpoints are saved as `state_dict` + config (no more pickle)
- [ ] `batch_size` from the CLI actually changes training
- [ ] Each sequence in a batch comes from a different random place in the data
- [ ] Pre-LN block implemented, and loss still decreases
- [ ] Parameter count printed per component; you can explain why the embedding is so large
- [ ] Diagram redrawn as decoder-only
- [ ] Project is in a git repository

✅ **Done when:** `python train.py --batch_size 16` trains, saves `ckpt.pt`, and `python sample.py` loads it and generates text without pickle.

---

## M1: Fill the math and deep learning gaps

**Goal:** Understand *why* the code works, so you can debug it when it doesn't.

**Knowledge to learn**
- **Linear algebra:** vectors, matrices, matmul shapes, dot product as similarity, transpose
- **Calculus:** derivatives, the chain rule, gradients, **backpropagation**
- **Probability:** softmax, cross-entropy, negative log-likelihood, **perplexity** = `exp(loss)`, sampling (temperature, top-k, top-p)
- **Optimization:** SGD → Momentum → Adam → AdamW (you started this in `bigram.ipynb`), learning rate, weight decay
- **Training dynamics:** overfitting vs. underfitting, train vs. val loss, dropout, initialization, vanishing/exploding gradients, normalization layers, residual connections

**Steps**
1. Watch 3Blue1Brown's *Essence of Linear Algebra* (chapters 1–4, 9) and *Neural Networks* series (all chapters, including the Transformer and attention videos).
2. Build **micrograd** by following Karpathy's *"The spelled-out intro to neural networks and backpropagation"*: an autograd engine in ~100 lines.
3. Follow *makemore* parts 2–4 (MLP, BatchNorm, manual backprop). Part 4 ("Becoming a Backprop Ninja") makes backprop click.
4. Reread your `Head.forward` and explain every line on paper, including the shape of every tensor and why we divide by `sqrt(head_size)`.

**Small goals**
- [ ] Can multiply two matrices by hand and predict the output shape of any `@`
- [ ] Built micrograd from scratch and trained a tiny MLP with it
- [ ] Can derive the gradient of softmax + cross-entropy (`probs - one_hot`)
- [ ] Can explain what loss 2.1 means and convert it to perplexity
- [ ] Can explain AdamW in your own words, including what "decoupled weight decay" means
- [ ] Can explain why residual connections and LayerNorm help deep networks train
- [ ] Wrote a one-page note: "How a GPT predicts the next token", end to end

✅ **Done when:** you can rebuild `gpt-v1.ipynb` from a blank file without looking at the tutorial.

---

## M2: Build a real tokenizer (BPE)

**Goal:** Replace the character-level tokenizer with Byte-Pair Encoding, the method GPT-2/3/4, Llama, and almost every modern LLM use.

**Knowledge to learn**
- Unicode vs. UTF-8 bytes; why byte-level BPE never hits an "unknown character"
- The BPE algorithm: count pairs, merge the most frequent, repeat
- Regex pre-splitting (the GPT-2/GPT-4 split patterns)
- Special tokens (`<|endoftext|>`, and later `<|user|>`, `<|assistant|>`)
- Tradeoff: bigger vocabulary means shorter sequences but larger embedding tables

**Steps**
1. Watch Karpathy's *"Let's build the GPT Tokenizer"*.
2. Implement BPE yourself (follow the `karpathy/minbpe` exercise): `train`, `encode`, `decode`.
3. Train your tokenizer on ~100 MB of your OpenWebText data with a vocabulary of 8k–32k.
4. Compare it with `tiktoken` (`gpt2` encoding) and Hugging Face `tokenizers`. For real training, use one of these fast libraries; yours is for learning.
5. Measure **compression**: characters per token on your validation text.

**Small goals**
- [ ] Explained the difference between a character, a byte, and a token
- [ ] Implemented BPE train/encode/decode, and `decode(encode(s)) == s` passes on random text
- [ ] Added and handled an `<|endoftext|>` special token
- [ ] Trained a tokenizer on your data and saved/loaded it
- [ ] Measured ~3–4.5 characters per token (vs. 1.0 now)
- [ ] Learned to use `tiktoken` or HF `tokenizers` for speed

✅ **Done when:** your GPT trains on BPE tokens and you can explain why `vocab_size` changed what the model spends its parameters on.

---

## M3: Build a production data pipeline

**Goal:** Feed the GPU fast, clean, shuffled data. The quality of your data matters more than any architecture trick.

**Knowledge to learn**
- Pretokenizing: tokenize once, store as a flat array of `uint16`/`uint32` in `.bin` shards
- `np.memmap` (an upgrade of the `mmap` you already used)
- Data quality: deduplication (exact and near-duplicate / MinHash), language filtering, quality filtering, removing boilerplate and PII
- Dataset choice: **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`) is much higher quality than OpenWebText for small models
- Data mixtures (web, code, books, math) and why they matter
- Licensing and the legal side of training data

**Steps**
1. Download a sample of FineWeb-Edu with the Hugging Face `datasets` library (start with the `sample-10BT` subset, or an even smaller slice).
2. Tokenize with multiprocessing, append `<|endoftext|>` between documents, and write ~100M-token shards (`train_000.bin`, ...). Keep one shard for validation.
3. Write a `DataLoader` that reads shards with `np.memmap` and returns `(x, y)`:
   ```python
   data = np.memmap("train_000.bin", dtype=np.uint16, mode="r")
   ix = torch.randint(len(data) - block_size - 1, (batch_size,))
   x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
   y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
   ```
4. Use `pin_memory()` and `.to(device, non_blocking=True)`.
5. Write a small dedup + filter step (exact-hash dedup, drop documents under N characters) and record how much it removes.
6. Delete your 36 GB `train_split.txt`. Token shards are smaller and faster.

**Small goals**
- [ ] Downloaded a FineWeb-Edu subset
- [ ] Tokenized into `.bin` shards with a fixed train/val split
- [ ] Data loader uses `np.memmap` and random offsets across shards
- [ ] Data loading is no longer the bottleneck (GPU utilization in `nvidia-smi` stays high)
- [ ] Wrote a dedup/filter step and measured its effect
- [ ] Can name 3 data-quality problems and how to fix each one
- [ ] Read the FineWeb paper or blog post on how they filtered data

✅ **Done when:** you can pretokenize a new dataset with one command and train on it.

---

## M4: Modernize the architecture

**Goal:** Upgrade your 2017-style GPT to the design that current open models (Llama, Qwen, Mistral) use.

**Knowledge to learn**

| Upgrade | Replaces | Why |
|---|---|---|
| Fused attention: `F.scaled_dot_product_attention(q, k, v, is_causal=True)` | Your manual `Head` loop + `tril` | FlashAttention: faster, less memory |
| Batched multi-head attention (one `qkv` Linear, reshape into heads) | `nn.ModuleList` of heads | One big matmul instead of N small ones |
| **Weight tying** (`lm_head.weight = tok_emb.weight`) | Separate output matrix | Saves `vocab × n_embd` parameters |
| GELU, then **SwiGLU** FFN | ReLU | Better quality per parameter |
| **RMSNorm** | LayerNorm | Simpler and slightly faster |
| **RoPE** (rotary position embeddings) | Learned `position_embedding_table` | Better at long context and can be extended |
| **GQA** (grouped-query attention) | Full multi-head K/V | A much smaller KV cache when serving |
| **KV cache** in `generate()` | Recomputing the whole sequence every step | Generation goes from O(n²) to O(n) per token |
| Dropout = 0 for pretraining | Dropout 0.2 | Large-data pretraining rarely overfits |

**Steps**
1. Watch Karpathy's *"Let's reproduce GPT-2 (124M)"* and read `karpathy/nanoGPT` → `model.py`. Compare it line by line with your `training.py`.
2. Apply the upgrades **one at a time**. After each one, train for 500 steps and check that loss is the same or better. If it gets worse, you have a bug.
3. Implement a KV cache in `generate()` and measure tokens/second before and after.
4. Read the Llama paper's architecture section and list every difference from GPT-2.
5. Stretch: make your model **Llama-compatible** (same layer names and shapes) so you can export it to Hugging Face format. Then the whole hosting ecosystem (vLLM, llama.cpp) can run it.

**Small goals**
- [ ] Attention uses `F.scaled_dot_product_attention` with a single `qkv` projection
- [ ] Weight tying implemented, and you can explain the parameter savings
- [ ] SwiGLU FFN implemented
- [ ] RMSNorm implemented
- [ ] RoPE implemented (positional embedding table removed)
- [ ] GQA implemented (`n_kv_head < n_head`)
- [ ] KV cache implemented, with a measured speedup
- [ ] An ablation table (change → val loss → speed) saved in your notes

✅ **Done when:** your `model.py` has the same building blocks as a Llama-style model and you can explain each one.

---

## M5: Train like a professional

**Goal:** Make training fast, stable, resumable, and observable. These are the engineering skills companies pay for.

**Knowledge to learn**
- **Mixed precision:** `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Your RTX 4060 supports bf16.
- `torch.compile(model)`, TF32 (`torch.set_float32_matmul_precision("high")`)
- **Gradient accumulation:** simulate a large batch (for example 0.5M tokens) on 8 GB of VRAM
- **LR schedule:** linear warmup, then cosine decay (or WSD: warmup-stable-decay)
- **Gradient clipping** (`clip_grad_norm_(…, 1.0)`)
- Weight decay only on 2D weights (not biases or norms), AdamW betas (0.9, 0.95)
- Memory math: AdamW + mixed precision ≈ **16 bytes per parameter** plus activations; activation (gradient) checkpointing
- **Checkpointing** model + optimizer + step + RNG state, so a crash doesn't lose days of work
- Experiment tracking: Weights & Biases or TensorBoard
- Reproducibility: seeds, logging configs, versioning data

**Steps**
1. Add bf16 autocast plus `torch.compile` and measure tokens/sec before and after.
2. Add gradient accumulation so `total_batch_tokens = micro_batch × block_size × accum_steps` is about 0.25M–0.5M.
3. Add warmup + cosine LR, gradient clipping, and weight-decay parameter groups.
4. Save a full checkpoint every N steps, and add a `--resume` flag. Test it by killing training (Ctrl+C) and resuming.
5. Log loss, LR, grad norm, tokens/sec, and GPU memory to W&B/TensorBoard.
6. Train a **~30–50M parameter** model on your 4060 for a few hours on ~1B FineWeb-Edu tokens. Compare it with your old char-level model.

**Small goals**
- [ ] bf16 + `torch.compile`, with the speedup measured
- [ ] Gradient accumulation working (same loss curve as a bigger batch)
- [ ] Warmup + cosine LR schedule
- [ ] Gradient clipping, and grad norm logged
- [ ] Training can be killed and resumed from a checkpoint with no loss spike
- [ ] All runs tracked in W&B/TensorBoard with their configs
- [ ] Can estimate the VRAM needed for a model of N parameters before running it
- [ ] Trained a ~30–50M BPE model locally; samples are clearly better than `model-01`

✅ **Done when:** you can start a multi-hour run, walk away, and trust the result.

---

## M6: Scale up: reproduce GPT-2 (124M) on cloud GPUs

**Goal:** Train a real, well-known model size the way the industry does it: on multiple GPUs, against a budget.

**Knowledge to learn**
- **Scaling laws:** Kaplan and **Chinchilla** (≈ 20 tokens per parameter is compute-optimal), and why modern small models train on far more tokens than that
- Compute estimate: **FLOPs ≈ 6 × parameters × tokens**, plus MFU (model FLOPs utilization)
- **Distributed training:** DDP with `torchrun`, `all_reduce`, rank and world size; what FSDP / ZeRO are for (models too large for one GPU)
- Cloud GPUs: renting H100/A100 machines (RunPod, Lambda, Vast.ai, and so on), SSH, `tmux`, moving data, shutting machines down so you don't burn money
- Linux command line, if you don't know it yet (you're on Windows; learn WSL2)

**Steps**
1. Set up **WSL2 (Ubuntu)** on your PC and get your training code running there. Cloud machines are Linux.
2. Add DDP support (`torchrun --nproc_per_node=N train.py`) and test it with 1 GPU locally.
3. Estimate the cost: 124M params × 10B tokens × 6 ≈ 7.4e18 FLOPs. At realistic H100 throughput that's a few hours on 8 GPUs (tens of dollars; prices change, so check).
4. Rent a multi-GPU machine, run the GPT-2 124M config on FineWeb-Edu (~10B tokens), and download the checkpoints.
5. Alternatively, follow `karpathy/nanochat`, which does tokenizer → pretraining → chat fine-tuning → web UI in one reproducible pipeline on one 8×H100 node.

**Small goals**
- [ ] Comfortable with Linux / WSL2, SSH, `tmux`, `scp`/`rsync`
- [ ] DDP training works with `torchrun`
- [ ] Wrote a cost estimate before renting (FLOPs → GPU-hours → $)
- [ ] Ran a 124M model on ≥ 5B tokens in the cloud
- [ ] Measured MFU and tokens/sec
- [ ] Downloaded the final checkpoint and **shut down the machine**
- [ ] Can explain Chinchilla scaling in one paragraph

✅ **Done when:** you have a 124M base model whose validation loss matches or beats the original GPT-2 124M on the same data.

---

## M7: Evaluate your model properly

**Goal:** Know how good your model is with numbers, not vibes. Production teams don't ship without evals.

**Knowledge to learn**
- Intrinsic metrics: validation loss, perplexity (only comparable with the **same tokenizer**)
- Benchmarks: HellaSwag (common sense), ARC, MMLU (knowledge), GSM8K (math), HumanEval (code). Know which ones are meaningful for small models.
- `EleutherAI/lm-evaluation-harness`
- **Data contamination:** benchmark questions leaking into training data
- Building your **own eval set** for your use case, often the most important eval
- LLM-as-a-judge, and its biases
- Statistical noise: run evals more than once and report confidence intervals

**Steps**
1. Add a HellaSwag eval to your training loop (nanoGPT/build-nanogpt shows how).
2. Run `lm-evaluation-harness` on your model, after exporting it to Hugging Face format if needed.
3. Write 50–100 hand-made test prompts for the task you care about, with expected behavior, and save them as JSONL.
4. Build a small `eval.py` that runs all of the above and writes a report. You'll reuse it in M8–M10.

**Small goals**
- [ ] Validation loss and perplexity reported for every checkpoint
- [ ] HellaSwag accuracy tracked during training
- [ ] Ran at least 2 benchmarks with lm-evaluation-harness
- [ ] Built a custom eval set (≥ 50 examples) for your use case
- [ ] Have one `eval.py` command that produces a report
- [ ] Can explain what data contamination is and how to check for it

✅ **Done when:** you can compare any two checkpoints and say which is better, with numbers.

---

## M8: Turn the base model into a chat model (post-training)

**Goal:** Turn "continues text" into "answers questions". This is the step your `chatbot.py` is missing.

**Knowledge to learn**
- **Base model vs. instruct/chat model**
- **Chat templates:** special tokens that mark the system, user, and assistant turns
- **SFT (Supervised Fine-Tuning):** train on (prompt, response) conversations and **mask the loss** so the model learns only from assistant tokens
- **Preference tuning:** RLHF (InstructGPT paper), then **DPO** (simpler, no reward model); RL with verifiable rewards (math/code, GRPO) at a high level
- Catastrophic forgetting, learning rates for fine-tuning (much lower than pretraining)
- Chat datasets: for example `HuggingFaceTB/smoltalk`; check each dataset's license

**Steps**
1. Add chat special tokens to your tokenizer, such as `<|user|>`, `<|assistant|>`, `<|end|>`.
2. Write a data formatter that turns conversations into token sequences plus a loss mask.
3. Fine-tune your 124M base model on a chat dataset (a few hours on the 4060 is enough for 124M).
4. Update `chatbot.py` to keep the **conversation history**, apply the chat template, and stop generating at `<|end|>`.
5. Optional: run DPO on a small preference dataset and compare the results with your eval set.
6. Compare base and SFT on your custom eval set from M7.

**Small goals**
- [ ] Can explain base vs. chat models and why SFT is needed
- [ ] Chat template with special tokens implemented
- [ ] Loss masking implemented (only assistant tokens are trained on)
- [ ] SFT run completed; the model answers in turns instead of rambling
- [ ] `chatbot.py` supports multi-turn conversation and stop tokens
- [ ] (Optional) DPO run completed and evaluated
- [ ] SFT model beats the base model on your eval set

✅ **Done when:** you can chat with *your own from-scratch model* in multiple turns. It'll be dumb, because it's tiny, but it behaves like a chatbot.

---

## M9: The production path: fine-tune an open-weight model

**Goal:** Build a model that's actually good enough for users, by standing on a strong pretrained base.

**Knowledge to learn**
- The Hugging Face stack: `transformers` (models), `datasets`, `peft` (LoRA), `trl` (SFT/DPO trainers), `accelerate`
- Choosing a base model: size, license (some open models restrict commercial use), language support, context length, base vs. instruct version
- **LoRA:** train small adapter matrices instead of all the weights
- **QLoRA:** LoRA on a 4-bit quantized base, which lets you fine-tune bigger models on 8 GB
- What fits on your RTX 4060 (8 GB): **1B–4B models comfortably with QLoRA; 7–8B is possible but tight** (short sequences, small batch). Use a cloud GPU (24–80 GB) for anything bigger.
- Hyperparameters for fine-tuning: LoRA rank and alpha, learning rate, epochs, sequence length
- Overfitting on small datasets; when to use **RAG instead of fine-tuning** (fine-tuning teaches behavior and style; RAG supplies facts)
- Merging LoRA adapters back into the base weights

**Steps**
1. Write down your **use case** (for example "a customer-support bot for X" or "a Vietnamese tutor for Y") and what "good" looks like.
2. Build a dataset: 500–5,000 high-quality conversations in your chat format. Quality beats quantity. Clean and dedupe it, and hold out 10% for eval.
3. Run the base or instruct model on your eval set first. **This is your baseline.**
4. Fine-tune with TRL `SFTTrainer` + PEFT LoRA/QLoRA on a 1B–4B model on your 4060.
5. Evaluate with your M7 `eval.py`. Fine-tuning must beat the baseline, or it isn't worth it.
6. Iterate on the data, not just the hyperparameters. Most gains come from better data.
7. Merge the adapter into the base model and save it in `safetensors` format.

**Small goals**
- [ ] Use case and success criteria written down
- [ ] Picked a base model and checked that its license allows your use
- [ ] Built and cleaned a fine-tuning dataset, with a held-out eval split
- [ ] Measured the baseline (before fine-tuning)
- [ ] QLoRA fine-tune completed on your RTX 4060
- [ ] Fine-tuned model beats the baseline on your eval set
- [ ] Tried at least 2 data improvements and recorded the results
- [ ] Adapter merged and exported as `safetensors`
- [ ] Can explain when to use RAG vs. fine-tuning vs. prompting

✅ **Done when:** you have a fine-tuned model that measurably beats its base model on your task.

---

## M10: Make the model ship-ready

**Goal:** Package the model so other people (and your hosting setup) can use it safely and reliably.

**Knowledge to learn**
- **Safety:** red-teaming, jailbreaks, refusal behavior, toxicity and bias evals, guard models (for example Llama Guard)
- **Hallucination:** how to measure it and how to reduce it (RAG, citations, "I don't know" training data)
- **Formats:** `safetensors` (safe and fast), Hugging Face model repo layout (`config.json`, `tokenizer.json`, `generation_config.json`, chat template in `tokenizer_config.json`)
- **Quantization for inference:** GGUF (llama.cpp/Ollama), AWQ/GPTQ (vLLM), FP8. Know the quality vs. size tradeoff.
- **Model card:** intended use, limitations, training data, eval results, license
- **Versioning:** model registry (Hugging Face Hub or MLflow), semantic versions, linking every model to its data, code commit, and eval report

**Steps**
1. Run a red-team session: 50+ adversarial prompts (jailbreaks, harmful requests, prompt injection). Fix the problems with data and re-evaluate.
2. Export to Hugging Face format and test that `AutoModelForCausalLM.from_pretrained(path)` plus `apply_chat_template` works.
3. Make quantized versions: a GGUF (Q4_K_M, Q8_0) and an AWQ or GPTQ. Run your eval on each to see how much quality you lose.
4. Write a model card (`README.md` in the model repo).
5. Push to the Hugging Face Hub (private repo) with version tags.
6. Write a short "release report": eval scores, safety results, known limitations.

**Small goals**
- [ ] Red-team test set (≥ 50 prompts) created and run
- [ ] Safety issues found → data fixed → re-evaluated
- [ ] Model loads with the standard `transformers` API and chat template
- [ ] GGUF and AWQ/GPTQ quantized versions created and evaluated
- [ ] Model card written
- [ ] Model versioned on the Hugging Face Hub (private)
- [ ] Release report written

✅ **Done when:** someone else could download your model, understand what it's for, and run it. **Next: `02_host_llm.md`.**

---

## Final "production-ready model" checklist

Before you call a model production-ready, all of these should be true:

- [ ] Clear use case and success metric
- [ ] Beats a baseline on a held-out, use-case-specific eval set
- [ ] Passes a safety/red-team review, with known failure modes documented
- [ ] Reproducible: data version + code commit + config + seed are recorded
- [ ] Stored in `safetensors`, loadable with standard tooling, chat template included
- [ ] Quantized variants evaluated
- [ ] License checked for the base model **and** for every dataset
- [ ] Model card + release report written
- [ ] Versioned in a registry, so you can roll back to the previous version

---

## Learning resources (in recommended order)

**Foundations**
- 3Blue1Brown: *Essence of Linear Algebra*, *Neural Networks* (YouTube)
- Andrej Karpathy: *Neural Networks: Zero to Hero* (YouTube playlist): micrograd → makemore → GPT → tokenizer → *Let's reproduce GPT-2 (124M)*

**Code to read**
- `github.com/karpathy/nanoGPT`, `karpathy/build-nanogpt`, `karpathy/minbpe`, `karpathy/nanochat`
- `github.com/rasbt/LLMs-from-scratch` (from Sebastian Raschka's book *Build a Large Language Model (From Scratch)*)

**Courses**
- Stanford **CS336: Language Modeling from Scratch** (lectures and assignments are online)
- Hugging Face **LLM Course** (`huggingface.co/learn`): transformers, datasets, fine-tuning

**Papers (read the abstract, intro, and method sections)**
1. *Attention Is All You Need* (2017), the original Transformer
2. *Language Models are Unsupervised Multitask Learners* (GPT-2)
3. *Language Models are Few-Shot Learners* (GPT-3)
4. *Training Compute-Optimal Large Language Models* (Chinchilla)
5. *LLaMA: Open and Efficient Foundation Language Models*
6. *RoFormer* (RoPE), *FlashAttention*, *GQA*
7. *Training language models to follow instructions with human feedback* (InstructGPT)
8. *Direct Preference Optimization* (DPO)
9. *LoRA* and *QLoRA*
10. The FineWeb dataset paper/blog

**Book for the production side**
- Chip Huyen, *AI Engineering*
