from train_gpt2 import GPT
import torch
import torch.nn.functional as F
model = GPT('gpt2')
model.eval()
model.to('cuda')

num_return_sequences = 5
max_length = 30
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode('hello world')
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

        ix = torch.multinomial(topk_probs, num_samples=1)

        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decode = enc.decode(tokens)
    print(">",decode)

