import torch
import torch.nn as nn
from torch.nn import functional as F
with open("./input.txt", "r",encoding="utf-8") as file:
    data = file.read()
    # print(data)
batch_size = 32      
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd=32


voc=sorted(list(set(data)))
# print(len(data))
# print(list(set(data)))
# print(set(data))
# print(voc)
chtoi={c:i for i,c in enumerate(voc)}
itoch={i:c for i,c in enumerate(voc)}
encoding=lambda s: [chtoi[c] for c in s]
decoding=lambda s: "".join([itoch[c] for c in s])

# print(encoding("hello world"))
# print(decoding(encoding("hello world")))

datatset=torch.tensor(encoding(data),dtype=torch.long)
# print(datatset[:100])
n=int(0.9*len(datatset))
trainset=datatset[:n]
testset=datatset[n:]
x_train=datatset[:block_size+1]
y_train=datatset[1:block_size+1]
# print(x_train)
# for i in range(block_size):
#     context=x_train[:i+1]
#     target=y_train[i]
    # print(f"context: {context}, target: {target}")
torch.manual_seed(42)

def getbatch(split):
    data=trainset if split=="train" else testset
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack(([data[i:i+block_size] for i in ix]))
    y=torch.stack(([data[i+1:i+block_size+1] for i in ix]))
    return x,y
xb,yb=getbatch("train")
# print(xb)
# print(yb)
voc_size=len(voc)
# for b in range(batch_size):
#     for a in range(block_size):
#         context=xb[b,:a+1]
#         target=yb[b,a]
#         print(f"context: {context}, target: {target}")

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size,n_embd=32, block_size=8):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  

      
        k = self.key(x)  
        q = self.query(x)  
        wei = q @ k.transpose(-2, -1) * C**-0.5  
        
        # Apply the triangular mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  
        
        wei = F.softmax(wei, dim=-1)  

        v = self.value(x)  
        out = wei @ v 

        return out    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(voc_size,n_embd)
        self.positional_emb=nn.Embedding(block_size,n_embd)
        self.lm_head=nn.Linear(n_embd,voc_size)
        # self.head=Head(n_embd)
        self.head=MultiHeadAttention(4,n_embd//4)
    
    def forward(self, xb,yb=None):
        B,T=xb.shape
        toke_embd=self.token_embedding_table(xb)
        pos_emb=self.positional_emb(torch.arange(T))
        x=toke_embd+pos_emb
        x=self.head(x)


        logits=self.lm_head(x)

        if yb is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            yb=yb.view(B*T)
            loss=F.cross_entropy(logits,yb)

      
        
        return logits,loss
        
    def generate(self, idx, max_new_tokens): 
        # idx is (B, T) array of indices in the current context 
        # for _ in range(max_new_tokens):  
        #     # get the predictions 
        #     logits, loss = self(idx)  

        #     # focus only on the last time step 
        #     print("logits")
        #     print(logits.shape)
        #     logits = logits[:, -1, :]  # becomes (B, C)  
        #     print(logits.shape)

        #     # apply softmax to get probabilities  
        #     probs = F.softmax(logits, dim=-1)  # (B, C)  

        #     # sample from the distribution  
        #     idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)  

        #     # append sampled index to the running sequence  
        #     idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1) 
        #     # print(idx)
        for _ in range(max_new_tokens):  
        
            idx_cond = idx[:, -block_size:]  
            logits, loss = self(idx_cond)  
            logits = logits[:, -1, :]  
            probs = F.softmax(logits, dim=-1)  
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, idx_next), dim=1)  

        return idx


m=BigramLanguageModel()
# logit,loss=m(xb,yb)
# print(logit.shape)
# print(loss)
        
# print(len(voc))
# t=m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()
# print(t)

# print(decoding(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))
optimizer=torch.optim.AdamW(m.parameters(),lr=0.001)

for steps in range(1000):
    xb, yb = getbatch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 100 == 0:
        print(f"Step {steps+1}, Loss: {loss.item():.4f}")
print(decoding(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=700)[0].tolist()))
