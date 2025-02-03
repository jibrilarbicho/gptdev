with open("./input.txt", "r",encoding="utf-8") as file:
    data = file.read()
    # print(data)
import torch

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
block_size=8
x_train=datatset[:block_size+1]
y_train=datatset[1:block_size+1]
# print(x_train)
for i in range(block_size):
    context=x_train[:i+1]
    target=y_train[i]
    # print(f"context: {context}, target: {target}")
torch.manual_seed(42)
batch_size=4
block_size=8
def getbatch(split):
    data=trainset if split=="train" else testset
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack(([data[i:i+block_size] for i in ix]))
    y=torch.stack(([data[i+1:i+block_size+1] for i in ix]))
    print(ix)
    return x,y
xb,yb=getbatch("train")
print(xb)
print(yb)

for b in range(batch_size):
    for a in range(block_size):
        context=xb[b,:a+1]
        target=yb[b,a]
        print(f"context: {context}, target: {target}")
    
