with open("./input.txt", "r",encoding="utf-8") as file:
    data = file.read()
    # print(data)

voc=sorted(list(set(data)))
# print(len(data))
# print(list(set(data)))
# print(set(data))
# print(voc)
chtoi={c:i for i,c in enumerate(voc)}
itoch={i:c for i,c in enumerate(voc)}
encoding=lambda s: [chtoi[c] for c in s]
decoding=lambda s: "".join([itoch[c] for c in s])


print(encoding("hello world"))
print(decoding(encoding("hello world")))