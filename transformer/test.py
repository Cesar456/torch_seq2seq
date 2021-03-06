import torch
import torch.nn as nn
import math

# 交叉熵计算
loss = nn.CrossEntropyLoss()
input = torch.randn(1, 5, requires_grad=True)
target = torch.empty(1, dtype=torch.long).random_(5)
output = loss(input, target)

print("输入为5类:")
print(input)
print("要计算loss的类别:")
print(target)
print("计算loss的结果:")
print(output)

first = 0
for i in range(1):
    first -= input[i][target[i]]
second = 0
for i in range(1):
    for j in range(5):
        second += math.exp(input[i][j])
res = 0
res += first + math.log(second)
print("自己的计算结果：")
print(res)

a = torch.tensor([[.1, .2, .3],
                  [1.1, 1.2, 1.3],
                  [2.1, 2.2, 2.3],
                  [3.1, 3.2, 3.3]])
print(a.argmax(dim=1))
print(a.argmax())

