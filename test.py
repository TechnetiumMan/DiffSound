import torch
a = torch.zeros([3], requires_grad=True).cuda()
b = a[1]
b.backward()