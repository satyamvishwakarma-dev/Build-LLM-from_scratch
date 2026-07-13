import torch

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1, 2, 3])
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(tensor0d)
print(tensor1d)
print(tensor2d)
print(tensor3d)