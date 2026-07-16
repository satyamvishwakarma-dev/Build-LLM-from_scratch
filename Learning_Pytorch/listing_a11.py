import torch

tensor_1 = torch.tensor([1, 2, 3])
tensor_2 = torch.tensor([4, 5, 6])

# print(f"tensor_1 + tensor_2 = {tensor_1 + tensor_2}")

tensor_1 = tensor_1.to("cuda:0")
tensor_2 = tensor_2.to("cuda:0")
print(f"tensor_1 + tensor_2 = {tensor_1 + tensor_2}")
