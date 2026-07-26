import torch
import torch.nn as nn

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)

mean = out.mean(dim=-1, keepdim = True)
var = out.var(dim=-1, keepdim = True)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim = True)
var = out_norm.var(dim=-1, keepdim = True)

torch.set_printoptions(sci_mode=False)

if __name__ == '__main__':
    # print(out)

    # print("mean\n", mean)

    # print("var\n", var)

    print("Mormalized output\n", out_norm)

    print("\nMean normalized output\n", mean)
    
    print("\nVariance normalized output\n", var)

    