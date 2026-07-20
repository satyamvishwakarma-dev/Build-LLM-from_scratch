from numpy import printoptions
import tiktoken
from torch import ne
from torch.utils.data import Dataset, DataLoader
from listing_5 import GPTDatasetV1



def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader

with open("E:\\Build_LLM\\working_with_text_data\\the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
# first_batch = next(data_iter)
# second_batch = next(data_iter)
inputs, targets = next(data_iter)

if __name__ == "__main__":
    # print(first_batch)
    # print(second_batch)
    print("--- INPUTS (What the model sees) ---")
    print(inputs)
    print("\n--- TARGETS (What the model predicts) ---")
    print(targets)
