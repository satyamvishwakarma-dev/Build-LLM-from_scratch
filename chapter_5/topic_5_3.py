import tiktoken

from chapter_5.listing_5_1 import token_ids
from chapter_5.topic_5_1 import model

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = 
