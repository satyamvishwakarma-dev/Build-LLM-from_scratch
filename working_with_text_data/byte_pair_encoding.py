import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     " of someunknownPlace."
)
integer = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integer)

strings = tokenizer.decode(integer)
print(strings)