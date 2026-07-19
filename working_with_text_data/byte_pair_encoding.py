import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     " of someunknownPlace."
)
unknown = "Akwir ier"
# integer = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
integer = tokenizer.encode(unknown)
print(integer)

strings = tokenizer.decode(integer)
print(strings)