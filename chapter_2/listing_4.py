import re
class SimpleTokenizerV2:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" 
            for item in preprocessed
            ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

if __name__ == "__main__":
    # Opening the text file and preprocessing it
    with open("E:\\Build_LLM\\working_with_text_data\\the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item for item in preprocessed if item.strip()]

    # Getting all the world in soeted list
    all_words = sorted(set(preprocessed))

    all_words.extend(["<|endoftext|>", "<|unk|>"])

    # Creating a vocabulary
    vocab = {token:integer for integer, token in enumerate(all_words)}
    
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit tarraces of the palace."
    text = " <|endoftext|> ".join([text1, text2])
    # print(text)

    tokanizer = SimpleTokenizerV2(vocab)
    print(tokanizer.encode(text))

    print(tokanizer.decode(tokanizer.encode(text)))