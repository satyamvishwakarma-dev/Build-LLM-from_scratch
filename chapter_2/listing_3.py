import re


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
if __name__ == "__main__":
    # Opening the text file and preprocessing it
    with open("E:\\Build_LLM\\working_with_text_data\\the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item for item in preprocessed if item.strip()]

    # Getting all the world in soeted list
    all_words = sorted(set(preprocessed))

    # Creating a vocabulary
    vocab = {token:integer for integer, token in enumerate(all_words)}

    # Creating a tokenizer
    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride"""
    
    # Encoding the text
    ids = tokenizer.encode(text)
    print("Encoded ids:", ids)
    # print(ids)
    print("Encoded text:", tokenizer.decode(ids))