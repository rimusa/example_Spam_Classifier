import pandas as pd
from bs4 import BeautifulSoup
from nltk.tokenize import TweetTokenizer
from copy import copy

tokenizer_dataloader = TweetTokenizer()


class DataLoader():

    def __init__(self, path, train_size=0.8, add_name=False, keep_case=False, unk_id=-1):

        self.unk_id = unk_id
        self.lowercase_flag = unk_id

        self._read_csv(path, train_size, add_name=False, keep_case=False)        


    def _read_csv(self, path, train_size=0.8, add_name=False, keep_case=False):

        to_keep = ["CONTENT", "CLASS"]

        if add_name:
            to_keep.append("VIDEO_NAME")

        df = pd.read_csv(path).sample(frac=1.)
        df = df[to_keep]


        df["CONTENT"] = df["CONTENT"].map(self._data_cleaning)


        all_words = {}
        word_count = {}

        self.word2idx = {}
        self.idx2word = {}

        counter = 0

        text = df["CONTENT"]
        for sentence in text:
            for token in sentence:
                if token not in all_words:
                    all_words[token] = 1
                else:
                    all_words[token] += 1

        for key in all_words:
            if all_words[key] > 5:
                word_count[key] = all_words[key]
                self.word2idx[key] = counter
                self.idx2word[counter] = key
                counter += 1

        n_datapoints = df.shape[0]

        self.data = {}
        self.data["train"] = df.iloc[:int(n_datapoints*train_size),:]
        self.data["valid"] = df.iloc[int(n_datapoints*train_size):,:]

        self.size = {key: split.shape[0] for key, split in self.data.items()}


    def _transform_text_to_id(self, sentence):
        sentence_ids = []
        for token in sentence:
            if token in self.word2idx:
                sentence_ids.append(self.word2idx[token])
            else:
                sentence_ids.append(self.unk_id)
        return sentence_ids


    def _data_cleaning(self, sentence):

        if self.lowercase_flag:
            sentence = sentence.lower()

        soup = str(BeautifulSoup(sentence, 'html.parser'))
        
        tokenized_sentence = tokenizer_dataloader.tokenize(soup)

        cleaned_tokens = []

        for token in tokenized_sentence:

            if token[:4] == "http":
                cleaned_tokens.append("[URL]")

            elif token[0] == "@":
                cleaned_tokens.append("[USERNAME]")

            elif token[0] == "#":
                if len(token) != 1:
                    cleaned_tokens.append("[HASHTAG]")
                else:
                    cleaned_tokens.append(token)

            else:
                
                cleaned_tokens.append(token)

        return cleaned_tokens


    def batcher(self, split, shuffle=True, batch_size=16):

        current_data = copy(self.data[split])

        if shuffle:
            current_data = current_data.sample(frac=1)

        batch_min = 0
        batch_max = batch_size

        while batch_min < self.size[split]:
            print(self.size[split])
            print(batch_max)
            print(min(batch_max,self.size[split]))
            yield current_data.iloc[[batch_min,min(batch_max,self.size[split])],:]
            stored_size = copy(batch_max)
            batch_min = stored_size
            batch_max = stored_size + batch_size
            print(batch_max)





if __name__ == "__main__":
    print("Hi!")
    path = "./Youtube-Spam-Dataset.csv"
    dataset = DataLoader(path)
    for _ in dataset.batcher("valid", batch_size=100):
        print("A")
    print("Finished running!")