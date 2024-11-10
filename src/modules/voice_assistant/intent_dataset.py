import json
import torch
from torch.utils.data import Dataset
import nltk
from nltk.stem import WordNetLemmatizer

class IntentDataset(Dataset):
    def __init__(self, intent_json_path: str, ignore_letters: tuple = ("!", "?", ",", ".")):
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

        self.words = []
        self.intents = []
        self.training_data = []
        self.lemmatizer = WordNetLemmatizer()

        self._prepare_intents_data(intent_json_path, ignore_letters)

    def _prepare_intents_data(self, intent_json_path: str, ignore_letters: tuple):
        with open(intent_json_path, 'r') as f:
            self.intents_data = json.load(f)
        
        documents = []
        
        for intent in self.intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])

            for pattern in intent["patterns"]:
                pattern_words = nltk.word_tokenize(pattern)
                self.words += pattern_words
                documents.append((pattern_words, intent["tag"]))

        # Lemmatize words and remove stopwords/punctuation
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(set(self.words))

        for document in documents:
            bag_of_words = self._get_bag_of_words(document[0])
            label_index = self.intents.index(document[1])
            self.training_data.append([bag_of_words, label_index])

        # Convert data to torch tensors
        self.X = torch.tensor([data[0] for data in self.training_data], dtype=torch.float32)
        self.y = torch.tensor([data[1] for data in self.training_data], dtype=torch.long)

    def _get_bag_of_words(self, pattern_words):
        return [1 if word in pattern_words else 0 for word in self.words]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_words_and_intents(self):
        return self.words, self.intents