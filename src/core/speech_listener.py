import random
import speech_recognition as sr
import torch
import nltk
from src.modules.voice_assistant.intent_dataset import IntentDataset
from src.modules.voice_assistant.intent_model import IntentModel
from src.modules.voice_assistant.intent_method import IntentMethod

class SpeechListener:
    def __init__(self, add_message_callback, model_path='trained_data/intent_model.pth', device='cpu'):
        self.add_message = add_message_callback
        self.recognizer = sr.Recognizer()
        self.intent_dataset = IntentDataset('data/speech/intents.json')
        self.words, self.intents = self.intent_dataset.get_words_and_intents()
        self.intent_method = IntentMethod()

        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.is_listening = False

    def load_model(self, model_path):
        model = IntentModel(input_size=len(self.words), output_size=len(self.intents))
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        model.eval()
        return model

    def start_listening(self):
        self.is_listening = True
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")

            while self.is_listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio)
                    self.add_message("You", text)
                    intent_response = self.process_input(text)
                    self.add_message("AI", f"{intent_response}")
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")

    def process_input(self, input_text):
        predicted_intent = self._predict_intent(input_text)

        try:
            response = self.intent_method.handle_intent(predicted_intent, input_text)
            return response
        except IndexError:
            return "I don't understand. Please try again."
        
    def _predict_intent(self, input_text):
        input_bag_of_words = self._text_to_tensor(input_text)

        # Assuming model is loaded and available
        with torch.no_grad():
            output = self.model(input_bag_of_words)
            _, predicted_idx = output.max(1)

        predicted_intent = self.intents[predicted_idx.item()]
        return predicted_intent

    def _text_to_tensor(self, input_text):
        input_words = nltk.word_tokenize(input_text)
        input_words = [self.intent_dataset.lemmatizer.lemmatize(w.lower()) for w in input_words]

        input_bag_of_words = [0] * len(self.words)
        for input_word in input_words:
            for i, word in enumerate(self.words):
                if input_word == word:
                    input_bag_of_words[i] = 1

        return torch.tensor([input_bag_of_words], dtype=torch.float32)

    def stop_listening(self):
        self.is_listening = False
        print("Stopped listening.")

# Example usage of SpeechListener
def add_message(speaker, message):
    print(f"{speaker}: {message}")

def main():
    speech_listener = SpeechListener(add_message)
    try:
        speech_listener.start_listening()
    except KeyboardInterrupt:
        speech_listener.stop_listening()

if __name__ == '__main__':
    main()