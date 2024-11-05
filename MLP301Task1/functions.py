import nltk
import re

# Define a mapping of sentiment labels to numeric encodings for easier processing.
sentiment_mapping = [
    {"name": "neutral", "encode": 1},
    {"name": "positive", "encode": 2},
    {"name": "negative", "encode": 3}
]
# Function to map sentiment labels to their respective numeric encodings.
def map_sentiment(value):
    for item in sentiment_mapping:
      if item["name"] == value:
          return item["encode"]
    return None 
  
STOP_WORDS = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.WordNetLemmatizer()

# Function to clean and preprocess a list of sentences.
def handle_clean_text(sentences):
  cleaned_sentences = []
  for sentence in sentences: 
    sentence = str(sentence).lower()
    # print("0. original Sentence: ", sentence)
    
    # 1. Removing URLs:
    sentence = re.sub(r"http\S+", "", sentence)
    # print("1. removed HTML Sentence: ", sentence)
    
    # 2. Removing HTML tag <>:
    html = re.compile(r'<.*?>')
    sentence = html.sub(r"", sentence)
    # print("2. removed HTML tag: ", sentence)
    
    # 3. Removing Special Character
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    # print("3. removed special character Sentence: ", sentence)
    
    # 4. Removing punctuations:
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
      sentence = sentence.replace(p, "")
    # print("4. removed punctuations:", sentence)
    
    # 5. Tokenize the sentence and remove stop words:
    sentence = [word.lower() for word in sentence.split() if word.lower() not in STOP_WORDS ]
    # print("5. Tokenized and removed stop words:", len(sentence), sentence)
    
    # 6. Lemmatising reduces words to their core meaning:
    sentence = [lemmatizer.lemmatize(word) for word in sentence]
    # print("6. Lemmantized sentence:", len(sentence), sentence)
    
    sentence = " ".join(sentence)
    
    # 7. Removing emoji:
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    
    sentence = emoji_pattern.sub(r'', sentence)
    # print("7. Removed Emoji sentence:", sentence)
    
    cleaned_sentences.append(sentence)
  return cleaned_sentences