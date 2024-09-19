"""This is a sample file for hw2. 
It contains the function that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

# Some dependencies
from collections import defaultdict
import nltk
import json

nltk.download('punkt')

# Global Variables
# Problem 1
ngrams_dict = defaultdict(int)

# Problem 2
vectorizer = CountVectorizer()
model = MultinomialNB()

"""
trainFile: a text file, where each line is arbitratry human-generated text
Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
"""
def calcNGrams_train(trainFile):
    global ngrams_dict
    with open(trainFile, 'r') as f:
      for line in f:
        #  tokenize each line
        tokens = nltk.word_tokenize(line.lower())
        #  create n-grams
        bigrams = list(nltk.bigrams(tokens))
        # keep count in dict
        for bigram in bigrams:
            ngrams_dict[bigram] += 1
"""
sentences: A list of single sentences. All but one of these consists of entirely random words.
Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is non-random.
"""
def calcNGrams_test(sentences):
    best_index = -1
    max_matches = -1
    
    for i, sentence in enumerate(sentences):
        tokens = nltk.word_tokenize(sentence.lower())
        bigrams = list(nltk.bigrams(tokens))
        match_count = sum(1 for bigram in bigrams if bigram in ngrams_dict) # count matches
        # keep track of bst matching sentence
        if match_count > max_matches:
            max_matches = match_count
            best_index = i
    return best_index

"""
trainFile: A jsonlist file, where each line is a json object. Each object contains:
	"review": A string which is the review of a movie
	"sentiment": A Boolean value, True if it was a positive review, False if it was a negative review.
"""
def calcSentiment_train(trainFile):
	global vectorizer, model
	reviews = []
	sentiments = []
	
	# Read the me 😊 (json) file
	with open(trainFile, 'r') as f:
		for line in f:
			data = json.loads(line)
			reviews.append(data['review']) # collect review text
			sentiments.append(1 if data['sentiment'] else 0) # collect sentiment True=1, False=0
   
	# Vectorize the reviews
	X = vectorizer.fit_transform(reviews)
 
	# Train the model
	model.fit(X, sentiments)
"""
review: A string which is a review of a movie
Return a boolean which is the predicted sentiment of the review.
Must run in under 120 seconds, and must use Naive Bayes
"""
def calcSentiment_test(review):
	global vectorizer, model
	# Transoform input review into vector
	X = vectorizer.transform([review])
	# Predict sentiment
	prediction = model.predict(X)
	return bool(prediction[0])