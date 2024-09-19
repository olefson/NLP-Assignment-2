"""This is a sample file for hw2. 
It contains the function that should be submitted,
except all it does is output a random value.
- Dr. Licato"""

from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.naive_bayes
import random

"""
trainFile: a text file, where each line is arbitratry human-generated text
Outputs n-grams (n=2, or n=3, your choice). Must run in under 120 seconds
"""
def calcNGrams_train(trainFile):
	pass #don't return anything from this function!

"""
sentences: A list of single sentences. All but one of these consists of entirely random words.
Return an integer i, which is the (zero-indexed) index of the sentence in sentences which is non-random.
"""
def calcNGrams_test(sentences):
	return random.randint(0, len(sentences)-1)

"""
trainFile: A jsonlist file, where each line is a json object. Each object contains:
	"review": A string which is the review of a movie
	"sentiment": A Boolean value, True if it was a positive review, False if it was a negative review.
"""
def calcSentiment_train(trainFile):
	pass #don't return anything from this function!

"""
review: A string which is a review of a movie
Return a boolean which is the predicted sentiment of the review.
Must run in under 120 seconds, and must use Naive Bayes
"""
def calcSentiment_test(review):
	return random.choice([True, False])