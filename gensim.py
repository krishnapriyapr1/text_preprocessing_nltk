import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
import nltk

# Download necessary resources for NLTK
nltk.download('punkt')

# Initialize the stemmer
stemmer = PorterStemmer()

def process_text(text):
    # Tokenization
    tokens = simple_preprocess(text)  # Gensim's simple_preprocess handles basic tokenization
    
    # Remove stop words using Gensim
    tokens_no_stopwords = remove_stopwords(' '.join(tokens)).split()
    
    # Stemming using NLTK
    stemmed_tokens = [stemmer.stem(token) for token in tokens_no_stopwords]
    
    # Lemmatization using Gensim's simple_preprocess (which provides a basic lemmatization)
    lemmatized_tokens = [token for token in tokens_no_stopwords]
    
    return tokens, stemmed_tokens, lemmatized_tokens

# Example usage
if __name__ == "__main__":
    # Sample text
    user_text = "Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle."
    
    # Process the user input
    tokens, stemmed_tokens, lemmatized_tokens = process_text(user_text)
    
    # Print results
    print("\nOriginal Tokens:")
    print(tokens)
    
    print("\nTokens after Stop Word Removal:")
    print(tokens_no_stopwords)
    
    print("\nStemmed Tokens:")
    print(stemmed_tokens)
    
    print("\nLemmatized Tokens:")
    print(lemmatized_tokens)
