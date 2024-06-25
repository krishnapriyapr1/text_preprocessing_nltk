import spacy

# Load the English language model in Spacy
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Tokenization
doc = nlp(text)
tokens = [token.text for token in doc]
print("Tokens:")
print(tokens)

# Stop words removal (using Spacy's stop words)
filtered_tokens = [token.text for token in doc if not token.is_stop]
print("\nTokens after Stop Word Removal:")
print(filtered_tokens)

# Lemmatization (Spacy performs lemmatization by default)
lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]
print("\nTokens after Lemmatization:")
print(lemmatized_tokens)
