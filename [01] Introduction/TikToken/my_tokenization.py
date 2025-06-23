# Tokenization simple implementation for understanding

# mapping = {
#     'A': 1,
#     'B': 2,
#     .....
# } # Instead of mapping values like this we can use ASCII

# tokenization by taking ASCII mapping
input_text = "Hello! World"

tokens = [ord(char) for char in input_text]
print(tokens)

# detokenization by taking ASCII mapping
decoded_text = "".join([chr(token) for token in tokens])
print(decoded_text)