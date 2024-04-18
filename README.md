# Partial Translation Tool for D&D

This Python script provides a tool for creating partial translations of text, simulating the effect of a character's limited understanding of a language in a Dungeons & Dragons (D&D) game. The script uses natural language processing (NLP) techniques to replace words with educated guesses or indications of uncertainty based on the character's skill level.

## Features

- Replaces words in the input text with guesses or indications of uncertainty based on the character's skill level.
- Uses the Natural Language Toolkit (NLTK) for part-of-speech tagging and accessing lexical resources like WordNet.
- Leverages the Brown Corpus from NLTK for retrieving words of specific parts of speech.
- Utilizes the Pattern library for word inflection and conjugation.
- Allows specifying specific terms that have a higher probability of being garbled.
- Provides options to customize the output, such as showing guesses explicitly or adjusting the minimum chance of guessing.

## Dependencies

- Python 3.x
- NLTK
- NumPy
- Pattern

## Installation

1. Clone the repository or download the script file.
2. Install the required dependencies using pip:
   ```
   pip install nltk numpy pattern
   ```
3. Download the necessary NLTK data by running the following commands in a Python shell:
   ```python
   import nltk
   nltk.download("wordnet")
   nltk.download("brown")
   nltk.download("universal_tagset")
   ```

## Usage

1. Import the `improved_garble_text` function from the script.
2. Provide the input text, character's skill level (a float between 0 and 1), and optionally, a list of specific terms to emphasize.
3. Call the `improved_garble_text` function with the required arguments.
4. The function will return the partially translated text.

Example:
```python
from partial_translation import improved_garble_text

text = "Your input text here"
skill_level = 0.5
specific_terms = ["term1", "term2"]

translated_text = improved_garble_text(text, skill_level, specific_terms)
print(translated_text)
```

## Customization

The script provides several options for customization:
- `CHANCE_OF_GUESSING`: The probability of attempting to guess a word (default: 0.75).
- `MIN_CHANCE`: The minimum probability of guessing a word correctly (default: 0.25).
- `SHOW_AS_GUESS`: Whether to explicitly indicate guessed words in the output (default: False).

## NLP Techniques Used

This script showcases the use of various NLP techniques and tools:
- Part-of-speech tagging using NLTK's `pos_tag` function.
- Accessing lexical resources like WordNet for synonyms, hypernyms, and hyponyms.
- Utilizing the Brown Corpus from NLTK for retrieving words of specific parts of speech.
- Leveraging the Pattern library for word inflection and conjugation.
- Text preprocessing techniques like tokenization and case handling.

## Acknowledgments

- The Natural Language Toolkit (NLTK) - https://www.nltk.org/
- The Pattern library - https://github.com/clips/pattern
- Princeton University "About WordNet." - https://wordnet.princeton.edu/

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, provide feedback, or report any issues you encounter while using this tool.
