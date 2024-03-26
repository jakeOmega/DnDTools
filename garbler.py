import nltk

from nltk.corpus import wordnet, brown
import random
import numpy as np
import re
from pattern.en import (
    conjugate,
    pluralize,
    singularize,
    lexeme,
)

# download needed nltk data
nltk.download("wordnet")
nltk.download("brown")
nltk.download("universal_tagset")

CHANCE_OF_GUESSING = 0.75
MIN_CHANCE = 0.25
SHOW_AS_GUESS = False

plural_tags = ["NNS", "NNPS", "JJS", "RBS"]
punctuation = [".", ",", "!", "?"]


synset_cache = {}
corpus_words_by_pos = {
    wordnet.NOUN: [
        word[0] for word in brown.tagged_words(tagset="universal") if word[1] == "NOUN"
    ],
    wordnet.VERB: [
        word[0] for word in brown.tagged_words(tagset="universal") if word[1] == "VERB"
    ],
    wordnet.ADJ: [
        word[0] for word in brown.tagged_words(tagset="universal") if word[1] == "ADJ"
    ],
    wordnet.ADV: [
        word[0] for word in brown.tagged_words(tagset="universal") if word[1] == "ADV"
    ],
}


def pattern_stopiteration_workaround():
    try:
        print(lexeme("gave"))
    except:
        pass


pattern_stopiteration_workaround()


def plural(word):
    if singularize(pluralize(word)) == word:
        return word
    else:
        return pluralize(word)


def singular(word):
    if pluralize(singularize(word)) == word:
        return word
    else:
        return singularize(word)


# Function to transform word based on POS tag
def transform_word(word, target_tag):
    if len(word.split()) > 1:
        return word
    if target_tag.startswith("VB"):  # If it's a verb
        if target_tag == "VB":
            # Base form, e.g. "give"
            return word
        elif target_tag == "VBD":
            # Past tense, e.g. "gave"
            return conjugate(word, "p")
        elif target_tag == "VBG":
            # Gerund / Present participle, e.g. "giving"
            return conjugate(word, "part")
        elif target_tag == "VBN":
            # Past participle, e.g. "given"
            return conjugate(word, "ppart")
        elif target_tag == "VBP":
            # Present tense, e.g. "give"
            return conjugate(word, "1sg")
        elif target_tag == "VBZ":
            # 3rd person singular present tense, e.g. "gives"
            return conjugate(word, "3sg")
    elif target_tag in ["NN", "NNS", "NNP", "NNPS"]:  # If it's a noun
        if target_tag in ["NNS", "NNPS"]:
            # Plural noun
            return plural(word)
    elif target_tag in ["JJ", "JJR", "JJS"]:  # If it's an adjective
        if target_tag in ["JJ", "JJR", "JJS"]:
            # Adjective
            return word
    elif target_tag in ["RB", "RBR", "RBS"]:  # If it's an adverb
        if target_tag == "RB":
            # Adverb
            return word
    return word


def get_synsets(word, pos):
    if (word, pos) in synset_cache:
        return synset_cache[(word, pos)]
    else:
        synsets = wordnet.synsets(word, pos=pos_to_wordnet_pos(pos))
        synset_cache[(word, pos)] = synsets
        return synsets


def get_synonyms(word, pos):
    unique_synonyms = set()
    synsets = get_synsets(word, pos)
    for synset in synsets:
        for lemma in synset.lemmas():
            # if not proper noun
            if not lemma.name().istitle():
                synonym = lemma.name().replace("_", " ").replace("\n", "")
                unique_synonyms.add(synonym)

    return unique_synonyms


def get_misleading_synonyms(word, pos):
    misleading_synonyms = []
    synsets = wordnet.synsets(word, pos=pos_to_wordnet_pos(pos))
    for synset in synsets:
        # antonyms
        for lemma in synset.lemmas():
            for antonym in lemma.antonyms():
                misleading_synonyms.append(
                    antonym.name().replace("_", " ").replace("\n", "")
                )

    if not misleading_synonyms:
        corpus_pos = pos_to_wordnet_pos(pos)
        if corpus_pos not in corpus_words_by_pos:
            print(f"no words in corpus for {pos}, word: {word}, using nouns instead")
            corpus_pos = wordnet.NOUN
        misleading_synonyms = random.choices(corpus_words_by_pos[corpus_pos], k=6)

    return misleading_synonyms


def get_hypernyms(word, pos):
    unique_hypernyms = set()
    synsets = get_synsets(word, pos)
    for synset in synsets:
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                # if not proper noun
                if not lemma.name().istitle():
                    hypernym = lemma.name().replace("_", " ").replace("\n", "")
                    unique_hypernyms.add(hypernym)

    return unique_hypernyms


def get_hyponyms(word, pos):
    unique_hyponyms = set()
    synsets = get_synsets(word, pos)
    for synset in synsets:
        for hyponym in synset.hyponyms():
            for lemma in hyponym.lemmas():
                # if not proper noun
                if not lemma.name().istitle():
                    hyponym = lemma.name().replace("_", " ").replace("\n", "")
                    unique_hyponyms.add(hyponym)

    return unique_hyponyms


def pos_to_wordnet_pos(penntag, returnNone=False):
    morphy_tag = {
        "NN": wordnet.NOUN,
        "JJ": wordnet.ADJ,
        "VB": wordnet.VERB,
        "RB": wordnet.ADV,
    }
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ""


def roll_to_skill_level(dice_roll_plus_modifiers, skill_dc):
    """
    Converts a dice roll to a skill level based on a DC. Scales from 0.25 at a roll of 1 to 1.0 at a roll of the DC.
    """
    return max(
        MIN_CHANCE,
        min(1.0, MIN_CHANCE + (1 - MIN_CHANCE) * dice_roll_plus_modifiers / skill_dc),
    )


def get_weights(words, POS, multiplier=1):
    if len(words) == 0:
        return []
    weights = [
        len(wordnet.synsets(word, pos=pos_to_wordnet_pos(POS))) for word in words
    ]
    total_weight = sum(weights)
    normalized_weights = [w * multiplier / total_weight for w in weights]
    return normalized_weights


def guess_word(word, POS, skill_level):
    """
    Attempts to make an educated guess about a word by finding its synonyms or related terms.
    Returns a string like "[unknown word]" or "[something related to 'word']".
    """
    word = word.lower()
    misleading_chance = 1 - skill_level
    if random.random() < misleading_chance:
        potential_synonyms = list(get_misleading_synonyms(word, POS))
        if not potential_synonyms:
            print(f"no misleading synonyms for {word}, {POS}")
    else:
        synonyms = list(get_synonyms(word, POS))
        hypernyms = list(get_hypernyms(word, POS))
        hyponyms = list(get_hyponyms(word, POS))
        potential_synonyms = synonyms + hypernyms + hyponyms
        if not potential_synonyms:
            print(f"no synonyms for {word}, {POS}")

    potential_synonyms = [s for s in potential_synonyms if s.lower() == s]
    if not potential_synonyms or random.random() >= CHANCE_OF_GUESSING:
        return None

    guess = np.random.choice(potential_synonyms)
    return guess


def improved_garble_text(text, skill_level=0.0, specific_terms=[]):
    """
    Improved version of the garble_text function.
    Replaces words with guesses or indications of uncertainty.
    """
    lines = text.split("\n")
    garbled_text = []
    translated_words = {}
    for line in lines:
        # treat the text as a list of words, punctuation as separate tokens
        words = re.findall(r"[\w']+|[.,!?;]", line)

        for word, POS in nltk.pos_tag(words):
            # skip stopwords and punctuation or words that aren't nouns, verbs, adjectives, or adverbs and proper nouns
            if (
                word.lower() in nltk.corpus.stopwords.words("english")
                or word in punctuation
                or pos_to_wordnet_pos(POS, returnNone=True) is None
                or POS in ["NNP", "NNPS"]
            ):
                translated_words[word.lower()] = word
                garbled_text.append(word)
                continue
            # Increase probability of garbling specific terms
            if any(term.lower() in word.lower() for term in specific_terms):
                prob = 1 - skill_level / 2  # Increase likelihood for specific terms
            else:
                prob = 1 - skill_level

            # Randomly decide whether to garble this word
            if word.lower() in translated_words.keys():
                translated_word = translated_words[word.lower()]
                if translated_word:
                    translated_word = translated_word.lower()
            elif random.random() < prob:
                # Guess the word based on synonyms or mark as unknown
                translated_word = guess_word(word, POS, skill_level)
            else:
                translated_word = word
            translated_words[word.lower()] = translated_word
            if translated_word is None:
                transformed_word = "[unknown word]"
            else:
                transformed_word = transform_word(translated_word, POS)
                # match case
                if word.istitle():
                    transformed_word = transformed_word.title()
                if translated_word != word and SHOW_AS_GUESS:
                    transformed_word = f"[possibly '{transformed_word}']"
            garbled_text.append(transformed_word)
        garbled_text.append("\n")
    output = " ".join(garbled_text)
    # replace space before punctuation (e.g. " , " -> ", ")
    for p in punctuation:
        output = output.replace(f" {p}", p)
    return output


example_text = """I had entered into a marriage
In the summer of my twenty-first year
And the bells rang for our wedding
Only now do I remember it clear
Alright, alright, alright
No more a rake and no more a bachelor
I was wedded and it whetted my thirst
Until her womb start spilling out babies
Only then did I reckon my curse
Alright, alright, alright
Alright, alright, alright
First came Eziah with his crinkled little fingers
Then came Charlotte and that wretched girl Dawn
Ugly Myfanwy died on delivery
Mercifully taking her mother along
Alright, alright, alright
What can one do when one is widower?
Shamefully saddled with three little pests
All that I wanted was the freedom of a new life
So my burden I began to divest
Alright, alright, alright
Alright, alright, alright
Charlotte, I buried after feeding her foxglove
Dawn was easy, she was drowned in the bath
Eziah fought but was easily bested
Burned his body for incurring my wrath
Alright, alright, alright
And that's how I came your humble narrator
To be living so easy and free
Expect you think that I should be haunted
But it never really bothers me
Alright, alright, alright
Alright, alright, alright"""

specific_terms = []

roll = 10

skill_level = roll_to_skill_level(roll, 30)
improved_garbled_example = improved_garble_text(
    example_text, skill_level, specific_terms=specific_terms
)
print(roll, improved_garbled_example)
