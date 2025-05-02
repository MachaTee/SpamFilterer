from dataclasses import dataclass, field
from typing import List

import string
import re
import pandas as pd
import datetime

#WORDS = set(words)
WORD_SEQUENCE_LENGTH = 5
REGEX_PARAMETERS = f"[{re.escape(string.punctuation)}]"
NORMALISATION_LEVEL = 1
SPAM_THRESHOLD = .70

#REGEX_IRI = "[[a-z](?:[-a-z0-9\+\.])*:(?:\/\/(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:])*@)?(?:\[(?:(?:(?:[0-9a-f]{1,4}:){6}(?:[0-9a-f]{1,4}:[0-9a-f]{1,4}|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3})|::(?:[0-9a-f]{1,4}:){5}(?:[0-9a-f]{1,4}:[0-9a-f]{1,4}|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3})|(?:[0-9a-f]{1,4})?::(?:[0-9a-f]{1,4}:){4}(?:[0-9a-f]{1,4}:[0-9a-f]{1,4}|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3})|(?:(?:[0-9a-f]{1,4}:){0,1}[0-9a-f]{1,4})?::(?:[0-9a-f]{1,4}:){3}(?:[0-9a-f]{1,4}:[0-9a-f]{1,4}|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3})|(?:(?:[0-9a-f]{1,4}:){0,2}[0-9a-f]{1,4})?::(?:[0-9a-f]{1,4}:){2}(?:[0-9a-f]{1,4}:[0-9a-f]{1,4}|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3})|(?:(?:[0-9a-f]{1,4}:){0,3}[0-9a-f]{1,4})?::[0-9a-f]{1,4}:(?:[0-9a-f]{1,4}:[0-9a-f]{1,4}|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3})|(?:(?:[0-9a-f]{1,4}:){0,4}[0-9a-f]{1,4})?::(?:[0-9a-f]{1,4}:[0-9a-f]{1,4}|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3})|(?:(?:[0-9a-f]{1,4}:){0,5}[0-9a-f]{1,4})?::[0-9a-f]{1,4}|(?:(?:[0-9a-f]{1,4}:){0,6}[0-9a-f]{1,4})?::)|v[0-9a-f]+\.[-a-z0-9\._~!\$&'\(\)\*\+,;=:]+)\]|(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])(?:\.(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])){3}|(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=])*)(?::[0-9]*)?(?:\/(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@]))*)*|\/(?:(?:(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@]))+)(?:\/(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@]))*)*)?|(?:(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@]))+)(?:\/(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@]))*)*|(?!(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@])))(?:\?(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@])|[\x{E000}-\x{F8FF}\x{F0000}-\x{FFFFD}\x{100000}-\x{10FFFD}\/\?])*)?(?:\#(?:(?:%[0-9a-f][0-9a-f]|[-a-z0-9\._~\x{A0}-\x{D7FF}\x{F900}-\x{FDCF}\x{FDF0}-\x{FFEF}\x{10000}-\x{1FFFD}\x{20000}-\x{2FFFD}\x{30000}-\x{3FFFD}\x{40000}-\x{4FFFD}\x{50000}-\x{5FFFD}\x{60000}-\x{6FFFD}\x{70000}-\x{7FFFD}\x{80000}-\x{8FFFD}\x{90000}-\x{9FFFD}\x{A0000}-\x{AFFFD}\x{B0000}-\x{BFFFD}\x{C0000}-\x{CFFFD}\x{D0000}-\x{DFFFD}\x{E1000}-\x{EFFFD}!\$&'\(\)\*\+,;=:@])|[\/\?])*)?$/i]"

# model = Model()

"""
    In a Hidden Markov Model, only the current node matters, previous nodes are not relevant.
"""
@dataclass(frozen=False, order=False)
class WordSequence:
    frequency: float
    is_spam: bool
    chain: field(default_factory = lambda :dict())
    """
        Chain, as a dictionary, strings mapped to frequencies
    """
    sequence: List[str] = field(default_factory = lambda: list())
    sequence_length: int = WORD_SEQUENCE_LENGTH
    already_exist: bool = False
    def __hash__(self):
        return hash((self.frequency, self.is_spam, " ".join(self.sequence)))

model = {}

# Load pandas
training_data = pd.read_csv('data\\spam_detection_training_data.csv').iloc[:2700, :]
test_data = pd.read_csv('data\\spam_detection_training_data.csv').iloc[2700:, :]

# test_data = pd.read_csv('data\\spam_detection_test_data.csv')

# I want a function to split strings into lists of WORD_SEQUENCE_LENGTH
def sliding_window(array: list, window_size: int = 5) -> list:
    # Dynamically adjust window size when nearing end
    array_length = len(array)
    hard_boundary = array_length - window_size

    output_array = []
    for x in range(array_length):
        if x > hard_boundary:
            window_size -= 1
        output_array.append(array[x:x+window_size])

    return output_array

def normalise_data(training_data: pd.core.frame.DataFrame = training_data, normalisation_level: int = NORMALISATION_LEVEL) -> pd.core.frame.DataFrame:
    match normalisation_level:
        case 0:
            return training_data
        case 1:
            training_data['text'] = training_data['text'].replace(r'[/\d+/g]', 'NUM', regex=True)
            return training_data

normalised_test_data = normalise_data(test_data)

# Now, I need to instantiate and populate WordSequence objects, how do I train?
def train(model: dict = model, training_data: pd.core.frame.DataFrame = training_data) -> dict:
    # Give them sequential IDs?
    """
        This is where pre-processing occurs
        Mostly just whitespace normalisation
    """
    normalised_data = normalise_data(training_data, 1).values

    for data_index in range(len(normalised_data)):
        data_values = normalised_data[data_index][0]

        # Deconstruct data into a list of sequences
        data_sequences = sliding_window(data_values.split())
        label_data = normalised_data[data_index][1]

        # Iterate over data_sequences and instantiate WordSequence objects
        # iterate by index because i need to access objects further along and index another array
        for sequence_index in range(len(data_sequences)):
            # Get sequence from index
            sequence = " ".join(data_sequences[sequence_index])
            next_sequence = ""

            try:
                next_sequence = " ".join(data_sequences[sequence_index + 1])
            except IndexError:
                pass

            # Check if sequence exists, if not, create new WordSequence
            if model.get(sequence):
                # Sequence exists, append to existing sequence
                existing_sequence = model[sequence]

                # Increment frequency by one as it was encountered
                existing_sequence.frequency += 1

                # RIGHT I only care about the current one, as it's an HMM
                # If next sequence exists within chain, increment it's frequency, otherwise append
                if existing_sequence.chain.get(next_sequence):
                    # if isinstance(existing_sequence.chain[next_sequence], str):
                        # print(existing_sequence)
                    if isinstance(existing_sequence.chain[next_sequence], str):
                        model[existing_sequence.chain[next_sequence]].frequency += 1
                    else:
                        existing_sequence.chain[next_sequence].frequency += 1
                else:
                    existing_sequence.chain[next_sequence] = next_sequence

            else:
                # Sequence does not exist, create new WordSequence object and append to model
                if not next_sequence:
                    new_word_sequence = WordSequence(1, bool(label_data), dict(), data_sequences[sequence_index], len(data_sequences[sequence_index]))
                else:
                    new_next_word_sequence = WordSequence(1, bool(label_data), dict(), next_sequence.split(), len(next_sequence))

                    if next_sequence in model:
                        new_next_word_sequence = model[next_sequence]
                        new_next_word_sequence.already_exist = True

                    # new_next_word_sequence.is_spam = bool(label_data)
                    new_word_sequence = WordSequence(1, bool(label_data), {next_sequence: new_next_word_sequence}, data_sequences[sequence_index], len(data_sequences[sequence_index]))

                model[sequence] = new_word_sequence

    return model

def discriminate(model: dict = model, test_data: pd.core.frame.DataFrame = test_data, normalised_test_data: pd.core.frame.DataFrame = normalised_test_data, spam_threshold: int = SPAM_THRESHOLD) -> pd.core.frame.DataFrame:
    # Discriminate based on training data
    # model is going to receive test data and classify it based on existing training data
    label_data = []

    # TODO: Delete before submission
    spam_data = []

    accuracy = 0
    for data_index in range(len(normalised_test_data.values)):
        spam_counter = {"spam" : 1, "not_spam" : 1}
        data_values = normalised_test_data.values[data_index][0]

        # Deconstruct data into a list of sequences
        data_sequences = sliding_window(data_values.split())
        for sequence_index in range(len(data_sequences)):
            sequence = " ".join(data_sequences[sequence_index])
            next_sequence = ""

            try:
                next_sequence = " ".join(data_sequences[sequence_index + 1])
            except IndexError:
                pass

            # Check if sequence exists in training data, if not pass
            if model.get(sequence):
                # Calculate increment, being the probability of a message being spam or otherwise
                if model[sequence].chain.get(next_sequence):
                    if model[sequence].is_spam:
                        spam_counter["spam"] += 1
                    else:
                        spam_counter["not_spam"] += 1

        if spam_counter["spam"] / spam_counter["not_spam"] > spam_threshold:
            label_data.append(1)
        else:
            label_data.append(0)

        spam_data.append(spam_counter["spam"] / spam_counter["not_spam"])

    # Insert label data into pandas
    test_data['test_label'] = label_data
    test_data['spam'] = spam_data

    now = datetime.datetime.now()

    for x, y in zip(test_data['test_label'], test_data['label']):
        # print(x, y)
        if (x == y): accuracy += 1

    total_accuracy = accuracy / len(test_data['test_label'])
    print(f"{spam_threshold}\t{total_accuracy}")
    return test_data

