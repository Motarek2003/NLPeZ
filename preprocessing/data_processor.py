import re
from typing import List, Tuple, Dict, Any

# --- Global Constants for Diacritics ---
# Define individual diacritic characters
DIACRITICS_CHARS = 'ًٌٍَُِّْ'
# Define the regex pattern for stripping (uses the character set)
DIACRITICS_PATTERN = r'[' + DIACRITICS_CHARS + r']'
NO_TASHKEEL = '<NT>'


class ArabicDiacritizationProcessor:
    """
    Handles data preprocessing, cleaning, and tokenization for Arabic Diacritization,
    serving as the pipeline's Data Ingestion and Preparation module.
    """
    def __init__(self):
        # 1. Individual diacritics
        base_diacritics = list(set(DIACRITICS_CHARS))

        # 2. Add common combined forms explicitly as single labels (Shadda + Vowel/Tanween)
        combined_diacritics = [
            'ًّ', # Shadda + Fatha Tanween (Normalized order: Shadda first)
            'ٌّ', # Shadda + Damma Tanween
            'ٍّ', # Shadda + Kasra Tanween
            'َّ', # Shadda + Fatha
            'ُّ', # Shadda + Damma (Note: using ُّ for Damma-Shadda)
            'ِّ', # Shadda + Kasra
            'ْ'    # Sukun (Single)
        ]

        # Combine all unique labels for the classification layer
        self.all_labels = sorted(list(set(base_diacritics + combined_diacritics)))
        self.all_labels.append(NO_TASHKEEL)

        # Create mapping dictionaries
        self.label_to_id = {label: i for i, label in enumerate(self.all_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        self.DIACRITICS_SET = DIACRITICS_CHARS # Used for lookup in split_char_diacritic
        self.NO_TASHKEEL = NO_TASHKEEL

    def clean_text(self, text: str) -> str:
        """Removes unwanted metadata, non-Arabic, and non-diacritic characters."""
        # Note: You may want to retain punctuation marks if they are relevant for context.
        # Here we follow the original cleaning logic:
        text = re.sub(r'[\(\)\[\]\{\}\<\>]+', '', text)
        text = re.sub(r'[a-zA-Z]+', '', text)
        return text.strip()

    def split_char_diacritic(self, sentence: str) -> Tuple[List[str], List[str]]:
        """
        Splits a diacritized Arabic sentence into parallel sequences of
        undiacritized characters and their corresponding diacritics.
        """
        characters: List[str] = []
        diacritics: List[str] = []
        i = 0
        while i < len(sentence):
            char = sentence[i]
            current_diacritic = NO_TASHKEEL
            next_i = i + 1
            diacritic_seq = ''

            # Aggregate all diacritics immediately following the character
            while next_i < len(sentence) and sentence[next_i] in self.DIACRITICS_SET:
                diacritic_seq += sentence[next_i]
                next_i += 1

            if diacritic_seq:
                current_diacritic = self.normalize_diacritic_sequence(diacritic_seq)

            if char not in self.DIACRITICS_SET:
                characters.append(char)
                diacritics.append(current_diacritic)

            i = next_i

        return characters, diacritics

    def normalize_diacritic_sequence(self, diac_seq: str) -> str:
        """
        Normalizes the order of combined diacritics (Shadda first) to match a
        pre-defined label (e.g., 'ٌّ').
        """
        if len(diac_seq) > 1:
            # Sorts to ensure Shadda (ّ) comes first, if present.
            normalized = ''.join(sorted(diac_seq, key=lambda x: 0 if x == 'ّ' else 1))

            # Check if the normalized sequence is one of the known combined labels
            # This handles cases where two diacritics that DON'T form a known combined
            # label are present. For now, we assume only Shadda combinations occur.
            if normalized in self.label_to_id:
                return normalized
            else:
                 # Fallback: If normalized form isn't a known label, use a single base diacritic
                 # This logic is a simplification and may need refinement based on corpus analysis.
                 return diac_seq[0] # Simplistic handling for unknown combinations

        return diac_seq

    # Define common Arabic diacritics for later stripping
    ARABIC_DIACRITICS = re.compile(r'[\u064b-\u065e\u0670\u0671]')
    def tokenize_to_words(self, diacritized_sentence: str) -> List[str]:
        """
        Takes a raw (diacritized) Arabic sentence and returns a list of
        undiacritized, cleaned word tokens. This output is used for FastText lookup.

        Args:
            diacritized_sentence: The fully vocalized string from the dataset.

        Returns:
            A list of word strings, normalized and undiacritized.
        """
        # 1. Clean the text (removes non-Arabic/non-punctuation, etc.)
        cleaned_text = self.clean_text(diacritized_sentence)

        # 2. Strip diacritics
        undiacritized_text = self.strip_diacritics(cleaned_text)

        # 3. Use standard space-based tokenization. Punctuation should typically
        #    be separated or removed during the initial clean_text phase if not needed.
        #    We filter out empty strings resulting from multiple spaces.
        words = [word for word in undiacritized_text.split() if word.strip()]

        # 4. Apply additional normalization (e.g., Hamza unification, if needed)
        #    Ensure these normalizations are consistent with what was used when training FastText.

        return words

    @staticmethod
    def strip_diacritics(text: str) -> str:
        """
        Static utility to remove diacritics from a single string.
        """
        return re.sub(DIACRITICS_PATTERN, '', text)

    def generate_undiacritized_corpus(self,
                                     input_file_path: str,
                                     output_file_path: str) -> str:
        """
        Reads a diacritized corpus file and generates an undiacritized version
        for use in unsupervised training (like FastText).
        """
        print(f"Generating undiacritized corpus from: {input_file_path}")
        undiacritized_lines: List[str] = []

        try:
            with open(input_file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # 1. Cleaning: Remove metadata/non-Arabic
                    cleaned_line = self.clean_text(line)

                    # 2. Stripping: Remove all diacritics for FastText input
                    undiacritized_line = self.strip_diacritics(cleaned_line)

                    # 3. Tokenization: Ensure lines are space-separated words
                    words = undiacritized_line.strip().split()

                    if words:
                        # FastText requires one long text file, typically space-separated words
                        undiacritized_lines.append(' '.join(words))

            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                for line in undiacritized_lines:
                    outfile.write(line + '\n')

            print(f"Successfully created undiacritized corpus at: {output_file_path}")
            return output_file_path

        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file_path}")
            return ""



# --- Example Usage for Testing the Split and Labeling ---
# processor = ArabicDiacritizationProcessor()
# diacritized_text = "ذَهَبَ عَلِيٌّ إِلَى الْمَدْرَسَةِ"
# cleaned = processor.clean_text(diacritized_text)
# char_seq, diac_seq = processor.split_char_diacritic(cleaned)
# print(f"Characters: {char_seq}")
# print(f"Diacritics: {diac_seq}")
# print(f"Labels (IDs): {[processor.label_to_id[d] for d in diac_seq]}")

processor = ArabicDiacritizationProcessor()
input = r'data/cleaned/trainc_data.txt'
output = "data/undiacritized/traincu_data.txt"
processor.generate_undiacritized_corpus(input, output)