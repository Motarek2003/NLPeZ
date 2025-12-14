import re
import os

class ArabicCleaner:
    def __init__(self):
        # 1. Tatweel Removal
        self.tatweel_pattern = re.compile(r'\u0640')

        # 2. Punctuation Removal (Based on CATT dataset observation)
        # Note: This removes diacritic-carrying punctuation. Only keep space.
        self.punctuation_pattern = re.compile(r'[.,:;?!()"\'\[\]{}<>+\-*/=&%#@$^\u0660-\u0669\u06F0-\u06F9]')

        # 3. Non-Arabic Noise (Latin letters, numbers, etc.)
        # This will remove noise like ( 24 / 355 ) as well.
        # We target characters that are NOT Arabic letters (0621-063A, 0641-064A)
        # or Diacritics (064B-065F) or standard space.
        self.noise_pattern = re.compile(r'[^ \u0621-\u063A\u0641-\u064A\u064B-\u065F]')

        # 4. Whitespace cleanup
        self.extra_space_pattern = re.compile(r'\s+')

    def normalize(self, text):
        # 1. Remove Tatweel
        text = self.tatweel_pattern.sub("", text)

        # 2. Remove Punctuation and Numbers (This is aggressive, but matches CATT's training strategy)
        text = self.punctuation_pattern.sub("", text)

        # 3. Remove General Non-Arabic Noise (Keep only Arabic, Diacritics, and spaces)
        text = self.noise_pattern.sub("", text)

        # 4. Whitespace Cleanup
        text = self.extra_space_pattern.sub(" ", text).strip()
        return text

    def clean_file(self, input_filepath, output_filepath):
        print(f"Cleaning: {input_filepath} -> {output_filepath}")
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:

            for line in infile:
                cleaned_line = self.normalize(line)
                if cleaned_line:  # Avoid writing empty lines
                    outfile.write(cleaned_line + '\n')
        print("Cleaning complete.")


# --- Execution Example (assuming your data is already split) ---
if __name__ == '__main__':
    cleaner = ArabicCleaner()

    # Assume your 50k train, 2.5k val, and test data are in data/train.txt, data/val.txt, data/test.txt
    # You must have split the data.txt into these three files first.

    # Ensure the target directory for the CATT script exists
    os.makedirs('data', exist_ok=True)

    # Process the files
    cleaner.clean_file('train.txt', 'cleaned_train_data.txt')
    cleaner.clean_file('val.txt', 'cleaned_val_data.txt')
    #cleaner.clean_file('data/test.txt', 'dataset/test/full_data.txt')