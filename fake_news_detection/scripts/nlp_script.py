import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LABELED_DIR = PROJECT_DIR / 'data' / 'processed' / 'labeled'
CLEANED_TEXT_DIR = PROJECT_DIR / 'data' / 'processed' / 'cleaned_text'

INPUT_FILES = [
	'train.processed.csv',
	'valid.processed.csv',
	'test.processed.csv',
	'train_binary.processed.csv',
	'valid_binary.processed.csv',
	'test_binary.processed.csv',
]

NON_LETTER_PATTERN = re.compile(r'[^a-z\s]')
MULTISPACE_PATTERN = re.compile(r'\s+')


def ensure_nltk_resources():
	resources = {
		'corpora/stopwords': 'stopwords',
		'corpora/wordnet': 'wordnet',
		'corpora/omw-1.4': 'omw-1.4',
	}
	for resource_path, download_name in resources.items():
		try:
			nltk.data.find(resource_path)
		except LookupError:
			nltk.download(download_name, quiet=True)


def clean_text(text, stop_words, lemmatizer):
	if pd.isna(text):
		return ''

	text = str(text).lower()
	text = NON_LETTER_PATTERN.sub(' ', text)
	text = MULTISPACE_PATTERN.sub(' ', text).strip()

	cleaned_tokens = []
	for token in text.split():
		if token in stop_words:
			continue
		cleaned_tokens.append(lemmatizer.lemmatize(token))

	return ' '.join(cleaned_tokens)


def process_file(file_name, stop_words, lemmatizer):
	source_path = LABELED_DIR / file_name
	destination_path = CLEANED_TEXT_DIR / file_name

	frame = pd.read_csv(source_path)
	frame['statement_clean'] = frame['statement'].map(
		lambda text: clean_text(text, stop_words, lemmatizer)
	)
	frame.to_csv(destination_path, index=False)


def main():
	CLEANED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
	ensure_nltk_resources()

	stop_words = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()

	for file_name in INPUT_FILES:
		process_file(file_name, stop_words, lemmatizer)


if __name__ == '__main__':
	main()
