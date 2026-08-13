"""
Transform tokeniser output into vectors
"""
import json
import pickle

from backend.lib.processor import BasicProcessor
from common.lib.compatibility import Compatibility
from common.lib.helpers import UserInput, convert_to_int

__author__ = "Stijn Peeters"
__credits__ = ["Stijn Peeters"]
__maintainer__ = "Stijn Peeters"
__email__ = "4cat@oilab.eu"

class Vectorise(BasicProcessor):
	"""
	Creates word vectors from tokens
	"""
	type = "vectorise-tokens"  # job type ID
	category = "Text analysis"  # category
	title = "Count words"  # title displayed in UI
	description = ("Counts how often a token or sequence of tokens (n-gram) appears in the dataset. This creates a bag "
				   "of words.")  # description displayed in UI
	extension = "zip"  # extension of result file, used internally and in UI

	# Allow processor on token sets
	compatibility = Compatibility(types={"tokenise-posts"}, preferred_followups=["vector-ranker"])

	@classmethod
	def get_options(cls, parent_dataset=None, config=None) -> dict:
		"""
		Get processor options

		:param parent_dataset DataSet:  An object representing the dataset that
			the processor would be or was run on. Can be used, in conjunction with
			config, to show some options only to privileged users.
		:param config ConfigManager|None config:  Configuration reader (context-aware)
		:return dict:   Options for this processor
		"""
		return {
			"n_size": {
				"type": UserInput.OPTION_CHOICE,
				"default": "1",
				"options": {
					"1": "unigrams (1)",
					"2": "bigrams (2)",
					"3": "trigrams (3)",
					"1-2": "uni- and bigrams (1-2)",
					"1-3": "uni-, bi- and trigrams (1-3)"
				},
				"help": "Number of words per n-gram",
				"tooltip": "N-grams are sequences of adjacent words. Counting bigrams or trigrams instead of single "
						   "words is useful to find multi-word expressions such as names. N-grams are joined with a "
						   "space and never span two items."
			},
			"min_frequency": {
				"type": UserInput.OPTION_TEXT,
				"default": 1,
				"min": 1,
				"help": "Minimum frequency",
				"tooltip": "Only keep n-grams occurring at least this often within a token set. Raising this is "
						   "recommended for bigrams and trigrams, which produce far more unique values than single "
						   "words."
			}
		}

	def process(self):
		"""
		Unzips token sets, vectorises them and zips them again.
		"""
		# determine which n-gram sizes to count; a range such as '1-3' counts
		# unigrams, bigrams and trigrams alongside each other
		n_size_parameter = str(self.parameters.get("n_size", "1"))
		if "-" in n_size_parameter:
			first, _, last = n_size_parameter.partition("-")
			first = max(1, convert_to_int(first, 1))
			last = max(first, convert_to_int(last, 1))
			n_sizes = list(range(first, last + 1))
		else:
			n_sizes = [max(1, convert_to_int(n_size_parameter, 1))]

		min_frequency = max(1, convert_to_int(self.parameters.get("min_frequency"), 1))

		# tokenising with 'only keep unique tokens per item' discards word order,
		# so any n-gram longer than one word would be an artefact of that
		warning = None
		if max(n_sizes) > 1 and self.source_dataset.parameters.get("only_unique"):
			warning = ("The token set this was based on was created with 'only keep unique tokens per item' enabled, "
					   "which discards word order. The n-grams counted here therefore do not reflect the original "
					   "texts.")

		# prepare staging area
		staging_area = self.dataset.get_staging_area()

		self.dataset.update_status("Processing token sets")
		vector_paths = []

		# go through all archived token sets and vectorise them
		index = 0
		for packed_tokens in self.source_dataset.iterate_items(self):
			if packed_tokens.file.name == '.token_metadata.json':
				# Skip metadata
				continue
			index += 1
			vector_set_name = packed_tokens.file.stem  # we don't need the full path
			self.dataset.update_status("Processing token set %i (%s)" % (index, vector_set_name))
			self.dataset.update_progress(index / self.source_dataset.num_rows)

			# we support both pickle and json dumps of vectors
			token_unpacker = pickle if vector_set_name.split(".")[-1] == "pb" else json
			write_mode = "wb" if token_unpacker is pickle else "w"

			# temporarily extract file (we cannot use ZipFile.open() as it doesn't support binary modes)
			with packed_tokens.file.open("rb") as binary_tokens:
				# these were saved as pickle dumps so we need the binary mode
				tokens = token_unpacker.load(binary_tokens)

				# all we need is a pretty straightforward frequency count, but
				# we count per item rather than over a flattened token list so
				# that n-grams never span the boundary between two items - for
				# unigrams both are equivalent
				vectors = {}
				for item_tokens in tokens:
					for n_size in n_sizes:
						if n_size == 1:
							grams = item_tokens
						else:
							grams = (" ".join(item_tokens[i:i + n_size]) for i in
									 range(len(item_tokens) - n_size + 1))

						for gram in grams:
							if gram not in vectors:
								vectors[gram] = 0
							vectors[gram] += 1

				# convert to vector list
				vectors_list = [[token, frequency] for token, frequency in vectors.items()
								if frequency >= min_frequency]

				# sort
				vectors_list = sorted(vectors_list, key=lambda item: item[1], reverse=True)

				# dump the resulting file via pickle
				vector_path = staging_area.joinpath(vector_set_name)
				vector_paths.append(vector_path)

				with vector_path.open(write_mode) as output:
					token_unpacker.dump(vectors_list, output)

		# create zip of archive and delete temporary files and folder
		self.write_archive_and_finish(staging_area, warning=warning)
