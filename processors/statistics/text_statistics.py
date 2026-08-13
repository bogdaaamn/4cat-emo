"""
Calculate vocabulary statistics for tokenised text, via textacy
"""
import itertools
import json
import math
import pickle

from collections import Counter

from natsort import natsorted

import spacy
from spacy.tokens import Doc

from textacy.text_stats import basics, diversity
from textacy.text_stats import utils as textacy_utils

from backend.lib.processor import BasicProcessor
from common.lib.compatibility import Compatibility
from common.lib.exceptions import ProcessorInterruptedException
from common.lib.helpers import UserInput, convert_to_int

__author__ = "Bogdan Covrig"
__credits__ = ["Bogdan Covrig"]
__maintainer__ = "Bogdan Covrig"
__email__ = "b.covrig@maastrichtuniversity.nl"


class TextStatistics(BasicProcessor):
	"""
	Calculate vocabulary statistics per token set, using textacy
	"""
	type = "text-statistics"  # job type ID
	category = "Statistics"  # category
	title = "Vocabulary statistics"  # title displayed in UI
	description = ("Calculates how large and how varied the vocabulary of a set of tokens is: token and type counts, "
				   "type-token ratios, lexical diversity measures that are not biased by text length (MATTR, MTLD, "
				   "HD-D), hapax legomena and concentration measures. One row per token set, so tokenising per month "
				   "or year gives a comparison over time. Uses textacy.")
	extension = "csv"  # extension of result file, used internally and in UI

	# Allow processor on token sets
	compatibility = Compatibility(types={"tokenise-posts"})

	references = [
		"[textacy text_stats documentation](https://textacy.readthedocs.io/en/latest/api_reference/text_stats.html)"
	]

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
			"segment_size": {
				"type": UserInput.OPTION_TEXT,
				"default": 100,
				"min": 10,
				"max": 1000,
				"help": "MATTR window size",
				"tooltip": "MATTR is the type-token ratio averaged over every window of this many consecutive tokens. "
						   "Token sets with fewer tokens than this get no MATTR value. 100 is conventional."
			},
			"round_to": {
				"type": UserInput.OPTION_TEXT,
				"default": 4,
				"min": 1,
				"max": 10,
				"help": "Decimal places",
				"tooltip": "How many decimal places to round the calculated ratios to."
			}
		}

	def process(self):
		"""
		Reads token sets and writes one row of vocabulary statistics per set
		"""
		segment_size = max(10, convert_to_int(self.parameters.get("segment_size"), 100))
		round_to = max(1, convert_to_int(self.parameters.get("round_to"), 4))

		# tokenising with 'only keep unique tokens per item' discards duplicate
		# tokens, which is precisely what these measures are counting
		warning = None
		if self.source_dataset.parameters.get("only_unique"):
			warning = ("The token set this was based on was created with 'only keep unique tokens per item' enabled, "
					   "which removes repeated tokens. Diversity is overstated as a result.")

		# textacy's measures take spaCy tokens, so the tokens are wrapped in
		# spaCy Docs below. Only the vocabulary is needed for that - the text
		# has already been tokenised, so no language model is loaded or
		# downloaded, and the multi-language pipeline avoids having to map
		# 4CAT's language names onto spaCy's language codes. The token
		# attributes textacy reads (is_punct, is_space) are character-class
		# checks that do not depend on the language.
		vocabulary = spacy.blank("xx").vocab

		self.dataset.update_status("Processing token sets")
		results = []

		index = 0
		for packed_tokens in self.source_dataset.iterate_items(self):
			if self.interrupted:
				raise ProcessorInterruptedException("Interrupted while calculating vocabulary statistics")

			if packed_tokens.file.name == ".token_metadata.json":
				# Skip metadata
				continue

			index += 1
			token_set_name = packed_tokens.file.stem  # we don't need the full path
			self.dataset.update_status("Processing token set %i (%s)" % (index, token_set_name))
			self.dataset.update_progress(index / self.source_dataset.num_rows)

			# we support both pickle and json dumps of tokens
			token_unpacker = pickle if packed_tokens.file.suffix == ".pb" else json

			with packed_tokens.file.open("rb") as binary_tokens:
				# these were saved as pickle dumps so we need the binary mode
				documents = token_unpacker.load(binary_tokens)

			statistics = self.get_statistics(documents, vocabulary, segment_size, round_to)
			if statistics:
				results.append({"date": token_set_name, **statistics})

		if not results:
			self.dataset.finish_as_empty("No tokens found to calculate statistics for")
			return

		# token sets are named after the interval they cover, so sorting by name
		# puts them in chronological order
		results = natsorted(results, key=lambda row: row["date"])

		self.write_csv_items_and_finish(results, warning=warning)

	def get_statistics(self, documents, vocabulary, segment_size, round_to):
		"""
		Calculate all vocabulary statistics for a single token set

		:param list documents:  List of documents, each a list of tokens
		:param Vocab vocabulary:  spaCy vocabulary to construct the documents with
		:param int segment_size:  Window size for the moving-average type-token ratio
		:param int round_to:  Decimal places to round ratios to
		:return dict:  Calculated statistics, or None if the set has no tokens
		"""
		# spaCy rejects empty strings as tokens, and empty documents have
		# nothing to contribute to any of the measures
		docs = []
		lengths = []
		for document in documents:
			words = [token for token in document if token and not token.isspace()]
			if words:
				docs.append(Doc(vocabulary, words=words))
				lengths.append(len(words))

		if not docs:
			return None

		# textacy's helpers consume the tokens they are given, so every measure
		# needs a freshly started iterator rather than a shared one
		def tokens():
			return itertools.chain.from_iterable(docs)

		total_tokens = self.measure(basics.n_words, tokens())
		if not total_tokens:
			return None

		# count the same words textacy counts, so that the measures calculated
		# here line up with the ones it reports
		frequencies = Counter(word.text for word in textacy_utils.get_words(tokens()))

		# how much of the text is accounted for by its ten most common types -
		# a high value points at repetitive or templated content, e.g. spam
		top_ten = sum(frequency for _, frequency in frequencies.most_common(10))

		# Simpson's D is the chance that two tokens drawn at random are of the
		# same type, so lower means more diverse
		if total_tokens > 1:
			simpson = sum(frequency * (frequency - 1) for frequency in frequencies.values()) / \
					  (total_tokens * (total_tokens - 1))
		else:
			simpson = None

		hapax = sum(1 for frequency in frequencies.values() if frequency == 1)
		dis = sum(1 for frequency in frequencies.values() if frequency == 2)

		lengths.sort()
		midpoint = len(lengths) // 2
		median_length = lengths[midpoint] if len(lengths) % 2 else (lengths[midpoint - 1] + lengths[midpoint]) / 2

		return {
			"documents": len(docs),
			"tokens": total_tokens,
			"types": self.measure(basics.n_unique_words, tokens()),
			"characters": self.measure(basics.n_chars, tokens()),
			"mean_tokens_per_document": self.rounded(total_tokens / len(docs), round_to),
			"median_tokens_per_document": self.rounded(median_length, round_to),
			"ttr": self.rounded(self.measure(diversity.ttr, tokens()), round_to),
			"root_ttr": self.rounded(self.measure(diversity.ttr, tokens(), variant="root"), round_to),
			"log_ttr": self.rounded(self.measure(diversity.log_ttr, tokens()), round_to),
			"mattr": self.rounded(
				self.measure(diversity.segmented_ttr, tokens(), segment_size=segment_size, variant="moving-avg"),
				round_to),
			"mtld": self.rounded(self.measure(diversity.mtld, tokens()), round_to),
			"hdd": self.rounded(self.measure(diversity.hdd, tokens()), round_to),
			"shannon_entropy": self.rounded(self.measure(basics.entropy, tokens()), round_to),
			"hapax_legomena": hapax,
			"hapax_percentage": self.rounded(100 * hapax / len(frequencies), round_to) if frequencies else "",
			"dis_legomena": dis,
			"simpsons_d": self.rounded(simpson, round_to),
			"top_10_types_coverage": self.rounded(100 * top_ten / total_tokens, round_to)
		}

	@staticmethod
	def measure(function, *args, **kwargs):
		"""
		Run a textacy measure, treating anything it cannot calculate as missing

		Most of these measures are undefined for very short or entirely
		repetition-free token sets, which they signal by raising or by returning
		a non-finite number. A token set that is too small for one measure is
		usually still fine for the others, so a failure is recorded as a missing
		value rather than failing the whole dataset.

		:param callable function:  textacy measure to call
		:return float|int|None:  Result, or None if it could not be calculated
		"""
		try:
			value = function(*args, **kwargs)
		except (ArithmeticError, ValueError, IndexError, TypeError):
			return None

		if value is None or not math.isfinite(value):
			return None

		return value

	@staticmethod
	def rounded(value, round_to):
		"""
		Round a value, passing through the empty string for undefined measures

		:param float|None value:  Value to round
		:param int round_to:  Decimal places
		:return float|str:  Rounded value, or an empty string if undefined
		"""
		if value is None:
			return ""

		# adding zero normalises the -0.0 that e.g. entropy produces for a
		# single-type token set, which would otherwise be written as '-0.0'
		return round(float(value), round_to) + 0.0
