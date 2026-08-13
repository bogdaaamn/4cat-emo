"""
Score the sentiment of items with VADER
"""
import csv

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from backend.lib.processor import BasicProcessor
from common.lib.compatibility import Compatibility
from common.lib.exceptions import ProcessorInterruptedException
from common.lib.helpers import UserInput, pad_interval, get_interval_descriptor

__author__ = "Bogdan Covrig"
__credits__ = ["Bogdan Covrig"]
__maintainer__ = "Bogdan Covrig"
__email__ = "b.covrig@maastrichtuniversity.nl"


class VaderSentiment(BasicProcessor):
	"""
	Score item sentiment with VADER
	"""

	type = "vader-sentiment"  # job type ID
	category = "Text analysis"  # category
	title = "Sentiment analysis (VADER)"  # title displayed in UI
	description = ("Scores the sentiment of each item with VADER, a rule-based model made for social media text: it "
				   "takes emoji, emoticons, capitalisation, punctuation, degree modifiers ('very') and negation "
				   "('not good') into account. Outputs a score per item, or the number of items per sentiment class "
				   "per timeframe. English only.")
	extension = "csv"  # extension of result file, used internally and in UI

	# both output shapes are csv; the frequency table is the one that can be
	# graphed, and 4CAT only offers those follow-ups when the file fits them
	compatibility = Compatibility(extensions={"csv", "ndjson"},
								  preferred_followups=["histogram", "render-rankflow", "descriptive_statistics"])

	references = [
		"[VADER: Valence Aware Dictionary and sEntiment Reasoner](https://github.com/cjhutto/vaderSentiment)",
		"[Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social "
		"Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14).]"
		"(https://ojs.aaai.org/index.php/ICWSM/article/view/14550)"
	]

	# VADER's own recommendation for classifying compound scores
	default_threshold = 0.05

	@classmethod
	def get_options(cls, parent_dataset=None, config=None):
		"""
		Get processor options

		:param parent_dataset DataSet:  An object representing the dataset that
			the processor would be or was run on.
		:param config ConfigManager|None config:  Configuration reader (context-aware)
		:return dict:  Options for this processor
		"""
		default_threshold = cls.default_threshold
		options = {
			"column": {
				"type": UserInput.OPTION_TEXT,
				"default": "body",
				"help": "Column with item text",
			},
			"language-notice": {
				"type": UserInput.OPTION_INFO,
				"help": "VADER's lexicon is English; scores for text in other languages are not meaningful. Give it "
						"the text as it was posted: it reads emoji, capitalisation and punctuation as sentiment "
						"signals, so cleaned or tokenised text scores differently.",
			},
			"output": {
				"type": UserInput.OPTION_CHOICE,
				"default": "item",
				"options": {
					"item": "Sentiment score per item",
					"overtime": "Number of items per sentiment class",
				},
				"help": "Output",
				"tooltip": "Per item gives the four VADER scores and a class for every item. The counts per class can "
						   "be graphed directly, e.g. as a histogram.",
			},
			"threshold": {
				"type": UserInput.OPTION_TEXT,
				"default": str(default_threshold),
				"help": "Neutral threshold",
				"tooltip": f"Items with a compound score above this value count as positive and below minus this "
						   f"value as negative; anything in between is neutral. {default_threshold} is the value "
						   f"VADER's authors recommend.",
			},
			"timeframe": {
				"type": UserInput.OPTION_CHOICE,
				"default": "all",
				"options": {
					"all": "Overall",
					"year": "Year",
					"month": "Month",
					"week": "Week",
					"day": "Day",
					"hour": "Hour",
				},
				"help": "Produce counts per",
				"requires": "output==overtime",
			},
			"pad": {
				"type": UserInput.OPTION_TOGGLE,
				"default": True,
				"help": "Include dates with zero items",
				"tooltip": "Makes the counts continuous. Only has effect when counting per timeframe.",
				"requires": "output==overtime",
			},
			"save_annotations": {
				"type": UserInput.OPTION_ANNOTATION,
				"label": "sentiment",
				"tooltip": "Adds the sentiment class and compound score as annotations to the source dataset.",
				"default": False,
				"to_parent": True,
			},
		}

		# Get the columns for the select columns option
		if parent_dataset and parent_dataset.get_columns():
			columns = parent_dataset.get_columns()
			options["column"]["type"] = UserInput.OPTION_CHOICE
			options["column"]["options"] = {v: v for v in columns}
			options["column"]["default"] = "body" if "body" in columns else columns[0]

		return options

	@classmethod
	def parse_threshold(cls, raw):
		"""
		Turn the user-provided neutral threshold into a usable compound bound

		A malformed or out-of-range value falls back to VADER's own
		recommendation rather than erroring, so a typo does not lose a run.

		:param raw:  Threshold as entered by the user
		:return float:  Threshold, between 0 and 1
		"""
		try:
			threshold = abs(float(raw))
		except (TypeError, ValueError):
			return cls.default_threshold

		return threshold if threshold < 1 else cls.default_threshold

	def process(self):
		"""
		Score each item's text with VADER, and write the scores per item or the
		number of items per sentiment class per timeframe.
		"""
		column = self.parameters.get("column")
		if not column:
			return self.dataset.finish_with_error("No column selected")
		column = column[0] if isinstance(column, list) else column

		output = self.parameters.get("output", "item")
		threshold = self.parse_threshold(self.parameters.get("threshold"))
		timeframe = self.parameters.get("timeframe", "all") if output == "overtime" else "all"
		save_annotations = self.parameters.get("save_annotations", False)

		# annotations are keyed by item ID, so without one they cannot be saved
		if save_annotations and "id" not in (self.source_dataset.get_columns() or []):
			save_annotations = False
			self.dataset.update_status("Cannot save annotations: the source dataset has no id column")

		analyzer = SentimentIntensityAnalyzer()

		if output == "overtime":
			return self.count_per_timeframe(analyzer, column, threshold, timeframe, save_annotations)

		return self.score_per_item(analyzer, column, threshold, save_annotations)

	def classify(self, compound, threshold):
		"""
		Turn a compound score into a sentiment class

		:param float compound:  VADER compound score, between -1 and 1
		:param float threshold:  Scores within ±threshold count as neutral
		:return str:  'positive', 'negative' or 'neutral'
		"""
		if compound > threshold:
			return "positive"
		if compound < -threshold:
			return "negative"

		return "neutral"

	def annotations_for(self, item, sentiment, compound):
		"""
		Build the annotations for a scored item

		:param item:  The source item
		:param str sentiment:  Sentiment class
		:param float compound:  VADER compound score
		:return list:  Annotations, empty if the item cannot be annotated
		"""
		if not item.get("id"):
			return []

		return [
			{"item_id": item["id"], "label": "sentiment", "value": sentiment},
			{"item_id": item["id"], "label": "sentiment compound", "value": compound},
		]

	def score_per_item(self, analyzer, column, threshold, save_annotations):
		"""
		Write the VADER scores and sentiment class for every item

		Items without text are written with empty scores, so the output keeps
		one row per source item.

		:param SentimentIntensityAnalyzer analyzer:  VADER analyzer
		:param str column:  Column with the text to score
		:param float threshold:  Scores within ±threshold count as neutral
		:param bool save_annotations:  Whether to annotate the source dataset
		"""
		# the source's own date column is kept so the scores stay usable as a
		# time series; anything else can be joined back on the item ID
		source_columns = self.source_dataset.get_columns() or []
		date_columns = [c for c in ("timestamp", "date") if c in source_columns]
		score_columns = ["sentiment", "compound", "positive", "neutral", "negative"]

		# a source column named e.g. 'sentiment' would otherwise be overwritten
		# by the score of the same name
		text_column = column
		if column in ("id", *date_columns, *score_columns):
			text_column = f"{column}_text"
		fieldnames = ["id", *date_columns, text_column, *score_columns]

		self.dataset.update_status("Scoring items")
		processed = 0
		scored = 0
		annotations = []

		with self.dataset.get_results_path().open("w", encoding="utf-8", newline="") as outfile:
			writer = csv.DictWriter(outfile, fieldnames=fieldnames)
			writer.writeheader()

			for item in self.source_dataset.iterate_items(self):
				if self.interrupted:
					raise ProcessorInterruptedException("Interrupted while scoring item sentiment")

				body = str(item.get(column) or "")
				row = {
					"id": item.get("id", processed + 1),
					**{date_column: item.get(date_column) for date_column in date_columns},
					text_column: body,
					"sentiment": "",
					"compound": "",
					"positive": "",
					"neutral": "",
					"negative": "",
				}

				if body.strip():
					scores = analyzer.polarity_scores(body)
					sentiment = self.classify(scores["compound"], threshold)
					row.update({
						"sentiment": sentiment,
						"compound": scores["compound"],
						"positive": scores["pos"],
						"neutral": scores["neu"],
						"negative": scores["neg"],
					})
					scored += 1

					if save_annotations:
						annotations.extend(self.annotations_for(item, sentiment, scores["compound"]))

				writer.writerow(row)
				processed += 1

				if processed % 2500 == 0:
					if annotations:
						self.save_annotations(annotations, source_dataset=self.source_dataset)
						annotations = []
					self.dataset.update_status(f"Scored {processed:,} of {self.source_dataset.num_rows:,} items.")
					self.dataset.update_progress(processed / self.source_dataset.num_rows)

		if annotations:
			self.save_annotations(annotations, source_dataset=self.source_dataset)

		if not processed:
			return self.dataset.finish_as_empty("No items to score")

		self.dataset.update_status(f"Finished, scored {scored:,} of {processed:,} items", is_final=True)
		self.dataset.finish(processed)

	def count_per_timeframe(self, analyzer, column, threshold, timeframe, save_annotations):
		"""
		Write the number of items per sentiment class as a frequency table

		:param SentimentIntensityAnalyzer analyzer:  VADER analyzer
		:param str column:  Column with the text to score
		:param float threshold:  Scores within ±threshold count as neutral
		:param str timeframe:  Interval to count per, or 'all'
		:param bool save_annotations:  Whether to annotate the source dataset
		"""
		# the classes are mutually exclusive and are emitted in this order
		categories = ["positive", "neutral", "negative", "no text"]

		intervals = {}
		compounds = {}
		first_interval = "9999"
		last_interval = "0000"

		self.dataset.update_status("Scoring items")
		processed = 0
		annotations = []

		for item in self.source_dataset.iterate_items(self):
			if self.interrupted:
				raise ProcessorInterruptedException("Interrupted while scoring item sentiment")

			body = str(item.get(column) or "")
			compound = None

			if body.strip():
				compound = analyzer.polarity_scores(body)["compound"]
				category = self.classify(compound, threshold)

				if save_annotations:
					annotations.extend(self.annotations_for(item, category, compound))
			else:
				category = "no text"

			if timeframe == "all":
				date = "overall"
			else:
				try:
					date = get_interval_descriptor(item, timeframe)
				except ValueError as e:
					return self.dataset.finish_with_error(f"{e}, cannot count items per {timeframe}")

				first_interval = min(first_interval, date)
				last_interval = max(last_interval, date)

			if date not in intervals:
				intervals[date] = {}
				compounds[date] = []
			intervals[date][category] = intervals[date].get(category, 0) + 1
			if compound is not None:
				compounds[date].append(compound)

			processed += 1
			if processed % 2500 == 0:
				if annotations:
					self.save_annotations(annotations, source_dataset=self.source_dataset)
					annotations = []
				self.dataset.update_status(f"Scored {processed:,} of {self.source_dataset.num_rows:,} items.")
				self.dataset.update_progress(processed / self.source_dataset.num_rows)

		if annotations:
			self.save_annotations(annotations, source_dataset=self.source_dataset)

		if not processed:
			return self.dataset.finish_as_empty("No items to score")

		# pad the intervals so the result can be graphed as a continuous series
		if self.parameters.get("pad") and timeframe != "all":
			missing, intervals = pad_interval(intervals, first_interval, last_interval)
			for date, counts in intervals.items():
				if isinstance(counts, int):
					intervals[date] = {}

		# items without text are noise in most datasets, so only report them
		# when they occur
		if not any(counts.get("no text") for counts in intervals.values()):
			categories.remove("no text")

		# 'total', 'percentage' and 'mean_compound' are extra context columns;
		# the visualisations read date/item/value by key and ignore the rest, so
		# adding them here does not affect charting (and a mean as an extra
		# *item* would, since counts and a -1 to 1 score do not share a scale)
		rows = []
		for date in sorted(intervals):
			# the classes are exhaustive, so their counts add up to the number
			# of items in the interval
			total = sum(intervals[date].values())
			scores = compounds.get(date, [])
			mean_compound = round(sum(scores) / len(scores), 4) if scores else ""
			for category in categories:
				value = intervals[date].get(category, 0)
				rows.append({
					"date": date,
					"item": category,
					"value": value,
					"total": total,
					"percentage": round(value / total * 100, 2) if total else 0,
					"mean_compound": mean_compound,
				})

		self.write_csv_items_and_finish(rows)
