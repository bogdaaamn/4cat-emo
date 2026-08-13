"""
Count emoji-only versus textual items, and group the textual ones by length
"""
import bisect
import re

import emoji

from backend.lib.processor import BasicProcessor
from common.lib.compatibility import Compatibility
from common.lib.helpers import UserInput, pad_interval, get_interval_descriptor

__author__ = "Bogdan Covrig"
__credits__ = ["Bogdan Covrig"]
__maintainer__ = "Bogdan Covrig"
__email__ = "b.covrig@maastrichtuniversity.nl"

# Zero-width joiners, variation selectors and skin tone modifiers can survive
# emoji stripping in some sequences; they carry no text of their own, so they
# should not make an item count as textual. Shaped as a str.translate() table so
# stripping them stays a single C-level pass per item.
INVISIBLE = {codepoint: None for codepoint in (
	*range(0x200b, 0x2010),  # zero width space through right-to-left mark
	0x2060,  # word joiner
	*range(0xfe00, 0xfe10),  # variation selectors
	0xfeff,  # zero width no-break space
	*range(0x1f3fb, 0x1f400),  # skin tone modifiers
)}


class CountEmojiText(BasicProcessor):
	"""
	Count emoji-only and textual items in a dataset
	"""

	type = "count-emoji-text"  # job type ID
	category = "Metrics"  # category
	title = "Count emoji-only vs. text items"  # title displayed in UI
	description = ("Counts how many items consist of emoji only and how many contain text, and divides the textual "
				   "items into groups by length. Useful to gauge how much of a dataset is non-verbal.")
	extension = "csv"  # extension of result file, used internally and in UI

	# Frequency table output, so the usual over-time visualisations apply
	compatibility = Compatibility(extensions={"csv", "ndjson"},
								  preferred_followups=["histogram", "render-rankflow"])

	@classmethod
	def get_options(cls, parent_dataset=None, config=None):
		"""
		Get processor options

		:param parent_dataset DataSet:  An object representing the dataset that
			the processor would be or was run on.
		:param config ConfigManager|None config:  Configuration reader (context-aware)
		:return dict:  Options for this processor
		"""
		options = {
			"column": {
				"type": UserInput.OPTION_TEXT,
				"default": "body",
				"help": "Column with item text",
			},
			"breakdown": {
				"type": UserInput.OPTION_CHOICE,
				"default": "length",
				"options": {
					"length": "Emoji-only, and text split into length groups",
					"simple": "Emoji-only versus text only",
				},
				"help": "Report",
				"tooltip": "Categories never overlap, so the counts always add up to the total number of items.",
			},
			"unit": {
				"type": UserInput.OPTION_CHOICE,
				"default": "characters",
				"options": {"characters": "Characters", "words": "Words"},
				"help": "Measure length in",
			},
			"buckets": {
				"type": UserInput.OPTION_TEXT,
				"default": "10, 25, 50, 100, 250, 500",
				"help": "Length group boundaries",
				"tooltip": "Comma-separated upper bounds. '10, 25' yields groups of 1-10, 11-25 and 26+. Lower these "
						   "when measuring length in words.",
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
			},
			"pad": {
				"type": UserInput.OPTION_TOGGLE,
				"default": True,
				"help": "Include dates with zero items",
				"tooltip": "Makes the counts continuous. Only has effect when counting per timeframe.",
			},
		}

		# Get the columns for the select columns option
		if parent_dataset and parent_dataset.get_columns():
			columns = parent_dataset.get_columns()
			options["column"]["type"] = UserInput.OPTION_CHOICE
			options["column"]["options"] = {v: v for v in columns}
			options["column"]["default"] = "body" if "body" in columns else columns[0]

		return options

	@staticmethod
	def parse_buckets(raw):
		"""
		Turn user-provided length group boundaries into a sorted list of bounds

		Anything that is not a positive number is ignored, so a malformed value
		degrades into fewer groups rather than an error.

		:param str raw:  Comma-separated upper bounds
		:return list:  Sorted, de-duplicated upper bounds
		"""
		bounds = set()
		for part in re.split(r"[^0-9.]+", str(raw)):
			if not part:
				continue
			try:
				bound = int(float(part))
			except ValueError:
				continue
			if bound > 0:
				bounds.add(bound)

		return sorted(bounds)

	@staticmethod
	def bucket_labels(bounds, unit):
		"""
		Build the ordered category labels for the length groups

		:param list bounds:  Sorted upper bounds
		:param str unit:  'characters' or 'words'
		:return list:  Labels, one per group, in ascending order
		"""
		labels = []
		previous = 0
		for bound in bounds:
			labels.append(f"text ({previous + 1}-{bound} {unit})")
			previous = bound

		labels.append(f"text ({previous + 1}+ {unit})")
		return labels

	def process(self):
		"""
		Classify each item as emoji-only, textual or empty, and write the
		results as a frequency table.
		"""
		column = self.parameters.get("column")
		if not column:
			return self.dataset.finish_with_error("No column selected")
		column = column[0] if isinstance(column, list) else column

		timeframe = self.parameters.get("timeframe", "all")
		unit = self.parameters.get("unit", "characters")
		breakdown = self.parameters.get("breakdown", "length")

		bounds = self.parse_buckets(self.parameters.get("buckets", ""))
		if breakdown == "length":
			text_categories = self.bucket_labels(bounds, unit)
		else:
			text_categories = ["text"]

		# categories are mutually exclusive and are emitted in this order
		categories = ["emoji only", *text_categories, "no text"]

		intervals = {}
		first_interval = "9999"
		last_interval = "0000"

		self.dataset.update_status("Processing items")
		processed = 0

		for item in self.source_dataset.iterate_items(self):
			body = str(item.get(column) or "")

			# an item is emoji-only when nothing but emoji (and whitespace) is
			# left once the emoji are removed
			residue = emoji.replace_emoji(body, replace="").translate(INVISIBLE).strip()
			if not residue:
				category = "emoji only" if emoji.emoji_count(body) else "no text"
			elif breakdown == "simple":
				category = "text"
			else:
				length = len(body.split()) if unit == "words" else len(body.strip())
				category = text_categories[bisect.bisect_left(bounds, length)]

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
			intervals[date][category] = intervals[date].get(category, 0) + 1

			processed += 1
			if processed % 2500 == 0:
				self.dataset.update_status(f"Counted {processed:,} of {self.source_dataset.num_rows:,} items.")
				self.dataset.update_progress(processed / self.source_dataset.num_rows)

		if not processed:
			return self.dataset.finish_as_empty("No items to count")

		# pad the intervals so the result can be graphed as a continuous series
		if self.parameters.get("pad") and timeframe != "all":
			missing, intervals = pad_interval(intervals, first_interval, last_interval)
			for date, counts in intervals.items():
				if isinstance(counts, int):
					intervals[date] = {}

		# 'no text' is noise in most datasets, so only report it when it occurs
		if not any(counts.get("no text") for counts in intervals.values()):
			categories.remove("no text")

		# 'total' and 'percentage' are extra context columns; the visualisations
		# read date/item/value by key and ignore the rest, so adding them here
		# does not affect charting (and a total as an extra *item* would, since
		# it would double the height of a stacked graph)
		rows = []
		for date in sorted(intervals):
			# the categories are exhaustive, so their counts add up to the
			# number of items in the interval
			total = sum(intervals[date].values())
			for category in categories:
				value = intervals[date].get(category, 0)
				rows.append({
					"date": date,
					"item": category,
					"value": value,
					"total": total,
					"percentage": round(value / total * 100, 2) if total else 0,
				})

		self.write_csv_items_and_finish(rows)
