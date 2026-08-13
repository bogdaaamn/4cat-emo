"""
Visualises topic models as an interactive HTML page
"""

from common.lib.helpers import UserInput
from backend.lib.processor import BasicProcessor
from common.lib.compatibility import Compatibility
from common.lib.exceptions import ProcessorInterruptedException

import json
import pickle
import html as html_lib
import numpy as np

__author__ = "Bogdan Covrig"
__credits__ = ["Bogdan Covrig"]
__maintainer__ = "Bogdan Covrig"
__email__ = "b.covrig@maastrichtuniversity.nl"


class TopicModelVisualiser(BasicProcessor):
    """
    Visualises topic models as an interactive HTML page
    """
    type = "topic-model-visualiser"  # job type ID
    category = "Text analysis"  # category
    title = "Visualise topic models"  # title displayed in UI
    description = "Creates an interactive HTML page showing the top words per topic, representative " \
                  "documents per topic, a topic similarity heatmap, and (if there is more than one " \
                  "token set) how topic prevalence shifts over time."
    extension = "html"  # extension of result file, used internally and in UI

    # Allow processor on topic models
    compatibility = Compatibility(types={"topic-modeller"})

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
            "topic_size": {
                "type": UserInput.OPTION_TEXT,
                "min": 1,
                "max": 50,
                "default": 10,
                "help": "Words per topic",
                "tooltip": "This many of the most relevant words will be shown per topic"
            },
            "num_documents": {
                "type": UserInput.OPTION_TEXT,
                "min": 0,
                "max": 10,
                "default": 3,
                "help": "Representative documents per topic",
                "tooltip": "This many documents most strongly associated with each topic will be shown, " \
                           "represented by their top terms (since original text is not stored in the " \
                           "topic model). Set to 0 to disable."
            }
        }

    def process(self):
        """
        Extracts topics, top words, representative documents, and topic
        similarity per model, plus topic prevalence per token set (based on
        document-topic predictions), then renders it all as a self-contained
        interactive HTML page.
        """
        self.dataset.update_status("Unpacking topic models")
        staging_area = self.unpack_archive_contents(self.source_file)
        topic_size = self.parameters.get("topic_size")
        num_documents = self.parameters.get("num_documents")

        # load predictions (per-document topic weights) per token set, if available
        # this file is written by the topic-modeller processor
        predictions_by_label = {}
        metadata_file = staging_area.joinpath(".model_metadata.json")
        if metadata_file.exists():
            with metadata_file.open(encoding="utf-8") as infile:
                model_metadata = json.load(infile)

            for key, entry in model_metadata.items():
                if key == "parameters" or not isinstance(entry, dict) or "predictions" not in entry:
                    continue

                model_label = entry.get("model_file", key)
                if model_label.endswith(".model"):
                    model_label = model_label[:-len(".model")]

                predictions_by_label[model_label] = entry["predictions"]

        model_files = sorted(staging_area.glob("*.model"))
        num_models = len(model_files) or 1

        # { model_label: { topic_number: [ {word, weight}, ... ] } }
        words_data = {}
        # { model_label: { topic_number: avg_weight } }
        prevalence_data = {}
        # { model_label: { "topics": [1, 2, ...], "matrix": [[...], ...] } }
        similarity_data = {}
        # { model_label: { topic_number: [ {doc_index, terms: [...]}, ... ] } }
        representative_docs = {}

        processed = 0
        for model_file in model_files:
            if self.interrupted:
                raise ProcessorInterruptedException("Interrupted while extracting topic model tokens")

            self.dataset.update_status("Extracting topics from model '%s'" % model_file.stem)
            self.dataset.update_progress(processed / num_models)
            processed += 1

            with model_file.open("rb") as infile:
                model = pickle.load(infile)

            with model_file.with_suffix(".features").open("rb") as infile:
                features = pickle.load(infile)

            vectors = None
            vectors_file = model_file.with_suffix(".vectors")
            if vectors_file.exists():
                with vectors_file.open("rb") as infile:
                    vectors = pickle.load(infile)

            model_label = model_file.stem
            words_data[model_label] = {}

            topic_index = 0
            for topic in model.components_:
                topic_index += 1
                weighted_features = {features[i]: float(weight) for i, weight in enumerate(topic)}
                top_words = sorted(weighted_features, key=lambda k: weighted_features[k], reverse=True)[:topic_size]

                words_data[model_label][topic_index] = [
                    {"word": word, "weight": weighted_features[word]} for word in top_words
                ]

            # topic similarity: cosine similarity between topics' word-weight vectors
            self.dataset.update_status("Calculating topic similarity for '%s'" % model_file.stem)
            components = np.asarray(model.components_, dtype=float)
            norms = np.linalg.norm(components, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalised = components / norms
            similarity_matrix = normalised @ normalised.T

            similarity_data[model_label] = {
                "topics": list(range(1, len(model.components_) + 1)),
                "matrix": similarity_matrix.round(3).tolist()
            }

            # topic prevalence and representative documents, based on per-document predictions
            predictions = predictions_by_label.get(model_label, {})
            if predictions:
                self.dataset.update_status("Calculating topic prevalence and representative documents for '%s'" % model_file.stem)

                topic_totals = {}
                num_docs = 0
                # doc_index (str) -> { topic_index_0based (str): weight }
                doc_topic_weights = {}

                for doc_index_str, doc_predictions in predictions.items():
                    num_docs += 1
                    doc_total = sum(doc_predictions.values()) or 1
                    for topic_index_str, weight in doc_predictions.items():
                        normalised_weight = weight / doc_total
                        topic_number = str(int(topic_index_str) + 1)
                        topic_totals[topic_number] = topic_totals.get(topic_number, 0) + normalised_weight
                        doc_topic_weights.setdefault(topic_number, []).append((doc_index_str, normalised_weight))

                if num_docs:
                    prevalence_data[model_label] = {
                        topic_number: total / num_docs for topic_number, total in topic_totals.items()
                    }

                if num_documents and vectors is not None:
                    representative_docs[model_label] = {}
                    for topic_number, doc_weights in doc_topic_weights.items():
                        top_docs = sorted(doc_weights, key=lambda pair: pair[1], reverse=True)[:num_documents]
                        doc_entries = []

                        for doc_index_str, weight in top_docs:
                            try:
                                doc_index = int(doc_index_str)
                                row = vectors[doc_index]
                                term_weights = {features[i]: float(v) for i, v in zip(row.indices, row.data)}
                                top_terms = sorted(term_weights, key=lambda k: term_weights[k], reverse=True)[:12]
                            except (IndexError, ValueError, AttributeError):
                                continue

                            doc_entries.append({
                                "doc_index": doc_index_str,
                                "weight": weight,
                                "terms": top_terms
                            })

                        if doc_entries:
                            representative_docs[model_label][topic_number] = doc_entries

        if not predictions_by_label:
            self.dataset.update_status("Note: no .model_metadata.json / predictions found in the archive - "
                                        "topic prevalence over time will be unavailable")
        elif len(prevalence_data) <= 1:
            self.dataset.update_status("Note: only %i token set(s) produced prevalence data - "
                                        "topic prevalence over time needs at least 2 to show a trend" % len(prevalence_data))

        self.dataset.update_status("Rendering visualisation")
        html_output = self.render_html(words_data, prevalence_data, similarity_data, representative_docs)

        with self.dataset.get_results_path().open("w", encoding="utf-8") as outfile:
            outfile.write(html_output)

        num_topics = sum(len(topics) for topics in words_data.values())
        self.dataset.finish(num_topics)

    def render_html(self, words_data, prevalence_data, similarity_data, representative_docs):
        """
        Build a self-contained HTML page (inline CSS/JS, no external
        dependencies) with tabs for: top words per topic (with
        representative documents), topic similarity, and (if available)
        topic prevalence over time.

        :param dict words_data:  { model_label: { topic_number: [ {word, weight}, ... ] } }
        :param dict prevalence_data:  { model_label: { topic_number: avg_weight } }
        :param dict similarity_data:  { model_label: { "topics": [...], "matrix": [[...]] } }
        :param dict representative_docs:  { model_label: { topic_number: [ {doc_index, weight, terms}, ... ] } }
        :return str:  Full HTML document
        """
        model_labels = list(words_data.keys())
        words_json = json.dumps(words_data)
        prevalence_json = json.dumps(prevalence_data)
        similarity_json = json.dumps(similarity_data)
        representative_json = json.dumps(representative_docs)
        options_html = "\n".join(
            '<option value="{0}">{1}</option>'.format(html_lib.escape(label), html_lib.escape(label))
            for label in model_labels
        )

        template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Topic model visualisation</title>
<style>
  :root {
    --accent: #3a6ea5;
    --accent-rgb: 58, 110, 165;
    --bg: #fafafa;
    --card-bg: #ffffff;
    --border: #e0e0e0;
    --text: #222;
    --muted: #777;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
  }
  h1 { font-size: 1.4em; margin-bottom: 4px; }
  .subtitle { color: var(--muted); margin-bottom: 20px; font-size: 0.9em; }
  .tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .tab-button {
    background: none;
    border: none;
    padding: 8px 14px;
    font-size: 0.95em;
    cursor: pointer;
    color: var(--muted);
    border-bottom: 2px solid transparent;
  }
  .tab-button.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    font-weight: 600;
  }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
  }
  select {
    padding: 6px 10px;
    font-size: 0.95em;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: white;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px;
  }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .card h2 {
    font-size: 1em;
    margin: 0 0 10px 0;
    color: var(--accent);
  }
  .bar-row {
    display: grid;
    grid-template-columns: 90px 1fr 60px;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    font-size: 0.85em;
  }
  .bar-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar-track {
    background: #eef1f5;
    border-radius: 4px;
    height: 14px;
    overflow: hidden;
  }
  .bar-fill {
    background: var(--accent);
    height: 100%;
    border-radius: 4px;
  }
  .bar-weight {
    text-align: right;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .empty {
    color: var(--muted);
    font-style: italic;
  }
  .doc-section {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px dashed var(--border);
  }
  .doc-section h3 {
    font-size: 0.8em;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--muted);
    margin: 0 0 8px 0;
  }
  .doc-item {
    font-size: 0.82em;
    margin-bottom: 6px;
    line-height: 1.4;
  }
  .doc-item .doc-id {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .prevalence-chart {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    height: 360px;
    padding: 10px 10px 0 10px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow-x: auto;
  }
  .prevalence-column {
    display: flex;
    flex-direction: column-reverse;
    width: 36px;
    min-width: 36px;
    height: 100%;
    cursor: pointer;
    border-radius: 4px 4px 0 0;
    overflow: hidden;
  }
  .prevalence-segment {
    width: 100%;
  }
  .prevalence-labels {
    display: flex;
    gap: 6px;
    padding: 6px 10px 0 10px;
    overflow-x: auto;
  }
  .prevalence-label {
    width: 36px;
    min-width: 36px;
    font-size: 0.7em;
    color: var(--muted);
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    max-height: 90px;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px 16px;
    margin-top: 16px;
    font-size: 0.85em;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    display: inline-block;
  }
  .hint {
    color: var(--muted);
    font-size: 0.85em;
    margin-top: 10px;
  }
  .heatmap-wrap {
    overflow-x: auto;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
  }
  .heatmap-table {
    border-collapse: collapse;
  }
  .heatmap-table th, .heatmap-table td {
    width: 28px;
    height: 28px;
    text-align: center;
    font-size: 0.65em;
    padding: 0;
  }
  .heatmap-table th {
    color: var(--muted);
    font-weight: 400;
  }
  .heatmap-cell {
    border-radius: 3px;
    color: white;
    font-variant-numeric: tabular-nums;
  }
</style>
</head>
<body>

<h1>Topic model visualisation</h1>
<div class="subtitle">Explore the words and documents behind each topic, how similar topics are to each other, and (if applicable) how they shift over time</div>

<div class="tabs">
  <button class="tab-button active" data-tab="words" onclick="switchTab('words')">Top words per topic</button>
  <button class="tab-button" data-tab="similarity" onclick="switchTab('similarity')">Topic similarity</button>
  <button class="tab-button" data-tab="prevalence" onclick="switchTab('prevalence')">Topic prevalence over time</button>
</div>

<div id="tab-words" class="tab-panel active">
  <div class="controls">
    <label for="model-select"><strong>Token set:</strong></label>
    <select id="model-select" onchange="onModelChange(this.value)">
      __OPTIONS_HTML__
    </select>
  </div>
  <div id="grid" class="grid"></div>
</div>

<div id="tab-similarity" class="tab-panel">
  <div class="subtitle">How similar each pair of topics is, based on cosine similarity of their word-weight vectors. Darker = more similar.</div>
  <div class="heatmap-wrap">
    <div id="heatmap"></div>
  </div>
</div>

<div id="tab-prevalence" class="tab-panel">
  <div class="subtitle">Each column is one token set; segment height shows that topic's average share of the documents in it. Click a column to inspect its top words.</div>
  <div id="prevalence-chart" class="prevalence-chart"></div>
  <div id="prevalence-labels" class="prevalence-labels"></div>
  <div id="legend" class="legend"></div>
</div>

<script>
  const WORDS_DATA = __WORDS_JSON__;
  const PREVALENCE_DATA = __PREVALENCE_JSON__;
  const SIMILARITY_DATA = __SIMILARITY_JSON__;
  const REPRESENTATIVE_DOCS = __REPRESENTATIVE_JSON__;

  const PALETTE = [
    '#3a6ea5', '#e07a5f', '#81b29a', '#f2cc8f', '#9b5de5',
    '#00b4d8', '#f15bb5', '#606c38', '#bc6c25', '#4361ee',
    '#ef476f', '#06d6a0', '#ffd166', '#118ab2', '#073b4c'
  ];

  function colorForTopic(topicNumber) {
    const index = (parseInt(topicNumber, 10) - 1) % PALETTE.length;
    return PALETTE[index >= 0 ? index : 0];
  }

  function switchTab(tab) {
    document.querySelectorAll('.tab-panel').forEach(function(panel) {
      panel.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(function(button) {
      button.classList.remove('active');
    });
    document.getElementById('tab-' + tab).classList.add('active');
    document.querySelector('.tab-button[data-tab="' + tab + '"]').classList.add('active');
  }

  function onModelChange(label) {
    renderWords(label);
    renderSimilarity(label);
  }

  function renderWords(label) {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';

    const topics = WORDS_DATA[label];
    if (!topics) {
      grid.innerHTML = '<div class="empty">No topics found for this token set.</div>';
      return;
    }

    const docsForModel = REPRESENTATIVE_DOCS[label] || {};

    Object.keys(topics).forEach(function(topicNumber) {
      const words = topics[topicNumber];
      const maxWeight = Math.max.apply(null, words.map(function(w) { return w.weight; })) || 1;

      const card = document.createElement('div');
      card.className = 'card';

      const heading = document.createElement('h2');
      heading.textContent = 'Topic ' + topicNumber;
      heading.style.color = colorForTopic(topicNumber);
      card.appendChild(heading);

      words.forEach(function(item) {
        const row = document.createElement('div');
        row.className = 'bar-row';

        const label = document.createElement('div');
        label.className = 'bar-label';
        label.textContent = item.word;
        label.title = item.word;

        const track = document.createElement('div');
        track.className = 'bar-track';
        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.width = (100 * item.weight / maxWeight).toFixed(1) + '%';
        fill.style.background = colorForTopic(topicNumber);
        track.appendChild(fill);

        const weight = document.createElement('div');
        weight.className = 'bar-weight';
        weight.textContent = item.weight.toFixed(2);

        row.appendChild(label);
        row.appendChild(track);
        row.appendChild(weight);
        card.appendChild(row);
      });

      const docs = docsForModel[topicNumber];
      if (docs && docs.length) {
        const section = document.createElement('div');
        section.className = 'doc-section';

        const heading3 = document.createElement('h3');
        heading3.textContent = 'Representative documents (top terms)';
        section.appendChild(heading3);

        docs.forEach(function(doc) {
          const docItem = document.createElement('div');
          docItem.className = 'doc-item';
          const idSpan = document.createElement('span');
          idSpan.className = 'doc-id';
          idSpan.textContent = '#' + doc.doc_index + ': ';
          docItem.appendChild(idSpan);
          docItem.appendChild(document.createTextNode(doc.terms.join(', ')));
          section.appendChild(docItem);
        });

        card.appendChild(section);
      }

      grid.appendChild(card);
    });
  }

  function renderSimilarity(label) {
    const container = document.getElementById('heatmap');
    container.innerHTML = '';

    const data = SIMILARITY_DATA[label];
    if (!data || !data.topics || !data.topics.length) {
      container.innerHTML = '<div class="empty">No similarity data for this token set.</div>';
      return;
    }

    const topics = data.topics;
    const matrix = data.matrix;

    const table = document.createElement('table');
    table.className = 'heatmap-table';

    const headerRow = document.createElement('tr');
    headerRow.appendChild(document.createElement('th'));
    topics.forEach(function(t) {
      const th = document.createElement('th');
      th.textContent = t;
      headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    topics.forEach(function(rowTopic, rowIndex) {
      const tr = document.createElement('tr');
      const rowHeader = document.createElement('th');
      rowHeader.textContent = rowTopic;
      tr.appendChild(rowHeader);

      topics.forEach(function(colTopic, colIndex) {
        const value = matrix[rowIndex][colIndex];
        const td = document.createElement('td');
        const cell = document.createElement('div');
        cell.className = 'heatmap-cell';
        const alpha = Math.max(0, Math.min(1, value));
        cell.style.background = 'rgba(58, 110, 165, ' + alpha.toFixed(2) + ')';
        cell.style.width = '100%';
        cell.style.height = '100%';
        cell.title = 'Topic ' + rowTopic + ' vs Topic ' + colTopic + ': ' + value.toFixed(2);
        td.appendChild(cell);
        tr.appendChild(td);
      });

      table.appendChild(tr);
    });

    container.appendChild(table);
  }

  function renderPrevalence() {
    const chart = document.getElementById('prevalence-chart');
    const labelsRow = document.getElementById('prevalence-labels');
    const legend = document.getElementById('legend');
    chart.innerHTML = '';
    labelsRow.innerHTML = '';
    legend.innerHTML = '';

    const dateLabels = Object.keys(PREVALENCE_DATA).sort();
    if (dateLabels.length === 0) {
      chart.innerHTML = '<div class="empty">No topic prevalence data was found. This usually means ' +
        'the .model_metadata.json file was missing from the topic model archive, or it had no ' +
        '"predictions" for any token set. Check the topic-modeller dataset log for errors.</div>';
      return;
    }
    if (dateLabels.length === 1) {
      chart.innerHTML = '<div class="empty">Only one token set (' + dateLabels[0] + ') has prevalence ' +
        'data, so there is no trend to show yet. If you expected more, check that the topic model ' +
        'ran over multiple token sets and that each produced predictions.</div>';
      return;
    }

    const allTopics = new Set();
    dateLabels.forEach(function(label) {
      Object.keys(PREVALENCE_DATA[label]).forEach(function(topicNumber) {
        allTopics.add(topicNumber);
      });
    });
    const topicNumbers = Array.from(allTopics).sort(function(a, b) { return parseInt(a) - parseInt(b); });

    dateLabels.forEach(function(dateLabel) {
      const topicShares = PREVALENCE_DATA[dateLabel];
      const total = Object.values(topicShares).reduce(function(sum, v) { return sum + v; }, 0) || 1;

      const column = document.createElement('div');
      column.className = 'prevalence-column';
      column.title = dateLabel;
      column.onclick = function() {
        const select = document.getElementById('model-select');
        if (WORDS_DATA[dateLabel]) {
          select.value = dateLabel;
          onModelChange(dateLabel);
          switchTab('words');
        }
      };

      topicNumbers.forEach(function(topicNumber) {
        const share = topicShares[topicNumber] || 0;
        const segment = document.createElement('div');
        segment.className = 'prevalence-segment';
        segment.style.height = (100 * share / total).toFixed(2) + '%';
        segment.style.background = colorForTopic(topicNumber);
        segment.title = dateLabel + ' — Topic ' + topicNumber + ': ' + (100 * share / total).toFixed(1) + '%';
        column.appendChild(segment);
      });

      chart.appendChild(column);

      const labelDiv = document.createElement('div');
      labelDiv.className = 'prevalence-label';
      labelDiv.textContent = dateLabel;
      labelsRow.appendChild(labelDiv);
    });

    topicNumbers.forEach(function(topicNumber) {
      const item = document.createElement('div');
      item.className = 'legend-item';
      const swatch = document.createElement('span');
      swatch.className = 'legend-swatch';
      swatch.style.background = colorForTopic(topicNumber);
      const text = document.createElement('span');
      text.textContent = 'Topic ' + topicNumber;
      item.appendChild(swatch);
      item.appendChild(text);
      legend.appendChild(item);
    });
  }

  // initial render
  const firstLabel = document.getElementById('model-select').value;
  if (firstLabel) {
    renderWords(firstLabel);
    renderSimilarity(firstLabel);
  }
  renderPrevalence();
</script>

</body>
</html>
"""

        return (template
                .replace("__OPTIONS_HTML__", options_html)
                .replace("__WORDS_JSON__", words_json)
                .replace("__PREVALENCE_JSON__", prevalence_json)
                .replace("__SIMILARITY_JSON__", similarity_json)
                .replace("__REPRESENTATIVE_JSON__", representative_json))
