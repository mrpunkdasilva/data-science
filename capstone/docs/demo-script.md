# Final Demo Script - TST Jurisprudence Explorer

**Author:** Gustavo Henrique de Jesus da Silva
**Format:** live presentation, ~28 minutes + questions
**Slides:** [`slides.md`](slides.md) (Marp). Render with `make slides`.
**Principle:** the slides carry facts; you carry the story. Never read a slide
aloud word by word - explain it in your own words.

This script is written so you can present **without having read any of the
references**. Every concept is explained in plain language here and on the
slides. If you are asked about a paper, the backup slide gives a one-line
explanation of each one.

---

## Timeline at a glance

| Time | Segment | Slides |
|---|---|---|
| 00:00 - 01:00 | Title and agenda | 1-2 |
| 01:00 - 04:00 | Problem and domain | 3-4 |
| 04:00 - 06:00 | Dataset | 5 |
| 06:00 - 12:00 | Concepts + architecture | 6-12 |
| 12:00 - 20:00 | **Live demo** (dashboard + CLI) | 13 |
| 20:00 - 25:00 | Results | 14-22 |
| 25:00 - 26:30 | Engineering and reproducibility | 23-24 |
| 26:30 - 28:00 | Limitations, future work, takeaways | 25-27 |

Followed by questions. If you fall behind, skim the concept slides (6-9, 14, 16,
19, 21) - they are explanatory aids, not results.

---

## Pre-demo checklist (15 minutes before)

- [ ] Dashboard loads `data/final/tst.jsonl`:
      `cd capstone && make dashboard` (or `streamlit run src/app/dashboard.py`).
- [ ] `capstone stats` shows **1058** records.
- [ ] `capstone search "hora extra" -k 5` returns results.
- [ ] Figures exist under `capstone/docs/figures/`.
- [ ] Slides render: `make slides` (keep `docs/slides.pdf` ready as backup).
- [ ] **Plan B ready:** screenshots of the three dashboard tabs + a saved terminal
      transcript.
- [ ] Terminal font large; notifications off; heavy apps closed.

---

## Opening (00:00 - 01:00)

**Slide 1 - Title.** "Good morning. Our capstone is the **TST Jurisprudence
Explorer**: an interactive application to navigate a large corpus of Brazilian
labour-court jurisprudence using **embeddings and clustering**."

**Slide 2 - Agenda.** "I will cover the problem, the data, the key concepts, the
architecture, then **demo it live**, and finish with results, engineering and
limitations."

---

## Problem and domain (01:00 - 04:00)

**Slide 3 - The problem.**
"The Tribunal Superior do Trabalho publishes a book of Súmulas, Orientações
Jurisprudenciais and Precedentes Normativos. It is a **579-page PDF in
Portuguese**. Each statement is buried among its own **historical redactions**
and cancellation notes. Searching it by hand is slow, and there is no
machine-readable version to search **by meaning**."

- *What is a "verbete"?* One numbered normative statement, with a type, a body
  and a theme. The book has 1058 of them.

**Slide 4 - Requirements.**
"The brief asked for four things: an **ingestion pipeline**, a **knowledge
base**, an **exploration interface**, and **quality control**. Each row of this
table maps a requirement to the component that delivers it. The domain is an
**open-access official act**, so we use no copyrighted documents."

---

## Dataset (04:00 - 06:00)

**Slide 5 - Dataset.**
"From the PDF we segment **1058 verbetes**: **703 OJs**, **235 Súmulas**, **120
Precedentes Normativos**. Each keeps only its **current wording** - history is
dropped. Lengths range from 22 to 4204 characters, median 315. We also ship a
small **offline fixture** with 22 verbetes so CI and Docker run without network."

- *Why highlight the class imbalance?* It explains later why tiny classes score
  badly; being upfront about it is a strength, not a weakness.

---

## Concepts + architecture (06:00 - 12:00)

Use these slides to teach, then point to the architecture.

**Slide 6 - What is an embedding?**
"An embedding is a **list of numbers that represents the meaning of a text**.
Our model turns each verbete into **384 numbers**. Texts with similar meaning end
up nearby in that 384-dimensional space."

**Slide 7 - How we embed.**
"We use `paraphrase-multilingual-MiniLM-L12-v2`. It is **multilingual** and
**pre-trained** on a huge general corpus, so it already understands Portuguese
legal text - we do **not** train it ourselves. That reuse of a pre-trained model
is exactly what **transfer learning** means. It runs on CPU, 384 dimensions, 32
at a time. We embed the **theme plus the text**, and normalise the vectors."

**Slide 8 - Cosine similarity and vector search.**
"To compare two embeddings we use **cosine similarity**, which measures the
**angle** between them: **1.0 means same direction (very similar)**, 0 means
unrelated. To search, we embed the query and return the closest verbetes. That is
the core of **semantic search**: we compare **meaning**, not exact keywords."

**Slide 9 - Vector database.**
"A normal database stores rows; a **vector database** stores vectors and finds the
**nearest neighbours** to a query, fast. We use **ChromaDB** on disk with an
**HNSW** index - a graph that speeds up nearest-neighbour search - using
**cosine** distance. Each record also keeps **metadata** (type, body, code,
theme, quality label, cluster, anomaly score), so we can **filter** as well as
search."

**Slide 10 - Architecture.** Walk the diagram left to right: "Ingestion produces
a table; we embed; store in ChromaDB; cluster with UMAP and HDBSCAN; run quality
control; export an enriched JSONL that powers the dashboard."

**Slide 11 - Pipeline stages.** "Six stages, each runnable alone or as one
`pipeline` command - important because embedding is the slow part and we do not
want to redo it."

**Slide 12 - Ingestion.** "We parse with `pdfplumber`, cut at `Histórico:` and
repeated headers, and stop at the index. Input can be PDF or plain text,
detected by extension."

---

## Live demo (12:00 - 20:00) - the centerpiece

Start with **search** because it is the most intuitive.

1. **Search tab.** Type `adicional de insalubridade`. "These results are ranked
   by **cosine similarity**, not by keyword matching - notice the top results are
   about the meaning of the query even when the exact words differ." Apply a
   `tipo = oj` filter and show the list shrink.
2. **Cluster map tab.** "Each dot is a verbete, placed by **UMAP** and coloured
   by **HDBSCAN** cluster. There is one **dominant thematic cluster**, a smaller
   **historical** one, and grey **noise** points."
3. **Quality & anomalies tab.** "Here is the label mix, the text-length
   histogram, and our **most anomalous** verbetes - mostly rare precedentes."

Then the **CLI** to prove the engine is shared:

```bash
capstone search "hora extra" -k 5
capstone clusters --top-terms 5
capstone anomalies --limit 10
capstone stats
```

Narrate the output while it runs. **Timing guard:** stop at 20:00 even if the
demo is still open.

---

## Results (20:00 - 25:00)

**Slide 14 - Concept: UMAP.** "We cannot plot 384 dimensions, so **UMAP**
flattens them to 2D while keeping nearby points nearby. `n_neighbors=15` balances
local versus global structure; `min_dist=0.1` controls how tightly points pack."

**Slide 15 - Clustering results.** Point at the scatter.
- *What the map is:* a **scatter plot** where **each dot is one verbete**.
- *Axes:* `x` = UMAP-1 and `y` = UMAP-2. These are **abstract coordinates with no
  units** - they only encode **similarity**: dots close together have similar
  meaning; distant dots are unrelated.
- *Colours / legend:* each **colour is one HDBSCAN cluster**; **grey is noise**
  (points that fit no dense group). The legend shows the size `n` of each group.
- *What to say:* "One dominant cluster of **1009**, a smaller cluster of **35**,
  and **14 noise** points. The big blob is the general body of the book; the small
  group on the edge is the historical rural-work precedentes."
- *Likely pushback (full answers in "How to read each figure"):* the axes have
  **no units** and UMAP is **lossy**; "only two clusters?" is expected because the
  corpus shares one vocabulary; **noise is a review queue, not garbage**; the map
  is **reproducible** (`random_state=42`).

**Slide 16 - Concept: HDBSCAN.** "**HDBSCAN** groups densely packed points
**without being told how many clusters to expect**, handles clusters of
different sizes, and marks sparse points as **noise**. That honesty about
uncertainty is why we chose it over k-means."

**Slide 17 - Cluster sizes.** "Bars = **number of documents per cluster** (noise
is excluded by design). Cluster 1 has 1009 verbetes - the general body of the
book. Cluster 0 has 35 - historical rural-work homologation language. And 14 are
noise."

**Slide 18 - Quality: heuristic labels.** "*How to read it:* a **grouped bar
chart** - the `x` axis lists the labels (completo, cancelado, curto, duplicado);
for each label the **blue** bar is the **heuristic rule** count and the **orange**
bar is the **classifier prediction** count; the bar height is the number of
documents. Similar heights mean the classifier agrees with the rules. First we
label with **transparent rules**: cancellation keywords, duplicates, very short
text. Result: 625 complete, 405 cancelled, 26 short, 2 duplicate."

**Slide 19 - Concept: TF-IDF + logistic regression.** "**TF-IDF** turns text into
numbers, giving rare informative words more weight (we use single words and
pairs, up to 5000 features). **Logistic regression** is a simple, fast linear
classifier that learns which words point to each label. This is **supervised
learning on labels we created with rules** - sometimes called weak labels."

**Slide 20 - Classifier results.** "Accuracy is **0.91**. The cancelled and
complete classes are strong (F1 0.91 and 0.92). The tiny classes - short with 5
examples and duplicate with 1 - are **not learnable**; we report that honestly
instead of hiding it."

**Slide 21 - Concept: IsolationForest.** "It builds many random trees that split
the data. **Anomalies get isolated in fewer splits**, so they receive a higher
score. `contamination=5%` sets the expected outlier rate. It is **unsupervised**:
nobody told it what an anomaly looks like."

**Slide 22 - Anomalies found.** "*How to read it:* an **anomaly-score histogram** -
`x` is the score from 0 to 1, `y` is the number of documents per bin, and the
**red dashed line at 0.7** is the review threshold. Most documents sit at low
scores; the right tail is the unusual ones. **47 verbetes (4.4%)** score above
0.7. The top hits are short, atypical **historical precedentes** - exactly the
items a human should review."

---

## How to read each figure (quick reference)

All figures are regenerated by `make docs` from `docs/figures/summary.json`. Under
each figure are the **points an audience or examiner may raise**, with a ready
answer, so nothing catches you off guard.

### `fig_tipo.png` - Documents by tipo
Bar chart; one bar per verbete type (`oj`, `sumula`, `precedente`); bar height =
number of documents. Shows the composition (703 / 235 / 120).
- *"Why so many OJs?"* The TST publishes far more Orientacoes Jurisprudenciais
  than Súmulas; the corpus simply reflects the source.

### `fig_text_length.png` - Text length distribution
Histogram; `x` = characters, `y` = documents. Most verbetes are a few hundred
characters, with a long right tail.
- *"Are the very short ones errors?"* Some short entries are genuine (cancelled or
  procedural notes); they are labelled `curto`/`ruido` rather than deleted.

### `fig_quality_labels.png` - Heuristic vs. predicted labels
Grouped bar chart; `x` = quality label; for each label the **blue** bar is the
heuristic rule and the **orange** bar is the classifier prediction; height =
documents. Similar heights mean the classifier reproduces the rules.
- *"Why do the two bars differ?"* The classifier generalises beyond the exact
  keywords, so it moves some records (for example, about 71 from `cancelado` to
  `completo`) when the lexical marker is not a reliable cancellation signal.
- *"Are these labels ground truth?"* No. They are **heuristic (weak) labels**; the
  classifier inherits their blind spots, which is why we report per-class metrics
  honestly.
- *"Why does `curto` barely change?"* It is tiny (26 heuristic, 27 predicted); the
  length rule dominates and the classifier cannot learn such a small class.

### `fig_anomaly_scores.png` - Anomaly score distribution
Histogram; `x` = IsolationForest score (0-1), `y` = documents; the **red dashed
line at 0.7** is the review threshold. The right tail is the flagged documents.
- *"Is 0.7 a principled cutoff?"* No, it is a practical **review threshold**. The
  scores are min-max normalised to this corpus, so they are **relative**, not
  absolute; 0.7 is a convenient operating point, not a law.
- *"Doesn't `contamination=5%` predefine the outlier count?"* The contamination
  sets the model's expected fraction, but the 0.7 cutoff is applied separately;
  the flagged set is coherent (rare historical precedentes), not an arbitrary
  slice.
- *"Does an anomaly mean the document is wrong?"* No. **Anomaly = unusual**, not
  low quality. It is a **review cue**, not a judgment.

### `fig_cluster_map.png` - UMAP projection with clusters
Scatter plot; each dot is one verbete; `x`/`y` = UMAP-1/UMAP-2 (**abstract
coordinates, no units**, encoding similarity only); **each colour is a cluster**,
**grey is noise**; the legend lists each group and its size `n`.

- *"What do the axes/units mean?"* They are **abstract UMAP coordinates with no
  units**. Only **relative proximity** matters: dots close together have similar
  meaning; the axes themselves have no interpretable scale.
- *"Can I trust the distances and the cluster areas?"* Not literally. UMAP
  compresses 384 dimensions into 2 and is **lossy**; use the map for **exploration**,
  not for measuring exact distances. Wide gaps can be visual artefacts.
- *"Only two clusters - are there really just two themes?"* The default HDBSCAN
  parameters (`min_cluster_size=15`) resolve only two. The corpus shares one
  dominant legal vocabulary, so most points form one dense blob; the small second
  cluster is the historic rural-work precedentes. Lowering `min_cluster_size`
  would split the big blob further but risks fragmenting it arbitrarily.
- *"Why is one cluster so much bigger?"* The book is dominated by modern OJs and
  Súmulas that share vocabulary; the smaller cluster uses distinct historic terms
  (`concede`, `homologação`, `sessão`).
- *"Are the grey noise points errors?"* Not necessarily. Noise means the point
  fits no dense region - some are genuine outliers, some may be parsing artefacts.
  It is a **manual-review queue**, not garbage.
- *"Did you cluster on the 384-D vectors or on the 2-D picture?"* On the **2-D UMAP
  coordinates** (as in `src/cluster/pipeline.py`). That keeps the colours visually
  consistent with the plot, but ties the clusters to the projection. Clustering
  directly on the 384-D vectors is a possible improvement.
- *"Could the big cluster be hiding sub-themes?"* Yes. A single cluster is a
  **coarse** view; hierarchical or topic modelling could refine it.
- *"Is this reproducible?"* Yes. UMAP uses a **fixed `random_state=42`** and HDBSCAN
  is deterministic for a given input, so re-running gives the same picture.

### `fig_cluster_sizes.png` - Cluster sizes
Bar chart; `x` = cluster label, `y` = documents (noise excluded by design).
- *"Where is the noise bar?"* Noise (14 points) is **excluded by design** so the
  cluster comparison is not distorted.

### `fig_architecture.png` - Architecture
Box-and-arrow diagram; blue boxes are data/ingest/embed, orange is the vector
store, green are the ML stages, pink is the export, yellow is the interface. Read
it left to right, following the arrows.
- *"Where does the dashboard get its data?"* From the exported JSONL, optionally
  enriched by live queries to the ChromaDB collection.
- *"Can a step be re-run alone?"* Yes - every box is a separate CLI command, and
  `pipeline` chains them with `--skip-*` flags.

If asked "what does the map's colour/legend mean?": colour identifies the
**cluster**, grey means **noise**, and the `n=...` in the legend is the **number
of verbetes** in that group.

---

## Engineering and reproducibility (25:00 - 26:30)

**Slide 23 - Engineering.** "Fourteen tests, including an **offline end-to-end**
pipeline run. CI on GitHub (lint and strict typing) and GitLab (full pipeline).
`black` and `mypy` clean. And a **Docker** image that builds the pipeline,
pre-loads the fixture and serves the dashboard with one command."

**Slide 24 - Reproducibility.** "Three commands rebuild everything:
`capstone pipeline`, `make docs`, `make dashboard`. The offline fixture means CI
and Docker need no network."

---

## Limitations, future work, close (26:30 - 28:00)

**Slide 25 - Limitations.** The honest list: heuristic labels, general-purpose
embeddings, weak tiny classes, density-sensitive clustering, and the
educational-only disclaimer.

**Slide 26 - Future work.** Domain-tuned embeddings, **active learning** with
human corrections, retrieval evaluation, larger corpora, richer entity
extraction.

**Slide 27 - Takeaways.** "To summarise: a complete pipeline from a raw legal PDF
to an interactive, quality-checked knowledge base - reproducible, tested and
containerised. Thank you - happy to take questions."

Keep the **backup slides** (references, commands, glossary) ready for questions.

---

## Plan B - if something fails

| Failure | Fallback |
|---|---|
| Dashboard won't start | Show the **screenshots** of each tab, then run the CLI. |
| Search/DB unavailable | Use `capstone search` on the pre-built JSONL, or the saved transcript. |
| Slide render fails | Use `docs/slides.pdf` or the Markdown in a plain viewer. |
| Machine too slow | Skip the CLI block; talk over the screenshots. |
| No network | Everything is local: fixture and artifacts are committed/cached. |

**Golden rule:** the demo is a story, not live debugging. Keep talking, keep the
audience oriented, and fall back early rather than fixing things on stage.

---

## Concepts you must be able to explain (cheat-sheet)

Say these in your own words if asked:

- **Embedding** - a list of numbers capturing the **meaning** of a text. Similar
  meanings sit close together. Ours: 384 numbers per verbete.
- **Cosine similarity** - similarity between two vectors by **angle**;
  1 = same direction (very similar), 0 = unrelated. Used to rank search results.
- **Vector database (ChromaDB) / HNSW** - stores vectors and finds nearest
  neighbours fast; HNSW is a graph index that trades a little accuracy for
  speed.
- **UMAP** - reduces the 384 dimensions to **2** for visualisation, preserving
  neighbourhoods.
- **HDBSCAN** - **density-based** clustering that needs no preset number of
  clusters and labels sparse points as **noise (-1)**.
- **TF-IDF** - numeric text features that weight **rare, informative** words more
  than common ones.
- **Logistic regression** - a simple linear classifier that outputs class
  probabilities; fast and interpretable.
- **IsolationForest** - anomaly detector; anomalies are **isolated in fewer
  random splits** and get higher scores.
- **Transfer learning** - reusing a model **pre-trained** on a large general
  corpus instead of training from scratch; why 1058 documents are enough.

---

## Likely questions and answers

- **Why a multilingual model?** Portuguese legal text works without fine-tuning,
  and it runs on CPU. A legal-domain model is future work.
- **Why UMAP + HDBSCAN instead of PCA + k-means?** UMAP keeps local structure
  better for visualisation, and HDBSCAN needs no preset `k` and can label
  outliers as noise.
- **Is the classifier trained on embeddings?** No - it uses **TF-IDF + logistic
  regression** on the text, with **balanced class weights**; the embeddings are a
  separate component. (This is a good, honest detail to know.)
- **How are the quality labels validated?** They are **heuristic** (rule-based).
  The classifier learns to reproduce them, and we report per-class metrics
  including the failures.
- **What does "anomaly" mean here?** A verbete whose embedding is **unusual**
  relative to the rest - often rare historical precedentes. It is a **review
  cue**, not a judgment.
- **Is this legal advice?** No - it is an educational data-engineering project.
- **How reproducible is it?** `capstone pipeline` plus `make docs` regenerate
  every artifact; CI and Docker run offline on the fixture.
- **What was hardest?** Aligning dependencies (the Streamlit/Pillow/pdfplumber
  conflict) and making a large legal PDF segment cleanly.
