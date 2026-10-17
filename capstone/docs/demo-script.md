# Final Demo Script - TST Jurisprudence Explorer

**Author:** Gustavo Henrique de Jesus da Silva
**Format:** live presentation, ~28 minutes followed by questions
**Slides:** [`slides.md`](slides.md) (Marp). Render with `make slides`.

**How to use this script.** The quoted blocks below are written exactly as I will
say them, in the first person and in full prose. I can loosen the wording, but the
numbers and the claims are precise, so I keep them intact. Lines in *italics* are
stage directions for me, not things to read out loud.

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

Followed by questions. If I fall behind, I skim the concept slides (6-9, 14, 16,
19, 21), because they are teaching aids and not results.

---

## Pre-demo checklist (15 minutes before)

- [ ] Dashboard loads `data/final/tst.jsonl`:
      `cd capstone && make dashboard` (or `streamlit run src/app/dashboard.py`).
- [ ] `capstone stats` shows **1058** records.
- [ ] `capstone search "hora extra" -k 5` returns results.
- [ ] Figures exist under `capstone/docs/figures/`.
- [ ] Slides render: `make slides` (keep `docs/slides.pdf` ready as a backup).
- [ ] **Plan B ready:** screenshots of the three dashboard tabs and a saved terminal
      transcript.
- [ ] Terminal font large; notifications off; heavy applications closed.

---

## Opening (00:00 - 01:00)

### Slide 1 - Title

> "Good morning, everyone. My name is Gustavo Henrique de Jesus da Silva, and this
> is my capstone project: the TST Jurisprudence Explorer. In one sentence, it is an
> interactive application that lets you navigate a large corpus of Brazilian
> labour-court jurisprudence using embeddings and clustering. Let me unpack that.
> Instead of searching the law by exact keywords, my system searches by meaning. On
> top of that, it organises the whole corpus into thematic groups, and it flags the
> entries that deserve a human look. The goal is simple: nobody should have to read
> hundreds of pages by hand to find the few rules that actually matter to a
> question."

*Stage: pause after the one-sentence pitch and let it land before moving on.*

### Slide 2 - Agenda

> "Here is how I will use the next half hour. I will start with the problem and the
> domain. Then I will describe the dataset I work with. After that I will explain
> the key concepts behind the system: embeddings, vector search, dimensionality
> reduction, clustering, and the quality model. I will show the architecture of the
> pipeline, and then I will run the demo live, because that is the centerpiece of
> this presentation. I will finish with the results, the engineering practices
> behind the project, and an honest list of what the system does not do yet."

---

## Problem and domain (01:00 - 04:00)

### Slide 3 - The problem

> "The Tribunal Superior do Trabalho, or TST, is one of the highest courts in
> Brazil, and it publishes a consolidated book of Súmulas, Orientações
> Jurisprudenciais, and Precedentes Normativos. This book is the settled
> interpretation of Brazilian labour law, which makes it authoritative for lawyers,
> judges, and students. The problem is the format. It ships as a single PDF of
> five-hundred and seventy-nine pages, written in Portuguese. Every statement is
> buried among its own historical redactions and its cancellation notes, and there
> is no machine-readable version that I can search by meaning. So if I want to know
> everything the court has ever said about one concept, I have to scroll, guess
> keywords, and hope. That is slow, it is error-prone, and it scales badly: the
> book keeps growing."

*Stage: point at the page count on the slide, then at the dense text sample.*

> "One word I will use throughout is verbete. A verbete is a single numbered
> normative statement. It has a type, a body, and a theme. This book contains one
> thousand and fifty-eight of them, and that number is the spine of everything I
> will show you today."

### Slide 4 - Domain and requirements

> "The project brief asked for four things, and I want to show that I answered all
> four. First, an ingestion pipeline that turns the raw PDF into clean, structured
> data. Second, a knowledge base that I can query. Third, an exploration interface
> for a human being. And fourth, quality control, because a legal database is only
> as good as its data. This table maps each requirement to the component that
> delivers it. One important note about the domain: this is an official
> open-access act, so I am working with public data and using no copyrighted
> documents."

---

## Dataset (04:00 - 06:00)

### Slide 5 - Dataset

> "From the PDF, I segment one thousand and fifty-eight verbetes. Seven hundred and
> three of them are Orientações Jurisprudenciais, two hundred and thirty-five are
> Súmulas, and one hundred and twenty are Precedentes Normativos. For each verbete I
> keep only the current wording and I drop the history, because that is the part
> that is actually in force. The text lengths range from twenty-two characters to
> four thousand two hundred and four, with a median of three hundred and fifteen.
> That huge spread is important, and I will come back to it when I talk about
> quality. I also ship a small offline fixture with twenty-two verbetes, so that
> the tests and the continuous integration run without any network access."

*Stage: the class imbalance will be a question, so address it now.*

> "You will notice the classes are very unbalanced. I mention that on purpose,
> because it explains something later: the tiny classes are simply not learnable,
> and I would rather be upfront about that than pretend otherwise."

---

## Concepts + architecture (06:00 - 12:00)

*Stage: these slides teach the audience; speak slowly and check for understanding.*

### Slide 6 - What is an embedding?

> "Let me start with the single most important idea in the whole project: the
> embedding. An embedding is just a list of numbers that represents the meaning of
> a piece of text. My model turns each verbete into three hundred and eighty-four
> numbers. The magic is the geometry: texts that mean similar things end up close
> to each other in that three-hundred-and-eighty-four-dimensional space, and texts
> that mean different things end up far apart. So once I have embeddings, the
> question 'which rules are about the same topic' becomes a question about
> distances, and computers are very good at distances."

### Slide 7 - How we embed

> "For the model, I use paraphrase-multilingual-MiniLM-L12-v2. Two properties
> matter here. First, it is multilingual, which means it already understands
> Portuguese legal text, so I do not have to teach it the language. Second, it is
> pre-trained on a huge general corpus, so I do not train it myself at all. That is
> what transfer learning means: I stand on the shoulders of a model someone else
> trained, and I simply use it. This is also the answer to the obvious question
> 'is one thousand documents enough to train a model?' It is not, and I do not try
> to. It runs on CPU, it produces three hundred and eighty-four dimensions, and I
> process thirty-two verbetes at a time. One small detail: I embed the theme
> together with the text, and I normalise every vector, so that comparing them is
> clean and consistent."

### Slide 8 - Cosine similarity and vector search

> "Once I have embeddings, I need a way to compare two of them. I use cosine
> similarity, which measures the angle between two vectors. A value of one point
> zero means the two vectors point in the same direction, so the texts are very
> similar. A value near zero means they are unrelated. To search, I embed the
> user's query the exact same way, and I return the verbetes whose embeddings are
> closest to it. That is the core of semantic search. I am comparing meaning, not
> exact words. So a query about 'overwork' can find a verbete that talks about
> 'excess hours' even though they share no keyword."

### Slide 9 - Vector database

> "A normal database stores rows and looks things up by exact values. A vector
> database stores vectors and finds the nearest neighbours of a query, fast. I use
> ChromaDB, stored on disk, with an HNSW index. HNSW is a graph that speeds up the
> nearest-neighbour search, trading a tiny amount of accuracy for a large amount of
> speed. I set it to use cosine distance, matching the similarity I just described.
> Crucially, each record also keeps metadata: the type, the body, the code, the
> theme, the quality label, the cluster, and the anomaly score. That is what lets
> me filter and search at the same time, which is exactly what the dashboard does."

### Slide 10 - System architecture

*Stage: walk the diagram from left to right with the pointer.*

> "Here is the whole system on one slide. On the left, ingestion produces a clean
> table of verbetes. Those get embedded into vectors. The vectors go into ChromaDB.
> Then clustering reduces the space with UMAP and groups it with HDBSCAN. Quality
> control labels the records with rules, learns a classifier, and scores anomalies.
> The result is exported as an enriched JSONL file, and that file is what powers the
> dashboard on the far right. Every arrow is a real step in the code, not a
> decoration."

### Slide 11 - Pipeline stages

> "There are six stages, and each one can run on its own or as part of a single
> pipeline command. That separation is deliberate. Embedding is the slow part, so I
> did not want to redo it every time I tweak the clustering. In practice, I run the
> expensive stage once, and I iterate on the cheap stages as much as I want. It
> also makes the pipeline reproducible: if something looks wrong downstream, I know
> exactly which stage produced it."

### Slide 12 - Ingestion

> "Ingestion is where the mess lives. I parse the PDF with pdfplumber. I cut each
> statement at the word Histórico and at repeated page headers, and I stop when I
> reach the subject index at the end of the book. I keep only the current wording.
> The nice part is that the loader accepts either a PDF or a plain text file, and
> it detects which one from the file extension, so the same pipeline works on both.
> That is also what makes the offline fixture possible."

---

## Live demo (12:00 - 20:00) - the centerpiece

*Stage: switch to the dashboard and keep talking. Start with search, because it is
the most intuitive tab.*

### Slide 13 - Live demo

> "Now I will stop describing and start showing. This is the dashboard, and the
> first tab is search. Let me type a realistic query: adicional de insalubridade,
> which means hazard pay. Look at the results. They are ranked by cosine
> similarity, not by keyword matching, so the top hits are about the meaning of my
> query even when they use different words. Now watch what happens when I add a
> filter: I set tipo to OJ, and the list shrinks to only Orientações
> Jurisprudenciais. That filtering happens inside the queries, using the metadata I
> stored next to each vector."

> "Next is the cluster map. Every dot here is one verbete. Its position comes from
> UMAP, and its colour comes from HDBSCAN. I can see one dominant thematic cluster,
> a smaller historical one, and a handful of grey noise points. This single picture
> answers a question that would otherwise take days: how does this corpus organise
> itself thematically?"

> "The third tab is quality and anomalies. Here I show the label mix, the text
> length distribution, and the most anomalous verbetes in the whole book. Notice
> that the strongest outliers are almost all rare historical precedentes, which is
> exactly the kind of document a human should double-check."

> "To prove that the interface is not doing anything special, let me run the same
> engine from the command line. Everything the dashboard shows is also a scripted
> command, and that matters, because a result I can only reproduce by clicking
> buttons is not really reproducible."

*Stage: run the commands below one at a time and narrate each output as it appears.
Keep the terminal font large, because the audience reads the fields with me.*

First, semantic search:

```bash
capstone search "hora extra" -k 5
```

> "This is the same search as the first tab. For every hit I print the theme, then
> the cosine similarity score, and then three lines of metadata. The score measures
> how close my query is to the verbete, so a value in the zero point eight range is
> a strong match. The next line gives the document type, the court body, the
> reference number and the internal id. The line after that gives the predicted
> quality label, the cluster the document fell into, and its anomaly score. The last
> line is the opening of the text itself, so I can judge the match without opening
> another window."

Then the clusters:

```bash
capstone clusters --top-terms 5
```

> "This prints one block per cluster discovered by HDBSCAN. For each cluster I show
> its size, its dominant quality label, the breakdown by document type, and the five
> most frequent meaningful words, which I call the top terms. Cluster one is the main
> body of the corpus: one thousand and nine verbetes, almost all OJs and súmulas,
> dominated by the completo label, and its vocabulary circles around trabalho,
> súmula, decorrência, cancelada and empregado. Cluster zero is the small historical
> group of thirty-five precedentes, all marked cancelado, and its words are concede,
> homologação, sessão, empregado and cláusula. Those words are a fast, honest summary
> of what each cluster is about, and they are stripped of stopwords, punctuation and
> dates so that what you read is a topic and not raw token noise."

Then the anomalies:

```bash
capstone anomalies --limit 10
```

> "Here I ask the Isolation Forest for the ten most unusual verbetes. An anomaly
> score near one means the model finds the document highly atypical, and near zero
> means ordinary. For each one I print the score, the theme, the same identifying
> metadata, and a snippet of the text. Read the themes and the pattern I mentioned
> on the quality tab shows up again: the strongest outliers are rare historical
> precedentes, many of them rural-work rulings from the SDC. That is exactly the
> short list a human reviewer should open first."

And finally the whole-corpus summary:

```bash
capstone stats
```

> "This is the whole-corpus summary. It confirms the numbers I quoted at the start:
> one thousand and fifty-eight verbetes in total, split into seven hundred and three
> OJs, two hundred and thirty-five súmulas and one hundred and twenty precedentes. It
> also prints the distribution by court body and the predicted quality mix, which is
> where the three hundred and thirty-four cancelled documents appear."

> "So the same engine answers the same questions from a script as from the
> dashboard. That is the point: the dashboard is a thin layer on top of a real,
> scriptable pipeline, and every number you just saw is reproducible with one
> command."

*Timing guard: stop the demo at 20:00 even if I still have things to show, because
the results section needs its time.*

---

## Results (20:00 - 25:00)

### Slide 14 - Concept: UMAP

> "Before the results, one more concept, because it explains the map you just saw.
> I cannot plot three hundred and eighty-four dimensions on a screen, so I use
> UMAP, which flattens them down to two dimensions while trying to keep nearby
> points nearby. It has two main knobs: n_neighbors, which balances local and
> global structure, and min_dist, which controls how tightly the points pack
> together. I set them to fifteen and zero point one, and I fix the random seed so
> the picture is reproducible."

### Slide 15 - Clustering results

*Stage: point at the scatter, then at the legend.*

> "This is the result of that projection. Each dot is one verbete. The axes are
> UMAP-one and UMAP-two, and here is the crucial caveat: those axes have no units.
> They are abstract coordinates, and the only thing they encode is similarity, so
> dots that are close together have similar meaning and dots that are far apart are
> unrelated. The colours are the HDBSCAN clusters, and grey means noise, that is,
> points that did not fit any dense group. The legend tells me the size of each
> group."

> "The headline is simple. There is one dominant cluster of one thousand and nine
> verbetes, a much smaller cluster of thirty-five, and fourteen noise points. The
> big blob is the general body of the book. The small group on the edge is the
> historical rural-work precedentes, which use a distinct vocabulary. I want to be
> honest about what this does and does not say: it is a coarse map, the distances
> are not literal, and two clusters is a starting point, not the final word on how
> many themes exist."

*If asked: see the figure appendix below for the full list of objections and
answers, including the units, the reliability of distances, the number of clusters,
and the grey noise.*

### Slide 16 - Concept: HDBSCAN

> "The clustering method is HDBSCAN. I chose it for three reasons. First, it groups
> densely packed points without me having to tell it how many clusters to expect,
> which matters because I genuinely did not know. Second, it handles clusters of
> different sizes, and as you saw, mine are wildly different in size. Third, and
> most importantly, it is allowed to say 'I do not know': it labels sparse points as
> noise instead of forcing them into a cluster. That honesty about uncertainty is
> exactly why I preferred it over k-means, which would have split everything into a
> fixed number of groups whether they fit or not."

### Slide 17 - Cluster sizes

*Stage: read the bars from left to right.*

> "This chart just makes the sizes concrete. Each bar is the number of documents in
> one cluster, and noise is excluded on purpose so the comparison is not distorted.
> Cluster one has one thousand and nine verbetes, the general body of the book.
> Cluster zero has thirty-five, the historical rural-work homologation language.
> And fourteen verbetes are noise, meaning they did not land in any dense group."

### Slide 18 - Quality: heuristic labels

*Stage: point at the pairs of bars.*

> "Now the quality side. How do I label one thousand verbetes when nobody has given
> me the answers? I start with transparent rules, which I call heuristic labels.
> Rules like: does the text contain cancellation keywords, is it a duplicate, is it
> extremely short. Reading this chart: each label on the x axis has two bars, a blue
> one for the rule count and an orange one for the classifier prediction, and the
> height is the number of documents. Similar heights mean the classifier agrees
> with the rules. With the rules alone, I get six hundred and twenty-five complete,
> four hundred and five cancelled, twenty-six short, and two duplicates. These
> rules are simple on purpose: they are easy to defend, and every single label can
> be traced back to a reason."

### Slide 19 - Concept: TF-IDF + logistic regression

> "Once I have those rule-based labels, I train a small classifier to reproduce and
> generalise them. It uses two classic ideas. TF-IDF turns text into numbers and
> gives rare, informative words more weight than common ones; I use single words
> and pairs, up to five thousand features. Logistic regression is a simple, fast,
> linear classifier that learns which words point to which label. This is
> supervised learning, but the labels are weak, because I created them with rules.
> That is an honest limitation and I will say it again in the limitations."

### Slide 20 - Classifier results

> "Here is how the classifier does, evaluated on a held-out twenty percent split.
> Accuracy is about zero point nine one. The two big classes are strong: cancelled
> has an F1 of zero point nine one, and complete has zero point nine two. And then
> the honest part: the short class has only five examples and the duplicate class
> has exactly one, so their F1 is effectively zero. No model can learn from one
> example. I report that instead of hiding it, because a quality report that only
> shows the good numbers is not a quality report at all."

### Slide 21 - Concept: IsolationForest

> "The last model is for anomalies, and it uses a method called IsolationForest. The
> intuition is elegant. It builds many random trees that split the data, and it
> notices that unusual points get isolated in very few splits, while normal points
> take many splits to separate. The fewer splits it takes to isolate a point, the
> higher its anomaly score. I set the expected contamination to five percent, and
> the method is fully unsupervised: nobody ever told it what an anomaly looks like.
> It found the unusual ones on its own."

### Slide 22 - Anomalies found

*Stage: point at the histogram and the red line.*

> "This is what it found. The x axis is the anomaly score from zero to one, the y
> axis counts documents, and the red dashed line at zero point seven is my review
> threshold. Most documents sit at low scores, and the right tail is the unusual
> ones. Forty-seven verbetes, about four point four percent, score above zero point
> seven. When I read them, the top hits are short, atypical historical precedentes,
> which is exactly what a human reviewer should look at first. And I want to be
> clear about the meaning: anomaly here means unusual, not wrong. It is a review
> cue, not a verdict."

---

## Engineering and reproducibility (25:00 - 26:30)

### Slide 23 - Engineering quality

> "Good results mean little if the code is not trustworthy, so let me show the
> engineering practices. There are seventeen automated tests, including a full
> offline end-to-end run of the pipeline, so the whole thing is verified without
> touching the network. Continuous integration runs on both GitHub and GitLab:
> formatting and strict type checking in one, the full pipeline and tests in the
> other. Black and mypy are clean across the project. The code is also packaged so
> that the same pipeline can be served as a container if needed."

### Slide 24 - Reproducibility

> "And everything is reproducible. Three commands rebuild the whole system: the
> pipeline builds the data and the models, the docs command regenerates the figures
> and the reports, and the dashboard command starts the interface. Because of the
> offline fixture, CI needs no network, so a fresh clone can be validated end to
> end. If you take away one thing from this section, let it be that: the results are
> not a one-off, they are a build."

---

## Limitations, future work, close (26:30 - 28:00)

### Slide 25 - Limitations

> "Now the honest part. First, the quality labels are heuristic, which means they
> are my own rules and not legally authoritative. Second, the embedding model is
> general-purpose, not fine-tuned for legal Portuguese. Third, the tiny classes are
> not learnable, and I report that. Fourth, the clustering is density-sensitive, so
> different parameters give different numbers of clusters. And finally, and most
> importantly: this is an educational tool. It is not legal advice."

### Slide 26 - Future work

> "If I kept going, these are the directions I would take. I would use a
> legal-domain or Portuguese-law embedding model. I would add active learning, so
> that when a human corrects a label, the classifier actually learns from it. I
> would evaluate the retrieval quality against a hand-labelled relevance set,
> instead of only eyeballing results. And I would scale to larger corpora and richer
> entity extraction."

### Slide 27 - Takeaways

> "Let me close with the summary. I built a complete pipeline that goes from a raw
> legal PDF all the way to an interactive, quality-checked knowledge base. It
> embeds, stores, visualises, clusters and quality-checks a real corpus, and it is
> reproducible and tested end to end. Thank you very much for your attention, and I
> am happy to take your questions."

*Stage: keep the backup slides open: the command reference and the glossary.*

---

## How to read each figure (quick reference)

All figures are regenerated by `make docs` from `docs/figures/summary.json`. Under
each figure are the **points an audience or examiner may raise**, with a ready
answer, so nothing catches me off guard.

### `fig_tipo.png` - Documents by tipo

Bar chart; one bar per verbete type (`oj`, `sumula`, `precedente`); bar height =
number of documents. Shows the composition (703 / 235 / 120).

- *"Why so many OJs?"* The TST publishes far more Orientações Jurisprudenciais
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
  classifier inherits their blind spots, which is why I report per-class metrics
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
  compresses 384 dimensions into 2 and is **lossy**; use the map for
  **exploration**, not for measuring exact distances. Wide gaps can be visual
  artefacts.
- *"Only two clusters - are there really just two themes?"* The default HDBSCAN
  parameters (`min_cluster_size=15`) resolve only two. The corpus shares one
  dominant legal vocabulary, so most points form one dense blob; the small second
  cluster is the historic rural-work precedentes. Lowering `min_cluster_size`
  would split the big blob further but risks fragmenting it arbitrarily.
- *"Why is one cluster so much bigger?"* The book is dominated by modern OJs and
  Súmulas that share vocabulary; the smaller cluster uses distinct historic terms
  (`concede`, `homologação`, `sessão`).
- *"Are the grey noise points errors?"* Not necessarily. Noise means the point
  fits no dense region; some are genuine outliers, some may be parsing artefacts.
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
**cluster**, grey means **noise**, and the `n=...` in the legend is the **number of
verbetes** in that group.

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
  neighbours fast; HNSW is a graph index that trades a little accuracy for speed.
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

Grouped by area, so you can jump to the topic he is probing. Where he asks "where
is the code?" the file and function are given, so you can open it immediately.

### Overview, domain and requirements

- **What is the one-line goal?** Turn the TST book PDF into a searchable,
  clustered, quality-checked knowledge base.
- **Who is the user?** Someone who needs to find related labour rulings fast
  (lawyer, researcher, citizen).
- **Why is the data usable?** Official public act, so there is no copyright issue.
- **What are the three document types?** Súmulas (TST), OJs
  (SBDI-1/SBDI-2/SDC/TP-OE/SBDI-1T), Precedentes Normativos (SDC).
- **What is a verbete?** One numbered statement: type, orgão, number, theme, body.
- **How many documents?** 1058 verbetes from a 579-page PDF.
- **Where is the requirement mapping?** The four parts of the brief map to ingest,
  ChromaDB, Streamlit and quality control.
- **Is this legal advice?** No - it is an educational data-engineering project.

### Ingestion (code: `src/ingest/tst.py`, command `cli.main.ingest`)

- **How is a heading detected?** Regexes `HEADING_SUMULA`, `HEADING_OJ`,
  `HEADING_PN`.
- **How is page furniture dropped?** `FURNITURE` and `BANNER_*` regexes.
- **How is only the current wording kept?** `HISTORY_BOUNDARY` cuts at `Histórico:`
  and `Redação ...`.
- **Where does parsing stop?** At `ÍNDICE REMISSIVO` (`BANNER_INDEX`).
- **How are hyphen splits fixed?** `_join_lines` merges `pe-` + `soal`.
- **How are orgãos normalized?** `ORG_MAP` maps `SDI1 -> SBDI-1`, and so on.
- **How are historical repeats skipped?** `_segment_verbetes` skips the same
  `tipo` + `codigo`.
- **How are heuristic labels assigned?** `_label_verbetes`: cancellation keywords,
  duplicates, then length.
- **What are the length thresholds?** `VERBETE_RUIDO=30`, `VERBETE_CURTO=120`.
- **PDF vs txt?** Same pipeline; `load_verbetes` picks `_read_pdf_lines` or
  `_read_text_lines` by extension.
- **How is `doc_id` built?** `{tipo}_{codigo}` for TST/SDC, else
  `{tipo}_{orgao}_{codigo}`.
- **What was hardest here?** Making a dense 579-page PDF segment cleanly (repeated
  headers, rotated text).

### Embeddings (code: `src/embed/model.py::get_model`)

- **Which model?** `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers).
- **Why this model?** Multilingual, 384-D, small and fast on CPU.
- **Why a multilingual model?** Portuguese legal text works without fine-tuning and
  runs on CPU. A legal-domain model is future work.
- **What text is embedded?** `tema` + `" "` + `text` (in `embed_reviews`).
- **Why include the theme?** The short theme adds a strong topical signal.
- **Why normalize vectors?** So cosine similarity becomes a fast dot product.
- **What is an embedding, really?** 384 learned numbers that encode meaning.
- **Batch size and threads?** 32 texts per batch; `torch.set_num_threads(8)`.
- **Is the model trained by us?** No - transfer learning from a general pre-trained
  model.
- **Why is 1058 documents enough?** We reuse pre-trained knowledge; we do not train
  the model.
- **Can we fine-tune?** Future work; it needs labelled data.

### ChromaDB and vector search (code: `src/store/client.py`)

- **Where is the client?** `get_chroma_client` (persistent, on disk).
- **Which index and space?** HNSW graph, cosine.
- **HNSW parameters?** `HNSW_M=32`, `HNSW_EF_CONSTRUCTION=200`.
- **What is HNSW?** Hierarchical Navigable Small World: an approximate
  nearest-neighbour graph.
- **Is it exact?** No - approximate, trading a little recall for a lot of speed.
- **Which metadata is stored?** `tipo`, `orgao`, `codigo`, `tema`, `quality_label`,
  `cluster_id`, `anomaly_score`, `umap_x/y`.
- **How is filtering done?** A `where` clause in `search_reviews`.
- **Why `$and`?** Chroma needs `{"$and": [...]}` when combining more than one
  condition.
- **How to rebuild?** Delete the collection, then `capstone pipeline`.

### UMAP (code: `src/cluster/pipeline.py::run_clustering`)

- **Why reduce dimensions?** We cannot plot or reason over 384-D directly.
- **Why UMAP?** Non-linear; keeps local and some global structure.
- **Why not PCA or t-SNE?** PCA is linear; t-SNE keeps only local structure and is
  slower.
- **What is `n_neighbors=15`?** The local vs global balance.
- **What is `min_dist=0.1`?** How tightly points may pack.
- **Why cosine metric?** It matches the embedding space.
- **Why `random_state=42`?** Reproducibility.
- **Are the axes meaningful?** No; UMAP-1 and UMAP-2 have no units.
- **Is the projection lossy?** Yes; distances are approximate, not measurements.

### HDBSCAN (code: same file, right after UMAP)

- **Why HDBSCAN over k-means?** No preset `k`, handles uneven densities, and can
  mark noise.
- **What is `min_cluster_size=15`?** The smallest group accepted as a cluster.
- **What is `min_samples=5`?** How conservative the density estimate is.
- **What does `-1` mean?** Noise: too sparse to belong to a cluster.
- **How many clusters?** Two dense clusters plus 14 noise points.
- **What are they?** Cluster 1 (1009, general body) and cluster 0 (35, historical
  rural-work precedentes).
- **Is a cluster a legal category?** No, only a thematic grouping.

### Quality labels and classifier (code: `src/quality/classifier.py`)

- **What are the labels?** completo, cancelado, curto, duplicado, ruido.
- **Where do they come from first?** Rules in `_label_verbetes` (weak supervision).
- **How are the quality labels validated?** They are heuristic (rule-based). The
  classifier learns to reproduce them, and I report per-class metrics including the
  failures.
- **Why a classifier at all?** The rules are rigid; the classifier generalises them.
- **Is the classifier trained on embeddings?** No - it uses TF-IDF + logistic
  regression on the text, with balanced class weights. The embeddings are a
  separate component.
- **Why TF-IDF + logistic regression?** Simple, fast, interpretable, and it works
  with few labels.
- **Why n-grams 1-2 and 5000 features?** Capture short phrases within a bounded
  size.
- **Why a 20% test split?** A held-out evaluation, seed 42.
- **What if a class has one sample?** Stratify is disabled, and that class cannot be
  learned.
- **Results?** Accuracy 0.91, weighted-F1 0.90; `curto` and `duplicado` fail (n=5
  and n=1).
- **Where is the model saved?** `models/quality_classifier.joblib`.
- **Are these legal labels?** No - heuristic and model predictions.

### Anomaly detection (code: `src/quality/anomaly.py::run_anomaly_detection`)

- **What does "anomaly" mean here?** A verbete whose embedding is unusual relative
  to the rest - often rare historical precedentes. A review cue, not a judgment.
- **Why IsolationForest?** Unsupervised, and it scales well.
- **How does it work?** Random splits isolate anomalies in fewer cuts, giving
  higher scores.
- **On what data?** The 384-D embeddings, so anomalies are semantic.
- **What is `contamination=0.05`?** The expected outlier rate (5%).
- **How are scores normalised?** Min-max to 0-1; higher means more anomalous.
- **Why is the score negated?** `decision_function` gives lower values for
  outliers.
- **Is 0.7 the right cutoff?** It is a display convention; the ranking matters more
  than the line.
- **How many flagged?** 47 verbetes (4.4%).
- **Top hits?** `precedente_106`, `precedente_63`, `precedente_107`.

### Search, export and dashboard (code: `src/search/engine.py`, `src/export/jsonl.py`, `src/app/dashboard.py`)

- **How does search work?** Embed the query, then ask ChromaDB for nearest
  neighbours.
- **How many results?** `min(k, MAX_SEARCH_K=50)`.
- **What is the score?** `1 - distance` (cosine similarity).
- **What is `get_stats` for?** Totals by type, orgão, quality, clusters, anomalies.
- **How are top terms computed?** `src/lib/text.py::top_terms`, which drops
  stopwords, punctuation and numbers.
- **What does export write?** `export_jsonl` writes one JSON object per verbete to
  JSONL.
- **Why does the dashboard read the JSONL, not the database?** A reproducible
  snapshot that also survives an empty database.
- **What if ChromaDB is missing?** The dashboard falls back to `lexical_search`.
- **How does the dashboard start?** `src/app/main.py` calls `streamlit run`.

### Engineering, tests and Docker

- **Where are the tests?** `tests/test_pipeline.py` (17 cases).
- **What does the end-to-end test do?** It runs the whole pipeline on an offline
  fixture (`tests/fixtures/tst-sample.pdf`, 22 verbetes).
- **What CI runs?** GitHub Actions (lint + strict typing) and GitLab CI (pipeline +
  tests).
- **Is Docker used in the demo?** No - the demo runs natively; Docker remains
  available.
- **How to reproduce?** `make test lint docs`, then `make dashboard`.

### Reproducibility and data

- **How reproducible is it?** `capstone pipeline` plus `make docs` regenerate every
  artifact; all seeds are fixed (`random_state=42`) and CI runs offline on the
  fixture.
- **Where is configuration?** `src/config.py`, overridable by `.env`.
- **What is committed vs generated?** Code and docs are committed; data, models and
  `chroma_db` are generated.
- **How big is the corpus?** 1058 verbetes; text length from 22 to 4204 characters.

### Hard and curveball questions

- **What was hardest?** Aligning dependencies (the Streamlit/Pillow/pdfplumber
  conflict) and making a large legal PDF segment cleanly.
- **What if we change the embedding model?** Re-embed everything; the dimension may
  change.
- **What if the corpus grows to a million documents?** Keep HNSW, batch ingestion,
  and consider sharding.
- **Why not use an LLM?** Cost, reproducibility and the offline constraint; a
  possible future step.
- **How do you know the clusters are correct?** There is no ground truth; they are
  qualitative and plausible.
- **Could the classifier be biased?** Yes - it inherits the rules' biases, which is
  why the labels are marked weak.
- **How is duplicate detection done?** Normalised exact-text equality, not fuzzy
  matching.
- **Is it production ready?** No - an educational capstone.
- **What would you do with more time?** A labelled retrieval benchmark and active
  learning.

### Where to stop and admit limits

Honest answers are stronger than guessing:

- **Can you prove retrieval quality?** Not yet; there is no labelled relevance set.
- **Is the quality label legally correct?** I cannot claim that; it is heuristic.
- **Why exactly two clusters?** That is what these density parameters found, not
  the only valid answer.
- **Would a legal expert agree with the top terms?** They are statistics, not legal
  analysis.
- **Does UMAP preserve true distances?** No, it is a lossy projection.
- **Is `curto` well modelled?** No, five examples are too few.
- **Can it replace a jurist?** No, it is a navigation aid.
