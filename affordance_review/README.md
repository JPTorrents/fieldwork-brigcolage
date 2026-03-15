# Pipeline:
	1.	ingest and audit
	2.	normalize the Scopus export into relational tables
	3.	parse and normalize cited references
	4.	derive analysis tables for coupling, co-citation, and time slices
	5.	export clean data products for coding and visualization
	6.	only then do clustering and theory reconstruction

Rationale: Faraj and Azad’s chapter is useful precisely because it warns against category conflation, feature-centrism, and static views of technology over time, then calls for a more relational treatment of affordances oriented to role, line of action, practice/routine, and artifact bundle. A normalized pipeline is the practical way to prevent reproducing the same flattening in your literature review.

# Directory structure:
affordance_review/
  data/
    raw/
      scopus_affordance_2010_2026.csv
    interim/
      references_split.csv
      references_parsed.csv
      references_cluster_candidates.csv
    processed/
      affordance_lit.sqlite
      documents_clean.csv
      cited_refs_clean.csv
      bibcoupling_edges.csv
      cocitation_edges.csv
      keyword_timeslices.csv
      core_org_subset.csv
  scripts/
    audit_csv.py
    init_db.py
    build_db.py
    parse_references.py
    normalize_references.py
    build_networks.py
    extract_core_subset.py
    make_tables.py
  notebooks/
    audit.ipynb
    clusters.ipynb
  outputs/
    figures/
    tables/
        missingness.csv
        duplicate_docs.csv
    logs/
        csv_audit.txt
  requirements.txt
  README.md


# Environment

Use SQLite.
Use Python 3.11+.
Core packages:
    pandas
    numpy
    sqlalchemy
    sqlite3
    python-dateutil
    rapidfuzz
    unidecode
    networkx
    scikit-learn
    python-louvain
    matplotlib
    pyarrow
    tqdm

Optional but useful:
    duckdb
    polars
    jupyter
    regex

Scripts:
    audit_csv.py
        •	read CSV
        •	audit columns
        •	export missingness and duplicate reports

    init_db.py
          •	create SQLite schema
    build_db.py
        •	insert documents
        •	split authors
        •	split keywords
        •	split raw references

    parse_references.py
        •	read references_raw
        •	extract DOI, year, first author, title candidates
        •	write cited_references

    normalize_references.py
        •	deduplicate cited references into reference_clusters
        •	write document_reference_links

    build_networks.py
        •	create bibliographic coupling edges
        •	create co-citation edges
        •	export node and edge CSVs

    extract_core_subset.py
        •	create organization-relevance flags
        •	export core_org_subset.csv

    make_tables.py
        •	trend tables
        •	top references
        •	cluster summaries
        •	timesliced keyword tables


# audit_csv.py results
- EID is present and unique across all 5,055 rows.
- Abstract coverage is complete: 5,055 / 5,055.
- References are present for 5,021 records, with 34 missing.
- DOI is present for 4,854 records, so 201 are missing DOI.


9. normalize_references.py

Normalize cited references into deduplicated works. This is the core of the pipeline. Different articles will cite the same foundational work in inconsistent strings. You need to collapse variants into one canonical cited work.
Use a tiered deduplication strategy.
- Tier 1: DOI exact match. If DOI exists, cluster by DOI. This is high confidence.
- Tier 2: exact normalized key. For references without DOI, create a key: first_author_norm + "_" + year + "_" + first_8_title_tokens_norm
- Tier 3: fuzzy match within blocking groups. Block candidates by:
	•	same first author surname
	•	year difference = 0
	•	title token overlap above threshold
Then use fuzzy similarity with rapidfuzz.

Recommended title normalization:
	•	lowercase
	•	remove punctuation
	•	remove stopwords
	•	transliterate accents
	•	collapse whitespace

Recommended thresholds:
	•	exact block key match: auto-cluster
	•	title similarity >= 92: likely same work
	•	85–91: flag for manual review
	•	< 85: keep separate

Store:
	•	cluster ID
	•	canonical title
	•	canonical first author
	•	canonical year
	•	canonical DOI
	•	cluster confidence

Keep a manual review table for ambiguous cases. This is where classics like Gibson 1979 and Markus & Silver 2008 must be merged correctly.

Build internal article-to-article citation links where possible. This is optional but useful.
Goal: identify whether papers in your corpus cite other papers also in your corpus.
Method:
	1.	normalize corpus DOIs and titles
	2.	match each deduplicated cited reference cluster to corpus documents
	•	first by DOI
	•	then by title similarity + year
	3.	create internal_citations
	•	citing_doc_id
	•	cited_doc_id
	•	match_method
	•	match_confidence

This allows direct citation analysis inside the affordance corpus. It is useful but not essential. Bibliographic coupling and co-citation matter more.

11. Build the network tables

Script: 04_build_networks.py

A. Bibliographic coupling

Two articles are coupled if they share cited works.

From document_reference_links:
	1.	for each pair of documents
	2.	count shared ref_cluster_id
	3.	compute edge weight

Recommended weighting:
	•	raw shared count
	•	cosine-normalized overlap
	•	Jaccard overlap

Store table:

bibcoupling_edges
	•	doc_id_1
	•	doc_id_2
	•	shared_refs_n
	•	jaccard
	•	cosine

Filter weak edges. For 5k docs, otherwise the graph explodes.

Reasonable thresholds:
	•	keep only edges with shared_refs_n >= 3
	•	then inspect network density
	•	for final clustering maybe raise to >= 5

B. Co-citation network

Two cited works are co-cited if a document cites both.

From document_reference_links:
	1.	within each document
	2.	create all unordered pairs of cited works
	3.	count co-citation frequency

Store table:

cocitation_edges
	•	ref_cluster_id_1
	•	ref_cluster_id_2
	•	cocite_n

This maps the intellectual structure.

C. Keyword co-occurrence

Lower priority, but still useful.

Store:

keyword_cooccurrence_edges
	•	keyword_id_1
	•	keyword_id_2
	•	cooccur_n

Only on cleaned author keywords, not raw index keywords alone.

12. Derive the analytical slices you actually need

Script: 06_make_tables.py

Outputs:

Yearly trend
	•	number of publications by year
	•	total citations by year
	•	mean citations by year

Journal trend
	•	top journals overall
	•	top journals by 2010–2014, 2015–2019, 2020–2026

Foundational cited works
	•	top 100 reference_clusters by number of citing docs

This table will show whether the post-2010 literature is anchored more in Gibson, Norman, Markus & Silver, Leonardi, Faraj & Azad, Hutchby, Orlikowski, platform studies, HCI, or social media research.

Cluster candidate tables

For each document cluster:
	•	cluster size
	•	top journals
	•	top keywords
	•	top cited references
	•	median year
	•	top 20 representative docs by centrality

This is what you interpret substantively.

13. Extract a focused organizational subset after the macro-map

Script: 05_extract_core_subset.py

Do not wait until the end.

Build a rule-based first-pass subset for organization/IS relevance using:

Journal filter

Create a list of core journals:
	•	Organization Science
	•	MIS Quarterly
	•	Information Systems Research
	•	Information and Organization
	•	Information Technology & People
	•	Information Systems Journal
	•	Journal of the Association for Information Systems
	•	European Journal of Information Systems
	•	Journal of Information Technology
	•	Organization Studies
	•	Academy of Management Annals
	•	Academy of Management Review
	•	Research Policy
	•	Technovation
	•	etc.

Abstract/title keywords

Keep papers mentioning:
	•	organization
	•	organizational
	•	work
	•	routine
	•	coordination
	•	practice
	•	knowledge
	•	implementation
	•	governance
	•	platform
	•	enterprise
	•	professional
	•	team
	•	managerial
	•	institution

Exclude likely peripheral domains in first pass

Do not delete them; just flag them:
	•	cyberbullying
	•	adolescent social media behavior
	•	consumer shopping
	•	sports coaching
	•	elementary pedagogy
	•	generic UX/mobile app studies

Create:
	•	documents.is_core_org_candidate
	•	documents.org_relevance_score

Then manually inspect top uncertain cases.

14. Use network clustering only after filtering weak edges

For bibliographic coupling:
	1.	build graph from bibcoupling_edges
	2.	remove isolated nodes if needed
	3.	apply Louvain or Leiden-style community detection
In Python, Louvain via community_louvain.best_partition

Store:

document_clusters
	•	doc_id
	•	cluster_id
	•	resolution
	•	cluster_algorithm

Then create cluster summaries:
	•	size
	•	top journals
	•	top years
	•	top references
	•	top keywords
	•	title n-grams

This gives labels such as:
	•	sociomaterial / IS organizational change
	•	social media and communication affordances
	•	HCI / design affordances
	•	educational technology affordances
	•	platform and algorithmic affordances
	•	healthcare / digital health affordances

That is the macro-evolution map.

15. Keep manual coding outside the main database, but linked to it

Do not bake theory coding into the raw ingest pipeline.

Create a separate table:

theory_coding
	•	doc_id
	•	coded_by
	•	coding_round
	•	definition_source
	•	affordance_location
	•	unit_of_analysis
	•	technology_conceptualization
	•	temporal_treatment
	•	organizational_outcome
	•	constraint_treatment
	•	notes

Use exported CSV for coding if easier.

Why this matters: Faraj and Azad explicitly push away from generic users and artifact-as-feature thinking toward relational enactment in practice.  Your review should directly measure whether later studies actually do that.

17. Concrete implementation order over 5 working sessions

Session 1

Build database and audit.

Deliverables:
	•	SQLite file
	•	documents
	•	document_authors
	•	document_keywords
	•	references_raw

Session 2

Build parser for cited references.

Deliverables:
	•	cited_references
	•	parse-quality summary
	•	sample error log

Session 3

Normalize references and inspect top cited works.

Deliverables:
	•	reference_clusters
	•	document_reference_links
	•	top 100 cited references table

This session is where you verify whether foundational affordance works are being captured correctly.

Session 4

Generate coupling and co-citation networks.

Deliverables:
	•	bibcoupling_edges.csv
	•	cocitation_edges.csv
	•	preliminary cluster labels

Session 5

Extract core organization subset and begin theory coding.

Deliverables:
	•	core_org_subset.csv
	•	coding template CSV
	•	first cluster memo

18. What not to do

Do not start with:
	•	full-text scraping
	•	embeddings
	•	a big Postgres system
	•	fancy dashboards
	•	generic topic modeling on abstracts
	•	citation counts alone

Those are secondary. The decisive asset is a reliable normalized citation structure.

19. Best practical standard for this project

Use this rule:

raw preserved, parsed reproducible, normalized reviewable, analysis exported

That means:
	•	never overwrite raw strings
	•	every transformation is script-based
	•	all ambiguous reference merges are inspectable
	•	all clustering inputs are exportable to CSV/Gephi/R

20. The first outputs worth producing for the paper

After the pipeline is operational, produce these tables first:
	1.	annual publication growth, 2010–2026
	2.	top 30 journals by period
	3.	top 50 cited references across the whole corpus
	4.	bibliographic-coupling clusters with labels
	5.	top 10 cited references within each cluster
	6.	organizational core subset versus peripheral subset
	7.	coding table: relational vs feature-centric uses of affordance

That sequence will already show whether the post-2010 literature deepened the relational promise of affordance theory or diluted it into a loose vocabulary of functionality and user possibility, which is the core theoretical issue Faraj and Azad leave open.

21. The right starting point

Start with:
	•	SQLite schema
	•	raw reference splitting
	•	reference normalization

Not with R plots.

R is fine later for bibliometrix comparisons or clean figures. It is not the right first instrument for this dataset.

22. Minimal next code target

The immediate deliverable is a working 01_build_db.py that:
	•	reads the Scopus CSV
	•	creates documents, authors, document_authors, keywords, document_keywords, references_raw
	•	writes affordance_lit.sqlite

After that, everything becomes inspectable and cumulative.
