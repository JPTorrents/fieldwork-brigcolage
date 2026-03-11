# Ranked scraping failure modes audit

Target notebook: `INESSS_cvs_publications_people_and_products_.ipynb`

## 1) CRITICAL — Notebook fails on clean top-to-bottom execution (execution-order dependence)
- **Where:** Validation cell runs before execution cell that defines `df_publications`, `df_people`, `df_products`.
- **Evidence:** Validation asserts on `df_publications` appears before scrape/execution block that creates DataFrames.
- **Impact:** Reproducibility failure; fresh kernel run errors before scrape pipeline completes.
- **Smallest fix:** Move validation cell below execution cell, or guard with `if 'df_publications' in globals(): ...` and fail explicitly otherwise.

## 2) CRITICAL — Description parser control-flow bug causing missing/incorrect extraction
- **Where:** `extract_description_after_santecom` area around nested helper and fallback logic.
- **Evidence:** `def collect_between(start_node):` and subsequent `if anchor:`/fallback lines are structurally inconsistent, making intended extraction path unreliable.
- **Impact:** Silent data loss/corruption in `description` field across publications.
- **Smallest fix:** Re-indent so `collect_between` is nested (or clearly scoped), then execute anchor branch and fallback in the same function body.

## 3) MAJOR — Over-broad DOM traversal can leak data across sections
- **Where:** Team and products extraction use `find_all_next()` from a header and then collect links/items until next header.
- **Evidence:** `extract_team_columns` and `extract_produits_connaissance` both iterate descendants beyond immediate section siblings.
- **Impact:** False positives (people/products from unrelated blocks), duplicated capture, unstable behavior on layout changes.
- **Smallest fix:** Traverse `next_siblings` within same parent container first; only fallback to broader scan when no section content is found.

## 4) MAJOR — Retries treat all exceptions similarly and ignore server hints
- **Where:** `polite_get` retry loop.
- **Evidence:** Catches broad `Exception`; retries with fixed exponential factor; does not read `Retry-After` for 429/503; raises synthetic `HTTPError` without response context.
- **Impact:** Inefficient backoff, avoidable request failures, and reduced resilience under throttling.
- **Smallest fix:** Catch `requests.RequestException`, preserve response object when status in {429,502,503}, and honor `Retry-After` when present.

## 5) MAJOR — Potential silent partial dataset with weak failure surfacing
- **Where:** Main loop appends errors but still writes outputs; only first 5 errors are printed.
- **Evidence:** `errors` list is not persisted to file; no hard threshold to fail job when error rate is high.
- **Impact:** Corrupted/incomplete outputs may look successful.
- **Smallest fix:** Persist `errors.csv` and print total failure rate; optionally abort if failures exceed threshold (e.g., >5%).

## 6) MAJOR — Deduplication key risks collision-driven record loss
- **Where:** Publication dedupe by `publication_id` only, derived from URL slug.
- **Evidence:** `make_publication_id` returns last path segment; downstream `drop_duplicates(subset=['publication_id'])`.
- **Impact:** Different URLs sharing slug can overwrite each other silently.
- **Smallest fix:** Dedupe publications on canonical URL first, keep `publication_id` as secondary key; or include normalized URL hash in ID.

## 7) MINOR — Throttling is global/static and not adaptive
- **Where:** `CRAWL_DELAY` loaded once from robots and applied uniformly.
- **Evidence:** Per-request sleep is constant + retry sleep factor; no jitter and no dynamic adaptation except FAST_MODE.
- **Impact:** Burst alignment risk and reduced friendliness under shared load.
- **Smallest fix:** Add small random jitter and adaptive increase on repeated 429/503.

## 8) MINOR — Pagination assumptions rely entirely on sitemap completeness
- **Where:** Crawl source is sitemap-only.
- **Evidence:** No in-page pagination crawling fallback.
- **Impact:** If sitemap omits pages, scrape has coverage gaps.
- **Smallest fix:** Keep sitemap strategy, but emit coverage diagnostics (e.g., count by year/theme trends) and log lastmod gaps; optional fallback only if required.

## 9) MINOR — Notebook outputs are committed, reducing reproducibility clarity
- **Where:** Notebook contains saved run outputs.
- **Evidence:** Output cells show prior run messages while `execution_count` is null.
- **Impact:** Harder to trust run provenance and detect stale state.
- **Smallest fix:** Clear outputs before commit and keep deterministic run instructions.
