# Agents

## Review guidelines

- Treat this repository as research scraping code that must prioritize correctness, reproducibility, and minimal edits.
- Review notebooks for hidden state, execution-order dependence, and hard-coded local paths.
- Flag any secrets, cookies, tokens, credentials, or personal data immediately.
- Prefer minimal, high-confidence fixes before larger refactors.
- For scraping code, check timeouts, retry logic, rate limiting, pagination, deduplication, selector brittleness, and silent data loss.
- Cite exact cells or code fragments when reporting issues.
- Do not invent behavior that is not visible in the repository.

## Task guidelines

- When asked to "modify", "patch", or "fix", produce code changes, not only a prose review.
- Prefer unified diffs and minimal edits.
- Do not stop at issue identification when the request explicitly asks for fixes.
