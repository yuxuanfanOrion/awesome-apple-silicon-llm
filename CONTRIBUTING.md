# Contributing

## What gets in

- It is about running, optimizing or understanding model inference on Apple Silicon. General ML resources that only happen to run on a Mac do not qualify.
- It is the canonical source: the original repository, not a fork or mirror; the original article, not a repost.
- It is maintained, or it is still the best reference on its topic. Mark old benchmarks with the year they were last updated.
- Claims in the description can be checked against the linked page.

## Format

One line per entry, in the section that matches the layer it belongs to:

```markdown
- [owner/repo](https://github.com/owner/repo) - What it does, and when you would use it.
```

- One sentence, specific: name the compute unit (GPU, ANE, CPU), the format or the API where it matters.
- No star counts, no marketing words, no emoji.
- Use a hyphen between link and description, not an em dash.

## Before opening a pull request

- Run `gh api repos/<owner>/<repo>` for every GitHub link and confirm `"fork": false`.
- Open every non-GitHub link and confirm it loads.
