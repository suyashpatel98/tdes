# Corpus snapshot

The raw corpus is intentionally small (13,343,489 bytes total) and is vendored so the demo
does not depend on network access. Exact provenance, sizes, licenses, and SHA-256 hashes are
in `SOURCES.json`.

The files were retrieved from the URLs in `SOURCES.json`. `raw/cpython_LICENSE.txt` contains
the CPython license, the Project Gutenberg text includes its license, and
`raw/dolly_README.md` contains Dolly's dataset card and license metadata.

Do not edit the raw files. `python3 run_demo.py` verifies every source before deterministic
cleaning and will reject modified bytes.
