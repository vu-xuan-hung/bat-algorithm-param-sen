# GitHub publication status — 2026-11-05

Destination: https://github.com/vu-xuan-hung/bat-algorithm

**Not pushed.** Shell Git could not resolve github.com; GitHub CLI could not connect to
api.github.com. No remote mutation succeeded. The parent agent's read-only web inspection
found a public repository with main, one existing commit and README.md containing
`# bat-algorithm`. The remote is not empty; its history must be preserved.

Executed remote checks:

- `git ls-remote https://github.com/vu-xuan-hung/bat-algorithm.git HEAD` failed with
  `Could not resolve host: github.com`.
- `gh repo view vu-xuan-hung/bat-algorithm --json nameWithOwner,defaultBranchRef,isEmpty`
  failed with `error connecting to api.github.com`.

A local allowlisted source snapshot commit and portable Git bundle were prepared in
`publication/`. The commit is an offline snapshot, **not** an existing-remote descendant.
Its SHA and branch are in publication/SNAPSHOT.json and can be inspected with
`git bundle list-heads publication/ba-clustering-continuation.bundle`.
No Git metadata was created in the project or parent; historical verification environments
therefore remain unchanged. The temporary snapshot repository alone has Git metadata.

Included: source, tests, configs, scripts, documentation/selected audit evidence, dependency
files, executed synthetic-example notebook, and the synthetic dataset/provenance.
Excluded: virtualenv/caches, parent research documents, user CSVs, bulk pilot results and
notebook-session raw outputs. The notebook embeds real plots; pilot audit evidence refers
to historical raw files preserved in the original workspace, not shipped here.
A staged-content scan detected no recognized private-key/token patterns; this is a bounded
scan, not proof of absence of every possible secret.

When network/authentication works, follow publication/README.md: clone the destination,
create a new branch based on origin/main, fetch the offline snapshot, apply only the
allowlisted paths, review/integrate the pre-existing README, commit and push that branch.
Never force-push or replace main history. Publishing and remote verification remain pending.


## Isolated snapshot verification

The allowlisted snapshot passed **372 tests in 5.18s** using the original interpreter
with PYTHONPATH explicitly set to the snapshot's src directory. An independent import
check resolved ba_clustering to /tmp/ba-publication-w1egjsld/src/ba_clustering/__init__.py.
This verifies that excluded bulk results/user data are not needed by the suite. Existing
dependencies were reused; fresh-install and Jupyter GUI limitations still apply.
Exact command/environment evidence: publication_snapshot_checks.txt.
