# Troubleshooting

Run from ba-clustering/, not its parent. Use .venv/bin/python and .venv/bin/ba-clustering;
an unactivated shell may otherwise select a different environment. --config and
--experiments accept explicit paths. Import errors imply missing installation or incorrect
interpreter; existing editable installation is verified, a new offline install is not.

'not a git repository': the original ba-clustering/ archive intentionally has no .git.
Use the sibling bat-algorithm-param-sen/ Git clone for status/branch/remote/push. Do not
initialize or reset the archive to fix this error. Historical records keep git_commit=null;
new runs inside the clone record its current commit.

'setuptools.build_meta' unavailable during --no-build-isolation: the build backend is
missing in that interpreter. The runtime lock is not a complete offline build bundle.
Provide setuptools>=68 and build dependencies before trying an offline install. The
earlier offline attempt failed here; the latest online fresh install passed after
installing those build tools. See REPRODUCIBILITY for the verified sequence.
uv may also fail acquiring its default cache lock because the user cache is read-only;
set UV_CACHE_DIR to a writable location for future setup. New cache does not supply wheels.

Matplotlib cache warning: default ~/.config/matplotlib is read-only in this session;
MPLCONFIGDIR=/tmp/ba-matplotlib is a writable alternative. Existing tests and plots pass;
no display is needed. Matplotlib may fall back to a temporary cache automatically.

Existing-results error: choose a new output root for checks. Resume requires the same
spec/data hash. Do not use overwrite to resolve a mismatch on research evidence. Preserve
historical files. Resume mutates manifests/JSONL and does not enforce environment identity.

Bad shapes/NaN/Infinity: clean numeric input rather than impute silently. K must be <=N
for fitting; X must be nonempty 2-D. Very large finite values can overflow arithmetic;
scale data. Empty clusters are valid SSE states and are reported without repair.

Flat best-SSE or tied frequency sweeps: inspect candidate branches/acceptance and loudness,
r0 extremes, clipping, max velocity and initialization. Use scripts/audit_frequency.py with
an unused output file and smoke seeds. Do not reverse the velocity sign to force better
results. At r0=1 no local search occurs until an acceptance, and at A0=0 none can occur.
A flat curve is not proof of an optimum. See SCIENTIFIC_AUDIT.md.

Blank comparison/undefined silhouette: check available completed paired seeds and distinct
labels. A one-sample summary has no SD or mean CI. Failed runs are excluded from summaries,
so inspect raw failures before interpreting aggregate statistics.
