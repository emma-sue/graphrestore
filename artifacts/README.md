# Audited release artifacts

The working experiment directory is ignored by default because it contains
approximately 11 GB of optimizer snapshots, migration backups, logs, metric
model caches, and formal prediction PNGs. The GitHub release force-adds only a
reviewed allowlist:

- the Stage-A warm-start and selected Stage0/1/3/4 EMA checkpoints;
- small checkpoint completion/run-contract/validation records;
- Stage2 relation/effect metadata, frozen thresholds, priors, and approvals;
- metric/degradation/data audits;
- formal inference and Table-1 CSV/JSON evidence, including score shards;
- migration receipts, but no migration checkpoint backups.

No raw dataset image, depth map, generated degradation, formal prediction PNG,
Python environment, third-party metric weight, optimizer recovery snapshot, or
training log is tracked. See `../REPRODUCIBILITY.md` and
`../WEIGHTS.sha256`.

The selected final lineage and formal result are self-contained at the
metadata/checkpoint level. Historical rejected-run and migration receipts may
refer to intentionally omitted read-only checkpoint backups; those abandoned
transitions are audit context and are not byte-for-byte replayable from this
release alone.
