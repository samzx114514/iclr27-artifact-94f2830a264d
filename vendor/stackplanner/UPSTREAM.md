# StackPlanner 1 snapshot

This directory is a fixed, source-only snapshot of the StackPlanner runtime used
by GitHarness. It derives from the open-source ByteDance **DeerFlow** project
(MIT-licensed, <https://github.com/bytedance/deer-flow>). The snapshot retains
the upstream MIT license in `LICENSE`; the original copyright and license terms
must not be removed.

Legacy server, general application builder, presentation, podcast, prose,
backup, FactStruct, demo, and upstream test sources were excluded because they
are not part of the five task-native graphs exposed by this repository.

Repository-local changes are limited to lazy imports needed after pruning,
environment-referenced model configuration, citation-label preservation at the
GitHarness boundary, and removal of machine-specific paths. These integration
changes are visible in this repository's Git history.
