# BIRD-SQL data notice

The SQL cases and model-visible schema/sample-row context in this directory are
adapted from the BIRD benchmark.

- Title: BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL)
- Project: <https://bird-bench.github.io/>
- Authors: Jinyang Li, Binyuan Hui, Ge Qu, and contributors
- License: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
- Changes: the source questions were organized into seven-turn evolving-intent
  trajectories; private gold SQL was removed; missing public schema context was
  deterministically extracted from the pinned SQLite databases.

The official project announced the CC BY-SA 4.0 data license on April 27, 2024.
This notice applies to the dataset material in `datasets/sql`; it does not
replace the repository's Apache-2.0 license for original software.

No SQLite database or gold SQL is redistributed in this repository.
