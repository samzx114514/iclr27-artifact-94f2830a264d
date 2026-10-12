# StackPlanner Multi-turn BrowseComp+ Search SOP

1. Append every public user turn, agent response, Search call, and Search result to the same checkpointed thread.
2. Treat an explicit later correction as authoritative while retaining non-conflicting earlier clues and evidence.
3. Reuse earlier passages and issue new targeted queries only for unresolved or changed clues.
4. Use only the injected read-only Search and Finish tools; never use a shell, arbitrary web access, benchmark labels, or grader feedback.
5. Finish every turn with the shortest current exact answer. Supporting explanation, citations, and confidence remain auxiliary metadata.
