### SWE-bench StackPlanner SOP

1. The Central Agent reads only the public problem statement and produces a
   concise investigation and implementation plan.
2. The SWE Coder follows that plan while treating the real `/testbed` workspace
   as the source of truth. It inspects code before editing, makes a minimal
   production fix, and runs focused tests through the isolated bash tool.
   It does not install packages or download external source.
3. The SWE Coder calls `finish` when the workspace is ready. It must not search
   for benchmark metadata, hidden tests, a gold patch, or grader feedback.
4. Submit deterministically exports the workspace's `git diff`; no agent rewrites
   or evaluates the patch.
