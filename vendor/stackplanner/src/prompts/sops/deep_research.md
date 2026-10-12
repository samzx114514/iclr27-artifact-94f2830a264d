### DeepResearch StackPlanner SOP

This profile uses a bounded report pipeline. Routing and termination belong to
the graph; agents must perform only their assigned stage.

1. Central Brief reads the chronological public user conversation and states
   the currently active research objective, constraints, language, and expected
   deliverable. It does not write an outline or report.
2. Outline produces one structured outline for that active brief.
3. Researcher performs the required web research once and returns evidence with
   the retrieved document identifiers.
4. Reporter writes one complete report in the configured locale. It cites
   retrieved evidence inline only as `【docid】` and must not create a References
   or 参考文献 section; deterministic post-processing owns the reference list.
5. Submit validates and exports that report, then terminates the graph. There is
   no model-driven critique or revision loop.

Later public user turns update the active brief chronologically. Evaluation
criteria, reference answers, grader feedback, and hidden benchmark fields must
never enter the graph state or prompts.
