# Synthetic Data Gen

Autodata-based synthetic test data generation using agentic data scientists.

## Reference

- Paper: [arXiv:2606.25996](https://arxiv.org/abs/2606.25996) - "Autodata: An agentic data scientist to create high quality synthetic data"
- Authors: Kulikov et al. (Meta AI, Jun 2026)
- **[Full paper notes (easy-consume edition)](autodata-paper-notes.md)** — TL;DR + section-by-section extraction of the whole PDF
- [Quick abstract/metadata reference](autodata-reference.md)

## Method

Agentic Self-Instruct: AI agents act as data scientists to build high-quality training/evaluation data, with meta-optimization improving the agent itself. Five roles: Challenger generates examples, weak + strong solvers attempt them, a Judge scores against rubrics, and an Orchestrator accepts/rejects and feeds failure analysis back to the Challenger. Acceptance signal: weak solver struggles, strong solver succeeds (the learnable band). See the paper notes for acceptance criteria per domain, meta-optimization details, and costs.

## TODO

- [ ] Implement Agentic Self-Instruct loop
- [ ] Define target domains for test data generation
- [ ] Set up evaluation metrics
