# Autodata Reference
## arXiv:2606.25996

**Title:** Autodata: An agentic data scientist to create high quality synthetic data

**Authors:** Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Yixin Nie, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, Jason Weston (Meta AI)

**Submitted:** 24 Jun 2026 (v1), revised 25 Jun 2026 (v2)

**Subjects:** Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG)

**Links:**
- Abstract: https://arxiv.org/abs/2606.25996
- PDF: https://arxiv.org/pdf/2606.25996
- DOI: https://doi.org/10.48550/arXiv.2606.25996

## Abstract

> We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.

## Key Concepts

1. **Agentic Data Scientist** - AI agent that acts as a data scientist to build training/evaluation data
2. **Meta-optimization** - Training the data scientist agent itself to create stronger data
3. **Agentic Self-Instruct** - Specific practical implementation described in the paper
4. **Test Domains:**
   - Computer science research tasks
   - Legal reasoning tasks
   - Reasoning with mathematical objects

## Application to Test Data Generation

This method can be used to generate synthetic test data by:
1. Defining the data scientist agent's task formulation
2. Implementing the Agentic Self-Instruct approach
3. Meta-optimizing the agent for your specific domain
4. Converting inference compute into higher quality training data

*Saved for reference: 2026-07-04*