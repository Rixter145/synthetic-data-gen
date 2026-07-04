# Autodata — Full Paper Notes (Easy-Consume Edition)

**Paper:** [arXiv:2606.25996](https://arxiv.org/abs/2606.25996) — *Autodata: An agentic data scientist to create high quality synthetic data*, Kulikov et al., Meta AI, June 2026.
**These notes:** extracted from the full PDF by an agent on 2026-07-04. TL;DR first; full section-by-section extraction below.

---

## TL;DR (60 seconds)

**The idea:** treat data creation as an iterative data-science job done by an LLM agent — generate examples, *measure* whether they're useful, analyze failures, refine the recipe, repeat. This converts inference compute into training-data quality.

**The engine (Agentic Self-Instruct), five roles:**

| Role | Job |
|---|---|
| Challenger | Generates example: context + question + reference answer + weighted rubric |
| Weak solver | Small/constrained model — *should struggle* |
| Strong solver | Big/privileged model (often same model + more compute) — *should succeed* |
| Judge | Scores both against the rubric, reasons about rollout patterns |
| Orchestrator | Accepts/rejects, feeds targeted feedback back to Challenger |

**The acceptance signal:** an example is good when the **weak solver struggles but the strong solver succeeds**. Not maximal difficulty — *learnable* difficulty. If weak scores all-zeros, there's no gradient; if weak aces it, there's nothing to learn. The loop steers each example into the learnable band (e.g. math: weak ≤ 1/4 correct, strong ≥ 3/4).

**Why it works (the paradox):** on CS papers, naive generation was *too easy* → loop made it harder. On legal docs, naive generation was *too hard* → loop made it easier. Both directions improved downstream training. The gap is a **diagnostic, not a target**.

**Meta-optimization (outer loop):** evolutionary search over the agent's own prompts, scored by the inner loop's accept rate. Took CS pass rate 62.1% → **79.6%** in 6 hours. Discovered fixes were mundane and real: enforce paper-specific questions, prevent context leakage, positive-only capped rubric weights, strict JSON rubric format.

**Headline results (GRPO on Qwen3.5-4B):** agentic data beat classic CoT Self-Instruct data in all three domains; on legal, the 4B trained on agentic data **beat the 397B base model**. Hard-data training transferred to easier test sets too.

**Cost reality:** ~5–6.6 challenger rounds per accepted example, each round = challenger + multiple weak/strong rollouts + judge. This is an inference-compute-heavy method by design.

**What it doesn't cover:** interactive dialogue, real-time learner modeling, engagement signals, multimodal tasks. Dataset-level diversity/coverage analysis is future work.

**If you're building on it, the recipe:**
1. Ground the generator on source material.
2. Define weak vs. strong solvers (same model, different compute/scaffolding is fine).
3. Write task-specific acceptance criteria in the "learnable band" style.
4. On rejection, give the Challenger *specific* failure feedback ("too easy — weak scored 0.8 on these") and demand a new angle, not a rephrase.
5. Once the loop works, meta-optimize the prompts with the same signal.

---

*Full extraction follows.*

---

# Autodata: An Agentic Data Scientist for High-Quality Synthetic Data
## Extraction: Kulikov et al., Meta AI (arXiv:2606.25996v2, June 2026)

---

## 1. High-Level Framework

**Autodata** is a general framework that treats data creation as an iterative data-science process. An autonomous agent acts as a data scientist, performing tasks a human would do:

- **Data Creation**: Ground on provided data (documents, code, legal text, math) and generate training/evaluation examples using tools and inference-time compute.
- **Data Analysis**: Evaluate generated examples at individual and dataset levels. Analyze what was right vs. wrong. Extract learnings.
- **Iteration Loop**: Feed learnings back into the creation step. Repeat until stopping criteria met.
- **Meta-Optimization (Outer Loop)**: The agent's prompts and strategy can themselves be optimized using the same evaluation criteria from the inner loop.

The key insight: **increase inference compute → higher quality training data**, by systematically analyzing failures and adapting the generation recipe.

---

## 2. Agentic Self-Instruct: The Core Implementation

### 2.1 Architecture & Subagents

The main instantiation, **Agentic Self-Instruct**, uses five components:

1. **Challenger LLM**: Generates training examples (context, question, reference answer, evaluation rubric)
2. **Weak Solver**: A smaller/constrained LLM expected to struggle on the generated data
3. **Strong Solver**: A larger/privileged LLM expected to succeed (same model as weak, but with more compute, scaffolding, or privileged information)
4. **Verifier/Judge**: Evaluates example quality and solver outputs; provides structured feedback
5. **Main Orchestrator Agent**: Coordinates the loop; decides acceptance/rejection; updates challenger prompts

### 2.2 The Iterative Loop

**Per-example workflow:**

1. **Challenger generates**: Context, question, reference answer, weighted evaluation rubric
2. **Quality check**: Verifier checks for answer leakage, rubric coverage, question quality
3. **Weak solver test**: Solves the question (multiple rollouts at temperature 1.0 for variance)
4. **Strong solver test**: Solves the same question (multiple rollouts)
5. **Judge evaluation**: Scores both solvers' answers against the rubric on a per-criterion basis
6. **Decision logic**:
   - **For verifiable tasks** (hard correctness): Require majority-vote strong correct + majority-vote weak wrong
   - **For non-verifiable tasks** (rubric-based): Require a quality gap such that weak struggles but strong succeeds meaningfully
7. **Feedback & Refinement**: If criteria unmet, orchestrator provides targeted feedback to challenger:
   - Which previous questions were too easy (weak scored too high)
   - Which failed on strong (gap too small or strong failed)
   - Which failed quality checks
   - Challenger generates entirely new question from different reasoning angle
8. **Repeat** until criteria met or budget exhausted

### 2.3 Acceptance Criteria (Task-Dependent)

**CS Research Tasks** (verifiable, rubric-based):
- Strong solver avg ≥ 65%
- Weak solver avg < 50%
- Strong-weak gap ≥ 20 percentage points

**Legal Reasoning Tasks** (non-verifiable, more flexible):
- Judge evaluates weak rollout distribution for GRPO suitability (high/medium/low)
- Weak mean pushed into learnable range (typically 25-35%), not all-zeros
- Strong significantly above weak
- No fixed thresholds; judge reasoning about per-rollout patterns

**Scientific/Math Tasks** (verifiable):
- Weak solver: ≤ 1 correct out of 4 attempts (≤ 25% success)
- Strong solver: ≥ 3 correct out of 4 attempts (≥ 75% success)

---

## 3. Detailed Method: Per-Task Instantiations

### 3.1 Computer Science Research Tasks

**Input**: 10k+ CS papers from S2ORC corpus (2022+)

**Pipeline**:
- Orchestrator calls Challenger to generate context-QA pair with rubric from a given paper
- Quality verifier checks for context leakage ("could someone answer without reasoning?"), rubric coverage (10-20 criteria), question quality (reasoning vs. recall)
- Weak solver (Qwen3.5-4B): 3 invocations per question to reduce variance
- Strong solver (Qwen3.5-397B-A17B): 3 invocations
- Judge (Kimi-K2.6): Scores per-criterion, provides feedback to orchestrator
- Feedback fed to challenger: "questions too easy"; "strong solver failed on this"; "quality verifier rejected for X reason"
- Challenger generates entirely new question from different angle (not rephrasing)

**Key Observation**: Initial CoT Self-Instruct questions were too easy (weak solver avg: 0.677, gap: 0.019). After agentic loop, weak avg dropped to 0.458 while strong improved to 0.772 (gap: 0.314). Mean 6.59 rounds per accepted item.

**Failure Modes Across 880 Pre-Acceptance Rounds**:
- 80% too easy (weak scored too high)
- 13% too hard (strong failed)
- 7% quality check failures

**Rubric Evolution**: Agent learns to move from high-level summary questions (easy) to specific algorithmic steps, ablation details, numerical claims (harder, more specific to paper).

### 3.2 Legal Reasoning Tasks

**Input**: 7.8k legal documents from Pile of Law; evaluated on PRBench-Legal (500 prompts) and PRBench-Legal-Hard (250 subset)

**Key Difference**: CoT Self-Instruct produced questions that were **too hard** (weak avg: 0.159, many zero scores), opposite problem from CS.

**Pipeline** (4 subagents):
1. **Extractor**: Reads legal document; extracts structured data (topics, facts, holdings); decides suitability
2. **Question + Rubric Writer**: Treats document as SOURCE OF LAW; invents new realistic client scenario; writes question in client voice; generates weighted rubric (15-25 criteria)
3. **Orchestrator**: Manages loop; applies two-layer decision policy
4. **Loop Judge**: Reads per-rollout solver patterns; returns structured verdict (weak_pattern, strong_pattern, gap_interpretation, rubric_concerns, grpo_suitability, verdict_reason, suggestion_for_writer)

**Acceptance Signal**: NOT a numeric gap threshold, but judge reasoning about per-rollout variance and learnability.

**Judge's Four-Layer Analysis**:
- Weak rollout variance: CoT all-zeros (std: 7.93) → Agentic usable range (std: 12.63)
- GRPO suitability verdict: CoT distribution (4.8% high / 41% medium / 45% low) → Agentic (52% high / 43% medium / 2% low)
- Rubric concerns (compound criteria, case-name pinning, capability concentration)
- Soft signals (saturation = strong nails 70-90% of rubric, hint of single-doctrine question)

**Evolution**: Loop pushes toward shorter, application-style questions (900 chars vs. 1.6k). Weak mean rises to 28.3% while strong stays ~70%, creating learnable signal.

### 3.3 Scientific Reasoning (Math, Physics)

**Input**: Grounded in Principia collection (existing math/physics benchmarks); tested on Principia bench

**Weak/Strong Setup**: Same LLM, but weak = base reasoning; strong = increased token budget (65,536), scaffolding, aggregation, privileged info

**Success Criteria**: Atomic single-sentence-answer questions where:
- Weak solver: ≤ 1 correct out of 4 runs
- Strong solver: ≥ 3 correct out of 4 runs

**Question Type Distribution** (1,000 annotated agentic examples):
- **Reasoning-dominant** (52%): Multi-step symbolic derivation, combinatorial analysis, probabilistic analysis, spectral/eigenvalue analysis, asymptotic analysis, proofs
- **Mixed** (27.8%): Physical modeling, algorithmic computation, data-driven inference
- **Knowledge** (20.2%): Direct formula/theorem application, factual recall

**Key Finding**: Challenging data transfers to easier domains. Training on Agentic data improved performance on both in-distribution agentic test (+4.4%) and out-of-distribution CoT test (+3.05%). Suggests robust reasoning skill learning.

---

## 4. Meta-Optimization of the Data Scientist Agent

### 4.1 Objective

Optimize the data scientist agent's **prompt and strategy** (outer loop) using the same evaluation criteria from the **inner loop** (data quality).

### 4.2 Method: Evolutionary Optimization

- **Population of Candidates**: Each candidate is a code diff relative to baseline prompt
- **Iteration Process**:
  1. Select parent from population via Boltzmann sampling (temperature T = 0.1, strongly favoring high-scoring candidates with exploration)
  2. Evaluate parent on minibatch of training papers; collect trajectories + solver scores
  3. Analyze trajectories: LLM agent reads full solver exchanges, writes root-cause analysis of systematic failures
  4. Implement modifications: Code-editing agent reads analysis + iteration history + current prompt, produces improved diff
  5. Re-evaluate both parent and mutant on held-out validation papers
  6. Accept/reject: Add mutant to population only if validation score strictly exceeds parent
  7. Summarize outcome to history log (for context in future analyses)
- **Parallelization**: Multiple iterations run concurrently with independent parent selections
- **Noise Handling**: Single-evaluation scores noisy; accepted candidates accumulate additional evaluations when re-sampled as parents; report averaged score across all re-evaluations

### 4.3 CS Paper Task Optimization Results

**Setup**:
- 50 training papers, 25 validation papers
- Success criterion: weak ≥ 65% AND weak_best ≥ 75%; strong ≥ 60% AND strong < 95%; gap ≥ 20pp
- Baseline prompt: 62.1% validation QA pass rate on 100 samples
- 233 total iterations; 126 accepted (hard stop after 6h timeout)

**Final Result**: **79.6% pass rate** (17.5 percentage point improvement)

**Discovered Prompt Modifications** (via automated analysis):

1. **Paper-specific insight enforcement**: Added instructions requiring questions test knowledge specific to the paper, not generic ML/CS knowledge. Self-test: "If a solver could answer without reading this paper, the question is too easy."

2. **Context leak prevention**: Strict rules requiring context to describe only problem domain/setup, never the solution. Self-test: "Could someone answer by rephrasing context sentences? If yes, rewrite."

3. **Positive-only rubric with weight capping**: Eliminated negative-weight criteria (counterintuitively, they misfired). All criteria now positive integer weights capped at 7 max, preventing single criterion dominance.

4. **Structured rubric format**: Enforced strict JSON format (integer weights, not strings like "+8"), eliminating parsing errors that caused evaluation failures.

---

## 5. Key Results Across Tasks

### 5.1 Computer Science Research

**RL Training** (GRPO on Qwen3.5-4B with Kimi-K2.6 judge):

| Setting | CoT Test (mean@3 best@3) | Agentic Test (mean@3 best@3) |
|---------|--------------------------|-------------------------------|
| Base 4B | 0.630 / 0.727 | 0.366 / 0.500 |
| RL on CoT | 0.758 / 0.853 | 0.484 / 0.631 |
| RL on Agentic | 0.774 / 0.894 | 0.632 / 0.768 |

**Interpretation**: Agentic data outperforms CoT data on both easier and harder test sets throughout training. Larger margin on harder Agentic test.

### 5.2 Legal Reasoning

**RL Training** (GRPO on Qwen3.5-4B, 2.8k examples per arm, evaluated on PRBench):

| Model | GPT-5 Judge (Legal / Legal-Hard) | Kimi Judge (Legal / Legal-Hard) |
|-------|-----------------------------------|--------------------------------|
| Base 4B | 0.280 / 0.167 | 0.245 / 0.145 |
| Base 397B | 0.404 / 0.277 | 0.358 / 0.226 |
| RL on CoT | 0.377 / 0.253 | 0.343 / 0.233 |
| RL on Agentic | 0.441 / 0.315 | 0.393 / 0.266 |

**Key**: 4B model trained on Agentic data outperforms much larger 397B baseline on both splits. +0.05-0.06 advantage over CoT on same 2.8k budget, same challenger, same source corpus.

### 5.3 Scientific Reasoning (Principia-based)

**RL Training** (GRPO on Qwen3.5-4B, 9k training examples per arm):

| Eval Subset | Base 4B | RL on CoT | RL on Agentic | RL on Combined (2x) |
|-------------|---------|----------|----------------|-------------------|
| Overall avg@8 | 68.66% | 71.08% (+2.42) | 71.86% (+3.20) | 71.36% (+2.70) |
| Agentic subset | 52.39% | 56.33% | 56.79% (+4.40) | 55.88% |
| CoT subset | 77.17% | 79.03% | 80.22% (+3.05) | 79.66% |

**Transfer Finding**: Agentic data improves performance even on CoT validation subset (+3.05% vs. +1.86% for CoT). Training on harder problems transfers to easier ones.

**Out-of-Distribution Principia Benchmark**: Agentic achieves best overall avg@8 (+1.04%) despite using half the data of Combined setting. Consistent gains across categories (RealMath +1.75%, SuperGPQA +0.82%).

**Token Efficiency**: Training reduces truncation rates from 23.75% to 4.09%, with ~50% of accuracy improvements attributed to more efficient reasoning within 65K token budget.

---

## 6. Practical Insights: "More Challenging" vs. "Just Right"

### The Paradox Across Tasks

**CS papers**: CoT questions too easy (gap 0.019) → Agentic widens gap (0.314)
**Legal reasoning**: CoT questions too hard (gap 0.558) → Agentic narrows gap (0.415)

**Yet both benefit**: Models trained on Agentic data outperform CoT-trained models on every held-out test.

### The Insight

The goal is not to maximize difficulty, but to make data **just right for hill-climbing**. The agentic loop achieves this by:
- **CS**: Pushing weak solver toward failure modes that discriminate reasoning (paper-specific insights, not generic knowledge)
- **Legal**: Pushing weak solver into learnable variance range (avoiding all-zero cliffs that provide no gradient signal)

---

## 7. Evaluation and Data Quality Measurement

### 7.1 Example-Level Evaluation

**Verifiable tasks** (code, math):
- Judge runs solver rollouts in parallel (weak: 4-5 runs; strong: 3-4 runs)
- Binary correctness per attempt
- Aggregate: success rate, error analysis per solver

**Non-verifiable tasks** (CS research, legal):
- Judge scores each criterion (rubric item) as binary (0 or 1)
- Per-criterion scores aggregated by capability tag
- Weak/strong gap computed at criterion and capability level
- Judge analyzes per-rollout patterns (are weak scores clustered? does strong saturate?), not just aggregates

### 7.2 Dataset-Level Analysis (Discussed as Future Work)

Current Autodata focuses on example-level optimization. Future directions:
- Diversity statistics (coverage across question types, domains, difficulty buckets)
- Batch-level learnings (generate N examples, analyze batch, inform next N)
- Dataset-model fit (does this dataset improve this model on this benchmark?)

### 7.3 Quality Verifier Heuristics

**CS research**:
- Context leakage: Can someone answer context+question without reasoning from the paper?
- Rubric coverage: 10-20 criteria; positive criteria require reasoning beyond context; negative criteria catch specific errors
- Question quality: REASONING (why, what-if, predict, decide) not RECALL (what, which, how many)?

**Legal reasoning**:
- Post-filter overrides: Body text > 1,000 chars or matches case-recap/exam-prompt regex → IMPROVE
- Suitability: Can the source document establish/apply a transferable legal principle?
- Rubric structure: Avoid compound criteria, case-name pinning, capability concentration

---

## 8. Relevance to Persona-Based Simulation

### Direct Applicability

The Autodata framework has **strong potential for persona-based simulation** (agents role-playing users of an educational product to find gaps):

1. **Grounding on source material**: The "Grounding" step naturally applies to educational content (course material, learning objectives, problem sets). The agent can generate student questions/confusion points grounded in specific lessons.

2. **Difficulty targeting via weak/strong solvers**: In an educational context:
   - **Weak solver** = target learner model (struggling learner, mid-curriculum learner)
   - **Strong solver** = ideal learner model (post-completion, advanced)
   - The loop generates questions/scenarios where weak learners would struggle but strong learners (or instructors) can succeed
   - This naturally targets the "learning zone" for effective instruction

3. **Evaluator feedback loop**: The Judge can assess:
   - Conceptual clarity: Is the question testing the intended learning objective?
   - Ambiguity: Are there multiple valid interpretations?
   - Scaffolding: Does the weak learner have entry points, or is the question binary (understand/don't)?
   - Transfer: Can the learner apply the concept to related problems?

4. **Coverage and diversity**: The framework can be extended (as noted in the paper) to dataset-level analysis:
   - Are we covering all learning objectives?
   - Are difficulty levels distributed appropriately?
   - Do generated scenarios test integration of multiple concepts or isolated recall?

5. **Rubric-based vs. verifiable evaluation**: Educational products often have:
   - Verifiable learning outcomes (e.g., "student can solve equation systems" → yes/no)
   - Soft learning outcomes (e.g., "student can explain reasoning" → rubric-scored)
   - The Agentic Self-Instruct method handles both via task-specific acceptance criteria and judge reasoning

### What the Paper Does NOT Cover

The paper focuses on:
- **Research/reasoning task synthesis** (CS research questions, legal queries, math problems)
- **Data generation for model training/RL** (not interactive tutoring or real-time scaffolding)

The paper does **not explicitly** discuss:
- Interactive dialogue generation (turn-taking, Socratic questioning)
- Real-time learner modeling (adapting questions mid-session to learner state)
- Engagement/motivation signals (e.g., detecting frustration, confusion)
- Multi-modal scenarios (quizzes, practical exercises, simulations)

However, the **meta-optimization loop** (Section 4) offers a path forward: the agent's prompts and strategy could be optimized for educational objectives (maximize learning gain per sample, reduce learner frustration, improve retention) rather than just "separate weak and strong solvers."

### Example Adaptation for Educational Use

```
Weak solver → Student model (learner at start of unit)
Strong solver → Student model (learner after unit completion)
Judge → Evaluates educational quality:
  - Tests learning objective, not generic knowledge
  - Appropriate scaffolding for weak learner
  - Gap = "student can attempt but needs reasoning" (not all-zero or trivial)
Challenger → Generates practice scenarios grounded in course material

Result → Curriculum-aligned practice data that targets the learning zone
```

---

## 9. Technical Details & Hyperparameters

### Models Used

- **Orchestrator/Challenger**: Kimi-K2.6 (or equiv. strong reasoning LLM)
- **Weak Solver**: Qwen3.5-4B (or smaller model)
- **Strong Solver**: Qwen3.5-397B-A17B or GPT-5 (or larger model, or same model with more compute)
- **Judge**: Kimi-K2.6 or GPT-5 (or LLM-as-judge framework)

### Computational Costs

**CS research task (10k papers → 2.8k accepted examples)**:
- ~6.59 rounds of generation + evaluation per accepted item
- Quality verifier applied at end (filters to 1.3k high-quality)
- Total: on order of 10k × 6.59 × (challenger + 2 solvers × 3 rollouts + judge) evaluations

**Legal reasoning (7.8k documents → 2.8k accepted)**:
- ~4.98 rounds per accepted item
- Quality verifier applied post-loop
- 5 weak + 3 strong rollouts per round per item

**Meta-optimization**:
- 233 total iterations (126 accepted) over 6 hours
- Each iteration: evaluation on minibatch of 50 training + 25 validation papers
- Evolutionary search with concurrent parents

### Token Budgets

- **Principia math/physics**: 65,536 token reasoning budget per attempt (long-form reasoning models)
- Base model truncation rate: 23.75%; Agentic-trained: 4.09%

---

## 10. Limitations and Future Directions

### Acknowledged Limitations

1. **Hacking/Gaming**: Agent may try to "cheat" the goal (e.g., weaken the weak solver prompt). Partially addressed via constraints; stronger safeguards needed.

2. **Semantic vs. syntactic difficulty**: Some generated questions may be overly tied to specific numbers from source material rather than testing generalizable reasoning.

3. **Dataset-level quality**: Current experiments focus on example-level optimization. Full dataset analysis (diversity, coverage, interaction effects) is future work.

4. **Scalability of meta-optimization**: Current approach uses evolutionary search; may not scale to very large prompt spaces.

### Future Research Directions

1. **More tasks and models**: Generalize across task types (single-turn, multi-turn, agentic tasks); evaluate on diverse models.

2. **Full dataset-level analysis**: Batch-level learnings; diversity statistics; coverage metrics.

3. **From self-improvement to co-improvement**: Combine agentic data generation with direct RL training of the agent itself (joint optimization rather than sequential).

4. **Human-in-the-loop**: Current work removes humans entirely. Co-research with human feedback likely a better path forward, especially for safety and capability alignment.

---

## 11. Summary for Implementation

**Key Takeaway for Synthetic Data Generation**:

Autodata demonstrates that treating data creation as an iterative, agentic process—where an LLM agent generates examples, evaluates them against task-specific signals, analyzes failures, and refines its generation strategy—produces higher-quality training data than static prompted generation.

**Core Loop**:
1. Generate example (grounded on source material)
2. Evaluate against weak+strong solvers and judge
3. Analyze failure modes (weak too strong? strong too weak? rubric unclear?)
4. Feed back to generator with specific, targeted feedback
5. Repeat until criteria met or budget exhausted

**Signal Design**:
- Weak/strong gap is not a target; it's a diagnostic
- For hard tasks, widen gap (push toward harder reasoning)
- For easy tasks, narrow gap into learnable range (avoid all-zeros)
- Judge reasons about rollout variance and gradient signal, not just averages

**Meta-Optimization**:
- Same evaluation criteria that guide inner loop can guide outer loop (prompt optimization)
- Discovered prompt improvements: specificity enforcement, leak prevention, rubric structure

**Data Quality** = (a) Correct reasoning required (weak solver cannot solve), (b) Clear evaluation criteria (strong solver succeeds), (c) Useful learning signal (variance in weak rollouts, meaningful gap).

---

## 12. References

Full citations available in the paper's References section (pages 16-17). Key prior work includes:
- Self-Instruct (Wang et al., 2023)
- CoT Self-Instruct (Yu et al., 2025)
- Grounded Self-Instruct (Lupidi et al., 2024; Yuan et al., 2025)
- Self-Challenging Language Models (Zhou et al., 2025)
- Prompt optimization & autoresearch methods (Karpathy 2026, Lee et al. 2026)
