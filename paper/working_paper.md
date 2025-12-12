# Composition Type Dissociation in Large Language Models: Preliminary Evidence for the Abstraction Primitive Hypothesis

**Hillary Danan, PhD**

*Working Paper - December 2025*

---

## Abstract

The Abstraction Primitive Hypothesis (APH) predicts that different composition types engage distinct computational mechanisms with different scaling properties. Specifically, recursive composition (Type 3c) should degrade faster with complexity than role-filler composition (Type 3b) in systems lacking iterative symbol-manipulation capabilities. We tested this prediction using a novel symbolic language paradigm that controls for training data confounds. Results from 500 trials with Claude Haiku showed significant dissociation: role-filler composition remained robust (98% accuracy) while recursive composition degraded substantially with embedding depth (100% → 68%). The slope difference was significant (p < 0.001, permutation test) with medium effect size (d = 0.63). These findings provide preliminary empirical support for the APH's claim that composition types are computationally distinct, with implications for understanding the architectural limitations of transformer-based language models.

**Keywords**: compositionality, recursion, large language models, cognitive architecture, abstraction

---

## 1. Introduction

What computational mechanisms underlie intelligent behavior? The Abstraction Primitive Hypothesis (APH; Danan, 2024) proposes that intelligence emerges from recursive interaction between symbol formation and compositional structure. A key claim of this framework is that **not all composition is computationally equivalent**.

The APH distinguishes four composition types:

1. **Concatenative (3a)**: Simple joining (A + B → AB)
2. **Role-filler (3b)**: Variable binding to structural slots (AGENT(x) + ACTION(y))
3. **Recursive (3c)**: Nested structures (A contains [B contains C])
4. **Analogical (3d)**: Structural mapping across domains

This taxonomy generates a testable prediction: systems that lack genuine recursive processing mechanisms should show **differential performance** on role-filler versus recursive composition tasks, even when controlling for surface complexity.

### 1.1 The Problem with Testing LLMs

Large language models (LLMs) present a challenge for testing compositional capabilities. Trained on billions of sentences, they may solve compositional tasks through statistical pattern matching rather than structural computation (Lake & Baroni, 2018; Fodor & Pylyshyn, 1988). Any test using natural language conflates structural processing with retrieval of memorized patterns.

### 1.2 The Present Study

We developed a novel symbolic language paradigm inspired by Berko's (1958) Wug Test. By using invented words with no semantic content, we force the model to process structure rather than retrieve associations. We then systematically varied:

- **Role-filler complexity**: Number of variable bindings and interference
- **Recursive complexity**: Depth of center-embedding

If the APH is correct, recursive depth should produce steeper degradation than role-filler complexity.

---

## 2. Method

### 2.1 Participants

We tested Claude Haiku (claude-haiku-4-5-20251001; Anthropic, 2024), a transformer-based LLM. Temperature was set to 0.0 for deterministic responses.

### 2.2 Materials

#### Novel Symbolic Language

**Entities** (n=16): blick, bleck, dax, dex, wug, wog, zop, zep, tiv, tev, nib, neb, gorp, gurp, frem, frim

Similar-sounding pairs (e.g., blick/bleck) were included for interference conditions at higher complexity levels.

**Relations** (n=8): zorps, bliffs, groms, plexes, kwins, frems, telks, norps

#### Condition 3b: Role-Filler Composition

Stimuli consisted of flat structures (no embedding) with increasing interference:

| Level | Example | Difficulty Source |
|-------|---------|-------------------|
| 1 | "The dax kwins the blick." | Baseline (2 entities, 1 relation) |
| 2 | "The zop bliffs the dax. The nib kwins the blick." | Multiple bindings (4 entities) |
| 3 | "The blick zorps the frem. The frem kwins the blick. The gorp bliffs the dax." | Reversal trap (A→B, B→A) |
| 4 | "The gorp groms the dax. The dax frems the zop. The zop plexes the tiv. The dax norps the gorp." | Chain + reversal |
| 5 | (8 similar-sounding entities, 5 relations) | Maximum interference |

#### Condition 3c: Recursive Composition

Stimuli used center-embedding with increasing depth:

| Level | Depth | Example |
|-------|-------|---------|
| 1 | 0 | "The frem bliffs the tiv." |
| 2 | 1 | "The dax that frems the tiv kwins the wug." |
| 3 | 2 | "The dax that kwins the blick that zorps the nib bliffs the frem." |
| 4 | 3 | "The gorp that frems the wug that groms the zop that bliffs the frem telks the dax." |
| 5 | 4 | "The wog that telks the tiv that groms the neb that frems the dex that kwins the dax bliffs the zep." |

#### Question Format

All questions used identical simple format: "Who [verbs] the [entity]?"

This controls for question complexity across conditions. Responses were four-choice multiple choice.

### 2.3 Procedure

Each trial presented:
1. Statement (premise)
2. Question
3. Four answer options (A-D)

The model was instructed to respond with only the letter. We ran 50 trials per level per condition (500 total per condition; 1000 trials overall).

### 2.4 Analysis

Primary analysis compared degradation slopes across conditions using:
1. Linear regression of accuracy on complexity level
2. Permutation test for slope difference (n=1000 permutations)
3. Effect size (Cohen's d)

---

## 3. Results

### 3.1 Accuracy by Condition and Level

| Level | Role-Filler (3b) | Recursive (3c) | Difference |
|-------|------------------|----------------|------------|
| 1 | 100% | 100% | 0% |
| 2 | 100% | 84% | 16% |
| 3 | 100% | 70% | 30% |
| 4 | 100% | 76% | 24% |
| 5 | 92% | 68% | 24% |
| **Mean** | **98.4%** | **79.6%** | **18.8%** |

### 3.2 Degradation Slopes

- Role-filler slope: -0.016 (essentially flat)
- Recursive slope: -0.072 (declining)
- Slope difference: 0.056

### 3.3 Statistical Tests

| Test | Statistic | p-value |
|------|-----------|---------|
| Z-test (slope comparison) | 2.27 | 0.012 |
| Permutation test (n=1000) | 0/1000 ≥ observed | **< 0.001** |
| Cohen's d | 0.63 | (medium effect) |

The permutation test found that **zero** of 1000 random label shuffles produced a slope difference as large or larger than observed.

### 3.4 Pattern Analysis

The recursive condition showed clear degradation from Level 1 (100%) to Level 5 (68%), a 32 percentage point drop. In contrast, role-filler accuracy remained at or near ceiling until Level 5 (92%).

The slight non-monotonicity in recursive (Level 4 = 76% > Level 3 = 70%) is within expected sampling variance and does not affect the overall pattern.

---

## 4. Discussion

### 4.1 Support for the APH Prediction

The results provide **preliminary support** for the APH prediction that composition types dissociate. Specifically:

1. **Role-filler composition was robust**: Even with reversal traps and similar-sounding entities, accuracy remained near ceiling (98.4%).

2. **Recursive composition degraded with depth**: Center-embedding beyond depth 1 produced substantial accuracy drops (84% → 68%).

3. **The dissociation was significant**: p < 0.001 with medium effect size.

This pattern is consistent with the hypothesis that LLMs can perform variable binding (role-filler) through attention mechanisms but lack the iterative stack maintenance required for deep recursive processing.

### 4.2 Relation to Prior Work

Our findings align with Lake & Baroni's (2018) demonstration that sequence-to-sequence models fail at systematic compositional generalization. However, we extend this work by:

1. **Distinguishing composition types**: Not all composition fails equally
2. **Using novel vocabulary**: Controlling for training distribution
3. **Testing a modern LLM**: Claude Haiku represents current capabilities

The recursive difficulty also parallels human psycholinguistic findings. Gibson (1998) showed that center-embedding creates processing difficulty that scales with depth, attributable to working memory limitations. Our results suggest LLMs face analogous (though possibly distinct) constraints.

### 4.3 Alternative Explanations

Several alternative explanations warrant consideration:

**Working memory**: The recursive difficulty might reflect general capacity limits rather than composition-specific mechanisms. However, this doesn't explain why role-filler stimuli with comparable information load remained easy.

**Training frequency**: Center-embedded structures may be rare in training data. But this would predict difficulty with all rare structures, not specifically recursive ones.

**Syntactic parsing**: The difficulty might be parsing relative clauses rather than recursive composition per se. Future work should test other recursive structures (e.g., mathematical expressions).

### 4.4 Limitations

1. **Ceiling effects**: Role-filler accuracy was near ceiling, preventing measurement of its degradation slope. Harder stimuli are needed.

2. **Single model**: We tested only Claude Haiku. Cross-architecture replication is essential.

3. **No human baseline**: We don't know if humans show the same pattern (they might, given Gibson's results on center-embedding).

4. **Limited recursion types**: We tested only center-embedding. Other recursive structures (right-branching, mathematical nesting) should be examined.

### 4.5 Implications

If replicated, these findings suggest:

1. **Architectural limitations**: Transformer attention may implement role-filler binding effectively but not true recursion.

2. **Training won't suffice**: The limitation may be architectural, not data-driven—more training won't fix it.

3. **Hybrid architectures**: Combining transformers with explicit stack mechanisms might be needed for human-like recursion.

---

## 5. Conclusion

We found significant dissociation between role-filler and recursive composition in an LLM, as predicted by the Abstraction Primitive Hypothesis. Recursive composition degraded substantially with embedding depth (100% → 68%) while role-filler composition remained robust (98%). This preliminary evidence supports the APH's claim that composition types are computationally distinct and suggests that current LLM architectures may lack mechanisms for genuine recursive processing.

---

## References

Berko, J. (1958). The child's learning of English morphology. *Word*, 14(2-3), 150-177.

Chomsky, N. (1957). *Syntactic structures*. Mouton.

Danan, H. (2024). Abstraction is all you need. https://github.com/HillaryDanan/abstraction-intelligence

Fodor, J. A., & Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: A critical analysis. *Cognition*, 28(1-2), 3-71.

Gibson, E. (1998). Linguistic complexity: Locality of syntactic dependencies. *Cognition*, 68(1), 1-76.

Lake, B., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *Proceedings of ICML*.

---

## Appendix A: Sample Stimuli

### Role-Filler Level 3 (Reversal Trap)

**Premise**: "The blick zorps the frem. The frem kwins the blick. The gorp bliffs the dax."

**Question**: "Who kwins the blick?"

**Options**: A) the blick, B) the frem, C) the gorp, D) the dax

**Correct**: B) the frem

### Recursive Level 4 (Depth 3)

**Premise**: "The gorp that frems the wug that groms the zop that bliffs the frem telks the dax."

**Question**: "Who telks the dax?"

**Options**: A) the gorp, B) the wug, C) the zop, D) the frem

**Correct**: A) the gorp

---

## Appendix B: Analysis Code

Available at: https://github.com/HillaryDanan/composition-type-dissociation

---

*Working paper. Comments welcome.*
