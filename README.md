# Composition Type Dissociation Test

**Empirical test of the Abstraction Primitive Hypothesis (APH) prediction that recursive composition (Type 3c) and role-filler composition (Type 3b) engage different computational mechanisms.**

[![Status](https://img.shields.io/badge/status-preliminary%20support-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)]()

---

## Key Finding

**The APH prediction is supported**: Recursive composition degrades significantly faster than role-filler composition as structural complexity increases.

| Metric | Value |
|--------|-------|
| Slope difference | 0.056 |
| p-value (permutation) | **< 0.001** |
| Cohen's d | **0.63** (medium) |
| N per condition | 250 (50 trials × 5 levels) |

```
Accuracy
100% ─●━━━━●━━━━●━━━━●━━━━○     Role-Filler (3b)
 90% ─                              
 80% ─     ○                    
 70% ─          ○    ○         
 60% ─               ↘    ○     Recursive (3c)
     ─────────────────────────
      1    2    3    4    5     Complexity Level
```

---

## Theoretical Background

### The Abstraction Primitive Hypothesis (APH)

The APH (Danan, 2024) proposes that intelligence emerges from recursive interaction between symbol formation and compositional structure. Critically, **not all composition is the same**:

| Type | Structure | Example | Mechanism |
|------|-----------|---------|-----------|
| **3a** | Concatenative | A + B → AB | Simple joining |
| **3b** | Role-filler | R(x) + S(y) | Variable binding |
| **3c** | Recursive | A[B[C]] | Stack maintenance |
| **3d** | Analogical | Structure mapping | Relational transfer |

### The Prediction

If 3b and 3c engage different computational mechanisms:

- **Role-filler (3b)**: Requires binding variables to slots. This may be parallelizable in attention mechanisms.
- **Recursive (3c)**: Requires maintaining a parse stack through nested structures. This may require iterative processing that transformers lack.

**Prediction**: Systems without true recursive mechanisms should show differential degradation—robust 3b, degraded 3c.

### Prior Literature

- **Fodor & Pylyshyn (1988)**: Systematicity and compositionality as tests of cognitive architecture
- **Lake & Baroni (2018)**: Compositional generalization failures in seq2seq models
- **Berko (1958)**: Wug test methodology for testing productive rules
- **Gibson (1998)**: Center-embedding difficulty scales with depth in humans
- **Chomsky (1957)**: Recursive structure as core to linguistic competence

---

## Experimental Design

### The Challenge: Training Data Confounds

LLMs have seen billions of sentences. Testing with natural language conflates structural processing with statistical pattern matching.

### Solution: Novel Symbolic Language

Inspired by Berko's Wug Test, we use:

- **Novel entities**: blick, dax, wug, zop, tiv, nib, gorp, frem (+ similar-sounding variants)
- **Novel relations**: zorps, bliffs, groms, plexes, kwins, frems, telks, norps
- **No semantic content**: Forces structural processing

### Condition 3b: Role-Filler Composition

Complexity scales via **number of bindings** and **interference** (no embedding):

| Level | Structure | Difficulty Source |
|-------|-----------|-------------------|
| 1 | A → B | Baseline |
| 2 | A → B, C → D | Multiple bindings |
| 3 | A → B, B → A, C → D | **Reversal trap** |
| 4 | A → B → C → D + reversal | Chain + trap |
| 5 | 8 similar-sounding entities | Max interference |

**Example Level 3**:
> "The blick zorps the frem. The frem kwins the blick. The gorp bliffs the dax."
> 
> Question: "Who kwins the blick?" → Answer: **the frem**

### Condition 3c: Recursive Composition

Complexity scales via **embedding depth** (center-embedding):

| Level | Depth | Structure |
|-------|-------|-----------|
| 1 | 0 | A verbs B |
| 2 | 1 | A [that verbs B] verbs C |
| 3 | 2 | A [that verbs B [that verbs C]] verbs D |
| 4 | 3 | Three levels |
| 5 | 4 | Four levels |

**Example Level 4**:
> "The gorp that frems the wug that groms the zop that bliffs the frem telks the dax."
> 
> Question: "Who telks the dax?" → Answer: **the gorp**

### Critical Design Features

1. **Simple questions throughout**: "Who [verbs] the [entity]?" (no meta-linguistic phrasing)
2. **Matched baselines**: Level 1 is identical across conditions
3. **Multiple choice format**: Forces discrete answer, prevents verbose reasoning
4. **Novel vocabulary**: Bypasses training data shortcuts

---

## Results

### Main Finding (n=50 per cell)

| Level | Role-Filler | Recursive | Δ |
|-------|-------------|-----------|---|
| 1 | 100% | 100% | 0% |
| 2 | 100% | 84% | +16% |
| 3 | 100% | 70% | +30% |
| 4 | 100% | 76% | +24% |
| 5 | 92% | 68% | +24% |

### Statistical Analysis

| Test | Result |
|------|--------|
| RF slope | -0.016 |
| RC slope | -0.072 |
| Slope difference | 0.056 |
| Z-statistic | 2.27 |
| p (parametric) | 0.012 |
| **p (permutation, n=1000)** | **< 0.001** |
| Cohen's d | 0.63 (medium) |

### Interpretation

The permutation test found **0 out of 1000** random label shuffles produced a slope difference as large as observed. This is robust evidence that the conditions differ.

---

## Replication

### Requirements

```bash
pip install anthropic>=0.40.0 numpy>=1.26.0 scipy>=1.11.0
```

### Run Experiment

```bash
export ANTHROPIC_API_KEY='your-key'

# Quick test (n=10)
python3 run_experiment.py --n-trials 10 --output ./results

# Full replication (n=50)
python3 run_experiment.py --n-trials 50 --seed 123 --output ./results

# Different seed for independent replication
python3 run_experiment.py --n-trials 50 --seed 456 --output ./results
```

### Output Files

- `results_[timestamp].json` - Raw trial data
- `analysis_[timestamp].json` - Statistical analysis
- `visualization_[timestamp].html` - Interactive chart

---

## Limitations & Alternative Explanations

### Not Yet Ruled Out

1. **Working memory hypothesis**: Center-embedding is hard for humans too (Gibson, 1998). The effect might reflect general memory limits, not composition-specific mechanisms.

2. **Training frequency**: Center-embedded structures may be rare in training data. The effect could reflect familiarity, not capability.

3. **Syntactic parsing**: The difficulty might be syntactic (parsing relative clauses) rather than compositional (recursive structure per se).

### Role-Filler Ceiling Effect

Role-filler accuracy remained near ceiling (98% mean). We're primarily measuring recursive degradation, not comparing two degradation curves. Harder role-filler stimuli needed.

### Single Model

Tested only Claude Haiku. Replication across architectures (GPT-4, Llama, etc.) needed.

---

## Next Steps

### Immediate
- [ ] Test additional models (Sonnet, GPT-4, Llama)
- [ ] Add human baseline for calibration
- [ ] Increase role-filler difficulty to break ceiling

### Future
- [ ] Right-branching vs. center-embedding comparison
- [ ] Fine-tuning experiments (train on recursive structures)
- [ ] Mechanistic interpretability (where does recursion fail?)

---

## Citation

If you use this work, please cite:

```bibtex
@software{composition_type_dissociation,
  author = {Danan, Hillary},
  title = {Composition Type Dissociation Test: Empirical Test of the APH},
  year = {2025},
  url = {https://github.com/HillaryDanan/composition-type-dissociation}
}
```

### Related Work

```bibtex
@article{danan2025aph,
  author = {Danan, Hillary},
  title = {Abstraction Is All You Need},
  year = {2025},
  url = {https://github.com/HillaryDanan/abstraction-intelligence}
}
```

---

## License

MIT

---

## Author

**Hillary Danan, PhD** · Cognitive Neuroscience

*"Abstraction is all you need ;)"*
