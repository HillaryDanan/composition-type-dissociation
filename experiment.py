"""
APH Composition Type Dissociation Test - v3
============================================

LESSONS FROM v2:
- Complex question phrasing triggered chain-of-thought mode
- Meta-linguistic questions test different abilities
- Must use SIMPLE DIRECT questions at all levels

v3 APPROACH:
- Simple questions throughout: "Who [verbs] the X?" / "What does X do to Y?"
- Difficulty via STRUCTURAL complexity, not question complexity
- More interference (similar entities, reversal traps)
- Longer premises (memory load)

Scientific Principle: Isolate the variable of interest (composition type)
by controlling everything else (question format, vocabulary, etc.)
"""

import anthropic
import json
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import re


# =============================================================================
# NOVEL SYMBOLIC LANGUAGE - DESIGNED FOR INTERFERENCE
# =============================================================================

# Similar-sounding pairs to create discrimination difficulty
ENTITY_PAIRS = [
    ("blick", "bleck"),  # Similar
    ("dax", "dex"),
    ("wug", "wog"),
    ("zop", "zep"),
    ("tiv", "tev"),
    ("nib", "neb"),
    ("gorp", "gurp"),
    ("frem", "frim"),
]

# All entities (including similar-sounding ones for harder levels)
ENTITIES_EASY = ["blick", "dax", "wug", "zop", "tiv", "nib", "gorp", "frem"]
ENTITIES_HARD = ["blick", "bleck", "dax", "dex", "wug", "wog", "zop", "zep", 
                 "tiv", "tev", "nib", "neb", "gorp", "gurp", "frem", "frim"]

# Relations (distinct sounding)
RELATIONS = ["zorps", "bliffs", "groms", "plexes", "kwins", "frems", "telks", "norps"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CompositionType(Enum):
    ROLE_FILLER = "3b"
    RECURSIVE = "3c"


@dataclass
class Stimulus:
    """A single test stimulus."""
    condition: CompositionType
    complexity_level: int
    premise: str
    question: str
    correct_answer: str
    distractor_answers: list
    stimulus_id: str = ""
    n_entities: int = 0
    n_relations: int = 0
    has_reversal: bool = False
    
    def __post_init__(self):
        if not self.stimulus_id:
            self.stimulus_id = f"{self.condition.value}_L{self.complexity_level}_{random.randint(1000,9999)}"


@dataclass
class Response:
    """Model's response to a stimulus."""
    stimulus: Stimulus
    raw_response: str
    extracted_answer: str
    is_correct: bool
    response_time_ms: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    n_trials_per_condition: int = 10
    complexity_levels: list = field(default_factory=lambda: [1, 2, 3, 4, 5])
    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.0
    seed: int = 42


# =============================================================================
# STIMULUS GENERATION - v3 (SIMPLE QUESTIONS, STRUCTURAL DIFFICULTY)
# =============================================================================

class StimulusGenerator:
    """
    v3 Generator: Simple direct questions, structural difficulty scaling.
    
    DIFFICULTY SCALING:
    - Level 1: 2 entities, 1 relation, simple
    - Level 2: 4 entities, 2 relations, interference
    - Level 3: 5 entities, 3 relations, reversal trap
    - Level 4: 6 entities, 4 relations, chain + reversal
    - Level 5: 8 entities (similar-sounding), 5 relations, max interference
    
    QUESTION FORMAT (CONSTANT):
    "Who [verbs] the [entity]?" or "What does [entity] do to [entity]?"
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        
    def _get_entities(self, n: int, hard: bool = False) -> list:
        """Sample n unique entities."""
        pool = ENTITIES_HARD if hard else ENTITIES_EASY
        return random.sample(pool, min(n, len(pool)))
    
    def _get_relations(self, n: int) -> list:
        """Sample n unique relations."""
        return random.sample(RELATIONS, min(n, len(RELATIONS)))

    # =========================================================================
    # TYPE 3b: ROLE-FILLER (Variable binding, no recursion)
    # =========================================================================
    
    def generate_role_filler_stimulus(self, level: int) -> Stimulus:
        """
        Role-filler stimuli with increasing interference.
        
        Structure is always flat (no embedding).
        Difficulty comes from:
        - Number of triplets to track
        - Similar entities (discrimination)
        - Reversal traps (A->B then B->A)
        """
        
        if level == 1:
            return self._rf_level_1()
        elif level == 2:
            return self._rf_level_2()
        elif level == 3:
            return self._rf_level_3()
        elif level == 4:
            return self._rf_level_4()
        elif level == 5:
            return self._rf_level_5()
        else:
            raise ValueError(f"Invalid level: {level}")
    
    def _rf_level_1(self) -> Stimulus:
        """2 entities, 1 relation. Baseline."""
        e = self._get_entities(2)
        r = self._get_relations(1)[0]
        
        premise = f"The {e[0]} {r} the {e[1]}."
        question = f"Who {r} the {e[1]}?"
        correct = f"the {e[0]}"
        distractors = [f"the {e[1]}", "nobody", f"both {e[0]} and {e[1]}"]
        
        return Stimulus(
            condition=CompositionType.ROLE_FILLER,
            complexity_level=1,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=2,
            n_relations=1
        )
    
    def _rf_level_2(self) -> Stimulus:
        """4 entities, 2 relations. Two independent triplets."""
        e = self._get_entities(4)
        r = self._get_relations(2)
        
        premise = f"The {e[0]} {r[0]} the {e[1]}. The {e[2]} {r[1]} the {e[3]}."
        
        # Ask about second triplet (must track both)
        question = f"Who {r[1]} the {e[3]}?"
        correct = f"the {e[2]}"
        distractors = [f"the {e[0]}", f"the {e[1]}", f"the {e[3]}"]
        
        return Stimulus(
            condition=CompositionType.ROLE_FILLER,
            complexity_level=2,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=4,
            n_relations=2
        )
    
    def _rf_level_3(self) -> Stimulus:
        """5 entities, 3 relations, with REVERSAL trap."""
        e = self._get_entities(5)
        r = self._get_relations(3)
        
        # A->B, B->A (reversal!), C->D
        premise = (
            f"The {e[0]} {r[0]} the {e[1]}. "
            f"The {e[1]} {r[1]} the {e[0]}. "  # REVERSAL
            f"The {e[2]} {r[2]} the {e[3]}."
        )
        
        # Ask about the reversed relationship (trap!)
        question = f"Who {r[1]} the {e[0]}?"
        correct = f"the {e[1]}"  # NOT e[0]!
        distractors = [f"the {e[0]}", f"the {e[2]}", f"the {e[3]}"]
        
        return Stimulus(
            condition=CompositionType.ROLE_FILLER,
            complexity_level=3,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=5,
            n_relations=3,
            has_reversal=True
        )
    
    def _rf_level_4(self) -> Stimulus:
        """6 entities, 4 relations, chain + reversal."""
        e = self._get_entities(6)
        r = self._get_relations(4)
        
        # Chain: A->B->C->D, plus reversal B->A
        premise = (
            f"The {e[0]} {r[0]} the {e[1]}. "
            f"The {e[1]} {r[1]} the {e[2]}. "
            f"The {e[2]} {r[2]} the {e[3]}. "
            f"The {e[1]} {r[3]} the {e[0]}."  # REVERSAL
        )
        
        # Ask about middle of chain
        question = f"Who {r[1]} the {e[2]}?"
        correct = f"the {e[1]}"
        distractors = [f"the {e[0]}", f"the {e[2]}", f"the {e[3]}"]
        
        return Stimulus(
            condition=CompositionType.ROLE_FILLER,
            complexity_level=4,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=6,
            n_relations=4,
            has_reversal=True
        )
    
    def _rf_level_5(self) -> Stimulus:
        """8 entities (similar-sounding!), 5 relations, max interference."""
        # Use similar-sounding pairs for max confusion
        e = self._get_entities(8, hard=True)
        r = self._get_relations(5)
        
        # Complex network with reversals
        premise = (
            f"The {e[0]} {r[0]} the {e[1]}. "
            f"The {e[2]} {r[1]} the {e[3]}. "
            f"The {e[1]} {r[2]} the {e[4]}. "
            f"The {e[3]} {r[3]} the {e[5]}. "
            f"The {e[4]} {r[4]} the {e[0]}."  # Cycle back
        )
        
        # Ask about a specific binding in the middle
        question = f"Who {r[2]} the {e[4]}?"
        correct = f"the {e[1]}"
        distractors = [f"the {e[0]}", f"the {e[3]}", f"the {e[4]}"]
        
        return Stimulus(
            condition=CompositionType.ROLE_FILLER,
            complexity_level=5,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=8,
            n_relations=5,
            has_reversal=True
        )

    # =========================================================================
    # TYPE 3c: RECURSIVE (Center embedding)
    # =========================================================================
    
    def generate_recursive_stimulus(self, level: int) -> Stimulus:
        """
        Recursive stimuli with increasing embedding depth.
        
        Structure uses center-embedding (hardest for humans too).
        Question is always simple: "Who [main-verb] the [final-entity]?"
        
        This requires tracking the OUTERMOST subject through nested clauses.
        """
        
        if level == 1:
            return self._rc_level_1()
        elif level == 2:
            return self._rc_level_2()
        elif level == 3:
            return self._rc_level_3()
        elif level == 4:
            return self._rc_level_4()
        elif level == 5:
            return self._rc_level_5()
        else:
            raise ValueError(f"Invalid level: {level}")
    
    def _rc_level_1(self) -> Stimulus:
        """Depth 0: Simple sentence (matched to RF level 1)."""
        e = self._get_entities(2)
        r = self._get_relations(1)[0]
        
        premise = f"The {e[0]} {r} the {e[1]}."
        question = f"Who {r} the {e[1]}?"
        correct = f"the {e[0]}"
        distractors = [f"the {e[1]}", "nobody", f"both"]
        
        return Stimulus(
            condition=CompositionType.RECURSIVE,
            complexity_level=1,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=2,
            n_relations=1
        )
    
    def _rc_level_2(self) -> Stimulus:
        """Depth 1: One embedded clause."""
        e = self._get_entities(3)
        r = self._get_relations(2)
        
        # "The A [that R1 the B] R2 the C"
        premise = f"The {e[0]} that {r[0]} the {e[1]} {r[1]} the {e[2]}."
        
        # Who does the MAIN verb?
        question = f"Who {r[1]} the {e[2]}?"
        correct = f"the {e[0]}"
        distractors = [f"the {e[1]}", f"the {e[2]}", "nobody"]
        
        return Stimulus(
            condition=CompositionType.RECURSIVE,
            complexity_level=2,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=3,
            n_relations=2
        )
    
    def _rc_level_3(self) -> Stimulus:
        """Depth 2: Two levels of embedding."""
        e = self._get_entities(4)
        r = self._get_relations(3)
        
        # "The A [that R1 the B [that R2 the C]] R3 the D"
        premise = f"The {e[0]} that {r[0]} the {e[1]} that {r[1]} the {e[2]} {r[2]} the {e[3]}."
        
        question = f"Who {r[2]} the {e[3]}?"
        correct = f"the {e[0]}"
        distractors = [f"the {e[1]}", f"the {e[2]}", f"the {e[3]}"]
        
        return Stimulus(
            condition=CompositionType.RECURSIVE,
            complexity_level=3,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=4,
            n_relations=3
        )
    
    def _rc_level_4(self) -> Stimulus:
        """Depth 3: Three levels of embedding."""
        e = self._get_entities(5)
        r = self._get_relations(4)
        
        premise = (
            f"The {e[0]} that {r[0]} the {e[1]} that {r[1]} the {e[2]} "
            f"that {r[2]} the {e[3]} {r[3]} the {e[4]}."
        )
        
        question = f"Who {r[3]} the {e[4]}?"
        correct = f"the {e[0]}"
        distractors = [f"the {e[1]}", f"the {e[2]}", f"the {e[3]}"]
        
        return Stimulus(
            condition=CompositionType.RECURSIVE,
            complexity_level=4,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=5,
            n_relations=4
        )
    
    def _rc_level_5(self) -> Stimulus:
        """Depth 4: Four levels (very hard even for humans)."""
        e = self._get_entities(6, hard=True)  # Similar-sounding entities
        r = self._get_relations(5)
        
        premise = (
            f"The {e[0]} that {r[0]} the {e[1]} that {r[1]} the {e[2]} "
            f"that {r[2]} the {e[3]} that {r[3]} the {e[4]} {r[4]} the {e[5]}."
        )
        
        question = f"Who {r[4]} the {e[5]}?"
        correct = f"the {e[0]}"
        distractors = [f"the {e[1]}", f"the {e[3]}", f"the {e[4]}"]
        
        return Stimulus(
            condition=CompositionType.RECURSIVE,
            complexity_level=5,
            premise=premise,
            question=question,
            correct_answer=correct,
            distractor_answers=distractors,
            n_entities=6,
            n_relations=5
        )
    
    def generate_stimulus_set(
        self, 
        n_per_level: int = 10,
        levels: list = [1, 2, 3, 4, 5]
    ) -> dict:
        """Generate complete stimulus set."""
        
        stimuli = {
            CompositionType.ROLE_FILLER: {level: [] for level in levels},
            CompositionType.RECURSIVE: {level: [] for level in levels}
        }
        
        for level in levels:
            for _ in range(n_per_level):
                stimuli[CompositionType.ROLE_FILLER][level].append(
                    self.generate_role_filler_stimulus(level)
                )
                stimuli[CompositionType.RECURSIVE][level].append(
                    self.generate_recursive_stimulus(level)
                )
        
        return stimuli


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Runs the experiment with consistent prompt format."""
    
    def __init__(self, api_key: str, config: ExperimentConfig):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config = config
        self.generator = StimulusGenerator(seed=config.seed)
        self.results = []
        
    def _format_prompt(self, stimulus: Stimulus) -> tuple:
        """
        Format prompt with STRONG answer format enforcement.
        
        Key: Make it very clear we want just a letter.
        """
        
        options = [stimulus.correct_answer] + stimulus.distractor_answers
        random.shuffle(options)
        options_str = "\n".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)])
        
        correct_letter = chr(65 + options.index(stimulus.correct_answer))
        
        # Strong format enforcement
        prompt = f"""Read the statement and answer the question.

STATEMENT: {stimulus.premise}

QUESTION: {stimulus.question}

{options_str}

Answer with just the letter (A, B, C, or D):"""

        return prompt, correct_letter
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer letter."""
        response = response.strip().upper()
        # Look for first letter A-D
        for char in response:
            if char in 'ABCD':
                return char
        return ""
    
    def run_single_trial(self, stimulus: Stimulus) -> Response:
        """Run single trial."""
        
        prompt, correct_letter = self._format_prompt(stimulus)
        
        start_time = time.time()
        
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=5,  # Even shorter - just need a letter
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_response = message.content[0].text
        except Exception as e:
            raw_response = f"ERROR: {str(e)}"
        
        response_time = (time.time() - start_time) * 1000
        
        extracted = self._extract_answer(raw_response)
        is_correct = extracted == correct_letter
        
        return Response(
            stimulus=stimulus,
            raw_response=raw_response,
            extracted_answer=extracted,
            is_correct=is_correct,
            response_time_ms=response_time
        )
    
    def run_experiment(self, verbose: bool = True) -> list:
        """Run full experiment."""
        
        if verbose:
            print("=" * 60)
            print("APH COMPOSITION TYPE DISSOCIATION TEST v3")
            print("Simple questions, structural difficulty")
            print("=" * 60)
            print(f"Model: {self.config.model}")
            print(f"Trials per condition per level: {self.config.n_trials_per_condition}")
            print(f"Complexity levels: {self.config.complexity_levels}")
            print("=" * 60)
        
        stimuli = self.generator.generate_stimulus_set(
            n_per_level=self.config.n_trials_per_condition,
            levels=self.config.complexity_levels
        )
        
        all_responses = []
        
        for condition in [CompositionType.ROLE_FILLER, CompositionType.RECURSIVE]:
            if verbose:
                print(f"\nRunning condition: {condition.value} ({condition.name})")
            
            for level in self.config.complexity_levels:
                if verbose:
                    print(f"  Level {level}: ", end="", flush=True)
                
                level_correct = 0
                for stimulus in stimuli[condition][level]:
                    response = self.run_single_trial(stimulus)
                    all_responses.append(response)
                    
                    if response.is_correct:
                        level_correct += 1
                    
                    if verbose:
                        print("✓" if response.is_correct else "✗", end="", flush=True)
                    
                    time.sleep(0.1)
                
                if verbose:
                    acc = level_correct / len(stimuli[condition][level])
                    print(f" ({acc:.0%})")
        
        self.results = all_responses
        return all_responses


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import os
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
    
    config = ExperimentConfig(
        n_trials_per_condition=10,
        complexity_levels=[1, 2, 3, 4, 5],
        model="claude-haiku-4-5-20251001",
        temperature=0.0,
        seed=42
    )
    
    runner = ExperimentRunner(api_key=api_key, config=config)
    results = runner.run_experiment(verbose=True)
    
    results_data = [
        {
            "stimulus_id": r.stimulus.stimulus_id,
            "condition": r.stimulus.condition.value,
            "complexity_level": r.stimulus.complexity_level,
            "premise": r.stimulus.premise,
            "question": r.stimulus.question,
            "correct_answer": r.stimulus.correct_answer,
            "raw_response": r.raw_response,
            "extracted_answer": r.extracted_answer,
            "is_correct": r.is_correct,
            "response_time_ms": r.response_time_ms,
            "n_entities": r.stimulus.n_entities,
            "n_relations": r.stimulus.n_relations,
            "has_reversal": r.stimulus.has_reversal
        }
        for r in results
    ]
    
    with open("results_raw.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to results_raw.json")
