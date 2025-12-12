#!/usr/bin/env python3
"""
APH Composition Type Dissociation Test - v3 Runner
===================================================

v3: Simple questions, structural difficulty
"""

import argparse
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment import ExperimentConfig, ExperimentRunner
from analysis import APHAnalyzer, create_visualization, NumpyEncoder


def print_header():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         APH COMPOSITION TYPE DISSOCIATION TEST v3               ║
║         Simple questions, structural difficulty                 ║
╠══════════════════════════════════════════════════════════════════╣
║  PREDICTION: Recursive composition (3c) degrades faster than    ║
║              role-filler composition (3b) with complexity       ║
║                                                                  ║
║  v3 FIX: Direct questions at all levels (no meta-linguistic)    ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def run_experiment(api_key: str, config: ExperimentConfig, output_dir: str = "."):
    """Run the full experiment pipeline."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[1/4] Running experiment...")
    runner = ExperimentRunner(api_key=api_key, config=config)
    results = runner.run_experiment(verbose=True)
    
    print("\n[2/4] Saving raw results...")
    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
    
    results_data = [
        {
            "stimulus_id": r.stimulus.stimulus_id,
            "condition": r.stimulus.condition.value,
            "condition_name": r.stimulus.condition.name,
            "complexity_level": int(r.stimulus.complexity_level),
            "premise": r.stimulus.premise,
            "question": r.stimulus.question,
            "correct_answer": r.stimulus.correct_answer,
            "distractor_answers": r.stimulus.distractor_answers,
            "raw_response": r.raw_response,
            "extracted_answer": r.extracted_answer,
            "is_correct": bool(r.is_correct),
            "response_time_ms": float(r.response_time_ms) if r.response_time_ms else None,
            "n_entities": int(r.stimulus.n_entities),
            "n_relations": int(r.stimulus.n_relations),
            "has_reversal": bool(r.stimulus.has_reversal)
        }
        for r in results
    ]
    
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    print(f"   Saved to: {results_file}")
    
    latest_file = os.path.join(output_dir, "results_latest.json")
    with open(latest_file, "w") as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    
    print("\n[3/4] Analyzing results...")
    analyzer = APHAnalyzer(results_data=results_data)
    analysis = analyzer.run_full_analysis()
    print(analysis.interpretation)
    
    analysis_file = os.path.join(output_dir, f"analysis_{timestamp}.json")
    analysis_data = {
        "timestamp": timestamp,
        "version": "v3",
        "config": {
            "n_trials_per_condition": int(config.n_trials_per_condition),
            "complexity_levels": [int(l) for l in config.complexity_levels],
            "model": config.model,
            "temperature": float(config.temperature),
            "seed": int(config.seed)
        },
        "slope_comparison": analysis.slope_comparison,
        "point_comparisons": {int(k): v for k, v in analysis.point_comparisons.items()},
        "effect_sizes": analysis.effect_sizes,
        "role_filler": {
            "accuracy_by_level": {int(k): float(v) for k, v in analysis.role_filler.accuracy_by_level.items()},
            "mean_accuracy": float(analysis.role_filler.mean_accuracy)
        },
        "recursive": {
            "accuracy_by_level": {int(k): float(v) for k, v in analysis.recursive.accuracy_by_level.items()},
            "mean_accuracy": float(analysis.recursive.mean_accuracy)
        }
    }
    
    with open(analysis_file, "w") as f:
        json.dump(analysis_data, f, indent=2, cls=NumpyEncoder)
    
    print("\n[4/4] Creating visualization...")
    viz_file = os.path.join(output_dir, f"visualization_{timestamp}.html")
    create_visualization(results_file, viz_file)
    print(f"   Saved to: {viz_file}")
    
    latest_viz = os.path.join(output_dir, "visualization_latest.html")
    create_visualization(latest_file, latest_viz)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Raw results: {results_file}")
    print(f"  - Analysis: {analysis_file}")
    print(f"  - Visualization: {viz_file}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Run APH Test v3")
    
    parser.add_argument("--api-key", type=str, default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--levels", type=str, default="1,2,3,4,5")
    parser.add_argument("--model", type=str, default="claude-haiku-4-5-20251001")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=".")
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("ERROR: No API key. Set ANTHROPIC_API_KEY or use --api-key")
        sys.exit(1)
    
    levels = [int(l.strip()) for l in args.levels.split(",")]
    
    print_header()
    
    config = ExperimentConfig(
        n_trials_per_condition=args.n_trials,
        complexity_levels=levels,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed
    )
    
    run_experiment(api_key=args.api_key, config=config, output_dir=args.output)


if __name__ == "__main__":
    main()
