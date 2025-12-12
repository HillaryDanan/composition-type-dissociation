"""
APH Experiment Analysis Module - v2
===================================

Fixed JSON serialization for numpy types.
"""

import json
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# JSON ENCODER FIX
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def safe_json_dump(data, f, **kwargs):
    """JSON dump with numpy type handling."""
    json.dump(data, f, cls=NumpyEncoder, **kwargs)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ConditionResults:
    """Results for one experimental condition."""
    condition_name: str
    levels: list
    accuracy_by_level: dict
    n_by_level: dict
    raw_correct: dict
    
    @property
    def mean_accuracy(self) -> float:
        total_correct = sum(sum(v) for v in self.raw_correct.values())
        total_n = sum(len(v) for v in self.raw_correct.values())
        return total_correct / total_n if total_n > 0 else 0
    
    @property
    def accuracies(self) -> list:
        return [self.accuracy_by_level[l] for l in sorted(self.levels)]


@dataclass
class AnalysisResults:
    """Complete analysis results."""
    role_filler: ConditionResults
    recursive: ConditionResults
    slope_comparison: dict
    point_comparisons: dict
    effect_sizes: dict
    interpretation: str


# =============================================================================
# ANALYZER
# =============================================================================

class APHAnalyzer:
    """Analyzes APH experiment results."""
    
    def __init__(self, results_file: str = None, results_data: list = None):
        if results_file:
            with open(results_file, 'r') as f:
                self.raw_data = json.load(f)
        elif results_data:
            self.raw_data = results_data
        else:
            raise ValueError("Must provide results_file or results_data")
        
        self._parse_results()
    
    def _parse_results(self):
        role_filler_data = [r for r in self.raw_data if r['condition'] == '3b']
        recursive_data = [r for r in self.raw_data if r['condition'] == '3c']
        
        self.role_filler = self._compute_condition_results(role_filler_data, "Role-Filler (3b)")
        self.recursive = self._compute_condition_results(recursive_data, "Recursive (3c)")
    
    def _compute_condition_results(self, data: list, name: str) -> ConditionResults:
        levels = sorted(set(r['complexity_level'] for r in data))
        
        accuracy_by_level = {}
        n_by_level = {}
        raw_correct = {}
        
        for level in levels:
            level_data = [r for r in data if r['complexity_level'] == level]
            correct = [1 if r['is_correct'] else 0 for r in level_data]
            
            raw_correct[level] = correct
            n_by_level[level] = len(correct)
            accuracy_by_level[level] = sum(correct) / len(correct) if correct else 0
        
        return ConditionResults(
            condition_name=name,
            levels=levels,
            accuracy_by_level=accuracy_by_level,
            n_by_level=n_by_level,
            raw_correct=raw_correct
        )
    
    def compute_degradation_slope(self, condition: ConditionResults) -> dict:
        x = np.array(sorted(condition.levels))
        y = np.array([condition.accuracy_by_level[l] for l in x])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'x': x.tolist(),
            'y': y.tolist()
        }
    
    def compare_slopes(self) -> dict:
        rf_slope = self.compute_degradation_slope(self.role_filler)
        rc_slope = self.compute_degradation_slope(self.recursive)
        
        slope_diff = rf_slope['slope'] - rc_slope['slope']
        
        se_diff = np.sqrt(rf_slope['std_err']**2 + rc_slope['std_err']**2)
        
        if se_diff > 0:
            z_stat = slope_diff / se_diff
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            z_stat = 0.0
            p_value = 1.0
        
        pooled_sd = np.std(self.role_filler.accuracies + self.recursive.accuracies)
        effect_size = slope_diff / pooled_sd if pooled_sd > 0 else 0
        
        perm_p = self._permutation_test_slopes(n_permutations=1000)
        
        return {
            'role_filler_slope': rf_slope,
            'recursive_slope': rc_slope,
            'slope_difference': float(slope_diff),
            'z_statistic': float(z_stat),
            'p_value_parametric': float(p_value),
            'p_value_permutation': float(perm_p),
            'effect_size_d': float(effect_size),
            'prediction_supported': bool(rc_slope['slope'] < rf_slope['slope'])
        }
    
    def _permutation_test_slopes(self, n_permutations: int = 1000) -> float:
        all_data = self.raw_data.copy()
        
        observed_diff = (
            self.compute_degradation_slope(self.role_filler)['slope'] -
            self.compute_degradation_slope(self.recursive)['slope']
        )
        
        np.random.seed(42)
        perm_diffs = []
        
        for _ in range(n_permutations):
            shuffled = [r.copy() for r in all_data]
            labels = [r['condition'] for r in shuffled]
            np.random.shuffle(labels)
            for i, r in enumerate(shuffled):
                shuffled[i]['condition'] = labels[i]
            
            rf_data = [r for r in shuffled if r['condition'] == '3b']
            rc_data = [r for r in shuffled if r['condition'] == '3c']
            
            rf_cond = self._compute_condition_results(rf_data, "perm_rf")
            rc_cond = self._compute_condition_results(rc_data, "perm_rc")
            
            rf_slope = self.compute_degradation_slope(rf_cond)['slope']
            rc_slope = self.compute_degradation_slope(rc_cond)['slope']
            
            perm_diffs.append(rf_slope - rc_slope)
        
        p_value = np.mean([d >= observed_diff for d in perm_diffs])
        
        return float(p_value)
    
    def point_comparisons(self) -> dict:
        comparisons = {}
        
        for level in self.role_filler.levels:
            rf_correct = self.role_filler.raw_correct.get(level, [])
            rc_correct = self.recursive.raw_correct.get(level, [])
            
            if not rf_correct or not rc_correct:
                continue
            
            rf_yes = sum(rf_correct)
            rf_no = len(rf_correct) - rf_yes
            rc_yes = sum(rc_correct)
            rc_no = len(rc_correct) - rc_yes
            
            table = [[rf_yes, rf_no], [rc_yes, rc_no]]
            
            odds_ratio, p_value = stats.fisher_exact(table)
            
            comparisons[level] = {
                'role_filler_accuracy': float(rf_yes / len(rf_correct)),
                'recursive_accuracy': float(rc_yes / len(rc_correct)),
                'difference': float((rf_yes/len(rf_correct)) - (rc_yes/len(rc_correct))),
                'odds_ratio': float(odds_ratio) if not np.isinf(odds_ratio) else "inf",
                'p_value': float(p_value),
                'n_role_filler': len(rf_correct),
                'n_recursive': len(rc_correct)
            }
        
        return comparisons
    
    def compute_effect_sizes(self) -> dict:
        rf_accs = []
        rc_accs = []
        
        for level in self.role_filler.levels:
            rf_accs.extend(self.role_filler.raw_correct.get(level, []))
            rc_accs.extend(self.recursive.raw_correct.get(level, []))
        
        rf_mean = np.mean(rf_accs)
        rc_mean = np.mean(rc_accs)
        
        pooled_std = np.sqrt(
            ((len(rf_accs)-1)*np.var(rf_accs) + (len(rc_accs)-1)*np.var(rc_accs)) /
            (len(rf_accs) + len(rc_accs) - 2)
        ) if len(rf_accs) + len(rc_accs) > 2 else 1.0
        
        cohens_d = (rf_mean - rc_mean) / pooled_std if pooled_std > 0 else 0
        
        rc_std = np.std(rc_accs)
        glass_delta = (rf_mean - rc_mean) / rc_std if rc_std > 0 else 0
        
        return {
            'cohens_d': float(cohens_d),
            'glass_delta': float(glass_delta),
            'role_filler_mean': float(rf_mean),
            'recursive_mean': float(rc_mean),
            'interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def run_full_analysis(self) -> AnalysisResults:
        slope_comp = self.compare_slopes()
        point_comp = self.point_comparisons()
        effects = self.compute_effect_sizes()
        
        interpretation = self._generate_interpretation(slope_comp, point_comp, effects)
        
        return AnalysisResults(
            role_filler=self.role_filler,
            recursive=self.recursive,
            slope_comparison=slope_comp,
            point_comparisons=point_comp,
            effect_sizes=effects,
            interpretation=interpretation
        )
    
    def _generate_interpretation(self, slope_comp: dict, point_comp: dict, effects: dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("APH COMPOSITION TYPE DISSOCIATION: RESULTS SUMMARY")
        lines.append("=" * 60)
        
        if slope_comp['prediction_supported']:
            lines.append("\n✓ PRIMARY PREDICTION SUPPORTED")
            lines.append("  Recursive depth shows steeper degradation than role-filler novelty")
        else:
            lines.append("\n✗ PRIMARY PREDICTION NOT SUPPORTED")
            lines.append("  Degradation patterns do not differ as predicted")
        
        lines.append(f"\nDEGRADATION SLOPES:")
        lines.append(f"  Role-Filler (3b): {slope_comp['role_filler_slope']['slope']:.4f}")
        lines.append(f"  Recursive (3c):   {slope_comp['recursive_slope']['slope']:.4f}")
        lines.append(f"  Difference:       {slope_comp['slope_difference']:.4f}")
        
        lines.append(f"\nSTATISTICAL TESTS:")
        lines.append(f"  Z-statistic:       {slope_comp['z_statistic']:.3f}")
        lines.append(f"  P-value (param.):  {slope_comp['p_value_parametric']:.4f}")
        lines.append(f"  P-value (perm.):   {slope_comp['p_value_permutation']:.4f}")
        
        sig_level = 0.05
        if slope_comp['p_value_permutation'] < sig_level:
            lines.append(f"  → Significant at α = {sig_level}")
        else:
            lines.append(f"  → Not significant at α = {sig_level}")
        
        lines.append(f"\nEFFECT SIZES:")
        lines.append(f"  Cohen's d: {effects['cohens_d']:.3f} ({effects['interpretation']})")
        
        lines.append(f"\nACCURACY BY LEVEL:")
        lines.append("  Level | Role-Filler | Recursive | Difference")
        lines.append("  " + "-" * 45)
        
        for level in sorted(point_comp.keys()):
            pc = point_comp[level]
            lines.append(
                f"    {level}   |    {pc['role_filler_accuracy']:.0%}      |   "
                f"{pc['recursive_accuracy']:.0%}     |   {pc['difference']:+.0%}"
            )
        
        lines.append("\n" + "=" * 60)
        lines.append("SCIENTIFIC INTERPRETATION")
        lines.append("=" * 60)
        
        if slope_comp['prediction_supported'] and slope_comp['p_value_permutation'] < sig_level:
            lines.append("""
The results provide SUPPORT for the APH prediction that recursive 
composition (Type 3c) and role-filler composition (Type 3b) engage 
different computational mechanisms with different scaling properties.

The steeper degradation with recursive depth suggests that LLMs may 
lack the iterative symbol-composition loop hypothesized by APH to 
underlie true recursive processing.

CAVEATS:
- This is a single experiment with one model
- The effect should replicate across models and stimulus sets
- Alternative explanations (e.g., working memory) need testing
""")
        else:
            lines.append("""
The results do NOT support the APH prediction. The degradation 
patterns are similar across composition types.

POSSIBLE INTERPRETATIONS:
1. The APH prediction is incorrect
2. The test lacks sensitivity (power)
3. The operationalization of recursion/role-filler is inadequate
4. The model has learned workarounds that mask the predicted difference

NEXT STEPS:
- Increase sample size for more power
- Test with different stimulus designs
- Compare across multiple models
""")
        
        return "\n".join(lines)
    
    def print_report(self):
        results = self.run_full_analysis()
        print(results.interpretation)
        return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(results_file: str, output_file: str = "aph_results.html"):
    analyzer = APHAnalyzer(results_file=results_file)
    analysis = analyzer.run_full_analysis()
    
    rf_slope = analysis.slope_comparison['role_filler_slope']
    rc_slope = analysis.slope_comparison['recursive_slope']
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>APH Composition Type Dissociation Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .stat {{
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            background: #f0f0f0;
            border-radius: 4px;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        .prediction-supported {{
            color: #2e7d32;
            font-weight: bold;
        }}
        .prediction-not-supported {{
            color: #c62828;
            font-weight: bold;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
        }}
    </style>
</head>
<body>
    <h1>APH Composition Type Dissociation Test</h1>
    
    <div class="card">
        <h2>Primary Prediction</h2>
        <p>
            <strong>Hypothesis:</strong> Recursive composition (Type 3c) degrades faster 
            with complexity than role-filler composition (Type 3b).
        </p>
        <p>
            <strong>Result:</strong> 
            <span class="{'prediction-supported' if analysis.slope_comparison['prediction_supported'] else 'prediction-not-supported'}">
                {'SUPPORTED' if analysis.slope_comparison['prediction_supported'] else 'NOT SUPPORTED'}
            </span>
        </p>
    </div>
    
    <div class="card">
        <h2>Degradation Curves</h2>
        <div class="chart-container">
            <canvas id="degradationChart"></canvas>
        </div>
    </div>
    
    <div class="card">
        <h2>Key Statistics</h2>
        <div class="stat">
            <div class="stat-value">{rf_slope['slope']:.4f}</div>
            <div class="stat-label">Role-Filler Slope</div>
        </div>
        <div class="stat">
            <div class="stat-value">{rc_slope['slope']:.4f}</div>
            <div class="stat-label">Recursive Slope</div>
        </div>
        <div class="stat">
            <div class="stat-value">{analysis.slope_comparison['slope_difference']:.4f}</div>
            <div class="stat-label">Difference</div>
        </div>
        <div class="stat">
            <div class="stat-value">p = {analysis.slope_comparison['p_value_permutation']:.4f}</div>
            <div class="stat-label">Permutation Test</div>
        </div>
        <div class="stat">
            <div class="stat-value">d = {analysis.effect_sizes['cohens_d']:.3f}</div>
            <div class="stat-label">Cohen's d ({analysis.effect_sizes['interpretation']})</div>
        </div>
    </div>
    
    <div class="card">
        <h2>Raw Data</h2>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background:#f0f0f0;">
                <th style="padding:10px; text-align:left;">Level</th>
                <th style="padding:10px; text-align:center;">Role-Filler Acc.</th>
                <th style="padding:10px; text-align:center;">Recursive Acc.</th>
                <th style="padding:10px; text-align:center;">Difference</th>
                <th style="padding:10px; text-align:center;">p-value</th>
            </tr>
            {''.join([f"""
            <tr>
                <td style="padding:10px;">{level}</td>
                <td style="padding:10px; text-align:center;">{pc['role_filler_accuracy']:.0%}</td>
                <td style="padding:10px; text-align:center;">{pc['recursive_accuracy']:.0%}</td>
                <td style="padding:10px; text-align:center;">{pc['difference']:+.0%}</td>
                <td style="padding:10px; text-align:center;">{pc['p_value']:.3f}</td>
            </tr>
            """ for level, pc in sorted(analysis.point_comparisons.items())])}
        </table>
    </div>
    
    <script>
        const ctx = document.getElementById('degradationChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {rf_slope['x']},
                datasets: [
                    {{
                        label: 'Role-Filler (3b)',
                        data: {rf_slope['y']},
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        fill: false,
                        tension: 0.1
                    }},
                    {{
                        label: 'Recursive (3c)',
                        data: {rc_slope['y']},
                        borderColor: '#F44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        fill: false,
                        tension: 0.1
                    }},
                    {{
                        label: 'Role-Filler Trend',
                        data: {[rf_slope['intercept'] + rf_slope['slope']*x for x in rf_slope['x']]},
                        borderColor: '#2196F3',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Recursive Trend',
                        data: {[rc_slope['intercept'] + rc_slope['slope']*x for x in rc_slope['x']]},
                        borderColor: '#F44336',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        title: {{
                            display: true,
                            text: 'Accuracy'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Complexity Level'
                        }}
                    }}
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Accuracy Degradation by Composition Type'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    return output_file


if __name__ == "__main__":
    import sys
    
    results_file = sys.argv[1] if len(sys.argv) > 1 else "results_raw.json"
    
    print(f"Analyzing results from: {results_file}")
    
    analyzer = APHAnalyzer(results_file=results_file)
    results = analyzer.print_report()
    
    viz_file = create_visualization(results_file)
    print(f"\nVisualization saved to: {viz_file}")
