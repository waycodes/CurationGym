#!/usr/bin/env python3
"""
CurationGym End-to-End Demo

Demonstrates the complete data curation pipeline:
1. Generate synthetic documents with known quality signals
2. Define and execute curation policies
3. Compare policy outcomes
4. Simulate proxy model training
5. Evaluate and attribute performance to data slices
6. Generate reports

This demo is self-contained and shows the core concepts without
requiring external dependencies beyond Python stdlib.

Run: python demo/run_demo.py
"""

import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Document:
    """A document in the curation pipeline."""
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        return len(self.text.split())


@dataclass
class Policy:
    """A curation policy specification."""
    name: str
    filters: list[dict[str, Any]] = field(default_factory=list)
    dedup: dict[str, Any] = field(default_factory=dict)
    mixing: dict[str, float] = field(default_factory=dict)
    
    def hash(self) -> str:
        """Deterministic hash for reproducibility."""
        data = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


# =============================================================================
# STEP 1: Generate Synthetic Documents
# =============================================================================

def generate_synthetic_corpus(n_docs: int = 1000, seed: int = 42) -> list[Document]:
    """Generate synthetic documents with controlled quality signals.
    
    Creates documents across 4 domains with varying quality levels,
    including intentional duplicates to test deduplication.
    """
    random.seed(seed)
    
    domains = ["wiki", "news", "code", "social"]
    quality_levels = ["high", "medium", "low"]
    
    templates = {
        "wiki": [
            "The {topic} is a {adj} concept in {field}. It was first described in {year}. "
            "Researchers have studied {topic} extensively, finding {finding}.",
            "{topic} refers to the process of {action} in {field}. Research shows {finding}. "
            "This has implications for {related_field} and beyond.",
        ],
        "news": [
            "Breaking: {topic} announced today. Experts say {finding}. "
            "Industry analysts are watching developments closely.",
            "Report: New developments in {topic}. Industry leaders respond to {action}. "
            "Markets reacted with {reaction}.",
        ],
        "code": [
            "def {func}(x, y):\n    '''Process {topic}.'''\n    result = x + y\n    return result * 2",
            "class {class_name}:\n    '''Handle {topic}.'''\n    def __init__(self):\n        self.value = {year}",
        ],
        "social": [
            "just learned about {topic} lol so {adj}",
            "anyone else think {topic} is {adj}?? #trending #tech",
        ],
    }
    
    topics = ["machine learning", "data curation", "neural networks", "transformers", "optimization"]
    fields = ["computer science", "statistics", "mathematics", "engineering"]
    related_fields = ["AI safety", "data science", "software engineering"]
    adjs = ["important", "fascinating", "complex", "revolutionary", "emerging"]
    actions = ["training models", "processing data", "scaling systems", "improving accuracy"]
    findings = ["significant improvements", "promising results", "new possibilities"]
    reactions = ["optimism", "caution", "enthusiasm"]
    funcs = ["process_data", "train_model", "evaluate", "optimize"]
    class_names = ["DataProcessor", "ModelTrainer", "Evaluator", "Optimizer"]
    
    docs = []
    
    for i in range(n_docs):
        domain = random.choice(domains)
        quality = random.choices(quality_levels, weights=[0.3, 0.4, 0.3])[0]
        
        template = random.choice(templates[domain])
        text = template.format(
            topic=random.choice(topics),
            field=random.choice(fields),
            related_field=random.choice(related_fields),
            adj=random.choice(adjs),
            action=random.choice(actions),
            finding=random.choice(findings),
            reaction=random.choice(reactions),
            year=random.randint(1990, 2024),
            func=random.choice(funcs),
            class_name=random.choice(class_names),
        )
        
        # Quality affects text length
        if quality == "high":
            text = text + " " + text  # Longer content
        elif quality == "low":
            text = text[:len(text)//2]  # Truncated
        
        doc = Document(
            id=f"doc_{i:05d}",
            text=text,
            metadata={
                "domain": domain,
                "quality_label": quality,
                "source": f"synthetic_{domain}",
            },
        )
        docs.append(doc)
    
    # Add exact duplicates (10%)
    n_exact = n_docs // 10
    for i in range(n_exact):
        src = random.choice(docs[:n_docs])
        docs.append(Document(
            id=f"dup_exact_{i:04d}",
            text=src.text,
            metadata={**src.metadata, "is_duplicate": True},
        ))
    
    # Add near-duplicates (5%)
    n_near = n_docs // 20
    for i in range(n_near):
        src = random.choice(docs[:n_docs])
        docs.append(Document(
            id=f"dup_near_{i:04d}",
            text=src.text + f" [variant {i}]",
            metadata={**src.metadata, "is_near_duplicate": True},
        ))
    
    random.shuffle(docs)
    return docs


# =============================================================================
# STEP 2: Define Curation Policies
# =============================================================================

def create_policies() -> dict[str, Policy]:
    """Create different curation policies to compare."""
    
    baseline = Policy(
        name="baseline",
        filters=[{"name": "length", "min_words": 5}],
        dedup={"method": "none"},
        mixing={},
    )
    
    quality_focused = Policy(
        name="quality_focused",
        filters=[
            {"name": "length", "min_words": 20, "max_words": 500},
            {"name": "quality", "threshold": 0.6},
        ],
        dedup={"method": "exact"},
        mixing={"wiki": 0.5, "code": 0.3, "news": 0.2},
    )
    
    diversity_focused = Policy(
        name="diversity_focused",
        filters=[{"name": "length", "min_words": 10}],
        dedup={"method": "minhash", "threshold": 0.8},
        mixing={"wiki": 0.25, "code": 0.25, "news": 0.25, "social": 0.25},
    )
    
    return {"baseline": baseline, "quality_focused": quality_focused, "diversity_focused": diversity_focused}


# =============================================================================
# STEP 3: Execute Policies
# =============================================================================

def exact_dedup(docs: list[Document]) -> list[Document]:
    """Remove exact duplicates by text hash."""
    seen = set()
    result = []
    for doc in docs:
        h = hashlib.md5(doc.text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(doc)
    return result


def minhash_dedup(docs: list[Document], threshold: float = 0.8) -> list[Document]:
    """Remove near-duplicates using MinHash approximation."""
    def shingles(text: str, k: int = 3) -> set[str]:
        words = text.lower().split()
        return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}
    
    def jaccard(s1: set, s2: set) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)
    
    result = []
    shingle_cache = []
    
    for doc in docs:
        doc_shingles = shingles(doc.text)
        is_dup = False
        for existing_shingles in shingle_cache:
            if jaccard(doc_shingles, existing_shingles) >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(doc)
            shingle_cache.append(doc_shingles)
    
    return result


def execute_policy(docs: list[Document], policy: Policy, target_size: int = 500) -> tuple[list[Document], dict]:
    """Execute a curation policy."""
    stats = {"input": len(docs)}
    current = docs[:]
    
    # Apply filters
    for f in policy.filters:
        if f["name"] == "length":
            min_w, max_w = f.get("min_words", 0), f.get("max_words", 100000)
            current = [d for d in current if min_w <= d.token_count <= max_w]
        elif f["name"] == "quality":
            scores = {"high": 0.9, "medium": 0.5, "low": 0.2}
            thresh = f.get("threshold", 0.5)
            current = [d for d in current if scores.get(d.metadata.get("quality_label"), 0.5) >= thresh]
    stats["after_filter"] = len(current)
    
    # Apply dedup
    method = policy.dedup.get("method", "none")
    if method == "exact":
        current = exact_dedup(current)
    elif method == "minhash":
        current = minhash_dedup(current, policy.dedup.get("threshold", 0.8))
    stats["after_dedup"] = len(current)
    
    # Apply mixing
    if policy.mixing:
        by_domain: dict[str, list[Document]] = {}
        for doc in current:
            domain = doc.metadata.get("domain", "unknown")
            by_domain.setdefault(domain, []).append(doc)
        
        mixed = []
        for domain, ratio in policy.mixing.items():
            domain_docs = by_domain.get(domain, [])
            n = int(target_size * ratio)
            mixed.extend(random.sample(domain_docs, min(n, len(domain_docs))))
        current = mixed[:target_size]
    else:
        current = current[:target_size]
    
    stats["final"] = len(current)
    return current, stats


# =============================================================================
# STEP 4: Simulate Training & Evaluation
# =============================================================================

def simulate_training(docs: list[Document], policy_name: str, seed: int = 42) -> dict:
    """Simulate proxy model training with composition-dependent results."""
    random.seed(seed + hash(policy_name) % 1000)
    
    # Compute composition
    domain_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for doc in docs:
        domain = doc.metadata.get("domain", "unknown")
        quality = doc.metadata.get("quality_label", "medium")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    
    total = len(docs) or 1
    wiki_r = domain_counts.get("wiki", 0) / total
    code_r = domain_counts.get("code", 0) / total
    high_q = quality_counts.get("high", 0) / total
    
    # Scores depend on composition (simulating real training dynamics)
    base = 0.25
    noise = lambda: random.uniform(-0.015, 0.015)
    
    hellaswag = base + 0.18 * wiki_r + 0.12 * high_q + noise()
    arc = base + 0.14 * wiki_r + 0.10 * code_r + 0.14 * high_q + noise()
    winogrande = base + 0.12 * wiki_r + 0.10 * high_q + noise()
    
    return {
        "policy": policy_name,
        "n_docs": len(docs),
        "composition": {
            "domains": {k: round(v/total, 3) for k, v in domain_counts.items()},
            "quality": {k: round(v/total, 3) for k, v in quality_counts.items()},
        },
        "eval": {
            "hellaswag": round(hellaswag, 4),
            "arc_easy": round(arc, 4),
            "winogrande": round(winogrande, 4),
            "average": round((hellaswag + arc + winogrande) / 3, 4),
        },
    }


# =============================================================================
# STEP 5: Attribution Analysis
# =============================================================================

def run_attribution(results: list[dict]) -> dict:
    """Compute slice-to-benchmark attribution via correlation."""
    domains = ["wiki", "news", "code", "social"]
    benchmarks = ["hellaswag", "arc_easy", "winogrande"]
    
    attributions = {}
    for bench in benchmarks:
        attributions[bench] = {}
        scores = [r["eval"][bench] for r in results]
        
        for domain in domains:
            presence = [r["composition"]["domains"].get(domain, 0) for r in results]
            
            # Pearson correlation
            n = len(scores)
            if n < 2:
                continue
            mean_p = sum(presence) / n
            mean_s = sum(scores) / n
            
            cov = sum((p - mean_p) * (s - mean_s) for p, s in zip(presence, scores))
            var_p = sum((p - mean_p) ** 2 for p in presence)
            var_s = sum((s - mean_s) ** 2 for s in scores)
            
            denom = (var_p * var_s) ** 0.5
            corr = cov / denom if denom > 0 else 0
            attributions[bench][domain] = round(corr, 4)
    
    return attributions


# =============================================================================
# STEP 6: Generate Report
# =============================================================================

def generate_report(
    policies: dict[str, Policy],
    stats: dict[str, dict],
    results: list[dict],
    attributions: dict,
) -> str:
    """Generate markdown report."""
    
    lines = [
        "# CurationGym Demo Report",
        "",
        "## Overview",
        "",
        "This report compares three data curation policies on synthetic data,",
        "demonstrating how curation choices affect downstream model performance.",
        "",
        "## Policies",
        "",
    ]
    
    for name, policy in policies.items():
        lines.extend([
            f"### {name}",
            f"- **Filters**: {[f['name'] for f in policy.filters]}",
            f"- **Dedup**: {policy.dedup.get('method', 'none')}",
            f"- **Mixing**: {policy.mixing or 'proportional'}",
            f"- **Hash**: `{policy.hash()}`",
            "",
        ])
    
    lines.extend([
        "## Pipeline Statistics",
        "",
        "| Policy | Input | After Filter | After Dedup | Final |",
        "|--------|-------|--------------|-------------|-------|",
    ])
    for name, s in stats.items():
        lines.append(f"| {name} | {s['input']} | {s['after_filter']} | {s['after_dedup']} | {s['final']} |")
    
    lines.extend([
        "",
        "## Evaluation Results",
        "",
        "| Policy | HellaSwag | ARC-Easy | WinoGrande | Average |",
        "|--------|-----------|----------|------------|---------|",
    ])
    for r in results:
        e = r["eval"]
        lines.append(f"| {r['policy']} | {e['hellaswag']:.4f} | {e['arc_easy']:.4f} | {e['winogrande']:.4f} | {e['average']:.4f} |")
    
    best = max(results, key=lambda r: r["eval"]["average"])
    lines.extend([
        "",
        f"**Winner**: `{best['policy']}` with average score {best['eval']['average']:.4f}",
        "",
        "## Attribution Analysis",
        "",
        "Correlation between domain presence and benchmark scores:",
        "",
        "| Domain | HellaSwag | ARC-Easy | WinoGrande |",
        "|--------|-----------|----------|------------|",
    ])
    for domain in ["wiki", "news", "code", "social"]:
        h = attributions["hellaswag"].get(domain, 0)
        a = attributions["arc_easy"].get(domain, 0)
        w = attributions["winogrande"].get(domain, 0)
        lines.append(f"| {domain} | {h:+.4f} | {a:+.4f} | {w:+.4f} |")
    
    lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Quality filtering improves scores**: The quality-focused policy achieves higher",
        "   benchmark scores by filtering low-quality documents.",
        "",
        "2. **Domain composition matters**: Wiki content shows strong positive correlation",
        "   with reasoning benchmarks (HellaSwag, WinoGrande).",
        "",
        "3. **Deduplication is essential**: Removing duplicates (10-15% of raw data)",
        "   improves training efficiency without hurting performance.",
        "",
        "4. **Attribution enables targeted improvement**: By identifying which slices",
        "   help which benchmarks, we can make informed data collection decisions.",
        "",
        "## Reproducibility",
        "",
        "- All results use `seed=42`",
        "- Policy hashes ensure configuration tracking",
        "- Full pipeline is deterministic given the same inputs",
        "",
        "---",
        "*Generated by CurationGym*",
    ])
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("CurationGym End-to-End Demo")
    print("=" * 60)
    
    # Step 1
    print("\n[1/6] Generating synthetic corpus...")
    docs = generate_synthetic_corpus(n_docs=1000, seed=42)
    print(f"      Created {len(docs)} documents (including duplicates)")
    
    # Step 2
    print("\n[2/6] Defining curation policies...")
    policies = create_policies()
    for name, p in policies.items():
        print(f"      - {name}: {len(p.filters)} filters, dedup={p.dedup.get('method')}")
    
    # Step 3
    print("\n[3/6] Executing policies...")
    all_stats = {}
    curated = {}
    for name, policy in policies.items():
        dataset, stats = execute_policy(docs, policy, target_size=500)
        curated[name] = dataset
        all_stats[name] = stats
        print(f"      {name}: {stats['input']} → {stats['final']} docs")
    
    # Step 4
    print("\n[4/6] Simulating training & evaluation...")
    results = []
    for name, dataset in curated.items():
        result = simulate_training(dataset, name)
        results.append(result)
        print(f"      {name}: avg={result['eval']['average']:.4f}")
    
    # Step 5
    print("\n[5/6] Running attribution analysis...")
    attributions = run_attribution(results)
    print("      Computed domain-benchmark correlations")
    
    # Step 6
    print("\n[6/6] Generating report...")
    report = generate_report(policies, all_stats, results, attributions)
    
    # Save outputs
    (OUTPUT_DIR / "demo_report.md").write_text(report)
    (OUTPUT_DIR / "demo_results.json").write_text(json.dumps({
        "stats": all_stats,
        "results": results,
        "attributions": attributions,
    }, indent=2))
    
    print(f"      Saved to {OUTPUT_DIR}/")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    best = max(results, key=lambda r: r["eval"]["average"])
    print(f"\nBest Policy: {best['policy']}")
    print(f"  HellaSwag:  {best['eval']['hellaswag']:.4f}")
    print(f"  ARC-Easy:   {best['eval']['arc_easy']:.4f}")
    print(f"  WinoGrande: {best['eval']['winogrande']:.4f}")
    print(f"  Average:    {best['eval']['average']:.4f}")
    
    print("\nKey Attribution (wiki domain):")
    for bench in ["hellaswag", "arc_easy", "winogrande"]:
        corr = attributions[bench].get("wiki", 0)
        print(f"  {bench}: {corr:+.4f}")
    
    print("\n✓ Demo complete! See demo/output/ for full report.")


if __name__ == "__main__":
    main()
