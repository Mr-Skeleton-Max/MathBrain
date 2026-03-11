"""MathBrain 评估框架 (适配模块化接口)"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class VoteContribution:
    slot_id: int
    shared_words: List[str]
    contribution: float
    phi_norm: float
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionTrace:
    context: List[str]
    target: str
    target_logit: float
    target_rank: int
    total_candidates: int
    top_k: List[Tuple[str, float]]
    source_votes: List[VoteContribution]
    decisive_factors: List[Tuple[str, float, str]]


@dataclass
class BenchmarkResult:
    category: str
    task_name: str
    accuracy: float
    total: int
    correct: int
    details: List[Dict] = field(default_factory=list)


class MathBrainEvaluator:
    """MathBrain 评估器 (模块化版本)"""

    def __init__(self, model):
        self.model = model
        self.a = model.a
        self.w2s = model.word_to_slots
        self.vocab = model.vocab

        # 反向映射: slot → words
        self.s2w: dict = defaultdict(list)
        for w, slots in self.w2s.items():
            for s in slots:
                self.s2w[int(s)].append(w)

    def _build_context(self, context_words: List[str]):
        """构建上下文 → (active_slots, phi)"""
        self.model.q.reset()
        for w in context_words:
            self.model._ensure_word(w)
            self.model.q.update(self.model.retina.encode(w))
        active_slots, Q_vals = self.model.q.get_active()
        if len(active_slots) == 0:
            return active_slots, np.zeros((0, self.model.config.D_COSINE), dtype=np.float32)
        phi = self.model.phi_encoder.encode(Q_vals)
        return active_slots, phi

    def _compute_word_scores(self, context_words: List[str]) -> dict:
        """上下文 → 词分数 dict"""
        preds = self.model.predict_next(context_words, k=len(self.vocab))
        return {w: s for w, s in preds}

    def trace_prediction(self, context: List[str], target: str) -> PredictionTrace:
        word_scores = self._compute_word_scores(context)
        sorted_words = sorted(word_scores.items(), key=lambda x: -x[1])
        top_k = sorted_words[:10]
        target_logit = word_scores.get(target, 0)
        target_rank = next((i+1 for i, (w, _) in enumerate(sorted_words) if w == target), -1)

        active_slots, phi = self._build_context(context)
        source_votes = self._analyze_source_votes(active_slots, phi, target)
        winner = top_k[0][0] if top_k else None
        decisive = self._find_decisive_factors(active_slots, phi, target, winner)

        return PredictionTrace(
            context=context, target=target,
            target_logit=target_logit, target_rank=target_rank,
            total_candidates=len(word_scores), top_k=top_k,
            source_votes=source_votes, decisive_factors=decisive,
        )

    def _analyze_source_votes(self, active_slots, phi, target: str) -> List[VoteContribution]:
        if not self.a.has_knowledge:
            return []
        target_slots = self.w2s.get(target, [])
        if len(target_slots) == 0:
            return []

        votes = []
        for i, src in enumerate(active_slots):
            src = int(src)
            if src not in self.a.E:
                continue
            e_v = self.a.E[src] @ self.a.W_V
            p_phi = self.a.P.T @ phi[i]
            contrib_vec = e_v * p_phi

            total_c = 0.0
            breakdown = {}
            for tgt_s in target_slots:
                tgt_s = int(tgt_s)
                if tgt_s in self.a.E:
                    e_q = self.a.E[tgt_s] @ self.a.W_Q
                    c = float(np.dot(e_q, contrib_vec))
                    total_c += c
                    breakdown[f"slot_{tgt_s}"] = c

            votes.append(VoteContribution(
                slot_id=src, shared_words=self.s2w[src][:5],
                contribution=total_c, phi_norm=float(np.linalg.norm(phi[i])),
                breakdown=breakdown,
            ))
        votes.sort(key=lambda x: -abs(x.contribution))
        return votes

    def _find_decisive_factors(self, active_slots, phi,
                               target: str, winner: Optional[str]) -> list:
        if not winner or winner == target:
            return []
        tv = {v.slot_id: v.contribution for v in self._analyze_source_votes(active_slots, phi, target)}
        wv = {v.slot_id: v.contribution for v in self._analyze_source_votes(active_slots, phi, winner)}
        factors = []
        for s in set(tv) | set(wv):
            diff = wv.get(s, 0) - tv.get(s, 0)
            if abs(diff) > 1:
                words = self.s2w[s][:3]
                label = f"有利于 '{winner}'" if diff > 0 else f"有利于 '{target}'"
                factors.append((f"Slot {s} ({words})", diff, label))
        factors.sort(key=lambda x: -x[1])
        return factors[:5]

    # ---- 基准测试 ----

    def run_benchmark(self, suite: str = "basic") -> List[BenchmarkResult]:
        results = []
        if suite in ["basic", "full"]:
            results.extend(self._benchmark_next_token())
        if suite == "full":
            results.extend(self._benchmark_semantic())
        return results

    def _benchmark_next_token(self) -> List[BenchmarkResult]:
        tests = [
            (["the", "dog", "eat"], "meat"),
            (["the", "cat", "eat"], "fish"),
            (["a", "bird", "can"], "fly"),
            (["a", "fish", "can"], "swim"),
        ]
        correct = 0
        details = []
        for ctx, expected in tests:
            preds = self.model.predict_next(ctx)
            pred = preds[0][0] if preds else "?"
            ok = pred == expected
            if ok:
                correct += 1
            details.append({
                "context": " ".join(ctx), "expected": expected,
                "predicted": pred, "correct": ok,
            })
        return [BenchmarkResult(
            category="next_token", task_name="基础预测",
            accuracy=correct / len(tests), total=len(tests),
            correct=correct, details=details,
        )]

    def _benchmark_semantic(self) -> List[BenchmarkResult]:
        tests = [
            (["dog", "is", "kind", "of"], "animal"),
            (["cat", "is", "kind", "of"], "pet"),
        ]
        correct = 0
        details = []
        valid = 0
        for ctx, expected in tests:
            if expected not in self.vocab or not all(w in self.vocab for w in ctx):
                continue
            valid += 1
            preds = self.model.predict_next(ctx)
            pred = preds[0][0] if preds else "?"
            ok = pred == expected
            if ok:
                correct += 1
            details.append({
                "context": " ".join(ctx), "expected": expected,
                "predicted": pred, "correct": ok,
            })
        if valid == 0:
            return []
        return [BenchmarkResult(
            category="semantic", task_name="类别归属",
            accuracy=correct / valid, total=valid,
            correct=correct, details=details,
        )]

    def compute_stats(self) -> Dict:
        stats = self.model.get_stats()
        if self.a.has_knowledge:
            eq_norms = [float(np.linalg.norm(v @ self.a.W_Q)) for v in self.a.E.values()]
            ev_norms = [float(np.linalg.norm(v @ self.a.W_V)) for v in self.a.E.values()]
            stats['eq_norm_mean'] = float(np.mean(eq_norms)) if eq_norms else 0
            stats['ev_norm_mean'] = float(np.mean(ev_norms)) if ev_norms else 0
        return stats


# ---- 格式化 ----

def format_trace(trace: PredictionTrace, verbose: bool = False) -> str:
    lines = [
        "=" * 60,
        f"预测追踪: '{' '.join(trace.context)}' → ?",
        "=" * 60,
        f"\n目标词: '{trace.target}'",
        f"  Logit: {trace.target_logit:.2f}",
        f"  排名: {trace.target_rank}/{trace.total_candidates}",
        f"\nTop-10 预测:",
    ]
    for i, (w, score) in enumerate(trace.top_k):
        marker = "←" if w == trace.target else ""
        lines.append(f"  {i+1}. {w}: {score:.2f} {marker}")
    lines.append(f"\n源槽位投票 (Top-10):")
    for vote in trace.source_votes[:10]:
        sign = "+" if vote.contribution > 0 else ""
        lines.append(f"  Slot {vote.slot_id} ({vote.shared_words[:3]}): {sign}{vote.contribution:.2f}")
    if trace.decisive_factors:
        lines.append(f"\n决定性因素:")
        for factor, diff, desc in trace.decisive_factors:
            sign = "+" if diff > 0 else ""
            lines.append(f"  {factor}: {sign}{diff:.2f} ({desc})")
    return "\n".join(lines)


def format_benchmark(results: List[BenchmarkResult]) -> str:
    lines = ["=" * 60, "基准测试结果", "=" * 60]
    by_cat: dict = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    for cat, rs in by_cat.items():
        lines.append(f"\n[{cat}]")
        for r in rs:
            lines.append(f"  {r.task_name}: {r.correct}/{r.total} ({r.accuracy*100:.1f}%)")
    total_c = sum(r.correct for r in results)
    total_t = sum(r.total for r in results)
    if total_t:
        lines.append(f"\n总体: {total_c}/{total_t} ({100*total_c/total_t:.1f}%)")
    return "\n".join(lines)
