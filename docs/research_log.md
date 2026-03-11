# Research Log: From Context Management to Categorical Voting

This document traces the intellectual journey that led to MathBrain, from initial frustration with LLM context limits through several failed approaches to the final architecture. It is written in roughly chronological order.

## Phase 0: The Trigger

The research began with a simple frustration: commercial AI assistants have terrible context management. Long conversations degrade, important information gets lost, and there is no principled way to manage what the model remembers.

This led to a question: **how should an intelligent system manage its activation history?**

## Phase 1: BubbleStream (Inference-Layer Solution)

The first attempt was an engineering solution called BubbleStream -- a staged inference framework that packages model outputs into indexed memory blocks and dynamically injects them into the model's context window as needed.

The idea: instead of keeping everything in context, let memory blocks "bubble up" on demand. The model processes in a rolling window, and relevant memory chunks appear when the retrieval system thinks they're needed.

**Result**: With only 16K effective context, BubbleStream achieved 94% memory precision across 32K cumulative dialogue. The remaining 6% could be recovered through prompt optimization.

**But**: During retrospective analysis, it became clear that precision was entirely determined by the memory block management mechanism. Embedding-based retrieval would inevitably degrade as the memory grew. A brief attempt at graph-based memory storage was abandoned -- it felt like patching, not solving.

The prototype lives at a separate repository and was not pursued further, but it validated that memory management is a real and solvable problem.

## Phase 2: GPT-2 Dissection

The next step was to understand how Transformers actually use context. Analysis of GPT-2's attention patterns revealed a striking finding:

> **99% of the variance in next-token prediction is explained by the most recent ~30 tokens.** Memory-related information explains only the last fraction -- but it plays a decisive role in ranking.

This led to a key reframing:

> Memory does not need to be complete. It only needs to provide enough voting evidence to tip the ranking at decision time. **Memory is hallucinated, not retrieved.**

## Phase 3: External Memory Matrix

Based on this insight, GPT-2 was modified with an external memory matrix that the model could read from and write to during inference.

**Result**: 10-20% perplexity reduction on the same dataset compared to vanilla GPT-2.

This confirmed that external memory could meaningfully improve prediction, but the approach was still fundamentally a Transformer patch -- not a new architecture.

## Phase 4: The Categorization Insight

A deeper investigation followed, drawing on findings from neuroimaging and psycholinguistics -- particularly research on categorical perception in speech.

The key observation from neuroscience:

> **The defining feature of human cognition is categorization.** All responses are discrete and finite. Phonemes, words, actions -- the brain does not produce continuous outputs. It selects from bounded categories.

This leads to a powerful implication:

> If all responses are already discretized, then in principle all possible experiences can be enumerated. Memory does not need to store complete episodes -- it only needs to mark states. The learning task becomes: **"given this compressed history state, what category should the system select?"**

## Phase 5: Sequence Encoding Attempts (Dead Ends)

With the categorization framework in mind, the focus shifted to encoding sequential context into a bounded representation.

**Attempt 1: Cosine chaos encoding.** A recursive cosine map was designed to fold positional information into bounded features. But excessive chaos made the encoding unlearnable -- chaotic systems are analytically intractable, and the model could not extract stable patterns.

**Attempt 2: Continuous rotational encoding on the vocabulary.** The idea was to apply rotary-style positional encoding directly to vocabulary embeddings, so that inner products would implicitly encode context order. This failed because it lacked cross-term interactions and had poor discriminability between positions.

**Critical realization at this point:**

> All previous approaches were still trapped in the **retrieval-based memory** paradigm -- assuming that memory requires exact pattern matching against stored episodes.

## Phase 6: The Voting Breakthrough

The breakthrough came from combining two observations:

1. **Memory is emergent from local patterns, not one-shot matching.** Remembering "ABCDEFG → H" does not require storing the full sequence. It only requires local patterns: A→B, AB→C, etc. And even these don't need exact storage -- B just needs to score slightly higher than competing candidates.

2. **Neural computation resembles voting.** Research in computational neuroscience suggests that the brain's decision process is well-modeled as a voting rule across local pattern detectors. The key enabler is categorical response -- because the output space is discrete and bounded, voting can work.

This reframing was decisive:

> **Replace "horizontal memory retrieval" (searching through stored episodes) with "vertical memory encoding" (writing context order directly into the vocabulary space).** Since categorical responses are fixed, and response order determines context, we only need to encode positional information into the vocabulary and compute inner products. Under this logic, inference becomes constant-time.

## Phase 7: EMA Encoding

The remaining problem was: how to encode position?

The answer came during a train ride home:

> **Exponential moving average (EMA) with multiple decay rates produces a unique encoding of history order.** Each word's EMA state is a vector of decayed activations across N timescales. This is bounded, monotonically decaying, and -- crucially -- uniquely identifies the temporal context without storing the sequence.

This directly solved the discriminability problem that killed the rotational encoding approach. EMA states are analytically tractable, bounded, and information-theoretically sufficient for the voting task.

## Phase 8: MathBrain

From the EMA encoding insight, the full architecture was designed in rapid succession:

1. **Hash Retina**: Deterministic character n-gram hashing for vocabulary encoding (no learned embeddings)
2. **Q State**: Multi-timescale EMA compression (N=4 decay rates)
3. **φ Encoder**: Cosine-chaos feature map at the edge of chaos (α=2) -- the chaos encoding from Phase 5 was rescued by operating at a specific regime
4. **B Memory**: Sparse hash-table storage with delta-rule online learning (hippocampus analog)
5. **A Knowledge**: Slice-wise low-rank factorization for long-term structure (neocortex analog)
6. **Wake-Sleep**: Online wake writes to B, offline sleep consolidates B→A

The implementation went through 8 iterations over the winter break, solving efficiency problems with hash tables, fixing norm suppression bugs, and validating on synthetic and small-scale corpora.

The core mathematical logic has remained stable since the initial design. What changed was engineering: making it fast, making it sparse, making it work.

## Current State

The architecture is validated at small scale (TinyStories, PTB). Multi-layer extensions show promising accuracy improvements (93% → 97%). Large-scale experiments are in progress.

This is a solo research project seeking collaborators and lab support. The goal is not to replace Transformers on broad benchmarks, but to demonstrate that a fundamentally different regime -- bounded state, categorical voting, online learning, full interpretability -- is viable and worth investigating.

## Side Products

Two other artifacts emerged from this research journey:

- **BubbleStream**: An inference-layer context management framework for existing LLMs. Functional prototype, not open-sourced.
- **Graph-based code indexer**: A structural code analysis tool based on graph representations. The underlying idea could extend to vibe-coding editors, PCB layout tools, and other structured-document applications.
