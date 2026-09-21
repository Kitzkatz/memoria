# Project Manifesto

## Memory Daemon

A fully local cognitive memory architecture.

---

## Goal

Provide an extensible long-term memory system for language models.

The memory system should remain independent of any individual LLM.

**The LLM is replaceable.**

**Memory is not.**

---

## Philosophy

The system is built around one idea:

> **Reasoning should not depend on memory.**
>
> **Memory should support reasoning.**
>
> **Never own it.**

### What This Means

**Reasoning should not depend on memory.**
- You should be able to reason without retrieving
- Reasoning is a separate cognitive function
- Memory is a substrate, not a crutch

**Memory should support reasoning.**
- Memory provides the facts, context, and history
- It should be fast, accurate, and relevant
- It should surface what matters, not everything

**Never own it.**
- The system should never claim to "know" something
- It should never make a claim without attribution
- It should always be able to cite its sources

---

## Vision

### V3: Reliable Memory
- Store and retrieve memories reliably
- Multiple retrieval strategies (FAISS, BM25, Graph)
- Feedback loop for improvement
- Local-first, no cloud dependencies

**Status: ✅ Complete**

### V4: Reasoning Infrastructure
- Goals and planning
- Blackboard architecture
- Parallel task execution
- Computation graphs
- Active reasoning over memory

**Status: 🚧 In Progress**

### V5: Cognitive Architecture
- Hierarchical memory representation
- Minimal reconstruction cost
- Compression as a side effect
- Research phase

**Status: 🔬 Research**

### Eventually: Operating System for Intelligent Agents
- A complete cognitive architecture
- Memory, reasoning, planning, execution
- Self-improving over time
- Fully local, fully private

**Status: 🔮 Vision**

---

## Core Principles

### 1. Local First
- No cloud dependencies
- Runs on your hardware
- Data stays on your machine
- Privacy by default

### 2. LLM Agnostic
- Swap models without changing the system
- Works with any LLM (Mistral, Llama, GPT, etc.)
- The LLM is a plugin, not the core

### 3. Observable
- Every decision is traceable
- Flight recorder for debugging
- Diagnostics and metrics

### 4. Extensible
- Pluggable components
- Custom signals and strategies
- Easy to add new retrieval methods

### 5. Efficient
- 150ms query latency on 4GB RAM
- CPU-only inference
- Optimized for local hardware

---

## What We Are Not

| We Are Not | Because |
|------------|---------|
| A vector database | Vector search is one part of memory, not the whole |
| A chat UI | We provide memory, not a chat interface |
| A replacement for your brain | Memory is a tool, not a person |
| A cloud service | Local-first, always |
| A product | We're building a system, not a product |

---

## What We Are

- A memory system for intelligent agents
- A research platform for cognitive architecture
- A local-first alternative to cloud memory
- An extensible framework for memory experiments

---

## The Big Idea

> Most AI systems treat memory as an afterthought.
>
> We treat memory as the foundation.
>
> Everything else is built on top.

---

## See Also

- `02_project_overview.md` — Project overview
- `05_Design_Principles.md` — Design decisions
- `06_roadmap.md` — Where we're going
