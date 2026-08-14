# ai-scheduler

Heterogeneous-computing-aware MoE expert offloading: scheduling + simulation.

This repository hosts **moesim** (`moesim/`), a domain-agnostic discrete-event
simulation engine paired with a heterogeneous-computing-aware scheduler for
Mixture-of-Experts (MoE) expert offloading. CPU compute capacity participates
in scheduling decisions, not just memory.

See `moesim/README.md` for quickstart, architecture, and the v2 roadmap. The
authoritative design spec is `moesim/docs/superpowers/specs/2026-08-09-moesim-design.md`.
