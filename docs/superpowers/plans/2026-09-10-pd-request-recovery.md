# PD Request Recovery Implementation Plan

**Goal:** Replace capacity-triggered segmented generation with recovery of the original decode request.
**Architecture:** Retain InferReq on D; restore missing KV via P using exact token IDs and epoch-scoped transfer tasks. P contexts retain multimodal resources until logical completion and local cleanup.
**Tech Stack:** Python, asyncio, ctypes shared requests, existing PD transport.
**Spec:** ../specs/2026-09-10-pd-request-recovery-design.md

## Global Constraints

Python 3.10+; no new dependencies; no committed unit test changes; wrap experiments with exp.

- [x] D scheduler: replace capacity finish with pause, process cancellation before paused state, add a recovery hook. Preserve token counters and sampling state; reserve decode headroom before receiving KV.
- [x] Transport: add recovery epoch and local owner ID to task metadata; transfer completion uses local owner, stale epochs are ignored; suppress recovery first-token injection.
- [x] P context: retain original multimodal allocations using work references; reuse IDs without tokenization; close only after work and shared-request cleanup.
- [x] Master: remove segmented generation; forward recovery metadata with fresh P work IDs; propagate errors to original request; cleanup on success, failure, disconnect and deadline.
- [x] Verify: existing CPU tests, external recovery checks, compile and repository pre-commit. Inspect final diff and document GPU validation limitations.

## Verification result

72 existing CPU checks passed. 21 external checks in `/tmp/test_pd_request_recovery.py` passed, covering exact IDs, two recovery epochs, local cache hits, transfer failure, cancellation, resource ownership and deadlines. Four existing dynamic-split assertions require the removed capacity-finish behavior and fail as expected; test files were not modified per user preference. Black and flake8 passed. No real GPU PD integration or performance run was performed.

Rebased onto upstream/main `cd2edb90`; recovery uses the new hybrid checkpoint API and `att_state` transfer page kind. Re-ran the 72 existing checks and all 21 external checks after adapting the interfaces.
