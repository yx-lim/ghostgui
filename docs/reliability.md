# Reliability And Quality Gates

GhostGUI treats edit-state integrity, recoverable project files, deterministic
robotics contracts, and clean process shutdown as release requirements.

## Automated Gates

Every change should pass:

```bash
python3 -m compileall -q application core gui scripts tests
python3 scripts/check_architecture.py
python3 scripts/check_docs.py
QT_QPA_PLATFORM=offscreen python3 -m unittest discover -s tests -v
python3 -m pip wheel . --no-deps --no-build-isolation --wheel-dir wheelhouse
python3 scripts/check_wheel.py --require-resources wheelhouse/*.whl
```

The architecture check enforces dependency direction:

```text
core <- application <- gui
```

Core code cannot depend on Qt or OpenGL. Application code cannot import GUI
modules. The launcher is the explicit composition-root exception that assembles
the Qt application. Compatibility facades may preserve old import paths, but
new ownership must follow these boundaries.

## Baseline

The roadmap baseline was captured on 2026-07-24 with Python 3.13 and the Qt
offscreen platform. The repository contained 206 tests. There was one expected
skip for an unavailable Z2 model and five failures:

- three playback assertions caused by a `0.20x` default that contradicted the
  documented and tested `1.00x` contract;
- Go2 accepted preview state did not become the committed state;
- a completed logical-target edit restored an FK-derived value instead of the
  value entered by the user.

The playback default was corrected with the initial guardrails. Document
ownership resolved the logical-target edit defect, and the shared contact
policy resolved the Go2 commit defect. Neither was hidden behind an
expected-failure marker.

## Regression Policy

Focused tests run before the full suite. A new failure in an unaffected
baseline test blocks the current change. A known baseline defect remains
visible until fixed; it is never hidden with an expected-failure marker.

Headless CI validates logic and widget contracts. Rendering changes also need
the manual checks in [Testing](testing.md) until deterministic visual fixtures
are available.

## Implementation Progress

The reliability roadmap is delivered in compatibility-preserving phases:

1. **Complete:** guardrails, CI, baseline, and initial packaging validation;
2. **Complete:** document, session, controller, command, and event ownership;
3. **Complete:** transactional persistence, schema migration, path safety, and
   installed resources;
4. **Complete:** shared robotics, trajectory, IK, collision, coordinate, and
   backend-selection contracts;
5. **Complete:** visualization context, displays, tools, panels, lifecycle
   contracts, and logical frame-pose services;
6. **Complete:** focused history, playback, loading, render-progress, status,
   IK-panel, trajectory-control, and camera components with compatibility
   facades;
7. **Complete:** cooperative job cancellation, late-callback suppression,
   deterministic window/session/process shutdown, coalesced render requests,
   and context-bound OpenGL resource cleanup;
8. **Complete:** isolated compatibility-matrix CI, persistence recovery and
   integration coverage, strict wheel validation, installed-package smoke
   tests, and Xvfb/software-OpenGL visual regression smoke;
9. **Complete:** final architecture decisions, contributor workflow, migration
   guide, operations and recovery runbook, changelog, and indexed navigation.

## Phase Verification Record

| Phase | Full-suite result | Additional gate |
| --- | --- | --- |
| 1 | 210 tests; two documented baseline defects, one platform skip | Architecture, docs, wheel structure |
| 2 | 220 tests; one documented baseline defect, one platform skip | Document/controller focused contracts |
| 3 | 231 tests; one documented baseline defect, one platform skip | 63.7 MB resource wheel installed from `/tmp` |
| 4 | 242 passed, one platform skip | Robotics, IK, collision, and fallback contracts |
| 5 | 248 passed, one platform skip | Visualization lifecycle and GUI refresh integration |
| 6 | 254 passed, one platform skip | History, playback, panels, IK UI, and camera integration |
| 7 | 260 passed, one platform skip | Cancellation, teardown, OpenGL, and render coalescing |
| 8 | 267 passed, two skips | Installed wheel smoke passed; visual smoke is Xvfb-gated |
| 9 | 269 passed, two skips | Final docs/architecture checks and installed wheel smoke passed |

The second Phase 8 skip is intentional when the ordinary offscreen suite does
not set `GHOSTGUI_VISUAL_TESTS`; CI runs that test separately under Xvfb. The
other skip is the existing unavailable-Z2 platform fixture.

This page records the current quality contract. Architectural ownership and
migration details live in [Architecture](architecture.md).
