# Contributing To GhostGUI

Thank you for improving GhostGUI. Keep changes focused, preserve model and data
contracts, and update the documentation that users rely on.

## Development Setup

Install the checkout using the platform instructions in
[`docs/install.md`](docs/install.md). Linux/Ubuntu is the primary development
and test platform.

For an existing checkout:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Launch a model through the packaged entry point:

```bash
ghostgui --model g1
```

Use `python3 scripts/run_gui.py --model g1` only when testing compatibility with
the direct checkout launcher.

## Branches And Scope

Use a short branch name that describes one outcome, such as:

```text
feature/model-import-status
fix/keyframe-export
docs/data-format-example
```

Avoid mixing formatting, generated assets, model changes, and behavior changes
in one pull request unless they are required for the same outcome.

## Code Changes

- Follow the surrounding Python style and keep application, core, and GUI
  responsibilities separated.
- Read [`docs/architecture.md`](docs/architecture.md) before moving ownership
  across layers.
- Route editor mutations through typed commands and keep compatibility facades
  thin.
- Implement reusable visualization behavior as a display, tool, panel, or
  frame-pose service with an explicit lifecycle.
- Put file and workflow logic in `application/` rather than directly in Qt
  callbacks when a reusable service is appropriate.
- Keep model-specific knowledge in the model registry or adapter instead of
  scattering robot-name checks across the UI.
- Add or update tests for behavior changes.
- Do not edit bundled robot assets or example CSV files casually; explain the
  source and validation for any necessary change.
- Preserve existing user work in the checkout and avoid destructive Git
  operations.

## Canonical Terminology

User-facing UI, help, tutorials, and documentation use:

| Use | Avoid |
| --- | --- |
| Keyframe | Legacy timeline synonyms |
| Commit Keyframe | Legacy save-action labels |
| Keyframe interval | Legacy step labels |
| Orange preview | Ambiguous “current robot” wording |
| Preview Path | Labels that imply the preview is saved |
| End Effector | Abbreviations in beginner-facing copy |
| Joint Angles | Implementation-specific joint-state wording |

Internal function names can be refactored separately when there is a technical
reason. Do not expose legacy internal names in new user-facing copy.

## Tests

Run the complete suite:

```bash
python3 scripts/run_test_suite.py
```

Run the static, architecture, and documentation gates:

```bash
python3 -m compileall -q application core gui scripts tests
python3 scripts/check_architecture.py
python3 scripts/check_docs.py
```

Use the focused commands and manual GUI checklist in
[`docs/testing.md`](docs/testing.md) while developing.

Changes to persistence, resources, packaging, rendering, or startup also require
the applicable recovery, installed-wheel, or Xvfb gates documented there.

## Documentation

Update documentation in the same change when behavior, UI labels, commands,
model support, or file formats change.

- Keep `README.md` concise and link to detailed pages.
- Put supported user behavior in the public pages under `docs/`.
- Do not publish one-off diagnosis notes; move still-relevant behavior into the
  focused public guide that owns it.
- Verify every command against the repository.
- Do not claim a platform or model is tested without evidence.
- Add new public pages to `docs/README.md`.
- Run link and terminology validation before review.

## Pull Request Checklist

- [ ] The change has one clear purpose.
- [ ] New behavior has automated tests where practical.
- [ ] Focused tests passed before the complete isolated suite.
- [ ] Architecture and documentation validation passed.
- [ ] Relevant manual GUI checks were completed.
- [ ] Public commands and links were verified.
- [ ] User-facing terminology is consistent.
- [ ] README and focused docs were updated when required.
- [ ] No generated caches, local projects, or unrelated assets were added.
- [ ] Known limitations and unverified platforms are described honestly.
