# Team Development Workflow

This document defines how we collaborate on this repository to keep work
organized, reduce merge conflicts, and maintain a stable `main` branch.

------------------------------------------------------------------------

# Branching Strategy

## Main Branch

-   `main` is always stable and runnable.
-   No one pushes directly to `main`.
-   All changes must go through a Pull Request (PR).

------------------------------------------------------------------------

## Feature Branches

Work is split by **feature**, not by person.

Each new task gets its own branch.

### Branch Naming

Use clear prefixes:

    feat/<feature-name>
    fix/<bug-name>
    chore/<non-feature-task>
    docs/<documentation-update>

Examples:

    feat/dataset-loader
    feat/train-smoke
    feat/onnx-export
    fix/windowing-bug
    chore/ci-setup

------------------------------------------------------------------------

# Daily Workflow

## 1. Start from Updated Main

``` bash
git checkout main
git pull
git checkout -b feat/your-feature
```

------------------------------------------------------------------------

## 2. Work and Commit Frequently

Make small, focused commits.

Example:

    feat: add 1D CNN model
    fix: correct batch dimension issue

------------------------------------------------------------------------

## 3. Push Branch

``` bash
git push -u origin feat/your-feature
```

------------------------------------------------------------------------

## 4. Open Pull Request

-   Open PR into `main`
-   Add description:
    -   What changed
    -   How to test
    -   Any relevant logs/screenshots

------------------------------------------------------------------------

## 5. Review and Merge

-   At least one teammate reviews
-   CI must pass
-   Use **Squash and Merge**
-   Delete branch after merge

------------------------------------------------------------------------

# Feature Definition

A feature should:

-   Be mergeable in 1--3 days
-   Add a concrete capability
-   Not depend on unfinished code
-   Leave the repo in a working state

------------------------------------------------------------------------

# Example Feature Breakdown for This Project

## Phase 1

-   `feat/project-skeleton`
-   `feat/dataset-loader`
-   `feat/preprocessing`
-   `feat/train-smoke-epoch`
-   `chore/ci-setup`
-   `feat/board-ssh-setup`

## Phase 2

-   `feat/eval-metrics`
-   `feat/macs-profiler`
-   `feat/inference-benchmark`

## Phase 3

-   `feat/model-depth-variant`
-   `feat/kernel-experiments`
-   `feat/hyperparam-sweep`

## Phase 4

-   `feat/export-onnx`
-   `feat/cpp-inference-runner`
-   `feat/npu-deployment`

## Phase 5

-   `feat/sensor-driver`
-   `feat/realtime-windowing`
-   `feat/live-inference-loop`

------------------------------------------------------------------------

# Folder Ownership (To Reduce Conflicts)

Assign general ownership areas:

-   ML Pipeline → `ml/`
-   Deployment / Board → `device/`
-   Sensor + C++ → `device/cpp/`
-   CI + Scripts → `.github/`

------------------------------------------------------------------------

# Pulling Updates While Working

Before pushing large updates:

``` bash
git checkout main
git pull
git checkout feat/your-feature
git merge main
```

Resolve conflicts early.

------------------------------------------------------------------------

# Team Rules

-   No direct pushes to `main`
-   Keep PRs small
-   Merge frequently
-   Keep branches short-lived
-   CI must stay green

------------------------------------------------------------------------

# Goal

-   Parallel development
-   Minimal merge conflicts
-   Clear history
-   Always-working `main`
