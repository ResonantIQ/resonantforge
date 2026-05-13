# Contributing to ResonantForge

Thank you for your interest in contributing to ResonantForge.

## Filing issues

Use [GitHub Issues](https://github.com/resonantiq/resonantforge/issues) for bug reports, feature requests, and documentation improvements. Please search existing issues before opening a new one.

The maintainer team uses Jira for internal sprint coordination. External contributors should use GitHub Issues — Jira is not accessible to the public and you do not need to create a Jira ticket to contribute.

## Types of contributions welcome

- **Bug reports** — clear reproduction steps, expected vs. actual behavior, Python version and OS
- **Feature requests** — describe the use case, not just the implementation
- **Documentation improvements** — typos, unclear explanations, missing examples
- **Test cases** — additional corpus properties, edge cases in validators, replay harness coverage
- **New profile implementations** — subclasses of `Profile` for domains beyond SaaS and Professional Services

## Pull request process

1. Fork the repository and create a branch off `main`.
2. Keep pull requests focused on one thing. A PR that fixes a bug and adds a feature is two PRs.
3. Add tests for any behavior you change or add. All existing tests must pass.
4. Update documentation if your change affects the public interface or CLI flags.
5. Open the PR against `main`. The description should explain what changed and why.

## Code style

ResonantForge uses [ruff](https://docs.astral.sh/ruff/) for formatting and linting.

Before submitting:

```bash
ruff format .
ruff check .
```

Both must be clean. The `pyproject.toml` contains the configuration.

## Running tests

```bash
cd harness
pip install -e .
pytest tests/
```

All tests must pass before a PR can be merged.
