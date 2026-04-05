# Contributing to Synapt

## Getting started

```bash
git clone https://github.com/synapt-dev/synapt.git
cd synapt
pip install -e ".[test]"
pytest tests/ -v
```

## Workflow

1. Fork the repo and create a feature branch
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Submit a pull request

## Code style

- Python 3.10+ with type annotations
- Use `from __future__ import annotations` in all modules
- Keep dependencies minimal

## Testing

Tests live in `tests/recall/` for recall features and `tests/` for top-level modules.

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/recall/test_core.py -v
```

## Plugin development

See the [README](README.md#plugins) for plugin creation docs. The plugin protocol is:

1. Export a `register_tools(mcp: FastMCP) -> None` function
2. Optionally set `PLUGIN_NAME` and `PLUGIN_VERSION` module attributes
3. Register via `[project.entry-points."synapt.plugins"]` in your `pyproject.toml`
