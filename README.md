# Project Template

View the `phonon_lifetime` package documentation [here](https://matt-ord.github.io/phonon_lifetime/).

## Setup

This project uses a **Dev Container** (via **VS Code**) and **[`uv`](https://github.com/astral-sh/uv)** for fast, reliable Python dependency management.

### Launch the Dev Container

1. Clone the project in VS Code.

2. In VS Code, open the Command Palette (`Ctrl+Shift+P`) and run:

   ```
   Dev Containers: Reopen in Container
   ```

   This will build the container if needed and connect you to the development environment.

### Setting up the Template

To setup the template

- Re-name the `my_project` folder to `your_project_name`
- Update references in `pyproject.toml`, including the project name, description, and python.analysis.include
- Update path of imports in `tests/import_test.py` and `examples/example.py`
- Update details in `docs/source/conf.py`
- Update details in `docs/source/index.rst`
- Update the GitHub workflow step to `Build Sphinx documentation` in `.github/workflows/deploy.yml`, and the link to docs in `README.md`
- Update python.analysis.include in `.devcontainer/devcontainer.json`

### Dependency Management

This project uses **[`uv`](https://github.com/astral-sh/uv)** for dependency management.
To add a package, use the following commands:

```bash
uv add <package-name>
uv add --dev <package-name>
```

This adds the package to `pyproject.toml` and updates the lockfile,
where `--dev` adds it to the development dependencies. To install
these dependencies, use:

```bash
# All dependencies
uv sync --all-extras
# Exclude dev dependencies
uv sync
```

### Spell checker, Linter, Type Checker and Test Runner

This project sets up tools for spell checking, linting, type checking, and testing,
which are automatically run in Github Actions on push and pull requests.
To run these tools locally, use the following commands:

```bash
# Linting and formatting
ruff check .
# Type checking
ty check .
# Testing
pytest
# Spell checking
cspell .
```

To add words to the spell checker's dictionary,
add them to the `.vscode/cspell.json` file under
the `words` or `ignoreWords` section.
