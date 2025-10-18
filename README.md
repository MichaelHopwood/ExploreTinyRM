# ExploreTinyRM

Minimal, reproducible environment for exploring tiny reasoning models in PyTorch.
This repository is intentionally lightweight: notebooks plus a small importable `src` package, no heavy scaffolding.

## 1. Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

That’s it. Use your existing notebook setup (e.g., VS Code) and select this virtual environment when running notebooks.

## 2. Reproducibility

We use two layers:

- `pyproject.toml` - **loose version ranges** for fast iteration.
- `requirements-lock.txt` - **exact pins** for team synchronization.

After your first successful local install:

```bash
pip freeze --exclude-editable > requirements-lock.txt
```

**Collaborators** can reproduce the environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e .
```

If dependencies change, regenerate and commit a new `requirements-lock.txt`.

## 3. Using Notebooks

Create notebooks under `notebooks/` and ensure your editor uses this repo’s virtual environment.
Import algorithm code from `src/exploretinyrm/` as you add it.

## 4. Repository Layout

```
ExploreTinyRM/
├─ .gitignore
├─ .gitattributes         # only relevant if you choose to track large model files with LFS
├─ LICENSE
├─ README.md
├─ pyproject.toml
├─ requirements-lock.txt  # generated and committed after first install
├─ src/
│  └─ exploretinyrm/
│     └─ __init__.py
├─ notebooks/
├─ scripts/
└─ tests/
```

