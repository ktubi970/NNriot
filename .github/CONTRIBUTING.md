# Contributing to NNriot

Thank you for your interest in contributing to **NNriot**! We welcome contributions from developers, data scientists, and League of Legends gaming enthusiasts.

---

## Code of Conduct

Please treat all community members with respect and professionalism. Maintain constructive and supportive communication in issues, PRs, and discussions.

---

## How to Contribute

### 1. Reporting Bugs
Before opening a new issue, please search the existing issue tracker to see if the issue has already been reported. When filing a bug report:
- Use the **Bug Report** template.
- Include Python version, OS details, and steps to reproduce.
- Provide log tracebacks where applicable.

### 2. Suggesting Enhancements
Feature requests are welcome!
- Use the **Feature Request** template.
- Clearly describe the problem you want to solve and your proposed feature design.

### 3. Submitting Pull Requests (PRs)
1. **Fork & Branch:** Create a feature or fix branch from `main`:
   ```bash
   git checkout -b feature/my-cool-feature
   ```
2. **Environment Setup:** Set up your virtual environment and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. **Run Tests:** Ensure all unit tests pass before committing:
   ```bash
   pytest tests/
   ```
4. **Clean Commits:** Keep commit messages clear, descriptive, and focused.
5. **Open a PR:** Fill out the PR template completely.

---

## Code Style & Standards

- Follow **PEP 8** guidelines for Python code.
- Write clean, modular, and self-documenting code with concise docstrings.
- Ensure new features are accompanied by corresponding unit tests in `tests/`.
- Do NOT commit sensitive data, `.env` files, API keys, or large `.db` / `.keras` model binaries.

---

## Questions & Support

If you have questions regarding architecture or analytics models, please refer to:
- [README.md](../README.md)
- [Analytics Documentation](../docs/analytics.md)
