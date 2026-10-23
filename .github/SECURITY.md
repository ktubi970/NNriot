# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |

---

## Reporting a Vulnerability

We take the security of **NNriot** seriously. If you discover a security vulnerability or credential leak issue, please **DO NOT** create a public issue on GitHub.

Instead, report it privately through GitHub's [private vulnerability reporting](https://github.com/ktubi970/NNriot/security/advisories/new):
1. Provide a detailed summary of the issue, including proof of concept or steps to reproduce.
2. Outline the potential impact of the vulnerability.

If you accidentally commit a Riot API key, revoke it immediately on the [Riot Developer Portal](https://developer.riotgames.com/) and regenerate a new one; removing it from Git history alone is not enough.

### What NOT to share publicly:
- Live Riot API keys or private tokens
- Database dump credentials or production server configurations

We will acknowledge your report within 48 hours and work with you to patch any confirmed security issues before public disclosure.
