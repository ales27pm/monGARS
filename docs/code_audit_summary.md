# Code Audit Summary — 2024-11-26

## Scope
- Triggered full unit test suite with `pytest` to surface runtime defects.
- Focused remediation on authentication stack where import-time user bootstrapping exercises password hashing routines.

## Findings
- `monGARS/core/security.py` instantiated a `CryptContext` with the `bcrypt` scheme. In environments lacking the optional C backend, Passlib raises `ValueError: password cannot be longer than 72 bytes` during backend detection.
- The error prevented API modules from importing, causing every FastAPI test module to fail during collection.

## Remediation
- Switched password hashing to `pbkdf2_sha256` with 390,000 rounds. The algorithm is implemented in pure Python and avoids the brittle backend detection path while maintaining a high work factor.
- Added inline documentation explaining the rationale so future maintainers understand the security and portability trade-offs.

## Recommendations
- If legacy bcrypt hashes exist in production data, plan a rolling re-hash to PBKDF2 on next successful login, or ensure the `bcrypt` wheel is bundled with the deployment image.
- Consider pinning `Passlib` in `requirements.txt` and capturing backend availability in CI to avoid future regressions.
