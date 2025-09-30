# Code Audit Summary — 2024-11-26

## Scope
- Executed the full `pytest` suite to surface runtime defects.
- Focused on the authentication stack where import-time user bootstrapping
  exercises password hashing routines.

## Findings
- `monGARS/core/security.py` instantiated a `CryptContext` with the `bcrypt`
  scheme. In environments lacking the optional C backend, Passlib raised
  `ValueError: password cannot be longer than 72 bytes` during backend detection,
  preventing FastAPI modules from importing.

## Remediation
- Switched password hashing to `pbkdf2_sha256` with 390k iterations—a pure-Python
  implementation that avoids brittle backend detection while maintaining a strong
  work factor.
- Added inline documentation clarifying the trade-offs so future maintainers know
  why bcrypt was replaced.

## Recommendations
- If legacy bcrypt hashes exist, rehash to PBKDF2 on the next successful login or
  bundle the `bcrypt` wheel in deployment images.
- Pin `passlib` in `requirements.txt` and record backend availability in CI to
  prevent regressions.
- Periodically rerun the audit after dependency upgrades or authentication flow
  changes.
