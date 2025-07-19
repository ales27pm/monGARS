# monGARS Contribution Guide

This repository contains the source code for **monGARS**, a modular, privacy-first AI system. The project includes Python services, a Django-based web interface, container orchestration and deployment scripts.

## Code Style

- **Python**: Follow [PEP 8](https://peps.python.org/pep-0008/) conventions with 4 space indentation.
- Use `black` for formatting and `isort` for import organization.
- Type hints are encouraged throughout the codebase.
- Commit messages should be concise and in present tense: `Add feature` / `Fix bug` / `Refactor module`.

## Tests

- Run `pytest` to execute the unit and integration tests found in the `tests/` directory.
- All new features should include accompanying tests when practical.

## Containers & Deployment

- Use `docker-compose up` to launch required services (PostgreSQL, Redis, and optional ML components).
- Kubernetes manifests are provided in `k8s/` for cluster deployment. Ensure RBAC and resource limits are adjusted for your environment.

## Documentation

 - Update relevant documentation when modifying or adding functionality.
 - High-level architectural notes are located in `monGARS_structure.txt` and the project ROADMAP.
 - Keep the `README.md` up to date when behaviour or setup steps change.

## Pull Requests

1. Keep PRs focused on a single topic.
2. Include a clear description of your change and testing performed.
3. Ensure `pytest` passes and code is formatted with `black` before requesting review.
4. Name branches using `feature/<topic>` or `fix/<issue>` for clarity.

Issue and pull request templates are provided in `.github/`. Use them to maintain a consistent workflow.

For questions or discussion, open an issue or reach out to the maintainers.
