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
- Keep this `AGENTS.md` and the `ROADMAP.md` synchronized with the current
  project state. Document new modules, tasks and design decisions as they are
  introduced.
- Hardware-specific optimisations automatically adjust worker count using
  `monGARS.utils.hardware`. Physical cores are preferred but the helper
  falls back to logical CPUs when necessary. See `README.md` for details.
- Use `build_embedded.sh` to create and push multi-arch images for Raspberry Pi and Jetson using Docker Buildx. Ensure you are logged in to your registry before running the script.
- Use `build_native.sh` to build an optimized x86_64 image leveraging all CPU cores on a developer workstation.
- Kubernetes RBAC policies were tightened. Refer to `rbac.yaml`.
- A `PeerCommunicator` module provides encrypted message passing between nodes. Use `/api/v1/peer/message` to receive messages. The route requires authentication and the JSON body `{ "payload": "..." }`.
 - Record common errors and the strategies developed to resolve them here so
  future contributors don't repeat the same investigation.
 - Keep a running log of new ideas and experimental results. Note what works well
  and what doesn't so the team can build on prior lessons.

## Pull Requests

1. Keep PRs focused on a single topic.
2. Include a clear description of your change and testing performed.
3. Ensure `pytest` passes and code is formatted with `black` before requesting review.
4. Name branches using `feature/<topic>` or `fix/<issue>` for clarity.

Issue and pull request templates are provided in `.github/`. Use them to maintain a consistent workflow.

For questions or discussion, open an issue or reach out to the maintainers.
