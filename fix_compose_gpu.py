#!/usr/bin/env python3
import yaml
from pathlib import Path

COMPOSE_FILE = Path("docker-compose.yml")
BACKUP_FILE  = Path("docker-compose.yml.bak")
SERVICES     = ["rayserve", "api", "embedded", "worker"]

def main():
    text = COMPOSE_FILE.read_text()
    data = yaml.safe_load(text)

    # Backup
    BACKUP_FILE.write_text(text)

    # Remove version
    data.pop("version", None)

    # Merge & inject per-service
    services = data.get("services", {})
    for name in SERVICES:
        if name not in services:
            continue
        svc = services[name]
        # Ensure gpus
        svc["gpus"] = "all"

        # Collect & merge env
        env = {}
        raw_env = svc.get("environment", {})
        # Handle list form or dict form
        if isinstance(raw_env, list):
            for item in raw_env:
                if isinstance(item, str) and "=" in item:
                    k, v = item.split("=", 1)
                    env[k] = v
        elif isinstance(raw_env, dict):
            env.update(raw_env)

        # Inject NVIDIA vars
        env.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
        env.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

        svc["environment"] = env

    # Write back
    COMPOSE_FILE.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"✅ Fixed {COMPOSE_FILE} (backup at {BACKUP_FILE})")
    print("Next: run `docker compose config` to validate.")

if __name__ == "__main__":
    main()
