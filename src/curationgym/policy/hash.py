"""Policy hashing for reproducibility and caching."""

import hashlib
import json
import subprocess
from typing import Any

from curationgym.policy.schema import Policy


def canonicalize_policy(policy: Policy) -> str:
    """Convert policy to canonical JSON string for hashing."""
    data = policy.to_dict()
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_policy_hash(policy: Policy) -> str:
    """Compute deterministic hash of policy configuration."""
    canonical = canonicalize_policy(policy)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def get_code_version() -> str:
    """Get current git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def compute_reproducibility_key(
    policy: Policy,
    input_signature: str,
    code_version: str | None = None,
) -> str:
    """Compute full reproducibility key for artifact lookup.

    Combines:
    - Policy configuration hash
    - Input data signature
    - Code version
    """
    policy_hash = compute_policy_hash(policy)
    code_ver = code_version or get_code_version()

    key_data = {
        "policy": policy_hash,
        "input": input_signature,
        "code": code_ver,
    }
    canonical = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def policy_to_yaml(policy: Policy) -> str:
    """Convert policy to YAML string."""
    import yaml
    return yaml.dump(policy.to_dict(), sort_keys=True, default_flow_style=False)


def policy_from_yaml(yaml_str: str) -> Policy:
    """Load policy from YAML string."""
    import yaml
    data = yaml.safe_load(yaml_str)
    return Policy.from_dict(data)
