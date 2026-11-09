"""The compose file encodes decisions that must not regress (hosts, seed, Ollama address)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
SERVICES = COMPOSE["services"]


def test_mlflow_host_validation_names_the_hosts_and_never_uses_a_wildcard() -> None:
    hosts = SERVICES["mlflow"]["environment"]["MLFLOW_SERVER_ALLOWED_HOSTS"].split(",")
    assert "mlflow:5000" in hosts  # the Host header containers send; the port is part of the entry
    assert "localhost:*" in hosts and "127.0.0.1:*" in hosts  # the host-side clients
    assert "*" not in [h.strip() for h in hosts]  # "*" would switch the protection off
    assert not any(h.strip().startswith("*") for h in hosts)
    command = " ".join(SERVICES["mlflow"]["command"])
    assert "--disable-security-middleware" not in command


def test_the_api_waits_for_the_seed_which_waits_for_a_healthy_registry() -> None:
    assert SERVICES["seed"]["depends_on"]["mlflow"]["condition"] == "service_healthy"
    assert SERVICES["api"]["depends_on"]["seed"]["condition"] == "service_completed_successfully"
    assert SERVICES["seed"]["restart"] == "no"  # a one-shot step
    assert SERVICES["seed"]["command"] == ["python", "-m", "failsense.models.seed_registry"]


def test_every_client_talks_to_the_registry_by_its_service_name() -> None:
    for name in ("api", "seed"):
        assert SERVICES[name]["environment"]["FAILSENSE_MLFLOW_URI"] == "http://mlflow:5000"
    assert SERVICES["dashboard"]["environment"]["FAILSENSE_API_URL"] == "http://api:8000"
    assert "FAILSENSE_MLFLOW_URI" not in SERVICES["dashboard"]["environment"]  # API-only


def test_the_ollama_address_comes_from_the_environment_with_a_host_default() -> None:
    value = SERVICES["api"]["environment"]["OLLAMA_BASE_URL"]
    assert value == "${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
    assert "host.docker.internal:host-gateway" in SERVICES["api"]["extra_hosts"]  # Linux hosts


def test_env_example_documents_the_variable_and_holds_no_secret() -> None:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OLLAMA_BASE_URL=http://host.docker.internal:11434" in text
    assert "OLLAMA_HOST=0.0.0.0" in text  # the Windows gotcha is written down
    assert not any(k in text.lower() for k in ("password", "token", "secret=", "api_key"))


def test_ports_are_bound_to_localhost_only() -> None:
    for name in ("mlflow", "api", "dashboard"):
        assert all(p.startswith("127.0.0.1:") for p in SERVICES[name]["ports"]), name


def test_the_image_contains_the_bundled_model_the_seed_needs() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --chown=app:app models ./models" in dockerfile
    assert (ROOT / "models" / "failsense-q1" / "model.skops").is_file()
    assert "models" not in (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
