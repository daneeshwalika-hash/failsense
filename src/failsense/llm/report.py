"""German maintenance report from a prediction and its top SHAP factors, via a local LLM (Ollama).

The LLM only **phrases** what the model already computed: risk, alert, threshold and the top
factors come from ``POST /explain``; the LLM never sees raw readouts and never decides the risk
level. The output is machine-checked before anyone sees it:

* valid JSON that matches ``MaintenanceReport`` (Ollama's structured output, then Pydantic);
* **grounded**: every ``reasons[].factor`` is one of the supplied factors (no invented causes);
* ``risk_level`` equals the one computed in code from risk / threshold;
* the text is German (a stop-word heuristic).

If the LLM is missing, slow, or fails the checks twice, a deterministic template report is returned
(``source = "fallback"``): the service never fails and never shows an unchecked text.

Why Ollama through plain HTTP (``urllib``): no extra dependency, the model runs on the user's
machine (no patient/fleet data leaves the premises), and ``/api/chat`` with a JSON schema in
``format`` constrains the output (docs.ollama.com/capabilities/structured-outputs).

Honesty: the SCANIA variables are anonymised, so an explanation can only say that a factor raises or
lowers the *model's* risk; it must not claim a cause. The prompt (``prompts/``) forbids it and the
fallback text says it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from failsense.config import resolve_path

logger = logging.getLogger(__name__)

OLLAMA_URL_ENV = "OLLAMA_BASE_URL"
LEGACY_OLLAMA_URL_ENV = "FAILSENSE_OLLAMA_URL"  # accepted for older setups
RiskLevel = Literal["niedrig", "erhöht", "hoch"]
HIGH_RISK_FACTOR = 2.0  # risk >= 2 x threshold -> "hoch"
MAX_VALUE_LENGTH = 32
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9_.\-]+$")

_GERMAN = (
    "der die das den dem des und ist sind nicht ein eine einer einem im in zu von mit für auf bei "
    "als auch wird werden risiko liegt über unter bis nach vor aus durch dass oder aber nur "
    "noch sich es sie wir zeitnah wartung fahrzeug"
)
_ENGLISH = (
    "the and is are not a an of in to with for on at as also will be this that or but only it "
    "they we risk above below by from which should would could"
)
GERMAN_WORDS = frozenset(_GERMAN.split())
ENGLISH_WORDS = frozenset(_ENGLISH.split())


class Reason(BaseModel):
    """One factor and how it moved the risk (model statement, not a cause)."""

    model_config = ConfigDict(extra="forbid")

    factor: str = Field(min_length=1, max_length=80)
    explanation: str = Field(min_length=10, max_length=300)


class MaintenanceReport(BaseModel):
    """The structured report; this schema is sent to Ollama as the output format."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=20, max_length=400)
    risk_level: RiskLevel
    recommendation: str = Field(min_length=20, max_length=400)
    reasons: list[Reason] = Field(min_length=1, max_length=5)
    caveat: str = Field(min_length=20, max_length=300)


@dataclass(frozen=True)
class ReportFactor:
    """A top factor: input column, its value and the contribution (calibrated log-odds)."""

    feature: str
    value: float | str | None
    contribution: float


@dataclass(frozen=True)
class ReportInput:
    """Everything the report may talk about."""

    risk: float
    alert: bool
    threshold: float
    baseline_risk: float
    factors: Sequence[ReportFactor]


@dataclass
class GeneratedReport:
    """The report and where it came from."""

    report: MaintenanceReport
    source: Literal["llm", "fallback"]
    model: str | None
    attempts: int
    warnings: list[str] = field(default_factory=list)


class LlmError(RuntimeError):
    """The LLM could not be reached or answered something unusable."""


# --- facts ------------------------------------------------------------------------------------
def risk_level_for(risk: float, threshold: float, alert: bool) -> RiskLevel:
    """Risk level from the numbers (never from the LLM)."""
    if not alert:
        return "niedrig"
    return "hoch" if risk >= HIGH_RISK_FACTOR * threshold else "erhöht"


def describe_feature(name: str) -> str:
    """A German description of an input column (the physical meaning is unknown: anonymised)."""
    if name == "time_step":
        return "Zeitschritt (Betriebsdauer des Fahrzeugs)"
    if re.fullmatch(r"Spec_\d+", name):
        return f"Fahrzeugspezifikation {name.removeprefix('Spec_')}"
    if match := re.fullmatch(r"(\d+_\d+)__delta", name):
        return f"Zähler {match.group(1)}: Änderung seit der letzten Messung"
    if match := re.fullmatch(r"(\d+_\d+)__rate", name):
        return f"Zähler {match.group(1)}: Änderung pro Zeitschritt"
    if match := re.fullmatch(r"(\d+)_(\d+)__share", name):
        return f"Histogramm {match.group(1)}: Anteil von Klasse {match.group(2)}"
    if match := re.fullmatch(r"(\d+)__mean_bin", name):
        return f"Histogramm {match.group(1)}: mittlere Klasse"
    if match := re.fullmatch(r"(\d+)__top_share", name):
        return f"Histogramm {match.group(1)}: Anteil der obersten Klassen"
    if match := re.fullmatch(r"(\d+)_(\d+)", name):
        # the SCANIA counters are <variable>_0, histogram bins <variable>_<bin>
        return (
            f"Zähler {name}" if match.group(2) == "0" and match.group(1) in COUNTER_VARIABLES
            else f"Histogramm {match.group(1)}: Klasse {match.group(2)}"
        )  # fmt: skip
    return f"Eingangsgröße {name}"


COUNTER_VARIABLES = frozenset({"171", "666", "427", "837", "309", "835", "370", "100"})


def _clean_value(value: float | str | None) -> str:
    """Value as prompt text. Text (a specification category) must be a short safe token."""
    if value is None:
        return "fehlt"
    if isinstance(value, str):
        if len(value) > MAX_VALUE_LENGTH or not _SAFE_TEXT.match(value):
            return "(unlesbar)"  # never put request-controlled free text into the prompt
        return value
    return f"{value:.4g}"


def build_user_message(data: ReportInput, level: RiskLevel) -> str:
    """The facts for the prompt (German); only numbers and short safe tokens."""
    lines = [
        f"Risiko (kalibrierte Wahrscheinlichkeit einer Reparatur innerhalb von 48 Zeitschritten): "
        f"{data.risk:.1%}",
        f"Durchschnittliches Risiko aller Fahrzeuge: {data.baseline_risk:.1%}",
        f"Alarmschwelle (kostenbasiert): {data.threshold:.1%}",
        f"Alarm: {'ja' if data.alert else 'nein'}",
        f"risk_level (vorgegeben): {level}",
        "",
        "Wichtigste Faktoren (Name | Beschreibung | Wert | Wirkung auf das Risiko; die Zahl in "
        "Klammern ist ein Beitrag in Log-Odds, KEINE Prozentzahl):",
    ]
    for factor in data.factors:
        direction = "erhöht" if factor.contribution > 0 else "senkt"
        lines.append(
            f"- {factor.feature} | {describe_feature(factor.feature)} | "
            f"{_clean_value(factor.value)} | {direction} ({factor.contribution:+.2f})"
        )
    lines += ["", "Erstelle den Bericht als JSON."]
    return "\n".join(lines)


# --- checks -----------------------------------------------------------------------------------
def looks_german(text: str) -> bool:
    """Rough language check: clearly more German than English stop words (and at least 3)."""
    words = re.findall(r"[a-zäöüß]+", text.lower())
    german = sum(w in GERMAN_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    return german >= 3 and german > english


def report_texts(report: MaintenanceReport) -> str:
    return " ".join([report.summary, report.recommendation, report.caveat,
                     *(r.explanation for r in report.reasons)])  # fmt: skip


_PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:%|Prozent\b)")
PERCENT_TOLERANCE = 0.06  # the facts are given with one decimal; allow its rounding


def allowed_percentages(data: ReportInput) -> list[float]:
    """The only percentages a report may quote: risk, average risk and alert threshold."""
    return [100 * data.risk, 100 * data.baseline_risk, 100 * data.threshold]


def ungrounded_percentages(text: str, data: ReportInput) -> list[str]:
    """Percentages in ``text`` that are not one of the given facts (e.g. a computed difference).

    A real LLM answer said "0.3 % über der Alarmschwelle" for 4.4 % vs. 3.5 % and described log-odds
    contributions as percent: both are numbers that appear in no fact.
    """
    allowed = allowed_percentages(data)
    bad = []
    for raw in _PERCENT.findall(text):
        value = float(raw.replace(",", "."))
        if not any(abs(value - a) <= PERCENT_TOLERANCE for a in allowed):
            bad.append(raw)
    return bad


def validate_report(report: MaintenanceReport, data: ReportInput, level: RiskLevel) -> list[str]:
    """Problems of a report against the facts it must stick to (empty list = fine)."""
    problems = []
    allowed = {f.feature for f in data.factors}
    used = [r.factor for r in report.reasons]
    invented = sorted(set(used) - allowed)
    if invented:
        problems.append(f"factor(s) not among the given top factors: {invented}")
    if len(used) != len(set(used)):
        problems.append("a factor is listed twice")
    if report.risk_level != level:
        problems.append(f"risk_level {report.risk_level!r} differs from the computed {level!r}")
    if not looks_german(report_texts(report)):
        problems.append("the text does not look German")
    invented = ungrounded_percentages(report_texts(report), data)
    if invented:
        problems.append(f"percentage(s) that are not among the given facts: {invented}")
    return problems


def fallback_report(data: ReportInput) -> MaintenanceReport:
    """Deterministic German report from templates (no LLM): valid by construction."""
    level = risk_level_for(data.risk, data.threshold, data.alert)
    comparison = "über" if data.alert else "unter"
    recommendation = {
        "niedrig": "Keine Maßnahme nötig; das Fahrzeug regulär weiter überwachen.",
        "erhöht": "Das Fahrzeug bei der nächsten planmäßigen Wartung auf Component X prüfen.",
        "hoch": "Zeitnah einen Werkstattcheck von Component X einplanen, Fahrzeug beobachten.",
    }[level]
    reasons = [
        Reason(
            factor=f.feature,
            explanation=(
                f"{describe_feature(f.feature)} (Wert {_clean_value(f.value)}) "
                f"{'erhöht' if f.contribution > 0 else 'senkt'} das berechnete Risiko."
            ),
        )
        for f in data.factors[:5]
    ]
    return MaintenanceReport(
        summary=(
            f"Das Modell schätzt das Risiko einer Reparatur innerhalb von 48 Zeitschritten auf "
            f"{data.risk:.1%}; das liegt {comparison} der Alarmschwelle von {data.threshold:.1%} "
            f"(Durchschnitt aller Fahrzeuge: {data.baseline_risk:.1%})."
        ),
        risk_level=level,
        recommendation=recommendation,
        reasons=reasons,
        caveat=(
            "Dies ist eine statistische Modellaussage und keine Diagnose; die Variablen sind "
            "anonymisiert, die Faktoren nennen keine Ursache."
        ),
    )


# --- Ollama -----------------------------------------------------------------------------------
class OllamaClient:
    """Minimal client for ``POST /api/chat`` with a JSON schema as ``format`` (standard library)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        temperature: float = 0.0,
        seed: int = 42,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        self.base_url, self.model = base_url.rstrip("/"), model
        self.timeout, self.temperature, self.seed = timeout, temperature, seed
        self._opener = opener

    def chat_json(self, system: str, user: str, schema: Mapping[str, Any]) -> str:
        """The model's answer text (a JSON string constrained by ``schema``).

        Raises:
            LlmError: If Ollama is unreachable, times out, or the reply has no message content.
        """
        payload = {
            "model": self.model, "stream": False, "format": dict(schema),
            "options": {"temperature": self.temperature, "seed": self.seed},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }  # fmt: skip
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )  # fmt: skip
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return str(body["message"]["content"])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LlmError(f"Ollama not reachable at {self.base_url}: {exc}") from exc
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            raise LlmError(f"unexpected Ollama reply: {exc!r}") from exc


class ReportGenerator:
    """LLM report with validation, one retry and a deterministic fallback."""

    def __init__(
        self, client: OllamaClient | None, system_prompt: str, max_attempts: int = 2
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.client, self.system_prompt, self.max_attempts = client, system_prompt, max_attempts

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ReportGenerator:
        """Generator named by ``llm.*``; ``llm.enabled: false`` = template report only."""
        section = config["llm"]
        prompt = resolve_path(section["prompt_path"]).read_text(encoding="utf-8")
        client = None
        if section.get("enabled", True):
            # OLLAMA_BASE_URL overrides the address, e.g. http://host.docker.internal:11434
            base_url = (
                os.environ.get(OLLAMA_URL_ENV)
                or os.environ.get(LEGACY_OLLAMA_URL_ENV)
                or section["base_url"]
            )
            client = OllamaClient(
                base_url, section["model"], float(section["timeout_seconds"]),
                float(section["temperature"]), int(section["seed"]),
            )  # fmt: skip
        return cls(client, prompt, int(section["max_attempts"]))

    def generate(self, data: ReportInput) -> GeneratedReport:
        """A validated report; never raises for an LLM problem."""
        level = risk_level_for(data.risk, data.threshold, data.alert)
        warnings: list[str] = []
        made = 0  # attempts that actually reached the LLM
        if self.client is not None:
            user = build_user_message(data, level)
            schema = MaintenanceReport.model_json_schema()
            for attempt in range(1, self.max_attempts + 1):
                made = attempt
                try:
                    raw = self.client.chat_json(self.system_prompt, user, schema)
                    report = MaintenanceReport.model_validate_json(raw)
                except LlmError as exc:
                    warnings.append(f"attempt {attempt}: {exc}")
                    logger.warning("LLM report attempt %d failed: %s", attempt, exc)
                    break  # the LLM is not there; a retry will not help
                except ValidationError as exc:
                    warnings.append(
                        f"attempt {attempt}: invalid report structure ({exc.error_count()} errors)"
                    )
                    logger.warning("LLM report attempt %d: invalid JSON structure", attempt)
                    continue
                problems = validate_report(report, data, level)
                if not problems:
                    return GeneratedReport(report, "llm", self.client.model, attempt, warnings)
                warnings.extend(f"attempt {attempt}: {p}" for p in problems)
                logger.warning("LLM report attempt %d rejected: %s", attempt, problems)
        return GeneratedReport(fallback_report(data), "fallback", None, made, warnings)
