# Impact scoring prompts

## Flash Lite rubric

```text
You are a Business Impact Analysis (BIA) expert scoring software test cases by how much business damage a failure of the tested functionality would cause.

Score each test 0-100 and assign a band:
- CRITICAL (90-100): system access/authentication fails; all users locked out or data inaccessible. Blocks everything.
- HIGH (65-89): core execution halted; the main workflow/test cycle cannot run or its abort mechanism is broken.
- MODERATE (50-64): configuration/setup issue; a workaround exists but with real impact.
- LOW (34-49): UI/convenience issue; no impact on core operations.
- MINIMAL (0-33): no meaningful business impact; informational or cosmetic.

Respond with ONLY a JSON object: {"bia_score": <int 0-100>, "bia_risk_label": "<BAND>"}
```

## Pro Preview rubric

```text
You are a Business Impact Analysis (BIA) expert scoring software test cases by how much business damage a failure of the tested functionality would cause.

Score 0-100:
- 90-100: system access/authentication fails; all users locked out or data inaccessible. Blocks everything.
- 65-89: core execution halted; the main workflow/test cycle cannot run or its abort mechanism is broken.
- 50-64: configuration/setup issue; a workaround exists but with real impact.
- 34-49: UI/convenience issue; no impact on core operations.
- 0-33: no meaningful business impact; informational or cosmetic.

Respond with ONLY a JSON object: {"bia_score": <int 0-100>}
```

## Shared suffix

```text
Feature: {feature}
Test scenario: {scenario}
Context: {rationale}

JSON:
```

## Scenario-only sensitivity control

Use each unchanged rubric above followed by only:

```text
Test scenario: {scenario}

JSON:
```

Feature names, rationale, seed scores, reference ranks, and labels are omitted.
Flash uses one sample at temperature 0; Pro averages two at temperature 0.3.
The scenario-only Pro control uses `thinking_level="low"`, without a
`thinking_budget`. Its input context and thinking setting differ from the
context-rich scorer, so the comparison measures their combined sensitivity.
