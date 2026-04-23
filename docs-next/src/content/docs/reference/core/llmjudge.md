---
title: LlmJudge
description: API reference for LlmJudge
---

# LlmJudge

```python
from modal_training_gym.common.llm_judge import LlmJudge
```

LLM-as-judge client for an OpenAI-compatible chat-completions endpoint.

## Constructor

```python
LlmJudge(model_name, base_url, max_score=10.0, max_tokens=100)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | *required* | Model identifier to pass in the chat-completions request (e.g. `"qwen3-4b"`). |
| `base_url` | `str` | *required* | Base URL of the OpenAI-compatible endpoint (e.g. `"http://localhost:8000"`). |
| `max_score` | `float` | `10.0` | Upper bound of the judge's numeric scale. Scores are normalized to [0, 1] by dividing by this value. Default `10.0`. |
| `max_tokens` | `int` | `100` | Maximum tokens in the judge response. Default `100`. |

## Methods

### `build_prompt(self, prompt: 'str', response: 'str', **kwargs: 'Any') -> 'str'`

Return the judge prompt for one (prompt, response) pair.

### `score(self, session: "'aiohttp.ClientSession'", prompt: 'str', response: 'str', **kwargs: 'Any') -> 'float'`

Score one (prompt, response) pair; returns a float in [0, 1].

**Source:** [`modal_training_gym/common/llm_judge.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/llm_judge.py)
