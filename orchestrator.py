"""
orchestrator.py - Gemini API Integration and Autonomous Data Analysis Agent Orchestrator.
Uses the Google GenAI SDK (pip install google-genai) and currently available Gemini models.
"""

import os
import re
import time
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

from google import genai
from google.genai import types
from helper import build_llm_data_context, execute_analysis_code, CodeExecutionResult


# ==============================================================================
# 1. SUPPORTED GEMINI MODELS CATALOG  (https://ai.google.dev/gemini-api/docs/models)
# ==============================================================================

DEFAULT_MODEL = "gemini-3.8-flash"

# Models Google has shut down or closed to new users. Any of these is replaced by DEFAULT_MODEL.
RETIRED_PREFIXES = (
    "gemini-1.5", "gemini-2.0", "gemini-2.5", "gemini-3-pro-preview", "gemini-3.1-flash-lite-preview",
)

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-3.8-flash": {
        "id": "gemini-3.8-flash",
        "display_name": "Gemini 3.8 Flash (Recommended)",
        "description": "Latest Flash model: fast, strong analytical reasoning and reliable code generation.",
        "category": "Flash",
        "default": True,
    },
    "gemini-3.7-flash": {
        "id": "gemini-3.7-flash",
        "display_name": "Gemini 3.7 Flash",
        "description": "Previous-generation Flash model with fast responses and solid code generation.",
        "category": "Flash",
        "default": False,
    },
    "gemini-3.6-flash": {
        "id": "gemini-3.6-flash",
        "display_name": "Gemini 3.6 Flash",
        "description": "Fast, general-purpose model for everyday data questions.",
        "category": "Flash",
        "default": False,
    },
    "gemini-3.5-flash": {
        "id": "gemini-3.5-flash",
        "display_name": "Gemini 3.5 Flash",
        "description": "Fast, general-purpose model for everyday data questions.",
        "category": "Flash",
        "default": False,
    },
    "gemini-3.5-flash-lite": {
        "id": "gemini-3.5-flash-lite",
        "display_name": "Gemini 3.5 Flash Lite",
        "description": "Lightweight, cost-effective model for rapid, simple queries.",
        "category": "Flash Lite",
        "default": False,
    },
    "gemini-3.1-pro-preview": {
        "id": "gemini-3.1-pro-preview",
        "display_name": "Gemini 3.1 Pro (Preview)",
        "description": "Deeper reasoning for complex multi-variable datasets. Preview model, may change.",
        "category": "Pro",
        "default": False,
    },
}


def get_available_models() -> List[str]:
    """Returns a list of supported Gemini model identifiers."""
    return list(AVAILABLE_MODELS.keys())


def get_model_details(model_id: str) -> Optional[Dict[str, Any]]:
    """Returns metadata and description for a given Gemini model ID."""
    return AVAILABLE_MODELS.get(model_id, None)


def resolve_model(model_name: Optional[str]) -> str:
    """Returns a usable model name; retired or empty names become DEFAULT_MODEL."""
    name = (model_name or "").strip()
    if name.startswith("models/"):
        name = name[len("models/"):]
    if not name or name.startswith(RETIRED_PREFIXES):
        return DEFAULT_MODEL
    return name


# ==============================================================================
# 2. AGENT SYSTEM PROMPT
# ==============================================================================

SYSTEM_PROMPT = """
You are "Antigravity Data Analyst", a professional, autonomous Senior Data Analyst and Business Intelligence Specialist.
Your mission is to explore, analyze, and extract deep strategic insights from user-provided datasets, then present findings clearly, cleanly, and actionably.

### Analytical Guidelines:
1. **Be Rigorous and Fact-Based**: Always reference actual columns, statistical metrics, distributions, and numbers from the provided dataset context.
2. **Business & Strategic Value**: Translate data patterns into tangible business impact (revenue, efficiency, churn, cost reduction, customer satisfaction).
3. **Actionable Visualizations**: Whenever a question benefits from visual explanation, provide a complete, clean, self-contained Python Plotly code snippet.
4. **Structured Markdown Output**: Structure your answers cleanly without using emojis:
   - **Executive Summary**: 2-3 concise, impactful bullet points summarizing key findings.
   - **Deep-Dive Insights**: Statistical findings, patterns, anomalies, and comparisons.
   - **Visual Analytics**: Concise explanation of what the generated chart illustrates.
   - **Strategic Recommendations**: 3-4 concrete business action items.

### Code Generation Rules:
- If generating code, use Python with `pandas` (as `pd`), `numpy` (as `np`), `plotly.express` (as `px`), and `plotly.graph_objects` (as `go`).
- Always assume the dataset is already loaded in variable `df`.
- **DO NOT** read from files (e.g. do not call `pd.read_csv`). Use `df` directly.
- Always store the final Plotly interactive chart in the variable named `fig` (e.g. `fig = px.bar(...)` or `fig = go.Figure(...)`).
- Use clean, professional minimalist color palettes in Plotly charts (e.g., `template="plotly_white"`).
- Put the executable Python code inside a single ```python ... ``` block.
- Keep the code clean, fast, and bug-free.
"""


# ==============================================================================
# 3. GEMINI CLIENT, ERROR HANDLING AND VALIDATION
# ==============================================================================

_API_KEY: Optional[str] = None
_CLIENTS: Dict[str, Any] = {}


def initialize_gemini(api_key: Optional[str] = None) -> bool:
    """Stores the API key (provided, or from GEMINI_API_KEY / GOOGLE_API_KEY). Returns False if none."""
    global _API_KEY
    key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        return False
    _API_KEY = key
    return True


def _client():
    if not _API_KEY:
        raise ValueError("Missing Gemini API key.")
    if _API_KEY not in _CLIENTS:
        _CLIENTS[_API_KEY] = genai.Client(api_key=_API_KEY)
    return _CLIENTS[_API_KEY]


def _has(msg: str, code: str) -> bool:
    return re.search(rf"\b{code}\b", msg) is not None


def _is_not_found(msg: str) -> bool:
    return "NOT_FOUND" in msg or _has(msg, "404") or "no longer available" in msg.lower()


def _is_overloaded(msg: str) -> bool:
    return _has(msg, "503") or "UNAVAILABLE" in msg or "overloaded" in msg.lower()


def friendly_error(e: Exception) -> str:
    """Turns a raw API exception into a short, readable message for the UI."""
    msg = str(e)
    low = msg.lower()
    if "api_key_invalid" in low or "api key not valid" in low or "unauthenticated" in low or _has(msg, "401"):
        return "The Gemini API key is invalid. Check the key and try again."
    if "permission_denied" in low or _has(msg, "403"):
        return "This API key does not have permission to use the selected model."
    if _is_not_found(msg):
        return f"The selected model is not available for this key. Choose {DEFAULT_MODEL} instead."
    if "resource_exhausted" in low or _has(msg, "429") or "quota" in low:
        return "Gemini quota or rate limit reached. Wait a minute and try again."
    if _is_overloaded(msg):
        return "Gemini is temporarily overloaded. Please try again in a moment."
    return msg[:200]


def _generate(model_name: str, prompt: str, system: Optional[str] = None,
              temperature: Optional[float] = None) -> Tuple[str, str]:
    """Calls Gemini. Retries when overloaded and falls back to DEFAULT_MODEL if the model is unavailable.
    Returns (text, model_actually_used)."""
    model = resolve_model(model_name)
    for attempt in range(3):
        kw: Dict[str, Any] = {}
        if system:
            kw["system_instruction"] = system
        if temperature is not None and not model.startswith("gemini-3"):
            kw["temperature"] = temperature  # Gemini 3 models are tuned for their default temperature
        try:
            resp = _client().models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**kw)
            )
            return (resp.text or ""), model
        except Exception as e:
            msg = str(e)
            if _is_overloaded(msg) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            if _is_not_found(msg) and model != DEFAULT_MODEL:
                model = DEFAULT_MODEL
                continue
            raise
    raise RuntimeError("Gemini did not respond.")


def validate_api_key(api_key: str, model_name: Optional[str] = None) -> Tuple[bool, str]:
    """Validates the API key by sending a tiny test prompt to Gemini."""
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty."
    key = api_key.strip()
    try:
        client = genai.Client(api_key=key)
        reply = client.models.generate_content(
            model=resolve_model(model_name), contents="Reply with the single word: ready"
        )
        if reply and reply.text:
            initialize_gemini(key)
            return True, "API key is valid and active."
        return False, "Gemini returned an empty response."
    except Exception as e:
        return False, friendly_error(e)


# ==============================================================================
# 4. CODE EXTRACTION UTILITY
# ==============================================================================

def extract_python_code(text: str) -> Optional[str]:
    """Extracts Python code block from LLM response."""
    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()  # the longest code block
    return None


def clean_markdown_text(text: str) -> str:
    """Cleans up raw markdown text."""
    return text.strip()


# ==============================================================================
# 5. CORE ORCHESTRATION ENGINE
# ==============================================================================

def run_agent_analysis(
    api_key: str,
    model_name: str,
    user_query: str,
    df: pd.DataFrame,
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """
    Autonomous agent workflow:
    1. Builds data context schema.
    2. Sends context + query to Gemini.
    3. Extracts and runs visualization / data transformation code safely.
    4. Returns text reasoning + interactive figure + execution logs.
    """
    if not initialize_gemini(api_key):
        return {
            "success": False,
            "error": "Missing Gemini API key. Enter a valid key in the sidebar.",
            "text_response": None, "figure": None, "code": None,
        }

    data_context = build_llm_data_context(df)

    user_prompt = f"""
{data_context}

---
### User Request / Analytical Goal:
{user_query}

### Instructions for Agent:
1. Answer the user request with data-driven depth and business context.
2. If a chart or visual would assist in answering this, write Python Plotly code that creates a chart and stores it in variable `fig`.
3. Provide key observations and business recommendations.
"""

    try:
        response_text, model_used = _generate(model_name, user_prompt, SYSTEM_PROMPT, temperature)
        response_text = response_text or "No response generated by model."

        code_block = extract_python_code(response_text)
        fig = None
        code_output = None
        exec_error = None

        if code_block:
            exec_res = execute_analysis_code(code_block, df)
            if exec_res.success:
                fig = exec_res.figure
                code_output = exec_res.output_text
            else:
                exec_error = exec_res.error
                # Self-healing retry if the code had a bug
                healed_fig, healed_code = retry_code_fix(model_used, code_block, exec_error, df, data_context)
                if healed_fig is not None:
                    fig = healed_fig
                    code_block = healed_code
                    exec_error = None

        return {
            "success": True,
            "text_response": response_text,
            "code": code_block,
            "code_output": code_output,
            "figure": fig,
            "execution_error": exec_error,
            "model_used": model_used,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": friendly_error(e),
            "text_response": None, "figure": None, "code": None,
        }


# ==============================================================================
# 6. SELF-HEALING CODE CORRECTION
# ==============================================================================

def retry_code_fix(
    model_name: str,
    broken_code: str,
    error_message: str,
    df: pd.DataFrame,
    data_context: str,
) -> Tuple[Optional[Any], Optional[str]]:
    """Feeds the runtime error back to Gemini to get corrected code."""
    try:
        fix_prompt = f"""
The following Python Plotly code encountered a runtime error:
```python
{broken_code}
```

### Runtime Error:
{error_message}

### Dataset Schema:
{data_context}

Please fix the Python code so that it runs without error on `df` and creates a Plotly figure stored in `fig`.
Return ONLY the corrected Python code inside a ```python ``` block.
"""
        text, _ = _generate(model_name, fix_prompt)
        fixed_code = extract_python_code(text) if text else None
        if fixed_code:
            exec_res = execute_analysis_code(fixed_code, df)
            if exec_res.success and exec_res.figure is not None:
                return exec_res.figure, fixed_code
    except Exception:
        pass
    return None, None


# ==============================================================================
# 7. AUTONOMOUS ONE-CLICK ROUTINES
# ==============================================================================

def generate_autonomous_eda(api_key: str, model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Autonomous Exploratory Data Analysis with automated chart & executive synthesis."""
    query = """
Perform an autonomous, high-level Exploratory Data Analysis (EDA) on this dataset.
Identify the primary target variables, distribution characteristics, key correlations, top outliers, and 3 key high-impact business insights.
Create a comprehensive visual chart (e.g. multi-bar, scatter matrix, or grouped distribution) stored in `fig`.
"""
    return run_agent_analysis(api_key, model_name, query, df, temperature=0.3)


def generate_anomaly_detection(api_key: str, model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Autonomous anomaly & outlier investigation."""
    query = """
Analyze this dataset for statistical anomalies, rare events, unusual variance, and outliers.
Explain why these anomalies exist, which dimensions are most volatile, and what business risks or opportunities they present.
Generate a Plotly chart (e.g., box plot, scatter with highlighted outliers, or residual plot) in `fig`.
"""
    return run_agent_analysis(api_key, model_name, query, df, temperature=0.2)


def generate_trends_and_correlations(api_key: str, model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Autonomous correlation matrix and trend analysis."""
    query = """
Perform a deep correlation and trend analysis across all numerical and categorical features.
Highlight strongest positive/negative relationships, hidden dependencies, and potential causation vs correlation warnings.
Create an interactive Plotly heatmap or multi-variable trend chart stored in `fig`.
"""
    return run_agent_analysis(api_key, model_name, query, df, temperature=0.2)


def generate_strategic_recommendations(api_key: str, model_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Autonomous executive action plan and growth strategies based on the data."""
    query = """
Act as a Principal Management Consultant. Based on this dataset, create a strategic action plan.
Provide 4 prioritized, high-ROI business initiatives, expected KPIs to monitor, potential risks, and a roadmap.
Create an insightful summary KPI or milestone chart in Plotly stored in `fig`.
"""
    return run_agent_analysis(api_key, model_name, query, df, temperature=0.4)
