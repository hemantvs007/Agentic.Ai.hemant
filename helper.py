"""
helper.py - Data Ingestion, Profiling, Cleaning, and Refining for Autonomous Data Analysis Agent.
"""

import io
import re
import sys
import contextlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, Any, List, Optional


# ==============================================================================
# 1. SAMPLE DATASET GENERATOR
# ==============================================================================

def get_sample_dataset(name: str = "E-Commerce Sales") -> pd.DataFrame:
    """Generates realistic sample datasets for instant demonstration and testing."""
    np.random.seed(42)
    
    if name == "E-Commerce Sales":
        n = 500
        categories = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Sports", "Beauty"]
        regions = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]
        segments = ["Consumer", "Corporate", "Small Business"]
        
        dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
        sales = np.random.exponential(scale=250, size=n) + 20
        quantity = np.random.randint(1, 15, size=n)
        discount = np.random.choice([0.0, 0.05, 0.1, 0.15, 0.2, 0.3], size=n, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
        profit = (sales * quantity * (1 - discount)) * np.random.uniform(0.1, 0.4, size=n) - np.random.uniform(0, 30, size=n)
        rating = np.round(np.random.normal(4.2, 0.6, size=n).clip(1.0, 5.0), 1)
        
        df = pd.DataFrame({
            "order_id": [f"ORD-{1000 + i}" for i in range(n)],
            "order_date": dates,
            "category": np.random.choice(categories, size=n),
            "region": np.random.choice(regions, size=n),
            "customer_segment": np.random.choice(segments, size=n),
            "unit_price": np.round(sales, 2),
            "quantity": quantity,
            "discount": discount,
            "profit": np.round(profit, 2),
            "customer_rating": rating,
            "is_returned": np.random.choice([True, False], size=n, p=[0.08, 0.92])
        })
        # Inject occasional missing values for realism
        df.loc[np.random.choice(df.index, 15, replace=False), "customer_rating"] = np.nan
        df.loc[np.random.choice(df.index, 8, replace=False), "profit"] = np.nan
        return df

    elif name == "Customer Churn":
        n = 450
        df = pd.DataFrame({
            "customer_id": [f"CUST-{2000 + i}" for i in range(n)],
            "age": np.random.randint(18, 70, size=n),
            "gender": np.random.choice(["Male", "Female"], size=n),
            "tenure_months": np.random.randint(1, 72, size=n),
            "contract_type": np.random.choice(["Month-to-Month", "One Year", "Two Year"], size=n, p=[0.55, 0.25, 0.20]),
            "monthly_charges": np.round(np.random.uniform(20, 120, size=n), 2),
            "total_charges": np.round(np.random.uniform(100, 6000, size=n), 2),
            "tech_support": np.random.choice(["Yes", "No", "No internet service"], size=n),
            "payment_method": np.random.choice(["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"], size=n),
            "churn": np.random.choice(["Yes", "No"], size=n, p=[0.27, 0.73])
        })
        df.loc[np.random.choice(df.index, 10, replace=False), "total_charges"] = np.nan
        return df

    elif name == "Employee Performance & HR":
        n = 400
        departments = ["Engineering", "Sales", "Marketing", "HR", "Product", "Operations"]
        df = pd.DataFrame({
            "employee_id": [f"EMP-{3000 + i}" for i in range(n)],
            "department": np.random.choice(departments, size=n),
            "years_at_company": np.random.randint(1, 15, size=n),
            "salary": np.random.randint(45000, 160000, size=n),
            "performance_score": np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.50, 0.20, 0.10]),
            "projects_completed": np.random.randint(2, 25, size=n),
            "overtime": np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7]),
            "attrition": np.random.choice(["Yes", "No"], size=n, p=[0.16, 0.84])
        })
        return df

    else:
        return get_sample_dataset("E-Commerce Sales")


# ==============================================================================
# 2. DATASET ACCEPTANCE & INGESTION
# ==============================================================================

def load_data(uploaded_file, file_type: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Accepts and parses diverse dataset file formats with intelligent encoding detection.
    Returns: (DataFrame, error_message)
    """
    if uploaded_file is None:
        return None, "No file uploaded."
    
    file_name = getattr(uploaded_file, "name", "uploaded_file").lower()
    
    try:
        if file_name.endswith(".csv") or file_type == "csv":
            # Attempt multiple common encodings
            encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252", "utf-16"]
            for enc in encodings:
                try:
                    if hasattr(uploaded_file, "seek"):
                        uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    return df, None
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
            return None, "Failed to parse CSV file. Please ensure it is a valid comma-separated format."

        elif file_name.endswith((".xlsx", ".xls")) or file_type in ["xlsx", "xls"]:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
            return df, None

        elif file_name.endswith(".json") or file_type == "json":
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_json(uploaded_file)
            return df, None

        elif file_name.endswith(".parquet") or file_type == "parquet":
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_parquet(uploaded_file)
            return df, None

        elif file_name.endswith((".tsv", ".tab")):
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep="\t")
            return df, None

        else:
            # Try generic csv
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            return df, None

    except Exception as e:
        return None, f"Error loading file: {str(e)}"


# ==============================================================================
# 3. DATA PROFILING & SUMMARY
# ==============================================================================

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produces a comprehensive, structured data profile of the DataFrame.
    """
    if df is None or df.empty:
        return {}
    
    num_rows, num_cols = df.shape
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    duplicates = int(df.duplicated().sum())
    dup_pct = round((duplicates / num_rows) * 100, 2) if num_rows > 0 else 0
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    total_cells = num_rows * num_cols
    total_missing = int(df.isnull().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0

    columns_info = []
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        null_p = round((null_cnt / num_rows) * 100, 2) if num_rows > 0 else 0
        nunique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)
        
        # Sample values
        sample_vals = df[col].dropna().unique()[:3].tolist()
        sample_vals_str = ", ".join([str(v) for v in sample_vals])
        
        columns_info.append({
            "column": col,
            "type": dtype_str,
            "null_count": null_cnt,
            "null_pct": null_p,
            "unique_count": nunique,
            "sample_values": sample_vals_str
        })

    # Outliers detection using IQR for numeric columns
    outliers_dict = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) >= 10:
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_cnt = int(((series < lower_bound) | (series > upper_bound)).sum())
            if outlier_cnt > 0:
                outliers_dict[col] = {
                    "count": outlier_cnt,
                    "pct": round((outlier_cnt / len(series)) * 100, 2),
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2)
                }

    # Skewness
    skewness = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 5:
            skewness[col] = round(float(series.skew()), 2)

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "memory_mb": memory_mb,
        "duplicates": duplicates,
        "duplicate_pct": dup_pct,
        "total_missing": total_missing,
        "missing_pct": missing_pct,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "bool_cols": bool_cols,
        "columns_info": pd.DataFrame(columns_info),
        "outliers": outliers_dict,
        "skewness": skewness,
        "describe_num": df[numeric_cols].describe().T.reset_index() if numeric_cols else pd.DataFrame(),
        "describe_cat": df[categorical_cols].describe().T.reset_index() if categorical_cols else pd.DataFrame()
    }


# ==============================================================================
# 4. DATA CLEANING & REFINING
# ==============================================================================

def clean_dataset(
    df: pd.DataFrame,
    clean_column_names: bool = True,
    remove_duplicates: bool = True,
    handle_missing: str = "auto",  # 'auto', 'mean_median', 'drop', 'none'
    convert_dates: bool = True,
    clip_outliers: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cleans and refines the DataFrame based on specified parameters.
    Returns: (cleaned_df, cleaning_log)
    """
    if df is None or df.empty:
        return df, ["Dataset is empty."]

    cleaned = df.copy()
    logs = []

    # 1. Clean column names
    if clean_column_names:
        original_cols = list(cleaned.columns)
        new_cols = []
        for col in original_cols:
            col_str = str(col).strip()
            # Replace whitespace and special chars with underscore
            col_str = re.sub(r"[^\w\s]", "", col_str)
            col_str = re.sub(r"\s+", "_", col_str).lower()
            new_cols.append(col_str)
        cleaned.columns = new_cols
        if original_cols != new_cols:
            logs.append(f"Standardized {len(new_cols)} column headers to clean snake_case format.")

    # 2. Remove duplicate rows
    if remove_duplicates:
        initial_count = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        dropped_dups = initial_count - len(cleaned)
        if dropped_dups > 0:
            logs.append(f"Removed {dropped_dups} duplicate row(s) ({(dropped_dups/initial_count)*100:.1f}%).")
        else:
            logs.append("Checked for duplicate rows: No duplicates found.")

    # 3. Detect & convert date columns
    if convert_dates:
        converted_date_cols = []
        for col in cleaned.select_dtypes(include=["object"]).columns:
            # Check if column name suggests date or sample matches date format
            is_date_col_name = any(kw in col.lower() for kw in ["date", "time", "year", "timestamp", "dt", "day"])
            sample = cleaned[col].dropna().astype(str).head(10)
            if not sample.empty:
                # Try parsing
                try:
                    if is_date_col_name or sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").all():
                        cleaned[col] = pd.to_datetime(cleaned[col], errors="ignore")
                        if pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                            converted_date_cols.append(col)
                except Exception:
                    pass
        if converted_date_cols:
            logs.append(f"Auto-converted date columns to datetime: {', '.join(converted_date_cols)}.")

    # 4. Handle Missing Values
    if handle_missing == "drop":
        before = len(cleaned)
        cleaned = cleaned.dropna().reset_index(drop=True)
        logs.append(f"Dropped rows with missing values (removed {before - len(cleaned)} rows).")
    elif handle_missing in ["auto", "mean_median"]:
        imputed_info = []
        for col in cleaned.columns:
            null_count = cleaned[col].isnull().sum()
            if null_count > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    # If skewed, use median; else mean
                    skew = cleaned[col].skew()
                    if abs(skew) > 1.0:
                        val = cleaned[col].median()
                        cleaned[col] = cleaned[col].fillna(val)
                        imputed_info.append(f"'{col}' (median={val:.2f})")
                    else:
                        val = cleaned[col].mean()
                        cleaned[col] = cleaned[col].fillna(val)
                        imputed_info.append(f"'{col}' (mean={val:.2f})")
                elif pd.api.types.is_string_dtype(cleaned[col]) or pd.api.types.is_object_dtype(cleaned[col]):
                    mode_val = cleaned[col].mode()
                    val = mode_val[0] if not mode_val.empty else "Unknown"
                    cleaned[col] = cleaned[col].fillna(val)
                    imputed_info.append(f"'{col}' (mode='{val}')")
                elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].ffill().bfill()
                    imputed_info.append(f"'{col}' (forward-fill)")
        if imputed_info:
            logs.append(f"Imputed missing values for: {', '.join(imputed_info)}.")
        else:
            logs.append("Missing value check: No missing values detected.")

    # 5. Outlier Clipping (optional)
    if clip_outliers:
        clipped_cols = []
        for col in cleaned.select_dtypes(include=[np.number]).columns:
            series = cleaned[col].dropna()
            if len(series) >= 20:
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                lb, ub = q25 - 1.5 * iqr, q75 + 1.5 * iqr
                outliers_exist = ((cleaned[col] < lb) | (cleaned[col] > ub)).any()
                if outliers_exist:
                    cleaned[col] = cleaned[col].clip(lower=lb, upper=ub)
                    clipped_cols.append(col)
        if clipped_cols:
            logs.append(f"Clipped IQR outliers for columns: {', '.join(clipped_cols)}.")

    return cleaned, logs


# ==============================================================================
# 5. LLM DATA CONTEXT BUILDER
# ==============================================================================

def build_llm_data_context(df: pd.DataFrame, max_sample_rows: int = 5) -> str:
    """
    Builds a high-density, structured schema & statistical context for the Gemini Agent.
    """
    if df is None or df.empty:
        return "Dataset is empty."

    num_rows, num_cols = df.shape
    columns = list(df.columns)
    
    # Column details
    col_details = []
    for col in columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        nunique = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:4].tolist()
        sample_str = str(sample_vals)[:60]
        col_details.append(f"- **{col}** ({dtype}): {nulls} nulls, {nunique} unique values. Samples: {sample_str}")

    # Helper for table conversion
    def _df_to_md(d: pd.DataFrame) -> str:
        try:
            return d.to_markdown(index=True)
        except Exception:
            return str(d.to_string())

    def _df_to_md_no_idx(d: pd.DataFrame) -> str:
        try:
            return d.to_markdown(index=False)
        except Exception:
            return str(d.to_string(index=False))

    # Numeric summary
    num_summary = ""
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        num_summary = "\n### Numerical Summary (describe):\n" + _df_to_md(num_df.describe().round(2))

    # Categorical summary
    cat_summary = ""
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        top_cats = []
        for col in cat_df.columns:
            top_val = cat_df[col].value_counts().head(3).to_dict()
            top_cats.append(f"- **{col}** top categories: {top_val}")
        cat_summary = "\n### Categorical Top Values:\n" + "\n".join(top_cats)

    # First N rows
    sample_table = _df_to_md_no_idx(df.head(max_sample_rows))

    context = f"""
### Dataset Context:
- **Total Records (Rows)**: {num_rows}
- **Total Features (Columns)**: {num_cols}
- **Column Schema**:
{chr(10).join(col_details)}

{num_summary}
{cat_summary}

### Sample Rows (Top {max_sample_rows}):
{sample_table}
"""
    return context.strip()


# ==============================================================================
# 6. SAFE CODE EXECUTION SANDBOX
# ==============================================================================

class CodeExecutionResult:
    def __init__(self, success: bool, output_text: str = "", figure: Optional[Any] = None, error: Optional[str] = None):
        self.success = success
        self.output_text = output_text
        self.figure = figure
        self.error = error


def execute_analysis_code(code_str: str, df: pd.DataFrame) -> CodeExecutionResult:
    """
    Safely executes data analysis code generated by Gemini Agent.
    Captures stdout and any generated Plotly figure (`fig`).
    """
    # Clean markdown backticks if any
    cleaned_code = re.sub(r"^```(?:python|py)?", "", code_str.strip(), flags=re.MULTILINE)
    cleaned_code = re.sub(r"```$", "", cleaned_code.strip(), flags=re.MULTILINE).strip()

    # Restricted local namespace
    local_vars = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "fig": None,
        "result": None
    }
    
    stdout_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(cleaned_code, globals(), local_vars)
        
        captured_output = stdout_buffer.getvalue()
        fig = local_vars.get("fig", None)
        
        # Check if result is a figure or df
        result_obj = local_vars.get("result", None)
        if fig is None and isinstance(result_obj, (go.Figure, px.scatter().__class__)):
            fig = result_obj
        elif result_obj is not None and not captured_output:
            captured_output = str(result_obj)

        return CodeExecutionResult(
            success=True,
            output_text=captured_output.strip(),
            figure=fig,
            error=None
        )
    except Exception as e:
        return CodeExecutionResult(
            success=False,
            output_text=stdout_buffer.getvalue().strip(),
            figure=None,
            error=f"{type(e).__name__}: {str(e)}"
        )
