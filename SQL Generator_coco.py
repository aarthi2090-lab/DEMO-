import json
import re
import pandas as pd
import snowflake.connector
from sqlalchemy import create_engine, text
import streamlit as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64
import os

src_generated_sql = []
tgt_generated_sql = []
# -----------------------------
# Fixed Snowflake connection
# -----------------------------
CORTEX_MODEL_NAME = "claude-3-7-sonnet"

SF_ACCOUNT   = "RMHNYOB-COGNIZANT_INDIA"
SF_USER      = "arthi.senthil@cognizant.com"
#SF_PASSWORD  = ""
SF_PROGRAMMATIC_ACCESS_TOKEN = os.getenv("SF_PROGRAMMATIC_ACCESS_TOKEN")
SF_WAREHOUSE = "SYSTEM$STREAMLIT_NOTEBOOK_WH"#"DEMO_WH"
SF_DATABASE  = "ARTHI_SENTHIL_COGNIZANT_COM_DB"
SF_SCHEMA    = "DBT_SCHEMA"

conn = snowflake.connector.connect(
    account=SF_ACCOUNT,
    user=SF_USER,
    #password=SF_PASSWORD,
    password=SF_PROGRAMMATIC_ACCESS_TOKEN,
    warehouse=SF_WAREHOUSE,
    database=SF_DATABASE,
    schema=SF_SCHEMA,
)

# -----------------------------
# Helpers for SQL reads
# -----------------------------
def sql_read(sql, params=None):
    return pd.read_sql(sql, conn, params=params)

# -----------------------------
# 1) Schema allow-list (Retrieve)
# -----------------------------
def fetch_schema_allowlist_json() -> str:
    """
    Returns JSON string mapping table -> [columns] for DBT_POC.PUBLIC.
    Uses Snowflake syntax: ARRAY_AGG ... WITHIN GROUP (ORDER BY ...).
    """
    q = f"""
    WITH cols AS (
      SELECT
        TABLE_NAME,
        ARRAY_AGG(COLUMN_NAME) WITHIN GROUP (ORDER BY ORDINAL_POSITION) AS COLS
      FROM {SF_DATABASE}.INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = '{SF_SCHEMA}'
      GROUP BY TABLE_NAME
    )
    SELECT OBJECT_AGG(TABLE_NAME, COLS) AS SCHEMA_JSON
    FROM cols;
    """
    df = sql_read(q)
    schema_obj = df.iloc[0, 0]
    if isinstance(schema_obj, str):
        schema_obj = json.loads(schema_obj)
    return json.dumps(schema_obj)

# -----------------------------
# 2) DDL chunking (Retrieve)
# -----------------------------
def ensure_chunks_table():
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SF_DATABASE}.{SF_SCHEMA}.DDL_CHUNKS (
      OBJECT_NAME STRING,
      CHUNK_INDEX NUMBER,
      CHUNK_TEXT  STRING
    );
    """
    with conn.cursor() as cur:
        cur.execute(ddl)

def fetch_schema_ddl() -> str:
    q = f"SELECT GET_DDL('SCHEMA', '{SF_DATABASE}.{SF_SCHEMA}', TRUE) AS DDL;"
    return sql_read(q).iloc[0, 0]

def refresh_schema_chunks(max_len: int = 3000):
    """
    Pull live schema DDL, split into CREATE TABLE blocks, sub-chunk if long,
    and store in DDL_CHUNKS.
    """
    ensure_chunks_table()
    full_ddl = fetch_schema_ddl()

    blocks = re.split(r"(?=CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+)", full_ddl, flags=re.IGNORECASE)
    rows = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        m = re.search(r"CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+([^\s(]+)", b, flags=re.IGNORECASE)
        if not m:
            continue
        object_name = m.group(1)
        for i in range(0, len(b), max_len):
            rows.append((object_name, i // max_len, b[i:i+max_len]))

    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {SF_DATABASE}.{SF_SCHEMA}.DDL_CHUNKS")
        ins = f"INSERT INTO {SF_DATABASE}.{SF_SCHEMA}.DDL_CHUNKS (OBJECT_NAME, CHUNK_INDEX, CHUNK_TEXT) VALUES (%s, %s, %s)"
        for r in rows:
            cur.execute(ins, r)

# -----------------------------
# 3) Simple keyword-based retrieval (Retrieve w/o embeddings)
# -----------------------------
def tokenize(text: str):
    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 2]

def retrieve_relevant_ddl_chunks(logic_text: str, k: int = 8) -> str:
    """
    Retrieve top-K DDL chunks by simple keyword overlap scoring with the logic text.
    Avoids use of SNOWFLAKE.CORTEX.EMBED_TEXT (not available in this account).
    """
    # Fetch all chunks
    df = sql_read(f"SELECT OBJECT_NAME, CHUNK_INDEX, CHUNK_TEXT FROM {SF_DATABASE}.{SF_SCHEMA}.DDL_CHUNKS;")
    if df.empty:
        return ""
    logic_tokens = tokenize(logic_text)
     # If no tokens, just return the first K chunks (stable)
    if not logic_tokens:
        top = df.sort_values(["OBJECT_NAME", "CHUNK_INDEX"]).head(k)
        return "\n\n".join(top["CHUNK_TEXT"].tolist())
# Score chunks by token overlap
    scores = []
    for _, row in df.iterrows():
        chunk = row["CHUNK_TEXT"] or ""
        chunk_lower = chunk.lower()
        score = 0
        for tok in logic_tokens:
            # Score chunks by token overlap
            if re.search(rf"\b{re.escape(tok)}\b", chunk_lower):
                score += 3
            elif tok in chunk_lower:
                score += 1
        scores.append(score)

    df["SCORE"] = scores
    top = df.sort_values(["SCORE", "OBJECT_NAME", "CHUNK_INDEX"], ascending=[False, True, True]).head(k)
    return "\n\n".join(top["CHUNK_TEXT"].tolist())

# -----------------------------
# 4) Optional: narrow table subset via LLM (Retrieve)
# -----------------------------
def select_relevant_tables(logic_text: str, schema_json_str: str):
    """
    Ask Cortex COMPLETE to pick a minimal subset of tables from the allow-list.
    """
    prompt = f"""
You are an expert Snowflake SQL assistant.

Given the business logic and this schema allow-list JSON (table -> columns),
return a JSON array of ONLY the table names that are relevant to the logic.
No extra text; just a JSON array.

Schema allow-list JSON:
{schema_json_str}

Business logic:
{logic_text}

Rules:
- Include only necessary tables.
- If unsure, prefer fewer tables.
- Output must be a valid JSON array, e.g. ["FACT_ORDERS","FACT_ORDER_ITEMS"].
"""
    with conn.cursor() as cur:
        cur.execute("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)", (CORTEX_MODEL_NAME, prompt))
        raw = (cur.fetchone()[0] or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").strip()
    try:
        arr = json.loads(raw)
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

# -----------------------------
# 5) Generation (Augment + Generate)
# -----------------------------
def sanitize_sql_output(text: str) -> str:
    sql = (text or "").strip()
    sql = sql.removeprefix("```sql").removeprefix("```").strip()
    if not sql.endswith(";"):
        sql += ";"
    return sql

def converting_english_sql(logic: str, top_k_chunks: int = 8) -> str:
    """
    Full RAG: allow-list + top-K DDL chunks (keyword retrieval) -> grounded prompt -> SQL via Cortex COMPLETE.
    """
    schema_json = fetch_schema_allowlist_json()
    # Narrow to relevant tables to keep prompt compact

    try:
        tables = select_relevant_tables(logic, schema_json)
    except Exception:
        tables = []

    if tables:
        full_schema = json.loads(schema_json)
        reduced = {t: full_schema.get(t, []) for t in tables}
        schema_context = json.dumps(reduced)
    else:
        schema_context = schema_json

    ddl_context = retrieve_relevant_ddl_chunks(logic, k=top_k_chunks)

    prompt = f"""
You are an expert Snowflake SQL developer.

Task:
Convert the following business logic into a single, valid Snowflake SQL statement.

Business Logic:
{logic}

Use ONLY these tables and columns (strict allow-list; JSON mapping table -> [columns]):
{schema_context}

Additional context (relevant DDL excerpts for reference):
{ddl_context}

Hard rules:
- Do NOT use any table or column not listed in the JSON allow-list.
- Fully qualify all tables with {SF_SCHEMA}.<TABLE>.
- Use Snowflake SQL syntax.
-Use ONLY the columns explicitly provided; do NOT add extra columns.
-Ensure source and target have exact 1:1 column mapping with equal count and order
- Output ONLY the SQL (no explanations, no code fences).
-Table aliases MUST follow: t1, t2, t3, ...
- End with a semicolon.
"""
    with conn.cursor() as cur:
        cur.execute("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)", (CORTEX_MODEL_NAME, prompt))
        out = cur.fetchone()[0] or ""

    return sanitize_sql_output(out)


def validation_sql(generated_sql_list):
    validated = []
    for sql in generated_sql_list:
        sql_clean = sql.replace("```sql", "").replace("```", "").strip()
        try:
            _ = sql_read(sql_clean)
            validated.append(sql)# keep original in output
        except Exception as e:
            validated.append(f"-- INVALID SQL\n-- {str(e)}\n{sql}")
    return validated

def process_row(row):
    src_logic = str(row["SRC_LOGIC"])
    tgt_logic = str(row["TARGET_LOGIC"])

    sql_src = converting_english_sql(src_logic, top_k_chunks=8)
    sql_tgt = converting_english_sql(tgt_logic, top_k_chunks=8)

    return sql_src, sql_tgt


if __name__ == "__main__":
    main()
