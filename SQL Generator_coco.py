import json
import re
import pandas as pd
import snowflake.connector
from sqlalchemy import create_engine, text
import streamlit as st
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64

src_generated_sql = []
tgt_generated_sql = []
# -----------------------------
# Fixed Snowflake connection
# -----------------------------
CORTEX_MODEL_NAME = "mistral-7b"#"claude-3-7-sonnet"

SF_ACCOUNT   = "RMHNYOB-COGNIZANT_INDIA"
SF_USER      = "arthi.senthil@cognizant.com"
#SF_PASSWORD  = ""
SF_PROGRAMMATIC_ACCESS_TOKEN = "
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

def main():
    refresh_schema_chunks()

    st.set_page_config(
        page_title="GEN AI SQL Generator",
        page_icon="⚙️",
        layout="wide"
    )

    # ---------------- Custom CSS with glow effects ---------------- #
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #020b1a 0%, #0a1628 40%, #0d1f35 70%, #081422 100%);
    }

    .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* HERO BANNER */
    .hero-banner {
        background: linear-gradient(135deg, #050d20, #0b1a2e, #0a2540);
        padding: 2.5rem 2rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 0 40px rgba(0, 255, 170, 0.08), 0 0 80px rgba(0, 170, 255, 0.05);
        border: 1px solid rgba(0, 255, 170, 0.1);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(0,255,170,0.03) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 42px;
        font-weight: 900;
        letter-spacing: 4px;
        color: #00ffaa;
        text-shadow:
            0 0 10px rgba(0, 255, 170, 0.6),
            0 0 20px rgba(0, 255, 170, 0.4),
            0 0 40px rgba(0, 255, 170, 0.3),
            0 0 80px rgba(0, 255, 170, 0.15);
        position: relative;
        z-index: 1;
        animation: glowPulse 3s ease-in-out infinite alternate;
    }

    @keyframes glowPulse {
        from {
            text-shadow:
                0 0 10px rgba(0, 255, 170, 0.6),
                0 0 20px rgba(0, 255, 170, 0.4),
                0 0 40px rgba(0, 255, 170, 0.3),
                0 0 80px rgba(0, 255, 170, 0.15);
        }
        to {
            text-shadow:
                0 0 15px rgba(0, 255, 170, 0.8),
                0 0 30px rgba(0, 255, 170, 0.5),
                0 0 60px rgba(0, 255, 170, 0.4),
                0 0 100px rgba(0, 255, 170, 0.2);
        }
    }

    .hero-title .sql-text {
        color: #00ccff;
        text-shadow:
            0 0 10px rgba(0, 204, 255, 0.6),
            0 0 20px rgba(0, 204, 255, 0.4),
            0 0 40px rgba(0, 204, 255, 0.3),
            0 0 80px rgba(0, 204, 255, 0.15);
    }

    .hero-sub {
        margin-top: 12px;
        color: rgba(255,255,255,0.5);
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 300;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }

    /* SPLIT LAYOUT CARDS */
    .split-card {
        background: linear-gradient(145deg, #0a1628, #0d1f35);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(0, 255, 170, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        min-height: 320px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .image-card {
        position: relative;
        overflow: hidden;
    }

    .image-card::after {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 16px;
        background: radial-gradient(ellipse at center, rgba(0,170,255,0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    .image-card img {
        max-width: 100%;
        max-height: 280px;
        border-radius: 12px;
        object-fit: contain;
        filter: drop-shadow(0 0 20px rgba(0, 170, 255, 0.15));
    }

    .upload-section-title {
        font-family: 'Inter', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .upload-section-title .icon {
        font-size: 22px;
    }

    /* FILE UPLOADER STYLING */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        border: 1px dashed rgba(0, 255, 170, 0.2);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 255, 170, 0.4);
        box-shadow: 0 0 20px rgba(0, 255, 170, 0.05);
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, #00ffaa, #00ccff);
        color: #07121f;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        border-radius: 12px;
        height: 50px;
        border: none;
        width: 100%;
        font-size: 16px;
        letter-spacing: 0.5px;
        box-shadow: 0 0 20px rgba(0, 255, 170, 0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(0, 255, 170, 0.4), 0 0 60px rgba(0, 255, 170, 0.15);
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #00aaff, #0077ff);
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        border-radius: 12px;
        height: 50px;
        border: none;
        width: 100%;
        font-size: 16px;
        box-shadow: 0 0 20px rgba(0, 170, 255, 0.2);
    }

    .stDownloadButton > button:hover {
        box-shadow: 0 0 30px rgba(0, 170, 255, 0.4), 0 0 60px rgba(0, 170, 255, 0.15);
    }

    /* METRICS */
    [data-testid="stMetric"] {
        background: rgba(0, 255, 170, 0.05);
        border: 1px solid rgba(0, 255, 170, 0.1);
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stMetricValue"] {
        color: #00ffaa;
        font-family: 'Orbitron', monospace;
    }

    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* SUCCESS / SPINNER */
        /* SUCCESS / SPINNER */
    .stSuccess, [data-testid="stNotification"] {
        background: rgba(0, 255, 170, 0.08) !important;
        border: 1px solid rgba(0, 255, 170, 0.2) !important;
        border-radius: 12px;
        color: #00ffaa !important;
    }

    .stSuccess p, [data-testid="stNotification"] p {
        color: #00ffaa !important;
    }

    /* GLOBAL TEXT VISIBILITY */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: rgba(255, 255, 255, 0.85);
    }

    /* TOGGLE / CHECKBOX LABELS */
    [data-testid="stCheckbox"] label span,
    .stToggle label span,
    [data-testid="stToggle"] label span {
        color: rgba(255, 255, 255, 0.85) !important;
    }

    /* METRIC LABELS */
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.6) !important;
    }

    [data-testid="stMetricLabel"] p {
        color: rgba(255, 255, 255, 0.6) !important;
    }

    /* FILE UPLOADER TEXT */
    /* FILE UPLOADER TEXT & ELEMENTS */
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploaderDropzone"] * {
        color: rgba(255, 255, 255, 0.7) !important;
    }

        [data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
        background: #0a1628 !important;
        border: 1px dashed rgba(0, 255, 170, 0.3) !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #00ffaa, #00ccff) !important;
        color: #07121f !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.4rem 1.5rem !important;
        min-width: 120px !important;
    }
    /* DOWNLOAD BUTTON TEXT */
    .stDownloadButton > button {
        color: white !important;
    }

    /* DIVIDER */
    hr {
        border-color: rgba(0, 255, 170, 0.1);
    }

    /* Placeholder image SVG */
    .placeholder-img {
        width: 100%;
        max-width: 360px;
        opacity: 0.9;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------- HERO BANNER ---------------- #
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            AI-POWERED <span class="sql-text">SQL</span> GENERATOR
        </div>
        <div class="hero-sub">
            Turn Business Logic into Optimized Snowflake SQL -Instantly 
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- SPLIT LAYOUT: Image | Upload ---------------- #
    col_img, col_upload = st.columns([1, 1], gap="large")

    with col_img:
        # Load background image via base64
        img_path = r"C:\Users\570666\OneDrive - Cognizant\Desktop\GEN AI POC\gen_ai_sql_bg.jpg.png"
        try:
            with open(img_path, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div class="split-card image-card">
                <img src="data:image/png;base64,{encoded_img}" alt="GEN AI SQL">
            </div>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.markdown("""
            <div class="split-card image-card" style="text-align:center;">
                <svg class="placeholder-img" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
                    <rect width="400" height="300" rx="16" fill="#0a1628"/>
                    <text x="200" y="130" text-anchor="middle" font-family="Orbitron,monospace" font-size="28" font-weight="900" fill="#00ffaa" style="filter:url(#glow)">GEN AI</text>
                    <text x="200" y="170" text-anchor="middle" font-family="Orbitron,monospace" font-size="28" font-weight="900" fill="#00ccff" style="filter:url(#glow2)">SQL</text>
                    <text x="200" y="210" text-anchor="middle" font-family="Inter,sans-serif" font-size="12" fill="rgba(255,255,255,0.4)">Powered by Snowflake Cortex</text>
                    <defs>
                        <filter id="glow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                        <filter id="glow2"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                    </defs>
                </svg>
            </div>
            """, unsafe_allow_html=True)

    with col_upload:
        st.markdown("""
        <div class="upload-section-title" style="font-size: 12px;">
         <span class="icon">📁</span> Upload your requirements
      </div>
     """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["xlsx", "xls", "csv"],
            label_visibility="collapsed"
        )

    # ---------------- FILE PROCESSING (below the split) ---------------- #
    if uploaded_file is not None:
        with st.spinner("Reading file..."):
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully ✅")

        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        if st.toggle("Preview data"):
            st.dataframe(df.head(1000), use_container_width=True, hide_index=True)

        st.divider()

        if st.button("🤖 Generate SQL"):
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(process_row, [row for _, row in df.iterrows()]))

            for sql_src, sql_tgt in results:
              src_generated_sql.append(sql_src)
              tgt_generated_sql.append(sql_tgt)
            src_validated_sql = validation_sql(src_generated_sql)
            tgt_validated_sql = validation_sql(tgt_generated_sql)
            df["Generated_Src_SQL"] = src_validated_sql
            df["Generated_Tgt_SQL"] = tgt_validated_sql

            csv = df.to_csv(index=False).encode("utf-8")
            st.success("SQL generation completed ✅")

            st.download_button(
                "⬇ Download SQL file",
                csv,
                "output_with_sql.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
