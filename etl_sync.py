"""
Cloud-Ready ETL Pipeline: Daily NPRA Pharmaceutical Products Sync
- Pulls latest CSV directly from MOH Malaysia (KKMNOW / data.gov.my)
- Filters strictly to Health Supplements (MAL...N) and Traditional Medicine / Natural Products (MAL...T),
  excluding OTC ('X') and other registration categories.
- Parses active ingredients, clean raw_material names, dosage strings, numeric dosage values & units
- Preserves full original formula under `all_active_ingredients` on every product & ingredient row
- Writes to Cloud PostgreSQL (Supabase/Neon) if DATABASE_URL is set, or SQLite locally
"""

import os
import re
import sqlite3
from datetime import datetime
import pandas as pd

KKMNOW_CSV_URL = "https://storage.data.gov.my/healthcare/pharmaceutical_products.csv"
DEFAULT_SQLITE_PATH = "npra_rnd.db"


def is_postgres() -> bool:
    url = os.getenv("DATABASE_URL", "").strip()
    return url.startswith("postgres://") or url.startswith("postgresql://")


def get_postgres_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def parse_single_ingredient(item_str: str) -> dict:
    item_str = item_str.strip()
    m = re.match(r"^([^\[]+)\[(.*)\]$", item_str)
    if not m:
        return {
            "raw_material": item_str,
            "dosage": "",
            "dosage_value": None,
            "dosage_unit": "",
            "base_unit_raw": "",
            "ingredient_with_dosage": item_str,
        }

    name = m.group(1).strip()
    dosage_inside = m.group(2).strip()
    tokens = [t.strip() for t in re.split(r"[,;]", dosage_inside) if t.strip()]
    first_dose = tokens[0] if tokens else ""
    base_unit = tokens[1] if len(tokens) > 1 else ""

    num_match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Zµμ/%]+)?", first_dose)
    if num_match:
        dosage_val = float(num_match.group(1))
        dosage_unit = (num_match.group(2) or "").strip()
    else:
        dosage_val = None
        dosage_unit = ""

    return {
        "raw_material": name,
        "dosage": dosage_inside,
        "dosage_value": dosage_val,
        "dosage_unit": dosage_unit,
        "base_unit_raw": base_unit,
        "ingredient_with_dosage": item_str,
    }


def split_ingredients_list(text) -> list[dict]:
    if pd.isna(text) or not str(text).strip():
        return []
    parts = re.split(r"(?<=\])\s*,\s*", str(text).strip())
    return [parse_single_ingredient(p) for p in parts if p.strip()]


def run_etl(source: str = KKMNOW_CSV_URL) -> dict:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching: {source}")
    df = pd.read_csv(source, low_memory=False)

    df["description"] = df["description"].astype(str).str.strip()
    df["reg_no"] = df["reg_no"].astype(str).str.strip()

    # Extract trailing registration category letters (e.g., N, NC, NCR, T, TC, X, XC)
    reg_suffix = df["reg_no"].str.extract(r"([A-Za-z]+)$")[0].fillna("").str.upper()

    # Keep ONLY Health Supplements (N...) and Traditional Medicine / Natural Products (T...), excluding OTC (X...)
    is_valid_desc = df["description"].isin(["HEALTH SUPPLEMENT", "NATURAL PRODUCT"])
    is_nt_only = reg_suffix.str.startswith(("N", "T")) & (~reg_suffix.str.startswith("X"))

    filtered = df[is_valid_desc & is_nt_only].copy()
    filtered.reset_index(drop=True, inplace=True)

    # Preserve full original formula explicitly as `all_active_ingredients`
    filtered["all_active_ingredients"] = filtered["active_ingredient"]

    parsed_lists = filtered["active_ingredient"].apply(split_ingredients_list)
    filtered["ingredient_count"] = parsed_lists.apply(len)

    pi_rows = []
    for idx, row in filtered.iterrows():
        reg_no = row["reg_no"]
        product_name = row["product"]
        status = row["status"]
        desc = row["description"]
        holder = row["holder"]
        manufacturer = row["manufacturer"]
        date_reg = row["date_reg"]
        date_end = row["date_end"]
        full_formula = row["all_active_ingredients"]

        for item in parsed_lists.iloc[idx]:
            pi_rows.append({
                "reg_no": reg_no,
                "product": product_name,
                "status": status,
                "description": desc,
                "holder": holder,
                "manufacturer": manufacturer,
                "date_reg": date_reg,
                "date_end": date_end,
                "raw_material": item["raw_material"],
                "raw_material_upper": item["raw_material"].upper(),
                "dosage": item["dosage"],
                "dosage_value": item["dosage_value"],
                "dosage_unit": item["dosage_unit"],
                "base_unit_raw": item["base_unit_raw"],
                "ingredient_with_dosage": item["ingredient_with_dosage"],
                "all_active_ingredients": full_formula,
            })

    pi_df = pd.DataFrame(pi_rows)

    rm_summary = (
        pi_df.groupby("raw_material_upper")
        .agg(
            raw_material=("raw_material", lambda s: s.value_counts().index[0]),
            total_products=("reg_no", "nunique"),
            hs_products=("reg_no", lambda s: s[pi_df.loc[s.index, "description"] == "HEALTH SUPPLEMENT"].nunique()),
            np_products=("reg_no", lambda s: s[pi_df.loc[s.index, "description"] == "NATURAL PRODUCT"].nunique()),
        )
        .reset_index()
        .sort_values(by=["total_products", "raw_material"], ascending=[False, True])
        .reset_index(drop=True)
    )

    meta = pd.DataFrame([{
        "last_synced": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": source,
        "total_products": len(filtered),
        "total_ingredient_links": len(pi_df),
        "unique_raw_materials": len(rm_summary),
    }])

    if is_postgres():
        from sqlalchemy import create_engine, text
        engine = create_engine(get_postgres_url())
        with engine.begin() as conn:
            filtered.to_sql("products", conn, if_exists="replace", index=False, chunksize=5000)
            pi_df.to_sql("product_ingredients", conn, if_exists="replace", index=False, chunksize=5000)
            rm_summary.to_sql("raw_material_summary", conn, if_exists="replace", index=False, chunksize=5000)
            meta.to_sql("sync_metadata", conn, if_exists="replace", index=False)

            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_prod_reg ON products(reg_no);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_prod_desc ON products(description);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pi_reg ON product_ingredients(reg_no);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pi_rm_upper ON product_ingredients(raw_material_upper);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rm_upper ON raw_material_summary(raw_material_upper);"))
        target_label = "Cloud PostgreSQL"
    else:
        conn = sqlite3.connect(DEFAULT_SQLITE_PATH)
        try:
            filtered.to_sql("products", conn, if_exists="replace", index=False)
            pi_df.to_sql("product_ingredients", conn, if_exists="replace", index=False)
            rm_summary.to_sql("raw_material_summary", conn, if_exists="replace", index=False)
            meta.to_sql("sync_metadata", conn, if_exists="replace", index=False)

            cur = conn.cursor()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_prod_reg ON products(reg_no);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_prod_desc ON products(description);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pi_reg ON product_ingredients(reg_no);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pi_rm_upper ON product_ingredients(raw_material_upper);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rm_upper ON raw_material_summary(raw_material_upper);")
            conn.commit()
        finally:
            conn.close()
        target_label = f"Local SQLite ({DEFAULT_SQLITE_PATH})"

    stats = {
        "total_products": len(filtered),
        "total_ingredient_links": len(pi_df),
        "unique_raw_materials": len(rm_summary),
        "target_db": target_label,
    }
    print("ETL Sync Complete:", stats)
    return stats


if __name__ == "__main__":
    local_csv = "pharmaceutical_products.csv"
    src = local_csv if os.path.exists(local_csv) else KKMNOW_CSV_URL
    run_etl(source=src)
