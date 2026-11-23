"""
Online NPRA R&D Product & Ingredient Intelligence Platform (Phase 1 + Phase 2 Functional Benefit Engine)
Deployable on Streamlit Community Cloud, Render, Railway, or Google Cloud Run.
"""

import os
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from etl_sync import run_etl, is_postgres, get_postgres_url, DEFAULT_SQLITE_PATH, KKMNOW_CSV_URL
from ai_benefit_engine import (
    CURATED_BENEFIT_PROFILES,
    analyze_positioning_with_llm,
    match_products_by_functional_ingredients,
    fetch_pubmed_evidence,
)

st.set_page_config(
    page_title="NPRA R&D Intelligence Platform",
    page_icon="🧬",
    layout="wide",
)

# Load secrets into environment variables if running on Streamlit Cloud
if hasattr(st, "secrets"):
    for secret_key in ("DATABASE_URL", "APP_PASSWORD", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        if secret_key in st.secrets and not os.getenv(secret_key):
            os.environ[secret_key] = st.secrets[secret_key]

# Optional Team Login Gate
app_password = os.getenv("APP_PASSWORD", "").strip()
if app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🔐 NPRA R&D Intelligence Platform")
        pwd = st.text_input("Enter team access password:", type="password")
        if st.button("Login"):
            if pwd == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()


def ensure_db_ready():
    needs_sync = False
    if is_postgres():
        from sqlalchemy import create_engine, inspect
        engine = create_engine(get_postgres_url())
        insp = inspect(engine)
        if "products" not in insp.get_table_names() or "product_ingredients" not in insp.get_table_names():
            needs_sync = True
        else:
            pi_cols = [c["name"] for c in insp.get_columns("product_ingredients")]
            if "dosage" not in pi_cols or "all_active_ingredients" not in pi_cols:
                needs_sync = True
            else:
                from sqlalchemy import text
                with engine.connect() as c_chk:
                    x_cnt = c_chk.execute(text("SELECT COUNT(*) FROM products WHERE reg_no ~ '[Xx][A-Za-z]*$'")).scalar()
                    if x_cnt and x_cnt > 0:
                        needs_sync = True
    else:
        if not os.path.exists(DEFAULT_SQLITE_PATH):
            needs_sync = True
        else:
            conn = sqlite3.connect(DEFAULT_SQLITE_PATH)
            cur = conn.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if "products" not in tables or "product_ingredients" not in tables:
                needs_sync = True
            else:
                pi_cols = [r[1] for r in cur.execute("PRAGMA table_info(product_ingredients)").fetchall()]
                if "dosage" not in pi_cols or "all_active_ingredients" not in pi_cols:
                    needs_sync = True
                else:
                    x_rows = cur.execute("SELECT COUNT(*) FROM products WHERE UPPER(reg_no) GLOB '*[0-9]X*'").fetchone()[0]
                    if x_rows and x_rows > 0:
                        needs_sync = True
            conn.close()

    if needs_sync:
        with st.spinner("Updating NPRA R&D database schema (raw_material, dosage, all_active_ingredients)..."):
            src = "pharmaceutical_products.csv" if os.path.exists("pharmaceutical_products.csv") else KKMNOW_CSV_URL
            run_etl(source=src)


ensure_db_ready()


@st.cache_data(ttl=3600)
def load_data(use_pg: bool):
    if use_pg:
        from sqlalchemy import create_engine
        engine = create_engine(get_postgres_url())
        with engine.connect() as conn:
            products = pd.read_sql("SELECT * FROM products", conn)
            pi = pd.read_sql("SELECT * FROM product_ingredients", conn)
            rm_sum = pd.read_sql("SELECT * FROM raw_material_summary", conn)
            meta = pd.read_sql("SELECT * FROM sync_metadata", conn)
    else:
        conn = sqlite3.connect(DEFAULT_SQLITE_PATH)
        products = pd.read_sql("SELECT * FROM products", conn)
        pi = pd.read_sql("SELECT * FROM product_ingredients", conn)
        rm_sum = pd.read_sql("SELECT * FROM raw_material_summary", conn)
        meta = pd.read_sql("SELECT * FROM sync_metadata", conn)
        conn.close()
    return products, pi, rm_sum, meta


products_df, pi_df, rm_summary_df, meta_df = load_data(is_postgres())

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧬 NPRA R&D Platform")
if not meta_df.empty:
    st.sidebar.caption(f"Last synced: **{meta_df.iloc[0]['last_synced']}**")
    st.sidebar.caption(
        f"Products: **{meta_df.iloc[0]['total_products']:,}** | "
        f"Ingredients: **{meta_df.iloc[0]['unique_raw_materials']:,}**"
    )

if st.sidebar.button("🔄 Sync Latest Data from KKMNOW", use_container_width=True):
    with st.spinner("Downloading latest dataset from storage.data.gov.my..."):
        try:
            run_etl(source=KKMNOW_CSV_URL)
            st.cache_data.clear()
            st.sidebar.success("Synced latest KKMNOW dataset!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Sync failed: {e}")

st.sidebar.divider()
category_filter = st.sidebar.multiselect(
    "Product Category",
    options=["HEALTH SUPPLEMENT", "NATURAL PRODUCT"],
    default=["HEALTH SUPPLEMENT", "NATURAL PRODUCT"],
)
status_options = sorted(products_df["status"].dropna().unique().tolist())
status_filter = st.sidebar.multiselect(
    "Registration Status",
    options=status_options,
    default=status_options,
)

with st.sidebar.expander("🤖 AI Engine Settings (Optional)"):
    st.caption("Add an OpenAI or Gemini API key to unlock unlimited custom health benefit & positioning discovery.")
    user_openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    user_gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    if user_openai_key:
        os.environ["OPENAI_API_KEY"] = user_openai_key
    if user_gemini_key:
        os.environ["GEMINI_API_KEY"] = user_gemini_key

reg_suf = products_df["reg_no"].astype(str).str.strip().str.extract(r"([A-Za-z]+)$")[0].fillna("").str.upper()
is_nt_reg = reg_suf.str.startswith(("N", "T")) & (~reg_suf.str.startswith("X"))
valid_products = products_df[
    products_df["description"].isin(category_filter)
    & products_df["status"].isin(status_filter)
    & is_nt_reg
]
valid_reg_nos = set(valid_products["reg_no"])
valid_pi = pi_df[pi_df["reg_no"].isin(valid_reg_nos)]

# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------
st.title("NPRA Product & Raw Material R&D Workbench")
st.markdown("Ingredient-level intelligence, functional benefit discovery, and scientific evidence benchmarking.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Filtered Products", f"{len(valid_products):,}")
c2.metric("Health Supplements (N)", f"{(valid_products['description'] == 'HEALTH SUPPLEMENT').sum():,}")
c3.metric("Traditional Medicine / Natural (T)", f"{(valid_products['description'] == 'NATURAL PRODUCT').sum():,}")
c4.metric("Active Raw Materials", f"{valid_pi['raw_material_upper'].nunique():,}")

tab1, tab2, tab3 = st.tabs([
    "🔍 Raw Material & Multi-Ingredient Search",
    "🧠 AI Positioning & Health Benefit Discovery",
    "📊 Co-Ingredient & White-Space Explorer"
])

# ===========================================================================
# TAB 1: Raw Material & Multi-Ingredient Search
# ===========================================================================
with tab1:
    st.subheader("Search Products by Raw Material (Single or Combination)")
    col_a, col_b = st.columns([2, 1])

    all_rm_choices = rm_summary_df["raw_material"].tolist()
    with col_a:
        selected_rms = st.multiselect(
            "Select one or more raw materials (AND logic — products containing ALL selected):",
            options=all_rm_choices,
            default=["BIOTIN"] if "BIOTIN" in all_rm_choices else all_rm_choices[:1],
        )
    with col_b:
        keyword_rm = st.text_input("Or type partial keyword (e.g., GINSENG, COLLAGEN, ZINC):", value="")

    if keyword_rm.strip():
        matched_pi = valid_pi[valid_pi["raw_material_upper"].str.contains(keyword_rm.strip().upper(), na=False)].copy()
        matched_regs = set(matched_pi["reg_no"])
    elif selected_rms:
        selected_uppers = [r.upper() for r in selected_rms]
        matched_regs = set(valid_reg_nos)
        for u in selected_uppers:
            regs_with_u = set(valid_pi[valid_pi["raw_material_upper"] == u]["reg_no"])
            matched_regs = matched_regs.intersection(regs_with_u)
        matched_pi = valid_pi[
            (valid_pi["reg_no"].isin(matched_regs)) & (valid_pi["raw_material_upper"].isin(selected_uppers))
        ].copy()
    else:
        matched_regs = set()
        matched_pi = pd.DataFrame()

    st.info(f"**{len(matched_regs):,}** registered products match your ingredient criteria.")

    if len(matched_regs) > 0:
        # Dosage distribution chart if numeric dosages exist
        dose_df = matched_pi.dropna(subset=["dosage_value"]).copy()
        dose_df = dose_df[dose_df["dosage_unit"] != ""]
        if not dose_df.empty:
            top_unit = dose_df["dosage_unit"].value_counts().index[0]
            unit_subset = dose_df[dose_df["dosage_unit"] == top_unit]
            if len(unit_subset) >= 3:
                fig_dose = px.histogram(
                    unit_subset,
                    x="dosage_value",
                    color="description",
                    nbins=25,
                    title=f"Dosage Distribution ({top_unit}) Across Matching Products",
                    labels={"dosage_value": f"Dosage ({top_unit})", "description": "Category"},
                    color_discrete_sequence=["#2C4D59", "#6B9080"],
                )
                st.plotly_chart(fig_dose, use_container_width=True)

        # Explicitly display `raw_material`, `dosage`, and `all_active_ingredients` side-by-side!
        display_cols_tab1 = [
            "reg_no",
            "product",
            "description",
            "status",
            "raw_material",
            "dosage",
            "all_active_ingredients",
            "holder",
            "manufacturer",
            "date_reg",
            "date_end",
        ]
        # Ensure all columns exist in matched_pi
        avail_cols = [c for c in display_cols_tab1 if c in matched_pi.columns]
        result_table = matched_pi[avail_cols].reset_index(drop=True)

        st.dataframe(
            result_table,
            use_container_width=True,
            height=440,
            column_config={
                "raw_material": st.column_config.TextColumn("raw_material", help="Clean ingredient name without brackets"),
                "dosage": st.column_config.TextColumn("dosage", help="Extracted dosage string"),
                "all_active_ingredients": st.column_config.TextColumn("all_active_ingredients", help="Full original product ingredient formula"),
            },
        )

        csv_export = result_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Matching Products with Raw Material & Dosage (CSV)",
            data=csv_export,
            file_name="npra_matched_products_dosage.csv",
            mime="text/csv",
        )

# ===========================================================================
# TAB 2: Functional Benefit & Product Positioning Discovery (Ingredient-Driven)
# ===========================================================================
with tab2:
    st.subheader("Discover Products by Health Benefit or Product Positioning")
    st.markdown(
        "Enter any health benefit or product positioning (e.g., **Brain Health**, **Sleep & Stress**, "
        "**Joint & Bone**, **Beauty & Skin**, **Gut & Immunity**, **Eye Health**, **Heart & Metabolic**, "
        "or custom concepts like *Women's Postpartum Recovery* or *Healthy Aging*). "
        "The system maps your positioning to **scientifically relevant active ingredients** and finds all NPRA "
        "products formulated with them—even if the product name never mentions the benefit."
    )

    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
    with col_p1:
        custom_positioning = st.text_input(
            "Enter target health benefit or product positioning:",
            value="Brain Health",
            placeholder="e.g., Brain Health, Sleep & Stress, Women's Hormonal Balance...",
        )
    with col_p2:
        preset_choice = st.selectbox(
            "Or quick-select a preset:",
            options=["(Use text box above)"] + [k.title() for k in CURATED_BENEFIT_PROFILES.keys()],
            index=0,
        )
    with col_p3:
        role_filter = st.multiselect(
            "Ingredient Role Filter:",
            options=["Primary Actives", "Secondary / Herbal", "Synergistic Co-Factors"],
            default=["Primary Actives", "Secondary / Herbal", "Synergistic Co-Factors"],
        )

    active_query = custom_positioning if preset_choice == "(Use text box above)" else preset_choice

    if st.button("🔬 Analyse Functional Ingredients & Matching Products", type="primary") or active_query:
        if active_query.strip():
            with st.spinner(f"Mapping functional raw materials for '{active_query}'..."):
                profile = analyze_positioning_with_llm(active_query, all_rm_choices)

            if "error" in profile:
                st.warning(profile["error"])
            else:
                st.caption(f"Analysis Engine: `{profile.get('engine_used', 'Scientific Knowledge Base')}`")
                st.info(f"**Functional Scope ({active_query}):** {profile.get('benefit_summary', '')}")

                filtered_specs = [
                    ing for ing in profile.get("ingredients", [])
                    if any(r.split()[0] in ing.get("role", "") for r in role_filter)
                ]
                filtered_profile = {**profile, "ingredients": filtered_specs}

                breakdown_df, ranked_products = match_products_by_functional_ingredients(
                    filtered_profile, valid_products, valid_pi
                )

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Functionally Relevant Ingredients", len(breakdown_df))
                k2.metric("Matching NPRA Products", f"{len(ranked_products):,}")
                k3.metric(
                    "Multi-Active Formulations (2+ Actives)",
                    f"{(ranked_products['functional_actives_count'] >= 2).sum():,}" if not ranked_products.empty else "0",
                )
                k4.metric(
                    "Health Supplement vs Natural",
                    f"{(ranked_products['description'] == 'HEALTH SUPPLEMENT').sum():,} HS / {(ranked_products['description'] == 'NATURAL PRODUCT').sum():,} NP"
                    if not ranked_products.empty else "0 / 0",
                )

                # Section A: Functional Ingredient Landscape
                st.markdown("### 1. Functionally Relevant Raw Materials in NPRA Database")
                if not breakdown_df.empty:
                    fig_ing = px.bar(
                        breakdown_df[breakdown_df["NPRA Products Found"] > 0].iloc[::-1],
                        x="NPRA Products Found",
                        y="Functional Ingredient",
                        color="Formulation Role",
                        orientation="h",
                        title=f"NPRA Product Penetration of {active_query}-Relevant Raw Materials",
                        color_discrete_map={
                            "Primary Actives": "#2C4D59",
                            "Secondary / Herbal": "#3E6B7B",
                            "Synergistic Co-Factors": "#84A59D",
                        },
                    )
                    st.plotly_chart(fig_ing, use_container_width=True)
                    st.dataframe(breakdown_df, use_container_width=True, height=340)

                # Section B: Live PubMed Scientific Evidence Check
                st.markdown("### 2. Live Peer-Reviewed Clinical Evidence Check (PubMed)")
                if not breakdown_df.empty:
                    ev_col1, ev_col2 = st.columns([2, 1])
                    with ev_col1:
                        selected_ev_ing = st.selectbox(
                            "Select a functional ingredient to verify peer-reviewed PubMed clinical studies:",
                            options=breakdown_df["Functional Ingredient"].tolist(),
                            index=0,
                        )
                    with ev_col2:
                        st.write("")
                        st.write("")
                        run_pubmed = st.button("📚 Fetch PubMed Clinical Studies")

                    if run_pubmed and selected_ev_ing:
                        with st.spinner(f"Querying NCBI PubMed for {selected_ev_ing} + {active_query}..."):
                            pm_res = fetch_pubmed_evidence(selected_ev_ing, active_query)
                        if "error" in pm_res:
                            st.error(f"PubMed lookup failed: {pm_res['error']}")
                        else:
                            st.success(f"Found **{pm_res['total_papers']:,}** indexed PubMed papers for `{selected_ev_ing}` + `{active_query}`.")
                            for art in pm_res.get("articles", []):
                                st.markdown(
                                    f"- **[PMID {art['pmid']}]({art['url']})** — {art['title']} "
                                    f"*(_{art['journal']}_, {art['pubdate']})*"
                                )

                # Section C: Ranked Matching NPRA Products (showing raw_material, dosage, and all_active_ingredients)
                st.markdown("### 3. Registered NPRA Products Containing These Functional Actives")
                if not ranked_products.empty:
                    min_actives = st.slider(
                        "Minimum number of functional actives per product (filter for synergistic multi-ingredient formulas):",
                        min_value=1,
                        max_value=max(1, int(ranked_products["functional_actives_count"].max())),
                        value=1,
                    )
                    filtered_ranked = ranked_products[ranked_products["functional_actives_count"] >= min_actives].copy()
                    if "all_active_ingredients" not in filtered_ranked.columns and "active_ingredient" in filtered_ranked.columns:
                        filtered_ranked["all_active_ingredients"] = filtered_ranked["active_ingredient"]

                    display_cols_tab2 = [
                        "reg_no",
                        "product",
                        "description",
                        "relevance_score",
                        "functional_actives_count",
                        "matched_raw_materials",
                        "matched_dosages",
                        "all_active_ingredients",
                        "holder",
                        "manufacturer",
                    ]
                    avail_cols_t2 = [c for c in display_cols_tab2 if c in filtered_ranked.columns]
                    display_prods = filtered_ranked[avail_cols_t2]

                    st.caption(f"Showing **{len(display_prods):,}** products ranked by functional formulation relevance score:")
                    st.dataframe(
                        display_prods,
                        use_container_width=True,
                        height=440,
                        column_config={
                            "matched_raw_materials": st.column_config.TextColumn("raw_material (matched)", help="Clean matched functional ingredient names without brackets"),
                            "matched_dosages": st.column_config.TextColumn("dosage (matched)", help="Matched ingredients with their extracted dosage strings"),
                            "all_active_ingredients": st.column_config.TextColumn("all_active_ingredients", help="Full original product ingredient formula"),
                        },
                    )

                    st.download_button(
                        f"⬇️ Download {active_query} Product Landscape (CSV)",
                        data=display_prods.to_csv(index=False).encode("utf-8"),
                        file_name=f"npra_{active_query.lower().replace(' ', '_')}_products.csv",
                        mime="text/csv",
                    )

# ===========================================================================
# TAB 3: Co-Ingredient & White-Space Explorer
# ===========================================================================
with tab3:
    st.subheader("Co-Ingredient Synergy & Niche Opportunity Finder")
    anchor_rm = st.selectbox(
        "Pick an anchor raw material to see what other ingredients it is most commonly paired with:",
        options=all_rm_choices,
        index=0,
    )
    if anchor_rm:
        anchor_regs = set(valid_pi[valid_pi["raw_material_upper"] == anchor_rm.upper()]["reg_no"])
        co_pi = valid_pi[
            (valid_pi["reg_no"].isin(anchor_regs)) & (valid_pi["raw_material_upper"] != anchor_rm.upper())
        ]
        co_counts = (
            co_pi.groupby("raw_material")["reg_no"]
            .nunique()
            .reset_index(name="co_occurrence_count")
            .sort_values("co_occurrence_count", ascending=False)
            .head(25)
        )
        col_left, col_right = st.columns([3, 2])
        with col_left:
            fig_co = px.bar(
                co_counts.iloc[::-1],
                x="co_occurrence_count",
                y="raw_material",
                orientation="h",
                title=f"Top Co-Formulated Ingredients with {anchor_rm}",
                color_discrete_sequence=["#3E6B7B"],
            )
            st.plotly_chart(fig_co, use_container_width=True)
        with col_right:
            st.markdown("**Co-Formulation Table**")
            st.dataframe(co_counts, use_container_width=True, height=450)
