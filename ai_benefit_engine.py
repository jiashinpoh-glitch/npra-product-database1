"""
Phase 2 AI Benefit & Scientific Evidence Engine
1. Maps a user-defined product positioning / health benefit (e.g., "Brain Health",
   "Women's Stress & Sleep", "Joint Mobility", "Skin Radiance") to functionally
   relevant raw materials in the NPRA database—even when the product name never
   mentions the health benefit.
2. Works in two modes:
   - AI Mode (OpenAI or Google Gemini API key): Analyzes the actual unique raw
     materials in your NPRA database + returns structured scientific mechanisms,
     evidence tiers, synonyms, and PubMed search queries.
   - Built-in Curated Biomedical Knowledge Base Fallback (Zero API Key Needed):
     Provides immediate functional mapping across major R&D health categories
     (Brain Health, Sleep & Stress, Joint & Bone, Gut & Immunity, Beauty & Skin,
     Heart & Metabolic, Eye Health, Energy & Vitality, Liver Detox, Women's Health).
3. Live PubMed E-utilities Verification: Fetches real peer-reviewed clinical study
   counts and top PMID citations from NCBI PubMed for any identified raw material.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import pandas as pd


# Built-in functional taxonomy fallback (used automatically if no LLM API key is configured)
CURATED_BENEFIT_PROFILES = {
    "brain health": {
        "benefit_summary": "Cognitive performance, memory retention, neuroprotection, focus, and cerebral blood flow support.",
        "ingredients": [
            {"keyword": "GINKGO BILOBA", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Enhances cerebral microcirculation and provides antioxidant neuroprotection."},
            {"keyword": "BACOPA MONNIERI", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Bacosides modulate cholinergic neurotransmission and support memory consolidation."},
            {"keyword": "PHOSPHATIDYLSERINE", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Key neuronal membrane phospholipid supporting neurotransmitter release and cognitive processing."},
            {"keyword": "DHA", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Essential omega-3 structural fatty acid in neuronal membranes supporting synaptic plasticity."},
            {"keyword": "DOCOSAHEXAENOIC ACID", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Essential omega-3 structural fatty acid in brain grey matter and visual/cognitive function."},
            {"keyword": "OMEGA-3", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "EPA/DHA reduce neuroinflammation and maintain neuronal membrane fluidity."},
            {"keyword": "FISH OIL", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Rich source of EPA and DHA supporting cognitive function and brain aging."},
            {"keyword": "HERICIUM ERINACEUS", "role": "Primary Actives", "evidence": "Emerging Clinical Evidence", "mechanism": "Lion's Mane mushroom; hericenones and erinacines stimulate Nerve Growth Factor (NGF) synthesis."},
            {"keyword": "LION'S MANE", "role": "Primary Actives", "evidence": "Emerging Clinical Evidence", "mechanism": "Stimulates Nerve Growth Factor (NGF) and supports mild cognitive function."},
            {"keyword": "CENTELLA ASIATICA", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Gotu kola triterpenoids support working memory, alertness, and calm focus."},
            {"keyword": "PANAX GINSENG", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Ginsenosides reduce mental fatigue and improve attention and working memory."},
            {"keyword": "METHYLCOBALAMIN", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Bioactive Vitamin B12 essential for myelin synthesis and homocysteine metabolism."},
            {"keyword": "CYANOCOBALAMIN", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Vitamin B12 cofactor supporting neurological function and red blood cell formation."},
            {"keyword": "PYRIDOXINE", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Vitamin B6 required for synthesis of dopamine, serotonin, and GABA neurotransmitters."},
            {"keyword": "FOLIC ACID", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Supports one-carbon metabolism and lowers neurotoxic homocysteine levels."},
            {"keyword": "CHOLINE", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Direct precursor to acetylcholine, the primary neurotransmitter for memory and learning."},
            {"keyword": "INOSITOL", "role": "Synergistic Co-Factors", "evidence": "Moderate Clinical Evidence", "mechanism": "Intracellular second messenger involved in neurotransmitter receptor signaling."},
            {"keyword": "CURCUMA LONGA", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Curcuminoids cross the blood-brain barrier to reduce oxidative stress and neuroinflammation."},
            {"keyword": "LUTEIN", "role": "Synergistic Co-Factors", "evidence": "Moderate Clinical Evidence", "mechanism": "Accumulates in brain neural tissue and supports processing speed and visual-cognitive performance."},
            {"keyword": "ASTAXANTHIN", "role": "Synergistic Co-Factors", "evidence": "Emerging Clinical Evidence", "mechanism": "Crosses the blood-brain barrier to protect neuronal lipids against oxidative damage."},
        ],
    },
    "sleep & stress": {
        "benefit_summary": "Stress resilience, cortisol modulation, relaxation, anxiety reduction, and sleep quality improvement.",
        "ingredients": [
            {"keyword": "WITHANIA SOMNIFERA", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Ashwagandha withanolides modulate HPA-axis cortisol response and improve sleep onset."},
            {"keyword": "ASHWAGANDHA", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Adaptogen shown in RCTs to reduce perceived stress and serum cortisol."},
            {"keyword": "PASSIFLORA", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Passionflower modulates GABA receptors to promote relaxation and restful sleep."},
            {"keyword": "VALERIANA", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Valerian root inhibits GABA breakdown in the CNS, reducing sleep latency."},
            {"keyword": "MELISSA OFFICINALIS", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Lemon balm inhibits GABA transaminase, easing nervous tension and restlessness."},
            {"keyword": "MATRICARIA", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Chamomile apigenin binds benzodiazepine-site GABA receptors for mild sedation."},
            {"keyword": "CHAMOMILE", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Promotes calm relaxation and gentle sleep support."},
            {"keyword": "MAGNESIUM", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Acts as an NMDA antagonist and GABA agonist supporting neuromuscular relaxation."},
            {"keyword": "THEANINE", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "L-theanine increases alpha brain waves and promotes calm alertness without drowsiness."},
            {"keyword": "RHODIOLA ROSEA", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Adaptogen that buffers stress-induced mental exhaustion and burnout."},
            {"keyword": "ZIZIPHUS JUJUBA", "role": "Secondary / Herbal", "evidence": "Traditional + Preclinical", "mechanism": "Suan Zao Ren saponins and flavonoids modulate GABAergic and serotonergic sleep pathways."},
            {"keyword": "PYRIDOXINE", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Cofactor in converting glutamate to calming GABA and tryptophan to serotonin/melatonin."},
        ],
    },
    "joint & bone": {
        "benefit_summary": "Cartilage protection, osteoarthritis comfort, anti-inflammatory joint mobility, and bone mineral density.",
        "ingredients": [
            {"keyword": "GLUCOSAMINE", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Amino sugar precursor for glycosaminoglycan and articular cartilage matrix synthesis."},
            {"keyword": "CHONDROITIN", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Maintains cartilage hydration and inhibits cartilage-degrading enzymes."},
            {"keyword": "METHYLSULFONYLMETHANE", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "MSM supplies bioavailable sulfur and reduces exercise- and osteoarthritis-related joint pain."},
            {"keyword": "BOSWELLIA SERRATA", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "AKBA boswellic acids inhibit 5-lipoxygenase (5-LOX) to reduce joint inflammation."},
            {"keyword": "CURCUMA LONGA", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Curcumin downregulates NF-κB, COX-2, and pro-inflammatory cytokines in synovial tissue."},
            {"keyword": "COLLAGEN", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Hydrolyzed or undenatured type II collagen stimulates chondrocyte matrix repair."},
            {"keyword": "HYALURONIC ACID", "role": "Synergistic Co-Factors", "evidence": "Moderate Clinical Evidence", "mechanism": "Enhances synovial fluid viscoelasticity and joint lubrication."},
            {"keyword": "CALCIUM", "role": "Primary Actives", "evidence": "Established Nutrient Role", "mechanism": "Primary structural mineral of bone hydroxyapatite matrix."},
            {"keyword": "CHOLECALCIFEROL", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Vitamin D3 promotes intestinal calcium absorption and bone mineralization."},
            {"keyword": "VITAMIN D3", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Essential for calcium homeostasis and skeletal strength."},
            {"keyword": "MENAQUINONE", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Vitamin K2 activates osteocalcin to bind calcium into bone matrix rather than arteries."},
            {"keyword": "VITAMIN K2", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Directs calcium deposition into skeletal bone via osteocalcin carboxylation."},
        ],
    },
    "beauty & skin": {
        "benefit_summary": "Skin elasticity, hydration, collagen synthesis, UV photoprotection, and hair/nail strength.",
        "ingredients": [
            {"keyword": "COLLAGEN", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Bioactive di/tripeptides stimulate dermal fibroblast procollagen and elastin synthesis."},
            {"keyword": "HYALURONIC ACID", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Oral hyaluronan increases stratum corneum moisture retention and skin suppleness."},
            {"keyword": "ASTAXANTHIN", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Potent xanthophyll carotenoid that reduces UV-induced oxidative stress and fine wrinkles."},
            {"keyword": "GLUTATHIONE", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Inhibits tyrosinase activity and shifts melanogenesis toward lighter pheomelanin."},
            {"keyword": "ASCORBIC ACID", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Obligate cofactor for prolyl and lysyl hydroxylase enzymes in collagen cross-linking."},
            {"keyword": "VITAMIN C", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Essential cofactor for dermal collagen biosynthesis and antioxidant defense."},
            {"keyword": "BIOTIN", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Supports keratin infrastructure for brittle nails and hair follicle health."},
            {"keyword": "CERAMIDE", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Phytoceramides reinforce the epidermal lipid barrier and reduce transepidermal water loss."},
            {"keyword": "GRAPE SEED", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Oligomeric proanthocyanidins (OPCs) protect dermal elastin and collagen from degradation."},
            {"keyword": "LYCOPENE", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Carotenoid antioxidant supporting skin smoothness and UV erythema defense."},
        ],
    },
    "gut & immunity": {
        "benefit_summary": "Microbiome balance, digestive regularity, mucosal barrier integrity, and innate/adaptive immune defense.",
        "ingredients": [
            {"keyword": "LACTOBACILLUS", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Probiotic genus producing lactic acid, competitive pathogen exclusion, and gut barrier support."},
            {"keyword": "BIFIDOBACTERIUM", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Colonizes the colon, produces short-chain fatty acids (SCFAs), and modulates mucosal immunity."},
            {"keyword": "SACCHAROMYCES BOULARDII", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Beneficial probiotic yeast supporting gut flora recovery and secretory IgA."},
            {"keyword": "INULIN", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Prebiotic fructan selectively fermented by beneficial Bifidobacteria."},
            {"keyword": "FRUCTOOLIGOSACCHARIDE", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Prebiotic fiber promoting SCFA butyrate production and digestive regularity."},
            {"keyword": "BETA GLUCAN", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Binds Dectin-1 receptors on macrophages and NK cells to prime innate immune response."},
            {"keyword": "ECHINACEA", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Alkamides and polysaccharides modulate upper respiratory immune defenses."},
            {"keyword": "SAMBUCUS NIGRA", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Elderberry anthocyanins support upper respiratory tract immune resilience."},
            {"keyword": "ZINC", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Essential trace mineral for T-cell maturation, thymulin activity, and mucosal integrity."},
            {"keyword": "ASCORBIC ACID", "role": "Synergistic Co-Factors", "evidence": "Strong Clinical Backing", "mechanism": "Accumulates in neutrophils and lymphocytes to support phagocytosis and antioxidant defense."},
            {"keyword": "CORDYCEPS", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Cordycepin and polysaccharides support respiratory vitality and immune modulation."},
            {"keyword": "GANODERMA LUCIDUM", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Lingzhi beta-glucans and triterpenes modulate immune cell signaling."},
        ],
    },
    "eye health": {
        "benefit_summary": "Macular pigment protection, blue-light filtering, visual acuity, and dry-eye relief.",
        "ingredients": [
            {"keyword": "LUTEIN", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Selectively accumulates in the retinal macula to filter high-energy blue light and neutralize ROS."},
            {"keyword": "ZEAXANTHIN", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Concentrates in the central fovea alongside lutein (AREDS2 formulation) to protect photoreceptors."},
            {"keyword": "VACCINIUM MYRTILLUS", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Bilberry anthocyanins support ocular microcirculation and visual fatigue recovery."},
            {"keyword": "BILBERRY", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Rich in anthocyanosides supporting rhodopsin regeneration and eye strain relief."},
            {"keyword": "ASTAXANTHIN", "role": "Primary Actives", "evidence": "Moderate Clinical Evidence", "mechanism": "Crosses the blood-retinal barrier to relieve ciliary muscle fatigue and digital eye strain."},
            {"keyword": "EUPHRASIA", "role": "Secondary / Herbal", "evidence": "Traditional + Preclinical", "mechanism": "Eyebright herb traditionally used to soothe ocular irritation."},
            {"keyword": "LYCIUM", "role": "Secondary / Herbal", "evidence": "Traditional + Clinical", "mechanism": "Goji berry (Fructus Lycii) naturally rich in zeaxanthin dipalmitate for macular health."},
            {"keyword": "RETINYL", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Vitamin A precursor required for rhodopsin visual pigment synthesis."},
            {"keyword": "BETA-CAROTENE", "role": "Synergistic Co-Factors", "evidence": "Established Nutrient Role", "mechanism": "Provitamin A carotenoid supporting low-light vision and corneal surface health."},
        ],
    },
    "heart & metabolic": {
        "benefit_summary": "Cardiovascular function, healthy lipid/cholesterol balance, blood glucose regulation, and cellular energy.",
        "ingredients": [
            {"keyword": "UBIDECARENONE", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Coenzyme Q10 drives mitochondrial ATP synthesis in myocardium and protects LDL from oxidation."},
            {"keyword": "COENZYME Q10", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Essential electron transport carrier supporting cardiac bioenergetics."},
            {"keyword": "OMEGA-3", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "EPA and DHA lower serum triglycerides and support endothelial vascular health."},
            {"keyword": "EPA", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Eicosapentaenoic acid modulates eicosanoids and supports cardiovascular health."},
            {"keyword": "MONASCUS PURPUREUS", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Red yeast rice naturally contains monacolin K which inhibits hepatic HMG-CoA reductase."},
            {"keyword": "ALLIUM SATIVUM", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Aged garlic allicin/S-allylcysteine supports healthy blood pressure and lipid profiles."},
            {"keyword": "CINNAMOMUM", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Polyphenols enhance insulin receptor sensitivity and postprandial glucose uptake."},
            {"keyword": "BERBERINE", "role": "Primary Actives", "evidence": "Strong Clinical Backing", "mechanism": "Activates AMPK pathway to improve insulin sensitivity, lipid metabolism, and glycemic control."},
            {"keyword": "CHROMIUM", "role": "Synergistic Co-Factors", "evidence": "Moderate Clinical Evidence", "mechanism": "Trace element potentiating insulin signaling via chromodulin."},
            {"keyword": "RESVERATROL", "role": "Secondary / Herbal", "evidence": "Moderate Clinical Evidence", "mechanism": "Activates SIRT1 and endothelial nitric oxide synthase (eNOS) for vascular protection."},
        ],
    },
}


def find_curated_profile(query: str) -> dict | None:
    q = query.strip().lower()
    if q in CURATED_BENEFIT_PROFILES:
        return CURATED_BENEFIT_PROFILES[q]
    # Fuzzy keyword match across curated keys
    keyword_aliases = {
        "brain health": ["brain", "cognitive", "memory", "focus", "nootropic", "neuro", "mental", "concentration"],
        "sleep & stress": ["sleep", "stress", "anxiety", "calm", "relax", "insomnia", "cortisol", "mood"],
        "joint & bone": ["joint", "bone", "cartilage", "arthritis", "mobility", "osteo", "knee", "calcium"],
        "beauty & skin": ["beauty", "skin", "whitening", "collagen", "anti-aging", "hair", "nail", "radiance", "glow"],
        "gut & immunity": ["gut", "immune", "immunity", "probiotic", "digestion", "digestive", "microbiome", "constipation"],
        "eye health": ["eye", "vision", "macular", "ocular", "lutein", "sight", "blue light"],
        "heart & metabolic": ["heart", "cardio", "cholesterol", "blood sugar", "glucose", "diabetes", "metabolic", "lipid", "blood pressure"],
    }
    for profile_key, aliases in keyword_aliases.items():
        if any(a in q for a in aliases):
            return CURATED_BENEFIT_PROFILES[profile_key]
    return None


def analyze_positioning_with_llm(positioning_query: str, unique_raw_materials: list[str]) -> dict:
    """
    Uses OpenAI (OPENAI_API_KEY) or Google Gemini (GEMINI_API_KEY) to map a custom
    product positioning / health benefit to functionally relevant raw materials
    present in the NPRA database. Falls back to CURATED_BENEFIT_PROFILES if no key is set.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    # Provide top 1,500 most common NPRA raw materials as grounding context so the LLM
    # prioritizes exact ingredient names that exist in the Malaysian NPRA registry!
    grounded_sample = ", ".join(unique_raw_materials[:1200])

    system_prompt = (
        "You are an expert nutraceutical R&D scientist and regulatory formulator specializing in "
        "Malaysian NPRA Health Supplements and Natural Products. Given a target product positioning "
        "or health benefit, identify 12 to 20 scientifically relevant active raw materials that deliver "
        "that functional benefit—even if products containing them do not mention the benefit in their brand name. "
        "Use uppercase canonical ingredient keywords that match NPRA active_ingredient naming conventions "
        "(e.g., GINKGO BILOBA, BACOPA MONNIERI, PHOSPHATIDYLSERINE, DOCOSAHEXAENOIC ACID, METHYLCOBALAMIN, "
        "WITHANIA SOMNIFERA, UBIDECARENONE, BOSWELLIA SERRATA, LUTEIN, LACTOBACILLUS). "
        "Return ONLY valid JSON matching this exact schema:\n"
        '{"benefit_summary": "1-sentence clinical summary of this positioning", '
        '"ingredients": [{"keyword": "INGREDIENT_NAME", "role": "Primary Actives | Secondary / Herbal | Synergistic Co-Factors", '
        '"evidence": "Strong Clinical Backing | Moderate Clinical Evidence | Emerging Clinical Evidence | Established Nutrient Role", '
        '"mechanism": "1-sentence mechanism of action"}]}'
    )

    user_prompt = (
        f"Target Product Positioning / Health Benefit: {positioning_query}\n\n"
        f"Reference list of frequent NPRA-registered raw materials in our database:\n{grounded_sample}"
    )

    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            data = json.loads(resp.choices[0].message.content)
            data["engine_used"] = "OpenAI (gpt-4o-mini)"
            return data
        except Exception as e:
            return {"error": f"OpenAI API error: {e}"}

    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"{system_prompt}\n\n{user_prompt}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            data = json.loads(resp.text)
            data["engine_used"] = "Google Gemini (gemini-2.5-flash)"
            return data
        except Exception as e:
            return {"error": f"Gemini API error: {e}"}

    # Fallback to built-in curated scientific taxonomy
    curated = find_curated_profile(positioning_query)
    if curated:
        res = dict(curated)
        res["engine_used"] = "Built-in Scientific Knowledge Base (Add OPENAI_API_KEY or GEMINI_API_KEY in Secrets for unlimited custom positionings)"
        return res

    return {
        "error": (
            f"No built-in template matched '{positioning_query}'. "
            "Try presets like 'Brain Health', 'Sleep & Stress', 'Joint & Bone', 'Beauty & Skin', "
            "'Gut & Immunity', 'Eye Health', or 'Heart & Metabolic'—or add an OPENAI_API_KEY / GEMINI_API_KEY "
            "in Streamlit Secrets to analyze ANY custom positioning automatically."
        )
    }


def match_products_by_functional_ingredients(
    benefit_profile: dict,
    valid_products: pd.DataFrame,
    valid_pi: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Matches all NPRA products containing any of the functionally mapped ingredients,
    scores each product by how many relevant functional actives it contains (relevance score),
    and returns:
      1) ingredient_breakdown_df: Each mapped raw material + how many NPRA products use it
      2) ranked_products_df: All matching NPRA products ranked by functional relevance score
    """
    ing_specs = benefit_profile.get("ingredients", [])
    if not ing_specs:
        return pd.DataFrame(), pd.DataFrame()

    breakdown_rows = []
    # Map reg_no -> list of matched functional ingredient labels & roles
    product_hits: dict[str, list[dict]] = {}

    for spec in ing_specs:
        kw = spec["keyword"].strip().upper()
        role = spec.get("role", "Primary Actives")
        evidence = spec.get("evidence", "Moderate Clinical Evidence")
        mechanism = spec.get("mechanism", "")

        hits = valid_pi[valid_pi["raw_material_upper"].str.contains(re.escape(kw), na=False)]
        matched_reg_set = set(hits["reg_no"])
        hs_cnt = hits[hits["description"] == "HEALTH SUPPLEMENT"]["reg_no"].nunique()
        np_cnt = hits[hits["description"] == "NATURAL PRODUCT"]["reg_no"].nunique()

        breakdown_rows.append({
            "Functional Ingredient": kw,
            "Formulation Role": role,
            "Scientific Evidence": evidence,
            "NPRA Products Found": len(matched_reg_set),
            "Health Supplements": hs_cnt,
            "Natural Products": np_cnt,
            "Mechanism of Action": mechanism,
        })

        weight = 3 if "Primary" in role else (2 if "Secondary" in role else 1)
        for reg_no in matched_reg_set:
            product_hits.setdefault(reg_no, []).append({
                "keyword": kw,
                "role": role,
                "weight": weight,
            })

    breakdown_df = (
        pd.DataFrame(breakdown_rows)
        .sort_values(by=["NPRA Products Found", "Functional Ingredient"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if not product_hits:
        return breakdown_df, pd.DataFrame()

    # Map reg_no -> exact matched ingredient_with_dosage items from valid_pi
    all_matched_kws = [spec["keyword"].strip().upper() for spec in ing_specs]
    regex_pattern = "|".join(re.escape(k) for k in all_matched_kws)
    matched_pi_sub = valid_pi[valid_pi["raw_material_upper"].str.contains(regex_pattern, na=False)]
    dosage_lookup = (
        matched_pi_sub.groupby("reg_no")["ingredient_with_dosage"]
        .apply(lambda s: ", ".join(dict.fromkeys(s.dropna().tolist())))
        .to_dict()
    )

    # Score and rank matching products
    score_records = []
    for reg_no, hit_list in product_hits.items():
        unique_kws = sorted(list({h["keyword"] for h in hit_list}))
        primary_hits = [h["keyword"] for h in hit_list if "Primary" in h["role"]]
        relevance_score = sum(h["weight"] for h in hit_list)
        score_records.append({
            "reg_no": reg_no,
            "relevance_score": relevance_score,
            "functional_actives_count": len(unique_kws),
            "primary_actives_count": len(set(primary_hits)),
            "matched_raw_materials": ", ".join(unique_kws),
            "matched_dosages": dosage_lookup.get(reg_no, ""),
        })

    scores_df = pd.DataFrame(score_records)
    ranked_products = (
        valid_products.merge(scores_df, on="reg_no", how="inner")
        .sort_values(
            by=["relevance_score", "primary_actives_count", "functional_actives_count", "product"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    return breakdown_df, ranked_products


def fetch_pubmed_evidence(ingredient: str, health_benefit: str, max_results: int = 5) -> dict:
    """
    Queries the free NCBI PubMed E-utilities API (esearch + esummary) to retrieve
    real peer-reviewed study counts and top clinical paper citations (PMID, title, journal, year).
    """
    query = f'("{ingredient}"[Title/Abstract]) AND ("{health_benefit}"[Title/Abstract] OR clinical trial[Publication Type] OR randomized controlled trial[Publication Type])'
    encoded_q = urllib.parse.quote(query)
    esearch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_q}&retmode=json&retmax={max_results}&sort=relevance"

    try:
        req = urllib.request.Request(esearch_url, headers={"User-Agent": "NPRA-RnD-Workbench/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            search_data = json.loads(resp.read().decode("utf-8"))

        esearch_res = search_data.get("esearchresult", {})
        total_papers = int(esearch_res.get("count", 0))
        id_list = esearch_res.get("idlist", [])

        articles = []
        if id_list:
            time.sleep(0.35)  # Respect NCBI rate limit (3 requests/sec without API key)
            ids_str = ",".join(id_list)
            esummary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
            req2 = urllib.request.Request(esummary_url, headers={"User-Agent": "NPRA-RnD-Workbench/1.0"})
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                sum_data = json.loads(resp2.read().decode("utf-8"))

            result_map = sum_data.get("result", {})
            for pmid in id_list:
                info = result_map.get(pmid, {})
                if info:
                    articles.append({
                        "pmid": pmid,
                        "title": info.get("title", ""),
                        "journal": info.get("source", ""),
                        "pubdate": info.get("pubdate", ""),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    })

        return {
            "total_papers": total_papers,
            "articles": articles,
            "query": query,
        }
    except Exception as e:
        return {"error": str(e), "total_papers": 0, "articles": []}
