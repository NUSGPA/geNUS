"""
NUSGPA DNA Engine
Computes rich insights from transcript data for the DNA feature.
Supports optional LLM narrative generation using RAG (Retrieval-Augmented Generation)
with NUSMods module data as the knowledge base.
"""

import pandas as pd
import numpy as np
from collections import Counter
import json
import os

# --- GEMINI MODEL ROTATION ---
# When one model hits its daily quota, the next one in the list is tried.
GEMINI_MODEL_ROTATION = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview"
]

def _call_gemini_with_fallback(api_key, prompt, models=None):
    """
    Attempt to generate content using a list of Gemini models in order.
    If a model returns a 429 / quota error, the next model is tried.
    Returns (text, model_used) on success, raises the last exception on total failure.
    """
    import google.generativeai as genai
    if not api_key:
        raise ValueError("Gemini API key is missing. Add GEMINI_API_KEY to .streamlit/secrets.toml.")
    if models is None:
        models = GEMINI_MODEL_ROTATION
    genai.configure(api_key=api_key)
    last_exc = None
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            return resp.text, model_name
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                last_exc = e
                continue   # try next model
            raise  # non-rate-limit errors bubble up immediately
    raise last_exc  # all models exhausted


# --- SUBJECT CATEGORY MAPPING ---
SUBJECT_CATEGORIES = {
    "Computing": ["CS", "IS", "IT", "CG", "CP", "BT", "DSA"],
    "Engineering": ["EE", "ME", "CE", "BME", "ESP", "TE", "IE", "CDE", "ES", "EG"],
    "Business": ["BZ", "ACC", "FIN", "MKT", "MNO", "BSP", "DAO", "DBA", "RE", "BMA"],
    "Mathematics & Physics": ["MA", "ST", "PC", "QF"],
    "Life Sciences": ["LSM", "BL", "CM", "PR", "SP"],
    "Humanities": ["EN", "HY", "PH", "GL", "CH", "JS", "TS", "TH", "EL", "LA"],
    "Social Sciences": ["PS", "SC", "SW", "EC", "GE", "SE"],
    "Design & Built Env": ["AR", "DT", "UD", "NM", "PF"],
    "Law": ["LL", "LC"],
    "Medicine & Health": ["MD", "NUR"],
    "General Education": ["GEA", "GEH", "GES", "GET", "GEQ", "GER", "GESS", "GEX", "GEN", "DTK", "HS"],
    "University": ["UTW", "CFG", "RVX", "IFS"],
}

def _get_category(prefix):
    if not prefix:
        return "Other"
    prefix = prefix.upper()
    for category, prefixes in SUBJECT_CATEGORIES.items():
        if prefix in prefixes:
            return category
    return "Other"


def compute_insights(courses_df, summary_df, grade_map, sem_mapping):
    """Compute all DNA insights from the user's transcript data."""
    ins = {}
    if courses_df.empty:
        return ins

    NON_GPA = ["CS", "CU", "IP", "S", "U"]

    # === BASIC STATS ===
    ins["total_modules"] = len(courses_df)
    ins["total_mcs"] = int(courses_df["Credits"].sum())  # Gross total MCs
    ins["completed_mcs"] = int(courses_df[courses_df["Grade"] != "IP"]["Credits"].sum())
    ins["ip_mcs"] = int(courses_df[courses_df["Grade"] == "IP"]["Credits"].sum())
    ins["graded_mcs"] = int(summary_df["Term Credits"].sum()) if not summary_df.empty else 0
    ins["num_semesters"] = len(summary_df) if not summary_df.empty else 0

    # === SEMESTER HIGHLIGHTS ===
    if not summary_df.empty:
        valid = summary_df[summary_df["Sem GPA"] > 0]
        if not valid.empty:
            best_i = valid["Sem GPA"].idxmax()
            ins["best_semester"] = valid.loc[best_i, "Semester"]
            ins["best_semester_gpa"] = round(float(valid.loc[best_i, "Sem GPA"]), 2)

            if len(valid) > 1:
                worst_i = valid["Sem GPA"].idxmin()
                ins["worst_semester"] = valid.loc[worst_i, "Semester"]
                ins["worst_semester_gpa"] = round(float(valid.loc[worst_i, "Sem GPA"]), 2)

                diffs = valid["Sem GPA"].diff()
                if diffs.max() > 0:
                    imp_i = diffs.idxmax()
                    ins["most_improved_sem"] = summary_df.loc[imp_i, "Semester"]
                    ins["improvement_delta"] = round(float(diffs.max()), 2)

            heaviest_i = summary_df["Mods"].idxmax()
            ins["heaviest_semester"] = summary_df.loc[heaviest_i, "Semester"]
            ins["heaviest_mods"] = int(summary_df.loc[heaviest_i, "Mods"])

    # === GRADE ANALYSIS ===
    graded = courses_df[~courses_df["Grade"].isin(NON_GPA)].copy()
    if not graded.empty:
        gc = graded["Grade"].value_counts()
        ins["most_common_grade"] = gc.index[0]
        ins["most_common_grade_count"] = int(gc.iloc[0])
        ins["grade_distribution"] = gc.to_dict()

        a_grades = graded[graded["Grade"].isin(["A+", "A", "A-"])]
        ins["a_count"] = len(a_grades)
        ins["a_percentage"] = round(len(a_grades) / len(graded) * 100, 1)

        high = graded[graded["Grade"].isin(["A+", "A", "A-", "B+"])]
        ins["high_grade_pct"] = round(len(high) / len(graded) * 100, 1)

        # Grade streak
        gs = graded.copy()
        gs["Sem_Int"] = gs["Semester"].map(sem_mapping).fillna(0).astype(int)
        gs = gs.sort_values("Sem_Int")
        streak, cur = 0, 0
        for gv in gs["Grade"].map(grade_map):
            if gv >= 4.5:
                cur += 1
                streak = max(streak, cur)
            else:
                cur = 0
        ins["best_streak"] = streak

    # === SUBJECT ANALYSIS ===
    cc = courses_df.copy()
    cc["Prefix"] = cc["Course"].astype(str).str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
    pc = cc["Prefix"].value_counts()

    if not pc.empty:
        ins["top_prefix"] = pc.index[0]
        ins["top_prefix_count"] = int(pc.iloc[0])
        ins["unique_prefixes"] = len(pc)

        cc["Category"] = cc["Prefix"].apply(_get_category)
        cat_counts = cc["Category"].value_counts()
        meaningful = cat_counts.drop(labels=["Other", "General Education", "University"], errors="ignore")
        ins["categories"] = cat_counts.to_dict()
        ins["dominant_category"] = meaningful.index[0] if not meaningful.empty else cat_counts.index[0]

        total = len(cc)
        if total > 0 and len(cat_counts) > 1:
            props = cat_counts / total
            entropy = -(props * np.log2(props + 1e-10)).sum()
            max_e = np.log2(len(cat_counts)) if len(cat_counts) > 1 else 1
            ins["diversity_score"] = round(float(entropy / max_e * 100), 0)
        else:
            ins["diversity_score"] = 0

    # === GPA PATTERNS ===
    if not summary_df.empty and len(summary_df) > 1:
        vg = summary_df["Sem GPA"][summary_df["Sem GPA"] > 0]
        if len(vg) > 1:
            std = float(vg.std())
            ins["gpa_std"] = round(std, 2)
            if std < 0.2:
                ins["consistency_label"] = "Rock Steady 🪨"
                ins["consistency_desc"] = "Your GPA barely moves. You know exactly what you're doing."
            elif std < 0.4:
                ins["consistency_label"] = "Smooth Sailing ⛵"
                ins["consistency_desc"] = "Minor fluctuations — the mark of a well-balanced student."
            elif std < 0.6:
                ins["consistency_label"] = "Adventurous 🌊"
                ins["consistency_desc"] = "Some highs, some lows — part of the journey."
            else:
                ins["consistency_label"] = "Roller Coaster 🎢"
                ins["consistency_desc"] = "Big swings! But your best semesters show what you're capable of."

    # === WORKLOAD PATTERN ===
    # Compares total module count in first half vs second half of your academic journey
    if not summary_df.empty and len(summary_df) > 1:
        sem_labels = summary_df["Semester"].tolist()
        mps = summary_df["Mods"].tolist()
        half = len(mps) // 2
        first_h, second_h = sum(mps[:half]), sum(mps[half:])
        first_sems = sem_labels[:half]
        second_sems = sem_labels[half:]
        first_label = f"{first_sems[0]}–{first_sems[-1]}" if len(first_sems) > 1 else first_sems[0]
        second_label = f"{second_sems[0]}–{second_sems[-1]}" if len(second_sems) > 1 else second_sems[0]
        std_m = pd.Series(mps).std()
        if std_m < 0.8:
            ins["workload_pattern"] = "Balanced ⚖️"
            ins["workload_desc"] = f"Even split across {first_label} and {second_label} — consistent planner!"
        elif first_h > second_h * 1.3:
            ins["workload_pattern"] = "Front-loaded 🚀"
            ins["workload_desc"] = f"Heavier in {first_label} ({first_h} mods), lighter in {second_label} ({second_h} mods)."
        elif second_h > first_h * 1.3:
            ins["workload_pattern"] = "Back-loaded 🏋️"
            ins["workload_desc"] = f"Lighter in {first_label} ({first_h} mods), ramped up in {second_label} ({second_h} mods)."
        else:
            ins["workload_pattern"] = "Balanced ⚖️"
            ins["workload_desc"] = f"Well-paced across {first_label} and {second_label}."

    # === GPA TRAJECTORY ===
    if not summary_df.empty and len(summary_df) > 1:
        gpas = summary_df["Sem GPA"][summary_df["Sem GPA"] > 0].tolist()
        if len(gpas) >= 2:
            t = max(1, len(gpas) // 3)
            first_t = np.mean(gpas[:t])
            last_t = np.mean(gpas[-t:])
            if last_t > first_t + 0.3:
                ins["trajectory"] = "📈 Rising Star"
                ins["trajectory_desc"] = "Your grades have been climbing. The best is yet to come!"
            elif last_t > first_t + 0.1:
                ins["trajectory"] = "📈 Gradual Ascent"
                ins["trajectory_desc"] = "Slow and steady improvement — you're finding your groove!"
            elif last_t < first_t - 0.3:
                ins["trajectory"] = "🔄 Time to Recharge"
                ins["trajectory_desc"] = "A dip — but awareness is the first step to a comeback."
            elif last_t < first_t - 0.1:
                ins["trajectory"] = "➡️ Slight Drift"
                ins["trajectory_desc"] = "A small dip — nothing a strong next semester can't fix."
            else:
                ins["trajectory"] = "🎯 Locked In"
                ins["trajectory_desc"] = "Remarkably consistent. You've found your rhythm."

    # === ARCHETYPE (Multi-factor scoring) ===
    scores = {
        "💻 The Technologist": 0, "📐 The Analyst": 0,
        "👔 The Strategist": 0, "🔬 The Researcher": 0,
        "🎨 The Creative": 0, "⚖️ The Advocate": 0,
        "🌍 The Renaissance Mind": 0, "🎓 The Scholar": 0,
        "🏥 The Healer": 0,
    }
    cat_map = {
        "Computing": "💻 The Technologist", "Engineering": "📐 The Analyst",
        "Mathematics & Physics": "📐 The Analyst", "Business": "👔 The Strategist",
        "Life Sciences": "🔬 The Researcher", "Design & Built Env": "🎨 The Creative",
        "Humanities": "🌍 The Renaissance Mind", "Social Sciences": "🌍 The Renaissance Mind",
        "Law": "⚖️ The Advocate", "Medicine & Health": "🏥 The Healer",
    }
    dom = ins.get("dominant_category", "")
    if dom in cat_map:
        scores[cat_map[dom]] += 4
    for cat, cnt in ins.get("categories", {}).items():
        if cat in cat_map:
            scores[cat_map[cat]] += min(cnt, 3)
    if ins.get("diversity_score", 0) > 70:
        scores["🌍 The Renaissance Mind"] += 3
    if ins.get("a_percentage", 0) > 70:
        scores["🎓 The Scholar"] += 3
    elif ins.get("a_percentage", 0) > 50:
        scores["🎓 The Scholar"] += 2
    if "Rock Steady" in ins.get("consistency_label", ""):
        scores["🎓 The Scholar"] += 2

    best_a = max(scores, key=scores.get)
    if scores[best_a] == 0:
        best_a = "🎓 The Scholar"
    ins["archetype"] = best_a

    descs = {
        "💻 The Technologist": "You live and breathe code. Your transcript reads like a changelog — always building, always shipping.",
        "📐 The Analyst": "Numbers don't lie, and neither does your transcript. You see elegant solutions where others see equations.",
        "👔 The Strategist": "Balance sheets, market trends, and big-picture thinking. You're preparing to lead.",
        "🔬 The Researcher": "Curiosity drives everything you do. You ask the questions others haven't thought of yet.",
        "🎨 The Creative": "Where others see constraints, you see a canvas. Your work blends form and function.",
        "⚖️ The Advocate": "Justice isn't just a concept — it's your calling. Every case study brings you closer to real change.",
        "🌍 The Renaissance Mind": "Your transcript is a world map of intellectual curiosity. No discipline is off-limits.",
        "🎓 The Scholar": "Consistent, dedicated, and always refining. Your record is a testament to discipline and love for learning.",
        "🏥 The Healer": "Driven by empathy and precision. Your path leads to making a tangible difference in people's lives.",
    }
    ins["archetype_description"] = descs.get(best_a, "A unique blend of strengths that defies easy categorization.")

    # === FUN FACTS ===
    ff = []
    if ins.get("best_streak", 0) >= 3:
        ff.append(f"🔥 You scored A- or above {ins['best_streak']} times in a row!")
    if ins.get("a_percentage", 0) > 50:
        ff.append(f"⭐ Over half your graded modules are A-range!")
    if ins.get("unique_prefixes", 0) >= 8:
        ff.append(f"🗺️ You've explored {ins['unique_prefixes']} different subject areas!")
    if ins.get("total_mcs", 0) >= 100:
        ff.append(f"💎 {ins['total_mcs']} MCs cleared — past the halfway mark!")
    elif ins.get("total_mcs", 0) >= 40:
        ff.append(f"📚 {ins['total_mcs']} MCs under your belt and counting!")
    if ins.get("improvement_delta", 0) > 0.5:
        ff.append(f"📈 Biggest comeback: +{ins['improvement_delta']} GPA jump in {ins.get('most_improved_sem', 'one semester')}!")
    ins["fun_facts"] = ff

    return ins


# === RAG: RETRIEVAL + AUGMENTATION ===

def enrich_modules_with_titles(courses_df, ay_list):
    """RAG Retrieval step: look up module titles from cached NUSMods JSONs."""
    enriched = courses_df.copy()
    all_mods = {}
    for ay in ay_list:
        fn = f"modules_lite_{ay}.json"
        if os.path.exists(fn):
            try:
                with open(fn, "r") as f:
                    data = json.load(f)
                for m in data:
                    code = m.get("moduleCode", "")
                    if code and code not in all_mods:
                        all_mods[code] = m.get("title", "")
            except Exception:
                pass
    enriched["Title"] = enriched["Course"].map(all_mods).fillna("")
    return enriched


def build_llm_prompt(insights, courses_df, current_gpa, enriched_courses=None):
    """RAG Augmentation step: build a context-rich prompt for the LLM."""
    if enriched_courses is not None and "Title" in enriched_courses.columns:
        lines = []
        for _, r in enriched_courses.iterrows():
            t = f" ({r['Title']})" if r.get("Title", "") else ""
            lines.append(f"- {r['Course']}{t}: {r['Grade']} in {r['Semester']}")
        mod_text = "\n".join(lines)
    else:
        mod_text = "\n".join(
            [f"- {r['Course']}: {r['Grade']} in {r['Semester']}" for _, r in courses_df.iterrows()]
        )

    return f"""You are writing a fun, engaging "Year in Review" for an NUS student — like Spotify Wrapped but for academics.

STUDENT PROFILE:
- Current CAP: {current_gpa:.2f}
- Modules: {insights.get('total_modules', 0)} | MCs: {insights.get('total_mcs', 0)} | Semesters: {insights.get('num_semesters', 0)}
- Best Semester: {insights.get('best_semester', 'N/A')} (GPA: {insights.get('best_semester_gpa', 'N/A')})
- Most Common Grade: {insights.get('most_common_grade', 'N/A')} ({insights.get('most_common_grade_count', 0)}x)
- A-rate: {insights.get('a_percentage', 0)}% | Best Streak: {insights.get('best_streak', 0)}
- Dominant Area: {insights.get('dominant_category', 'N/A')} | Diversity: {insights.get('diversity_score', 0)}/100
- Consistency: {insights.get('consistency_label', 'N/A')} | Trajectory: {insights.get('trajectory', 'N/A')}
- Archetype: {insights.get('archetype', 'N/A')}

MODULES:
{mod_text}

INSTRUCTIONS:
1. Write 3-4 short paragraphs, warm/witty/encouraging tone.
2. Reference SPECIFIC module codes AND titles for personalization.
3. Do NOT mention specific GPA/CAP numbers — keep it qualitative.
4. End with a punchy motivational one-liner.
5. Under 200 words. Use emoji sparingly (2-4 total)."""


def generate_narrative(prompt, api_key, provider="gemini"):
    """RAG Generation step: call LLM API."""
    try:
        if provider == "gemini":
            text, _ = _call_gemini_with_fallback(api_key, prompt)
            return text
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return resp.choices[0].message.content
    except ImportError:
        return "⚠️ Required package not installed. Run `pip install google-generativeai` for Gemini or `pip install openai` for OpenAI."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def generate_custom_archetype(courses_df, api_key):
    """Uses Gemini and RAG from filtered_raw.csv to generate a custom archetype."""
    import google.generativeai as genai
    import pandas as pd
    import re
    
    # 1. Get prefix counts
    prefixes = []
    for _, row in courses_df.iterrows():
        c = row["Course"]
        match = re.match(r"^[A-Za-z]+", str(c))
        if match:
            prefixes.append(match.group(0).upper())
    
    prefix_counts = pd.Series(prefixes).value_counts().to_dict()
    
    # 2. Add Context from CSV
    context = ""
    try:
        raw_df = pd.read_csv("filtered_raw.csv")
        if "Prefix" in raw_df.columns and "Title" in raw_df.columns:
            for p, count in prefix_counts.items():
                match_df = raw_df[raw_df["Prefix"].str.upper() == p]
                if not match_df.empty:
                    titles = match_df.iloc[0]["Title"]
                    context += f"- Prefix {p} (taken {count} times): {titles}\n"
                else:
                    context += f"- Prefix {p} (taken {count} times): (Unknown)\n"
    except Exception:
        for p, count in prefix_counts.items():
            context += f"- Prefix {p} (taken {count} times)\n"

    # 3. Build prompt
    prompt = f"""You are a fun and creative student profiler. A university student has taken modules spanning the following subject prefixes. 
The number of times a prefix was taken indicates the student's focus/weighting.
Here is the context about what these prefixes mean (list of module titles under each prefix):

{context}

Based on this weighted combination of subjects, generate a highly personalized, creative 1-4 word archetype name (must include one relevant emoji) and a 1-2 sentence description of their academic vibe.
Make sure the archetype heavily reflects the prefixes they took most frequently, while the description gives a nod to the prefixes they took less frequently.

Return the exact output in this format:
Archetype Name | Description"""

    try:
        text, _ = _call_gemini_with_fallback(api_key, prompt)
        text = text.strip().replace("**", "").replace("*", "")
        if "|" in text:
            name, desc = text.split("|", 1)
            return name.strip(), desc.strip()
        else:
            # Fallback parsing
            lines = text.split("\n")
            if len(lines) >= 2:
                return lines[0].strip(), lines[1].strip()
            return "🎯 The Enigma", text
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
            # All models exhausted — try a simpler prompt as last resort
            short_prompt = f"Profile a university student taking multiple modules with these prefix codes: {list(prefix_counts.keys())}. Form their academic identity into a 2-word archetype with emoji, pipe symbol '|', and 1 sentence description."
            try:
                text, _ = _call_gemini_with_fallback(api_key, short_prompt)
                text = text.strip().replace("**", "").replace("*", "")
                if "|" in text:
                    name, desc = text.split("|", 1)
                    return name.strip(), desc.strip()
            except Exception:
                pass
            return "⏳ Daily limit reached", "All Gemini models hit their daily quota. Try again tomorrow or add another API key."
        return "⚠️ AI Error", str(e)

