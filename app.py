import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import re 
import data_manager as dm
import matplotlib.pyplot as plt
import wrapped_engine as we

# --- SECTION 1: APP CONFIGURATION ---
st.set_page_config(page_title="geNUS", layout="wide")

# FIX: Opt-in to future pandas behavior to silence downcasting warnings
pd.set_option('future.no_silent_downcasting', True)

# Constants
grade_map = {
    "A+": 5.0, "A": 5.0, "A-": 4.5, "B+": 4.0, "B": 3.5, "B-": 3.0,
    "C+": 2.5, "C": 2.0, "D+": 1.5, "D": 1.0, "F": 0.0, 
    "CS": 0.0, "CU": 0.0, "IP": 0.0, "S": 0.0, "U": 0.0
}

# --- MAPPINGS ---

# 1. NEW Robust Semester Mapping (4 Terms per Year)
sem_mapping = {}
for year in range(1, 7): # Year 1 to 6
    base = (year - 1) * 4
    sem_mapping[f"Y{year} Sem 1"] = base + 1
    sem_mapping[f"Y{year} Sem 2"] = base + 2
    sem_mapping[f"Y{year} Special 1"] = base + 3
    sem_mapping[f"Y{year} Special 2"] = base + 4

# 2. OLD Semester Mapping (Legacy) used for migration
old_sem_mapping = {
    "Y1 S1": 1, "Y1 S2": 2, "Y2 S1": 3, "Y2 S2": 4,
    "Y3 S1": 5, "Y3 S2": 6, "Y4 S1": 7, "Y4 S2": 8,
    "Y5 S1": 9, "Y5 S2": 10, "Y6 S1": 11, "Y6 S2": 12,
    "Special Term": 99
}
# Map Integers -> New String Format
int_to_new_str_mapping = {
    1: "Y1 Sem 1", 2: "Y1 Sem 2", 3: "Y2 Sem 1", 4: "Y2 Sem 2",
    5: "Y3 Sem 1", 6: "Y3 Sem 2", 7: "Y4 Sem 1", 8: "Y4 Sem 2",
    9: "Y5 Sem 1", 10: "Y5 Sem 2", 11: "Y6 Sem 1", 12: "Y6 Sem 2",
    99: "Y1 Special 1" 
}

# --- HELPER: ROBUST MIGRATION ---
def migrate_old_data(df):
    """
    Bulletproof migration.
    Converts 1, 1.0, "1", "1.0" -> "Y1 Sem 1"
    Leaves "Y1 Sem 1" alone.
    """
    if "Semester" not in df.columns: return df
    
    def convert_val(val):
        if val in sem_mapping: return val # Already correct
        try:
            float_val = float(val)
            int_val = int(float_val)
            if int_val in int_to_new_str_mapping:
                return int_to_new_str_mapping[int_val]
        except (ValueError, TypeError):
            pass
        return val

    df["Semester"] = df["Semester"].apply(convert_val)
    return df

# --- SECTION 2: SESSION STATE ---
if "courses" not in st.session_state:
    st.session_state.courses = pd.DataFrame(columns=["Course", "Semester", "Grade", "Credits", "SU_Opt_Out"])
else:
    # ON RELOAD: Force migration
    st.session_state.courses = migrate_old_data(st.session_state.courses)

if "uploader_id" not in st.session_state: st.session_state.uploader_id = 0

keys_defaults = {
    "course_name_input": "", "credits_input": 4.0, "search_selection": None, 
    "sem_input_label": "Y1 Sem 1", "grade_input": "A"
}
for key, default in keys_defaults.items():
    if key not in st.session_state: st.session_state[key] = default

# --- SECTION 3: CALLBACKS ---
def on_module_select():
    current_ay = st.session_state.get("ay_selector", dm.get_current_acad_year())
    df_lookup = dm.get_modules_for_ay(current_ay)
    
    selection = st.session_state.search_selection
    if selection and not df_lookup.empty:
        matches = df_lookup[df_lookup["display_label"] == selection]
        if not matches.empty:
            row = matches.iloc[0]
            st.session_state.course_name_input = row["moduleCode"]
            st.session_state.credits_input = float(row["moduleCredit"])

def add_course_callback():
    name = st.session_state.course_name_input
    sem_label = st.session_state.sem_input_label 
    grade = st.session_state.grade_input
    credits = st.session_state.credits_input
    
    final_name = name if name else (st.session_state.search_selection if st.session_state.search_selection else "Unknown Course")
    
    new_row = pd.DataFrame([{
        "Course": final_name, "Semester": sem_label, "Grade": grade, 
        "Credits": credits, "SU_Opt_Out": False
    }])
    st.session_state.courses = pd.concat([st.session_state.courses, new_row], ignore_index=True)
    st.session_state.course_name_input = ""
    st.session_state.search_selection = None

def reset_app_callback():
    st.session_state.courses = pd.DataFrame(columns=["Course", "Semester", "Grade", "Credits", "SU_Opt_Out"])
    st.session_state.last_loaded_hash = None
    st.session_state.uploader_id += 1
    st.session_state.course_name_input = ""
    st.session_state.search_selection = None

# --- SECTION 4: SIDEBAR ---
st.sidebar.header("Data & Actions")

unique_key = f"uploader_{st.session_state.uploader_id}"
uploaded_file = st.sidebar.file_uploader("Load CSV", type=["csv"], key=unique_key, label_visibility="collapsed")
if uploaded_file is None: st.sidebar.caption("📂 Load History (CSV)")

if uploaded_file:
    f_hash = hash(uploaded_file.getvalue())
    if "last_loaded_hash" not in st.session_state or st.session_state.last_loaded_hash != f_hash:
        try:
            df_up = pd.read_csv(uploaded_file)
            required_cols = {"Course", "Semester", "Grade", "Credits", "SU_Opt_Out"}
            if required_cols.issubset(df_up.columns):
                if "SU_Opt_Out" in df_up.columns: df_up["SU_Opt_Out"] = df_up["SU_Opt_Out"].astype(bool)
                df_up = migrate_old_data(df_up) # Migrate immediately
                st.session_state.courses = df_up
                st.session_state.last_loaded_hash = f_hash
                st.rerun()
            else: st.error("❌ Invalid CSV columns")
        except Exception as e: st.error(f"Error: {e}")

if not st.session_state.courses.empty:
    st.sidebar.download_button(
        "📥 Download CSV", 
        st.session_state.courses.to_csv(index=False).encode('utf-8'), 
        "mygeNUS.csv", 
        "text/csv", 
        use_container_width=True
    )
else:
    st.sidebar.download_button("📥 Download CSV", "", disabled=True, use_container_width=True)

if st.sidebar.button("⚠️ Reset All", on_click=reset_app_callback, type="primary", use_container_width=True): pass

st.sidebar.markdown("---")

# Pre-fetch year options for the selectbox below
ay_list, default_ay = dm.get_ay_options()

with st.sidebar.expander("Add New Course", expanded=True):
    try: idx = ay_list.index(default_ay)
    except: idx = 0
    sel_ay = st.selectbox("AY Source", ay_list, index=idx, key="ay_selector", label_visibility="collapsed")
    
    modules_df = dm.get_modules_for_ay(sel_ay)
    opts = modules_df["display_label"].tolist() if not modules_df.empty else []

    st.caption(f"Searching **{sel_ay}** database:")
    st.selectbox("Search", opts, index=None, placeholder="Search (e.g. CS1010)...", key="search_selection", on_change=on_module_select, label_visibility="collapsed")
    
    st.caption("Or edit details manually:")
    
    st.text_input("Course Code", key="course_name_input", label_visibility="collapsed", placeholder="Course Code")
    st.selectbox("Semester", list(sem_mapping.keys()), key="sem_input_label", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1: st.number_input("Credits", min_value=0.0, step=1.0, key="credits_input", label_visibility="collapsed")
    with c2: st.selectbox("Grade", list(grade_map.keys()), key="grade_input", label_visibility="collapsed")
    st.button("Add Module", on_click=add_course_callback, use_container_width=True)

# Check/Update the database at the bottom of the sidebar to prevent UI jitter of top elements
with st.sidebar:
    if dm.ensure_all_years_cached(ay_list):
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data provided by [NUSMods](https://nusmods.com).")

# --- SECTION 5: TITLE ---
genus_title = """
<style>
.header-container { display: flex; align-items: baseline; justify-content: flex-start; gap: 15px; margin-bottom: 25px; }
.logo-text { 
    font-family: 'Helvetica Neue', sans-serif; 
    font-size: 4.5rem; 
    font-weight: 900; 
    color: white;
    filter: drop-shadow(0px 2px 2px rgba(0,0,0,0.3)); 
    margin: 0px; padding: 0px; line-height: 1; 
}
.gradient-nus { 
    background: linear-gradient(90deg, #FF7B00, #0073ff); 
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
}
.subtitle-text { color: #A0A0A0; font-size: 1.2rem; font-weight: 500; margin: 0px; padding: 0px; }
</style>
<div class="header-container">
    <h1 class="logo-text">ge<span class="gradient-nus">NUS</span></h1>
    <span class="subtitle-text">Beyond the bell curve.</span>
</div>
"""
st.markdown(genus_title, unsafe_allow_html=True)

# --- SECTION 6: ANALYTICS (TABS) ---
if not st.session_state.courses.empty:
    df = st.session_state.courses.copy()
    NON_GPA = ["CS", "CU", "IP", "S", "U"]
    
    # --- HELPER: Calculate GPA ---
    def calculate_stats(dataframe, honor_su=True):
        temp_df = dataframe.copy()
        
        # Sort logic
        temp_df["Sem_Int"] = temp_df["Semester"].map(sem_mapping).fillna(0).astype(int)
        
        temp_df["Grade Value"] = temp_df["Grade"].map(grade_map)
        
        def get_credits(row):
            is_su = row["SU_Opt_Out"] if honor_su else False
            if is_su or row["Grade"] in NON_GPA: return 0
            return row["Credits"]

        temp_df["Calc_Credits"] = temp_df.apply(get_credits, axis=1)
        temp_df["Q_Points"] = temp_df["Grade Value"] * temp_df["Calc_Credits"]
        
        total_pts = temp_df["Q_Points"].sum()
        total_creds = temp_df["Calc_Credits"].sum()
        gpa = (total_pts / total_creds) if total_creds > 0 else 0.0
        
        summ = temp_df.groupby(["Semester", "Sem_Int"]).apply(lambda x: pd.Series({
            "Term Credits": x["Calc_Credits"].sum(), 
            "Term Points": x["Q_Points"].sum(), 
            "Mods": len(x)
        }), include_groups=False).reset_index()
        
        summ = summ.sort_values("Sem_Int")
        summ["Sem GPA"] = (summ["Term Points"] / summ["Term Credits"]).fillna(0)
        summ["Cumulative GPA"] = (summ["Term Points"].cumsum() / summ["Term Credits"].cumsum()).fillna(0)
        
        return gpa, summ

    # 1. Calculate Statistics for Tabs
    current_gpa, summary_df = calculate_stats(df, honor_su=True)
    baseline_gpa, _ = calculate_stats(df, honor_su=False)
    
    # Honours Class Label
    def get_class(gpa):
        if gpa >= 4.5: return "🥇 First Class"
        elif gpa >= 4.0: return "🥈 Second Upper"
        elif gpa >= 3.5: return "🥉 Second Lower"
        elif gpa >= 3.0: return "🎓 Third Class"
        elif gpa >= 2.0: return "📜 Pass"
        return "⚠️ Below Req"

    # --- TABS UI ---
    st.divider()
    tab1, tab2, tab4 = st.tabs(["Dashboard", "S/U & Target Planner", "DNA"])

    # === TAB 1: DASHBOARD ===
    with tab1:
        # --- Dashboard CSS ---
        st.markdown("""
        <style>
        .dash-gpa-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1.8rem 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            margin-bottom: 0.5rem;
        }
        .dash-gpa-card .gpa-number {
            font-size: 3rem;
            font-weight: 900;
            color: #fff;
            line-height: 1.1;
        }
        .dash-gpa-card .gpa-class {
            font-size: 1rem;
            color: rgba(255,255,255,0.7);
            margin-top: 0.3rem;
        }
        .dash-gpa-card .gpa-label {
            font-size: 0.7rem;
            color: rgba(255,255,255,0.4);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.4rem;
        }
        .dash-stat {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.2rem 1rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            margin-bottom: 0.5rem;
        }
        .dash-stat .ds-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #fff;
        }
        .dash-stat .ds-label {
            font-size: 0.7rem;
            color: rgba(255,255,255,0.5);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 0.2rem;
        }
        .dash-section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            margin-top: 1.2rem;
        }
        .subject-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.2rem;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            text-align: center;
        }
        .sc-medal-0 {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.15) 0%, rgba(255, 255, 255, 0.08) 100%);
            border: 1px solid rgba(255, 215, 0, 0.3);
        }
        .sc-medal-1 {
            background: linear-gradient(135deg, rgba(192, 192, 192, 0.15) 0%, rgba(255, 255, 255, 0.08) 100%);
            border: 1px solid rgba(192, 192, 192, 0.3);
        }
        .sc-medal-2 {
            background: linear-gradient(135deg, rgba(205, 127, 50, 0.15) 0%, rgba(255, 255, 255, 0.08) 100%);
            border: 1px solid rgba(205, 127, 50, 0.3);
        }
        .subject-card .sc-medal { font-size: 1.8rem; }
        .subject-card .sc-name {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
            margin: 0.3rem 0 0.1rem;
        }
        .subject-card .sc-gpa {
            font-size: 1.4rem;
            font-weight: 800;
            color: #60b4ff;
        }
        .subject-card .sc-detail {
            font-size: 0.75rem;
            color: rgba(255,255,255,0.5);
            margin-top: 0.2rem;
        }
        .improvement-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.2rem 1.5rem;
            border: 1px solid rgba(255,255,255,0.15);
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        .improvement-card .imp-name {
            font-size: 1.1rem;
            font-weight: 700;
            color: #ff9f43;
        }
        .improvement-card .imp-detail {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.6);
            margin-top: 0.3rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # --- Transcript + Stats Side by Side ---
        dash_left, dash_right = st.columns([1, 1])
        with dash_left:
            st.caption("**Official Transcript.** Edit details below.")
            main_config = {
                "SU_Opt_Out": None, 
                "Grade": st.column_config.SelectboxColumn("Grade", options=list(grade_map.keys()), required=True),
                "Credits": st.column_config.NumberColumn("Credits", format="%.1f"),
                "Semester": st.column_config.SelectboxColumn("Semester", options=list(sem_mapping.keys()), required=True, width="medium")
            }
            edited_df = st.data_editor(
                st.session_state.courses,
                num_rows="dynamic",
                column_config=main_config,
                use_container_width=True,
                key="main_editor",
                height=380
            )
            if not edited_df.equals(st.session_state.courses):
                edited_df = edited_df.reset_index(drop=True)
                st.session_state.courses = edited_df
                st.rerun()

        with dash_right:
            graded_mc = int(summary_df['Term Credits'].sum())
            total_mc = int(df['Credits'].sum())
            mod_count = len(df)
            st.markdown(f'''
            <div style="display: flex; flex-direction: column; height: 380px;">
                <div class="dash-gpa-card" style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center; margin-bottom: 0.8rem;">
                    <div class="gpa-label">Cumulative GPA</div>
                    <div class="gpa-number">{current_gpa:.2f}</div>
                    <div class="gpa-class">{get_class(current_gpa)}</div>
                </div>
                <div style="display: flex; gap: 0.8rem;">
                    <div class="dash-stat" style="flex: 1; margin-bottom: 0;"><div class="ds-value">{graded_mc}</div><div class="ds-label">Graded MCs</div></div>
                    <div class="dash-stat" style="flex: 1; margin-bottom: 0;"><div class="ds-value">{total_mc}</div><div class="ds-label">Total MCs</div></div>
                    <div class="dash-stat" style="flex: 1; margin-bottom: 0;"><div class="ds-value">{mod_count}</div><div class="ds-label">Modules</div></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- Charts Row ---
        c_tr, c_gr, c_mx = st.columns(3)
        with c_tr:
            st.markdown("#### Performance Trend")
            t_data = summary_df.copy()
            y_domain = [0, 5]
            y_base = 0 
            if not t_data.empty:
                all_values = pd.concat([t_data["Sem GPA"], t_data["Cumulative GPA"]])
                non_zero = all_values[all_values > 0.0]
                if not non_zero.empty:
                    d_min = non_zero.min()
                    d_max = non_zero.max()
                    padding = 0.25
                    lower = max(0.0, d_min - padding)
                    upper = min(5.0, d_max + padding)
                    y_domain = [lower, upper]
                    y_base = lower
            t_data['Plot_Sem_GPA'] = t_data['Sem GPA'].apply(lambda x: max(x, y_base))
            t_data['y_base'] = y_base 
            base = alt.Chart(t_data).encode(x=alt.X('Semester', sort=alt.EncodingSortField(field="Sem_Int", order="ascending"), title=None))
            bar = alt.Chart(t_data).mark_bar(opacity=0.7, color='#60b4ff', cornerRadiusTopLeft=5, cornerRadiusTopRight=5, clip=True).encode(
                x=alt.X('Semester', sort=alt.EncodingSortField(field="Sem_Int", order="ascending"), title=None),
                y=alt.Y('Plot_Sem_GPA', scale=alt.Scale(domain=y_domain, zero=False), title="GPA"), 
                y2=alt.Y2('y_base', title=None), tooltip=['Semester', 'Sem GPA', 'Mods'])
            line = base.mark_line(color="#ff6b6b", point=True).encode(
                y=alt.Y('Cumulative GPA', scale=alt.Scale(domain=y_domain, zero=False)), tooltip=['Semester', 'Cumulative GPA'])
            st.altair_chart((bar+line).properties(height=280), use_container_width=True)

        with c_gr:
            st.markdown("#### Grade Distribution")
            def chart_grade(r):
                if r["SU_Opt_Out"]: return "CS" if grade_map.get(r["Grade"],0)>=2.0 else "CU"
                return r["Grade"]
            c_df = df.copy()
            c_df["C_Grade"] = c_df.apply(chart_grade, axis=1)
            dist = c_df["C_Grade"].value_counts().reset_index()
            dist.columns = ["Grade", "Count"]
            d_chart = alt.Chart(dist).mark_bar(color='#ffd93d', cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X('Grade', sort=list(grade_map.keys()), title=None), y=alt.Y('Count', title='Count', axis=alt.Axis(tickMinStep=1)), tooltip=['Grade', 'Count']
            ).properties(height=280)
            st.altair_chart(d_chart, use_container_width=True)

        with c_mx:
            st.markdown("#### Module Mix")
            m_df = df.copy()
            m_df["Prefix"] = m_df["Course"].astype(str).str.extract(r'^([A-Za-z]+)', expand=False).fillna("Other").str.upper()
            mix_dist = m_df["Prefix"].value_counts().reset_index()
            mix_dist.columns = ["Prefix", "Count"]
            mix_dist["Percentage"] = (mix_dist["Count"] / mix_dist["Count"].sum() * 100).round(1)
            
            mix_chart = alt.Chart(mix_dist).mark_arc(innerRadius=60, stroke="#111").encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Prefix", type="nominal", scale=alt.Scale(scheme='tableau20'), legend=None),
                tooltip=["Prefix", "Count", alt.Tooltip("Percentage", format=".1f", title="Mix %")]
            ).properties(height=280)
            
            st.altair_chart(mix_chart, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- Subject Insights ---
        st.markdown("#### Subject Insights")
        s_df = df.copy()
        s_df["Subject"] = s_df["Course"].astype(str).str.extract(r'^([A-Za-z]+)', expand=False)
        s_df["Subject"] = s_df["Subject"].fillna("Other").str.upper()
        graded_subs = s_df[~s_df["Grade"].isin(NON_GPA + ["S", "U"])].copy()
        
        if not graded_subs.empty:
            graded_subs["GradeVal"] = graded_subs["Grade"].map(grade_map)
            graded_subs["Weighted"] = graded_subs["GradeVal"] * graded_subs["Credits"]
            sub_stats = graded_subs.groupby("Subject").apply(lambda x: pd.Series({
                "Total Points": x["Weighted"].sum(), "Total Credits": x["Credits"].sum(), "Count": len(x), "Modules": ", ".join(x["Course"].tolist()) 
            }), include_groups=False).reset_index()
            sub_stats["Subject GPA"] = sub_stats["Total Points"] / sub_stats["Total Credits"]
            podium = sub_stats.sort_values(by=["Subject GPA", "Count"], ascending=[False, False]).head(3)
            
            if len(podium) > 0:
                medals = ["🥇", "🥈", "🥉"]
                cols = st.columns(len(podium))
                for i, (_, row) in enumerate(podium.iterrows()):
                    with cols[i]:
                        st.markdown(f'''
                        <div class="subject-card sc-medal-{i}">
                            <div class="sc-medal">{medals[i]}</div>
                            <div class="sc-name">{row['Subject']}</div>
                            <div class="sc-gpa">{row['Subject GPA']:.2f}</div>
                            <div class="sc-detail">{int(row['Count'])} modules</div>
                            <div class="sc-detail">{row['Modules']}</div>
                        </div>
                        ''', unsafe_allow_html=True)
        else:
            st.info("Add graded modules to see your strongest subjects.")

    # === TAB 2: S/U & TARGET PLANNER ===
    with tab2:
        if "sandbox_df" not in st.session_state:
            st.session_state.sandbox_df = df.copy() 
            st.session_state.sandbox_df["SU_Opt_Out"] = st.session_state.sandbox_df["SU_Opt_Out"].astype(bool)
        
        current_ay = st.session_state.get("ay_selector", dm.get_current_acad_year())
        lookup_df = dm.get_modules_for_ay(current_ay)
        
        full_sandbox = st.session_state.sandbox_df.copy()
        if "canSU" in lookup_df.columns:
            merged = full_sandbox.merge(lookup_df[["moduleCode", "canSU"]], left_on="Course", right_on="moduleCode", how="left")
            merged["canSU"] = merged["canSU"].fillna(False).infer_objects(copy=False).astype(bool)
        else:
            merged = full_sandbox
            merged["canSU"] = True 

        ELIGIBLE_GRADES = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "D+", "D"]
        mask_grade_ok = merged["Grade"].isin(ELIGIBLE_GRADES)
        mask_mod_ok = (merged["canSU"] == True)
        final_mask = mask_grade_ok & mask_mod_ok
        eligible_view = st.session_state.sandbox_df[final_mask]

        locked_mods = merged.loc[mask_grade_ok & ~mask_mod_ok, "Course"].tolist()
        bad_grade_mods = merged.loc[~mask_grade_ok, "Course"].tolist()
        total_hidden = len(locked_mods) + len(bad_grade_mods)

        # --- Table + Stats Side by Side ---
        su_left, su_right = st.columns([1, 1])
        
        with su_left:
            st.markdown("#### S/U Scenario Sandbox")
            st.caption("Select eligible modules to S/U. Only modules marked 'S/U-able' by NUS with grades A-D are shown.")
            if st.button("🔄 Reset Sandbox to Official Grades"):
                st.session_state.sandbox_df = df.copy()
                st.session_state.sandbox_df["SU_Opt_Out"] = st.session_state.sandbox_df["SU_Opt_Out"].astype(bool)
                st.rerun()

            edited_subset = st.data_editor(
                eligible_view,
                column_config={
                    "SU_Opt_Out": st.column_config.CheckboxColumn("Exercise S/U?", default=False),
                    "Course": st.column_config.TextColumn("Course", disabled=True),
                    "Grade": st.column_config.TextColumn("Grade", disabled=True),
                    "Semester": None, "Credits": None
                },
                disabled=["Course", "Grade", "Semester", "Credits"],
                hide_index=True, use_container_width=True, key="su_sandbox_editor"
            )
            if not edited_subset.equals(eligible_view):
                st.session_state.sandbox_df.update(edited_subset)
                st.rerun()
            if total_hidden > 0:
                with st.expander(f"ℹ️ See {total_hidden} hidden modules"):
                    if locked_mods: st.markdown(f"**{len(locked_mods)} Ineligible for S/U:**\n\n" + ", ".join(locked_mods))
                    if bad_grade_mods: st.markdown(f"**{len(bad_grade_mods)} Inapplicable Grades:**\n\n" + ", ".join(bad_grade_mods))

        with su_right:
            # --- AUTO-SOLVER ---
            st.markdown("#### Auto-Maximize GPA")
            st.caption("The solver will optimize for the **entire Academic Year** of the selected semester.")
            su_sem_col, su_budget_col = st.columns(2)
            with su_sem_col:
                su_target_sem = st.selectbox("Select Current Semester", list(sem_mapping.keys()), index=0)
            with su_budget_col:
                su_budget = st.slider("Max MCs to S/U", 0, 32, 20, 1)

            run_solver = st.button("Solve Best S/U Plan", use_container_width=True)
            
            if run_solver:
                temp_solve_df = st.session_state.sandbox_df.copy()
                eligible_indices = eligible_view.index.tolist()
                candidates = []
                target_year_prefix = su_target_sem.split(" ")[0]
                for idx in eligible_indices:
                    row = temp_solve_df.loc[idx]
                    if not str(row["Semester"]).startswith(target_year_prefix):
                        continue 
                    candidates.append({"id": idx, "grade_val": grade_map.get(row["Grade"], 0), "credits": row["Credits"], "course": row["Course"], "grade_str": row["Grade"]})
                if not candidates:
                    st.warning(f"No eligible S/U candidates found in **Academic Year {target_year_prefix}**.")
                else:
                    candidates.sort(key=lambda x: x["grade_val"])
                    used_mcs = 0
                    sued_list = []
                    for cand in candidates:
                        if used_mcs + cand["credits"] <= su_budget:
                            curr_gpa, _ = calculate_stats(temp_solve_df, honor_su=True)
                            if cand["grade_val"] < curr_gpa:
                                temp_solve_df.at[cand["id"], "SU_Opt_Out"] = True
                                used_mcs += cand["credits"]
                                sued_list.append(f"{cand['course']} ({cand['grade_str']})")
                            else:
                                break
                    if sued_list:
                        st.session_state.sandbox_df = temp_solve_df
                        st.success(f"Optimized! S/U'ed {len(sued_list)} modules in {target_year_prefix}: {', '.join(sued_list)}.")
                    else:
                        st.info(f"Your current plan for {target_year_prefix} is already optimal.")

            st.markdown("<br>", unsafe_allow_html=True)
            
            projected_gpa, _ = calculate_stats(st.session_state.sandbox_df, honor_su=True)
            diff = projected_gpa - current_gpa
            
            if diff > 0:
                impact_color = "rgba(40, 167, 69, 0.15)"
                impact_border = "rgba(40, 167, 69, 0.3)"
                impact_sign = "+"
            elif diff < 0:
                impact_color = "rgba(220, 53, 69, 0.15)"
                impact_border = "rgba(220, 53, 69, 0.3)"
                impact_sign = ""
            else:
                impact_color = "rgba(255, 255, 255, 0.08)"
                impact_border = "rgba(255, 255, 255, 0.15)"
                impact_sign = ""

            st.markdown(f'''
            <div style="display: flex; gap: 0.8rem; margin-bottom: 0.8rem;">
                <div class="dash-gpa-card" style="flex: 1; margin-bottom: 0;">
                    <div class="gpa-label">Official GPA</div>
                    <div class="gpa-number">{current_gpa:.2f}</div>
                </div>
                <div class="dash-gpa-card" style="flex: 1; margin-bottom: 0; background: rgba(220, 180, 140, 0.15); border: 1px solid rgba(220, 180, 140, 0.3);">
                    <div class="gpa-label" style="color: rgba(255, 220, 180, 0.8);">Sandbox GPA</div>
                    <div class="gpa-number" style="color: #ffdeb3;">{projected_gpa:.2f}</div>
                </div>
            </div>
            <div class="dash-gpa-card" style="background: {impact_color}; border: 1px solid {impact_border}; margin-bottom: 0;">
                <div class="gpa-label">S/U Impact</div>
                <div class="gpa-number">{impact_sign}{diff:.2f}</div>
            </div>
            ''', unsafe_allow_html=True)

        # --- Target Planner Section ---
        st.divider()
        st.markdown("#### Future Goal Calculator")
        c_tgt, c_rem = st.columns(2)
        with c_tgt: target_gpa = st.number_input("Goal GPA", min_value=0.0, max_value=5.0, value=4.5, step=0.01)
        with c_rem: remaining_mcs = st.number_input("Remaining MCs", min_value=0, value=40, step=1)

        curr_creds = summary_df["Term Credits"].sum()
        curr_pts = summary_df["Term Points"].sum()
        final_creds = curr_creds + remaining_mcs
        required_pts = target_gpa * final_creds
        needed_pts = required_pts - curr_pts
        
        if remaining_mcs == 0:
            st.warning("No MCs remaining!")
        else:
            req_avg = needed_pts / remaining_mcs
            if req_avg > 5.0: st.error(f"❌ **Impossible.** You need an average of **{req_avg:.2f}** (above 5.0).")
            elif req_avg <= 0.0: st.success(f"✅ **Done.** You are already above your target!")
            else:
                if req_avg > 4.5: grade_equiv = "A/A+"
                elif req_avg > 4.0: grade_equiv = "A-"
                elif req_avg > 3.5: grade_equiv = "B+"
                elif req_avg > 3.0: grade_equiv = "B"
                elif req_avg > 2.5: grade_equiv = "B-"
                elif req_avg > 2.0: grade_equiv = "C+"
                elif req_avg > 1.5: grade_equiv = "C"
                elif req_avg > 1.0: grade_equiv = "D+"
                else: grade_equiv = "D/F"
                st.info(f"To hit **{target_gpa}**, you need an average grade of **{req_avg:.2f}**.")
                st.caption(f"Suggested Target: **Strictly {grade_equiv} or better**")
                st.progress(min(max(req_avg/5.0, 0.0), 1.0))

    # === TAB 4: DNA ===
    with tab4:
        # --- Custom CSS ---
        st.markdown("""
        <style>
        .DNA-hero {
            background: linear-gradient(90deg, rgba(240, 147, 251, 0.07), rgba(245, 87, 108, 0.07), rgba(255, 217, 61, 0.07));
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
            margin-bottom: 1.5rem;
        }
        .DNA-hero h1 {
            font-size: 2.6rem;
            font-weight: 900;
            background: linear-gradient(90deg, #f093fb, #f5576c, #ffd93d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }
        .DNA-hero p {
            color: rgba(255,255,255,0.65);
            font-size: 1rem;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            padding: 1.4rem 1rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .stat-card .stat-value {
            font-size: 2rem;
            font-weight: 800;
            color: #fff;
        }
        .stat-card .stat-label {
            color: rgba(255,255,255,0.55);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-top: 0.3rem;
        }
        .insight-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            padding: 1.4rem;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .insight-card .insight-title {
            color: rgba(255,255,255,0.5);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 0.3rem;
        }
        .insight-card .insight-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #fff;
        }
        .insight-card .insight-desc {
            color: rgba(255,255,255,0.55);
            font-size: 0.85rem;
            margin-top: 0.3rem;
        }
        .archetype-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
            margin: 0;
            height: 100%;
        }
        .archetype-card .arch-label {
            color: rgba(255,255,255,0.45);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
        }
        .archetype-card .arch-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #fff;
            margin-bottom: 0.5rem;
        }
        .archetype-card .arch-desc {
            color: rgba(255,255,255,0.65);
            font-size: 1rem;
            max-width: 550px;
            margin: 0 auto;
            line-height: 1.6;
        }
        .funfact-card {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            padding: 1.4rem 1.2rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
            color: #fff;
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.8rem;
            height: 100%;
        }
        .narrative-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 2rem;
            border: 1px solid rgba(255,255,255,0.08);
            color: rgba(255,255,255,0.85);
            line-height: 1.7;
            font-size: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Compute insights
        ins = we.compute_insights(df, summary_df, grade_map, sem_mapping)

        # --- Hero ---
        st.markdown('''
        <div class="DNA-hero">
            <h1>geNUS DNA</h1>
            <p>Your academic journey, decoded.</p>
        </div>
        ''', unsafe_allow_html=True)

        # --- Stats Grid ---
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">📚 {ins.get("total_modules", 0)}</div><div class="stat-label">Modules</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">✅ {ins.get("completed_mcs", 0)}</div><div class="stat-label">Completed MCs</div></div>', unsafe_allow_html=True)
        with c3:
            ip_mcs = ins.get("ip_mcs", 0)
            ip_display = f"⏳ {ip_mcs}" if ip_mcs > 0 else "—"
            st.markdown(f'<div class="stat-card"><div class="stat-value">{ip_display}</div><div class="stat-label">In-Progress MCs</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card"><div class="stat-value">📅 {ins.get("num_semesters", 0)}</div><div class="stat-label">Semesters</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="stat-card"><div class="stat-value">⭐ {ins.get("a_percentage", 0)}%</div><div class="stat-label">A-Rate</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Highlights ---
        h1, h2, h3 = st.columns(3)
        with h1:
            best_sem = ins.get("best_semester", "—")
            best_gpa = ins.get("best_semester_gpa", "—")
            st.markdown(f'<div class="insight-card"><div class="insight-title">Best Semester</div><div class="insight-value">{best_sem}</div><div class="insight-desc">GPA: {best_gpa}</div></div>', unsafe_allow_html=True)
        with h2:
            traj = ins.get("trajectory", "—")
            traj_d = ins.get("trajectory_desc", "")
            st.markdown(f'<div class="insight-card"><div class="insight-title">Trajectory</div><div class="insight-value">{traj}</div><div class="insight-desc">{traj_d}</div></div>', unsafe_allow_html=True)
        with h3:
            con = ins.get("consistency_label", "—")
            con_d = ins.get("consistency_desc", "")
            st.markdown(f'<div class="insight-card"><div class="insight-title">Consistency</div><div class="insight-value">{con}</div><div class="insight-desc">{con_d}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        h4, h5, h6 = st.columns(3)
        with h4:
            streak = ins.get("best_streak", 0)
            st.markdown(f'<div class="insight-card"><div class="insight-title">🔥 Best A-Streak</div><div class="insight-value">{streak} in a row</div><div class="insight-desc">Consecutive A-/A/A+ grades</div></div>', unsafe_allow_html=True)
        with h5:
            wl = ins.get("workload_pattern", "—")
            wl_d = ins.get("workload_desc", "")
            st.markdown(f'<div class="insight-card"><div class="insight-title">📦 Workload Style</div><div class="insight-value">{wl}</div><div class="insight-desc">{wl_d}</div></div>', unsafe_allow_html=True)
        with h6:
            div = ins.get("diversity_score", 0)
            st.markdown(f'<div class="insight-card"><div class="insight-title">🌈 Subject Diversity</div><div class="insight-value">{div}/100</div><div class="insight-desc">{ins.get("unique_prefixes", 0)} unique subject areas explored</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- Archetype logic ---
        api_key = st.secrets.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
        
        # Invalidate archetype if the courses dataframe changed
        current_df_hash = hash(df.to_string())
        if st.session_state.get("custom_archetype_df_hash") != current_df_hash:
            if "custom_archetype_name" in st.session_state:
                del st.session_state["custom_archetype_name"]
                del st.session_state["custom_archetype_desc"]

        # --- 3-Column Format for Archetype, Fun Facts, and Report Card ---
        st.markdown("<br>", unsafe_allow_html=True)
        wrap_col_L, wrap_col_M, wrap_col_R = st.columns([1, 1, 1.2]) # give report card a bit more space
        
        with wrap_col_L:
            # --- Fun Facts ---
            ff = ins.get("fun_facts", [])
            if ff:
                st.markdown("#### Fun Facts")
                import re
                for fact in ff:
                    # Convert Markdown bold to HTML bold since st.markdown won't parse it inside a div
                    fact_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', fact)
                    st.markdown(f'<div class="funfact-card" style="display:flex; align-items:center; justify-content:center; text-align:center;">{fact_html}</div>', unsafe_allow_html=True)
            else:
                st.info("Add some courses to unlock fun facts!")

        with wrap_col_M:
            st.markdown("#### Your Academic Archetype")
            if "custom_archetype_name" in st.session_state:
                arch = st.session_state["custom_archetype_name"]
                arch_d = st.session_state["custom_archetype_desc"]
                st.markdown(f'''
                <div class="archetype-card" style="display:flex; flex-direction:column; justify-content:center;">
                    <div class="arch-label">Archetype</div>
                    <div class="arch-title">{arch}</div>
                    <div class="arch-desc">{arch_d}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.info("Generate your academic archetype.")
                if st.button("Generate Archetype", use_container_width=True):
                    if api_key == "YOUR_API_KEY_HERE":
                        st.warning("Please replace 'YOUR_API_KEY_HERE' in `.streamlit/secrets.toml` with your real Gemini API key!")
                    else:
                        with st.spinner("Generating Archetype... This usually takes up to 10 seconds."):
                            name, desc = we.generate_custom_archetype(df, api_key)
                            if "⚠️" not in name:
                                st.session_state["custom_archetype_name"] = name
                                st.session_state["custom_archetype_desc"] = desc
                                st.session_state["custom_archetype_df_hash"] = current_df_hash
                                st.rerun()
                            else:
                                st.error(f"{name}: {desc}")

        with wrap_col_R:
            st.markdown("#### Share Your Story")
            st.caption("Customize your academic report card and share it with the world!")
            
            if "custom_archetype_name" not in st.session_state:
                st.info("**Generate your archetype** to unlock your story card!")
            else:
                hide_gpa = st.checkbox("Hide GPA in shareable card", value=False)
                
                arch_title = st.session_state["custom_archetype_name"]
                arch_desc = st.session_state["custom_archetype_desc"]
                
                comp_mcs = ins.get('completed_mcs', 0)
                a_rate = ins.get('a_percentage', 0)
                best_sem = ins.get('best_semester', "N/A")
                a_streak = ins.get('best_streak', 0)
                num_mods = ins.get('total_modules', 0)
                num_sems = ins.get('num_semesters', 0)
                
                m_df = df.copy()
                m_df["Prefix"] = m_df["Course"].astype(str).str.extract(r'^([A-Za-z]+)', expand=False).fillna("Other").str.upper()
                prefix_dict = m_df["Prefix"].value_counts().head(3).to_dict()
                p_items_html = " ".join([f'<span class="chip">{k}</span>' for k in prefix_dict.keys()])
                
                gpa_html = ""
                if not hide_gpa:
                    gpa_html = f"""
                    <div class="main-display">
                        <div class="gpa-val">{current_gpa:.2f}</div>
                        <div class="gpa-lbl">Cumulative GPA</div>
                    </div>
                    """
                else:
                    gpa_html = """<div style="margin: 20px 0;"></div>"""
                
                html_code = f"""
                <!DOCTYPE html>
                <html>
                <head>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
                <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
                <style>
                    body {{
                        font-family: 'Outfit', sans-serif;
                        margin: 0; padding: 0; background: transparent;
                        display: flex; flex-direction: column; align-items: center;
                        -webkit-font-smoothing: antialiased;
                    }}
                    .card-container {{
                        background: #08080c;
                        width: 480px;
                        min-height: 700px;
                        height: auto;
                        border-radius: 0; 
                        padding: 40px 35px;
                        box-sizing: border-box;
                        display: flex;
                        flex-direction: column;
                        position: relative;
                        overflow: hidden;
                        color: white;
                        box-shadow: 0 40px 100px rgba(0,0,0,0.8);
                    }}
                    .mesh {{
                        position: absolute;
                        top: 0; left: 0; width: 100%; height: 100%;
                        z-index: 1;
                        background: 
                            radial-gradient(at 0% 0%, rgba(255, 123, 0, 0.3) 0px, transparent 50%),
                            radial-gradient(at 100% 0%, rgba(0, 115, 255, 0.3) 0px, transparent 50%),
                            radial-gradient(at 50% 100%, rgba(255, 123, 0, 0.15) 0px, transparent 50%);
                        filter: blur(40px);
                        opacity: 0.7;
                    }}
                    .content {{
                        position: relative;
                        z-index: 10;
                        height: 100%;
                        display: flex;
                        flex-direction: column;
                        text-align: center;
                        justify-content: flex-start;
                    }}
                    .top-label {{ font-size: 11px; font-weight: 600; letter-spacing: 4px; opacity: 0.4; text-transform: uppercase; margin-bottom: 5px; }}
                    
                    .brand-h1 {{ font-size: 42px; font-weight: 800; margin: 5px 0; letter-spacing: -1.5px; line-height: 1; }}
                    
                    .main-display {{ margin: 25px 0 15px; }}
                    .gpa-val {{ font-size: 64px; font-weight: 800; line-height: 1; letter-spacing: -2px; color: #fff; text-shadow: 0 10px 30px rgba(0,0,0,0.4); }}
                    .gpa-lbl {{ font-size: 14px; color: rgba(255,255,255,0.5); font-weight: 400; margin-top: 5px; text-transform: uppercase; letter-spacing: 2px; }}

                    .stats-grid {{
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 15px 10px;
                        margin: 10px 0;
                        padding: 20px 0;
                        border-top: 1px solid rgba(255,255,255,0.08);
                        border-bottom: 1px solid rgba(255,255,255,0.08);
                    }}
                    .s-unit {{ text-align: center; }}
                    .s-val {{ font-size: 20px; font-weight: 700; color: #fff; }}
                    .s-lbl {{ font-size: 9px; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; }}

                    .identity-section {{ margin-top: 20px; flex-grow: 1; }}
                    .id-label {{ font-size: 10px; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }}
                    .id-title {{ font-size: 24px; font-weight: 700; color: #fff; line-height: 1.2; word-wrap: break-word; }}
                    .id-desc {{ font-size: 11px; color: rgba(255,255,255,0.6); margin-top: 10px; line-height: 1.5; padding: 0 10px; }}
                    
                    .chips-row {{ margin-top: 15px; display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }}
                    .chip {{ 
                        background: rgba(255,255,255,0.05); 
                        padding: 6px 14px; border-radius: 100px; 
                        font-size: 10px; font-weight: 600; border: 1px solid rgba(255,255,255,0.08);
                        color: #FF7B00;
                    }}

                    #dl-btn {{
                        margin-top: 25px;
                        background: white;
                        color: black;
                        border: none;
                        padding: 14px 40px;
                        border-radius: 100px;
                        font-family: inherit;
                        font-weight: 700;
                        font-size: 14px;
                        cursor: pointer;
                        transition: all 0.3s ease;
                        box-shadow: 0 15px 30px rgba(0,0,0,0.4);
                        width: 480px;
                    }}
                    #dl-btn:hover {{ transform: scale(1.03); box-shadow: 0 20px 40px rgba(0,0,0,0.5); }}
                </style>
                </head>
                <body>
                    <div id="capture" class="card-container">
                        <div class="mesh"></div>
                        <div class="content">
                            <div class="top-label">2026 Academic Report</div>
                            
                            <div class="header">
                                <svg width="200" height="60" viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg">
                                    <defs>
                                        <linearGradient id="nugrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="40%" style="stop-color:#FF7B00;stop-opacity:1" />
                                            <stop offset="100%" style="stop-color:#0073ff;stop-opacity:1" />
                                        </linearGradient>
                                    </defs>
                                    <text x="100" y="45" text-anchor="middle" font-family="'Outfit', sans-serif" font-weight="800" font-size="42" letter-spacing="-1.5">
                                        <tspan fill="white">ge</tspan><tspan fill="url(#nugrad)">NUS</tspan>
                                    </text>
                                </svg>
                                <h1 class="brand-h1" style="font-size:32px; margin-top:-15px; opacity:0.7; font-weight:400; text-align:center;">DNA.</h1>
                            </div>

                            {gpa_html}

                            <div class="stats-grid">
                                <div class="s-unit">
                                    <div class="s-val">{num_mods}</div>
                                    <div class="s-lbl">Modules</div>
                                </div>
                                <div class="s-unit">
                                    <div class="s-val">{num_sems}</div>
                                    <div class="s-lbl">Semesters</div>
                                </div>
                                <div class="s-unit">
                                    <div class="s-val">{a_rate}%</div>
                                    <div class="s-lbl">A-Grade Rate</div>
                                </div>
                                <div class="s-unit">
                                    <div class="s-val">{comp_mcs}</div>
                                    <div class="s-lbl">Total MCs</div>
                                </div>
                                <div class="s-unit">
                                    <div class="s-val">{a_streak} M</div>
                                    <div class="s-lbl">Longest A Streak</div>
                                </div>
                                <div class="s-unit">
                                    <div class="s-val">{best_sem}</div>
                                    <div class="s-lbl">Best Semester</div>
                                </div>
                            </div>

                            <div class="identity-section">
                                <div class="id-label">Archetype</div>
                                <div class="id-title">{arch_title}</div>
                                <div class="id-desc">{arch_desc}</div>
                                
                                <div class="chips-row">
                                    {p_items_html}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <button id="dl-btn" onclick="takeShot()">SHARE YOUR STORY</button>
                    <div id="output" style="display:none; margin-top: 20px; width: 480px;"></div>
                    
                    <script>
                        function takeShot() {{
                            const card = document.getElementById('capture');
                            const btn = document.getElementById('dl-btn');
                            const out = document.getElementById('output');
                            btn.innerText = "CAPTURING...";
                            
                            html2canvas(card, {{
                                scale: 3, 
                                backgroundColor: "#08080c",
                                useCORS: true,
                                logging: false
                            }}).then(canvas => {{
                                let url = canvas.toDataURL("image/png");
                                let a = document.createElement("a");
                                a.href = url;
                                a.download = "mygeNUS.png";
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                
                                btn.innerText = "DOWNLOADED";
                                out.style.display = "block";
                                out.innerHTML = "<p style='color: white; font-size: 13px; margin-bottom:15px; opacity: 0.6; font-weight:700;'>YOUR DNA IS READY!</p><img src='" + url + "' style='width: 100%; border-radius: 0;' />";
                            }}).catch(err => {{
                                btn.innerText = "ERROR";
                                console.error(err);
                            }});
                        }}
                    </script>
                </body>
                </html>
                """
                import streamlit.components.v1 as components
                components.html(html_code, height=950)

else:
    st.info("**Welcome!** Start by adding your modules using the sidebar on the left.")
