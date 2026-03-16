import requests
import pandas as pd
import os
import json
import time
from datetime import datetime
import streamlit as st

# --- CONFIGURATION ---
START_YEAR = 2021 
CACHE_DURATION = 86400 # 24 Hours in seconds

def get_current_acad_year():
    now = datetime.now()
    year = now.year
    if now.month >= 6: return f"{year}-{year+1}"
    else: return f"{year-1}-{year}"

def get_ay_options():
    current = get_current_acad_year()
    current_start = int(current.split("-")[0])
    years = []
    for y in range(START_YEAR, current_start + 2):
        years.append(f"{y}-{y+1}")
    return sorted(years, reverse=True), current

def ensure_all_years_cached(ay_list):
    missing_or_old = []
    
    # 1. Identify files that need updating
    for ay in ay_list:
        filename = f"modules_lite_{ay}.json"
        
        # Check if file exists
        if not os.path.exists(filename):
            missing_or_old.append(ay)
        else:
            # Check if file is older than 24 hours
            file_age = time.time() - os.path.getmtime(filename)
            if file_age > CACHE_DURATION:
                missing_or_old.append(ay)
            
    # 2. Batch Download (Only if needed)
    if missing_or_old:
        with st.status("Refreshing module database...", expanded=True) as status:
            for i, ay in enumerate(missing_or_old):
                st.write(f"Updating data for AY {ay}...")
                url = f"https://api.nusmods.com/v2/{ay}/moduleInfo.json"
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        full_data = response.json()
                        lite_data = []
                        for mod in full_data:
                            attrs = mod.get("attributes", {})
                            can_su = attrs.get("su", False)
                            lite_data.append({
                                "moduleCode": mod.get("moduleCode"),
                                "title": mod.get("title"),
                                "moduleCredit": float(mod.get("moduleCredit", 0)),
                                "canSU": can_su
                            })
                        
                        with open(f"modules_lite_{ay}.json", "w") as f:
                            json.dump(lite_data, f)
                except Exception as e:
                    st.warning(f"Could not update {ay}. Keeping old data if available.")
            
            status.update(label="Database up to date!", state="complete", expanded=False)

@st.cache_data(show_spinner=False)
def get_modules_for_ay(ay):
    filename = f"modules_lite_{ay}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["display_label"] = df["moduleCode"] + ": " + df["title"]
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()