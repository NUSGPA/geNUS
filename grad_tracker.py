"""
Graduation Tracker Engine for geNUS.

Data Model:
- A "plan" has a name and a list of "buckets" (categories like Core, UE, GE).
- Each bucket has a name and a list of "requirements" (individual graduation requirements).
- Each requirement has a TYPE that determines how it's fulfilled:
    1. "specific"    — must complete a specific course (e.g., "CS2040S")
    2. "either"      — must complete ONE of several courses (e.g., "CS1010 or CS1101S")
    3. "list"        — must complete N courses from a pool (e.g., "pick 3 from [CS3230, CS3233, ...]")
    4. "description" — free-text requirement, manually marked as done (e.g., "Any 4k CS elective")
"""
import json
import copy


# ---------------------------------------------------------------------------
# DATA MODEL
# ---------------------------------------------------------------------------

def create_plan(name="My Graduation Plan"):
    """Create a blank graduation plan."""
    return {"plan_name": name, "buckets": []}


def create_bucket(name):
    """Create an empty bucket (category)."""
    return {"name": name, "requirements": []}


def create_requirement(req_type="specific", course="", courses=None,
                       pick=1, mcs=4, description="", done=False):
    """
    Create a single requirement.
    
    req_type: "specific" | "either" | "list" | "description"
    course:   course code (for "specific")
    courses:  list of course codes (for "either" / "list")
    pick:     how many to pick (for "list")
    mcs:      modular credits this requirement is worth
    description: free-text label (for "description", also used as display label)
    done:     manually marked as done (for "description" type)
    """
    req = {
        "type": req_type,
        "mcs": mcs,
        "description": description,
    }
    if req_type == "specific":
        req["course"] = course.upper().strip()
    elif req_type == "either":
        req["courses"] = [c.upper().strip() for c in (courses or [])]
    elif req_type == "list":
        req["courses"] = [c.upper().strip() for c in (courses or [])]
        req["pick"] = max(1, pick)
    elif req_type == "description":
        req["done"] = done
    return req


# ---------------------------------------------------------------------------
# MATCHING ENGINE
# ---------------------------------------------------------------------------

def evaluate_progress(plan, transcript_df):
    """
    Match transcript courses against the plan's requirements.
    
    Returns a list of bucket results. Each bucket result contains:
    - name: bucket name
    - requirements: list of requirement results, each with:
        - ...original requirement fields...
        - status: "done" | "partial" | "pending"
        - matched: list of matched course codes
        - fulfilled_mcs: MCs fulfilled for this requirement
    - total_mcs: sum of all requirement MCs in the bucket
    - fulfilled_mcs: sum of fulfilled MCs
    - percentage: completion percentage
    
    A course can only be claimed once across all requirements (first match wins).
    """
    # Build available courses pool
    available = {}  # code -> {credits, grade, claimed}
    if transcript_df is not None and not transcript_df.empty:
        for _, row in transcript_df.iterrows():
            code = str(row.get("Course", "")).strip().upper()
            if code and code not in available:
                available[code] = {
                    "credits": float(row.get("Credits", 0)),
                    "grade": str(row.get("Grade", "")),
                    "claimed": False
                }
    
    bucket_results = []
    
    for bucket in plan.get("buckets", []):
        req_results = []
        bucket_fulfilled_mcs = 0
        bucket_in_progress_mcs = 0
        bucket_total_mcs = 0
        
        for req in bucket.get("requirements", []):
            r = copy.deepcopy(req)
            rtype = r.get("type", "specific")
            mcs = r.get("mcs", 0)
            bucket_total_mcs += mcs
            
            if rtype == "specific":
                course = r.get("course", "").upper()
                if course in available and not available[course]["claimed"]:
                    available[course]["claimed"] = True
                    grade = available[course]["grade"]
                    if grade == "IP":
                        r["status"] = "in_progress"
                        r["fulfilled_mcs"] = 0
                        bucket_in_progress_mcs += mcs
                    else:
                        r["status"] = "done"
                        r["fulfilled_mcs"] = mcs
                        bucket_fulfilled_mcs += mcs
                    r["matched"] = [course]
                    r["matched_detail"] = [(course, grade)]
                else:
                    r["status"] = "pending"
                    r["matched"] = []
                    r["matched_detail"] = []
                    r["fulfilled_mcs"] = 0
            
            elif rtype == "either":
                courses = r.get("courses", [])
                matched = None
                for c in courses:
                    if c in available and not available[c]["claimed"]:
                        matched = c
                        break
                if matched:
                    available[matched]["claimed"] = True
                    grade = available[matched]["grade"]
                    if grade == "IP":
                        r["status"] = "in_progress"
                        r["fulfilled_mcs"] = 0
                        bucket_in_progress_mcs += mcs
                    else:
                        r["status"] = "done"
                        r["fulfilled_mcs"] = mcs
                        bucket_fulfilled_mcs += mcs
                    r["matched"] = [matched]
                    r["matched_detail"] = [(matched, grade)]
                else:
                    r["status"] = "pending"
                    r["matched"] = []
                    r["matched_detail"] = []
                    r["fulfilled_mcs"] = 0
            
            elif rtype == "list":
                courses = r.get("courses", [])
                pick = r.get("pick", 1)
                matched = []
                matched_detail = []
                for c in courses:
                    if len(matched) >= pick:
                        break
                    if c in available and not available[c]["claimed"]:
                        available[c]["claimed"] = True
                        matched.append(c)
                        matched_detail.append((c, available[c]["grade"]))
                
                r["matched"] = matched
                r["matched_detail"] = matched_detail
                if len(matched) >= pick:
                    r["status"] = "done"
                    r["fulfilled_mcs"] = mcs
                    bucket_fulfilled_mcs += mcs
                elif len(matched) > 0:
                    r["status"] = "partial"
                    r["fulfilled_mcs"] = int(mcs * len(matched) / pick)
                    bucket_fulfilled_mcs += r["fulfilled_mcs"]
                else:
                    r["status"] = "pending"
                    r["fulfilled_mcs"] = 0
            
            elif rtype == "description":
                # Check for linked course matching
                linked = r.get("linked_course", "").upper().strip()
                if linked and linked in available and not available[linked]["claimed"]:
                    available[linked]["claimed"] = True
                    grade = available[linked]["grade"]
                    if grade == "IP":
                        r["status"] = "in_progress"
                        r["fulfilled_mcs"] = 0
                        bucket_in_progress_mcs += mcs
                    else:
                        r["status"] = "done"
                        r["fulfilled_mcs"] = mcs
                        bucket_fulfilled_mcs += mcs
                    r["matched"] = [linked]
                    r["matched_detail"] = [(linked, grade)]
                elif r.get("done", False):
                    r["status"] = "done"
                    r["fulfilled_mcs"] = mcs
                    bucket_fulfilled_mcs += mcs
                    r["matched"] = []
                    r["matched_detail"] = []
                else:
                    r["status"] = "pending"
                    r["matched"] = []
                    r["matched_detail"] = []
                    r["fulfilled_mcs"] = 0
            
            req_results.append(r)
        
        pct = (bucket_fulfilled_mcs / bucket_total_mcs * 100) if bucket_total_mcs > 0 else 0
        ip_pct = (bucket_in_progress_mcs / bucket_total_mcs * 100) if bucket_total_mcs > 0 else 0
        bucket_results.append({
            "name": bucket.get("name", "Unnamed"),
            "requirements": req_results,
            "total_mcs": bucket_total_mcs,
            "fulfilled_mcs": bucket_fulfilled_mcs,
            "in_progress_mcs": bucket_in_progress_mcs,
            "percentage": min(pct, 100),
            "ip_percentage": min(ip_pct, 100 - pct)
        })
    
    return bucket_results


def get_overall_progress(plan, transcript_df):
    """Returns (fulfilled_mcs, in_progress_mcs, total_mcs, percentage, ip_percentage)."""
    results = evaluate_progress(plan, transcript_df)
    total = sum(b["total_mcs"] for b in results)
    fulfilled = sum(b["fulfilled_mcs"] for b in results)
    in_progress = sum(b["in_progress_mcs"] for b in results)
    pct = (fulfilled / total * 100) if total > 0 else 0
    ip_pct = (in_progress / total * 100) if total > 0 else 0
    return fulfilled, in_progress, total, min(pct, 100), min(ip_pct, 100 - pct)


# ---------------------------------------------------------------------------
# SERIALIZATION
# ---------------------------------------------------------------------------

def plan_to_json(plan):
    """Serialize plan to JSON string."""
    return json.dumps(plan, indent=2)


def plan_from_json(json_str):
    """Deserialize plan from JSON string."""
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# BUILT-IN TEMPLATES
# ---------------------------------------------------------------------------

CS_TEMPLATE = {
    "plan_name": "Computer Science (AY2024)",
    "buckets": [
        {
            "name": "CS Foundation",
            "requirements": [
                {"type": "specific", "course": "CS1101S", "mcs": 4, "description": "Programming Methodology"},
                {"type": "specific", "course": "CS1231S", "mcs": 4, "description": "Discrete Structures"},
                {"type": "specific", "course": "CS2030S", "mcs": 4, "description": "Programming Methodology II"},
                {"type": "specific", "course": "CS2040S", "mcs": 4, "description": "Data Structures & Algorithms"},
                {"type": "specific", "course": "CS2100", "mcs": 4, "description": "Computer Organisation"},
                {"type": "specific", "course": "CS2101", "mcs": 4, "description": "Effective Communication"},
                {"type": "specific", "course": "CS2103T", "mcs": 4, "description": "Software Engineering"},
                {"type": "specific", "course": "CS2106", "mcs": 4, "description": "Intro to Operating Systems"},
                {"type": "specific", "course": "CS2109S", "mcs": 4, "description": "Intro to AI & ML"},
            ]
        },
        {
            "name": "Math & Sciences",
            "requirements": [
                {"type": "specific", "course": "MA1521", "mcs": 4, "description": "Calculus for Computing"},
                {"type": "specific", "course": "MA1522", "mcs": 4, "description": "Linear Algebra for Computing"},
                {"type": "specific", "course": "ST2334", "mcs": 4, "description": "Probability & Statistics"},
            ]
        },
        {
            "name": "Industry Experience",
            "requirements": [
                {"type": "either", "courses": ["CP3880", "CP3200", "CP3202", "IS4010", "CP4101"],
                 "mcs": 12, "description": "Internship / ATAP / Dissertation"},
            ]
        },
        {
            "name": "CS Breadth & Depth",
            "requirements": [
                {"type": "description", "mcs": 4, "description": "Focus Area Primary 1", "done": False},
                {"type": "description", "mcs": 4, "description": "Focus Area Primary 2", "done": False},
                {"type": "description", "mcs": 4, "description": "Focus Area Primary 3 (4k+)", "done": False},
                {"type": "description", "mcs": 4, "description": "CS Elective 1 (4k+)", "done": False},
                {"type": "description", "mcs": 4, "description": "CS Elective 2 (4k+)", "done": False},
                {"type": "description", "mcs": 4, "description": "CS Elective 3", "done": False},
            ]
        },
        {
            "name": "General Education",
            "requirements": [
                {"type": "either", "courses": ["GEA1000", "GEA1000N"], "mcs": 4, "description": "Data Literacy"},
                {"type": "specific", "course": "GEC1000", "mcs": 4, "description": "Cultures & Connections"},
                {"type": "specific", "course": "GESS1000", "mcs": 4, "description": "Singapore Studies"},
                {"type": "either", "courses": ["GEN1000", "GEN2000"], "mcs": 4, "description": "Communities & Engagement"},
                {"type": "specific", "course": "ES2660", "mcs": 4, "description": "Critique & Expression"},
                {"type": "either", "courses": ["GEH1000", "GEH1001", "GEH1002", "GEH1003", "GEH1004"],
                 "mcs": 4, "description": "Humanities"},
                {"type": "description", "mcs": 4, "description": "Interdisciplinary 1", "done": False},
                {"type": "description", "mcs": 4, "description": "Interdisciplinary 2", "done": False},
                {"type": "description", "mcs": 4, "description": "AI & Design Thinking", "done": False},
                {"type": "description", "mcs": 4, "description": "Digital Literacy (CS1101S counts)", "done": False},
            ]
        },
        {
            "name": "Unrestricted Electives",
            "requirements": [
                {"type": "description", "mcs": 4, "description": "UE Slot 1", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 2", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 3", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 4", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 5", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 6", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 7", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 8", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 9", "done": False},
                {"type": "description", "mcs": 4, "description": "UE Slot 10", "done": False},
            ]
        }
    ]
}

BUILTIN_TEMPLATES = {
    "Computer Science (AY2024)": CS_TEMPLATE,
}


def get_builtin_template_names():
    return list(BUILTIN_TEMPLATES.keys())


def get_template(name):
    return copy.deepcopy(BUILTIN_TEMPLATES.get(name))
