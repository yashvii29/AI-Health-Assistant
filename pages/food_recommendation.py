
import random
from itertools import combinations

import pandas as pd
import streamlit as st

# ============================================================
# OPTIONAL GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Food Recommendation System",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 AI Food Recommendation System")
st.write(
    "Personalized meal planning using Constraint Satisfaction "
    "Problems (CSP), content-based ranking, nutritional scoring, "
    "and optional AI explanations."
)

st.info(
    "This is an educational meal-planning tool. Calorie estimates "
    "and food nutrition values are approximate catalogue values. "
    "The output is not a prescribed diet or medical nutrition plan."
)

# ============================================================
# FOOD DATABASE
# Catalogue values are illustrative estimates per listed serving.
# Allergens must always be verified against actual product labels.
# ============================================================

FOOD_DB = [
    # BREAKFAST
    {
        "meal": "Breakfast",
        "name": "Oatmeal with Fruits",
        "diet": "Vegetarian",
        "calories": 350,
        "protein": 12,
        "carbs": 55,
        "fat": 8,
        "score": 92,
        "allergens": [],
    },
    {
        "meal": "Breakfast",
        "name": "Vegetable Sandwich",
        "diet": "Vegetarian",
        "calories": 300,
        "protein": 10,
        "carbs": 40,
        "fat": 9,
        "score": 88,
        "allergens": [],
    },
    {
        "meal": "Breakfast",
        "name": "Boiled Eggs and Toast",
        "diet": "Non-Vegetarian",
        "calories": 380,
        "protein": 22,
        "carbs": 28,
        "fat": 15,
        "score": 90,
        "allergens": ["Egg"],
    },
    {
        "meal": "Breakfast",
        "name": "Tofu Scramble",
        "diet": "Vegan",
        "calories": 320,
        "protein": 18,
        "carbs": 22,
        "fat": 12,
        "score": 91,
        "allergens": ["Soy"],
    },

    # LUNCH
    {
        "meal": "Lunch",
        "name": "Brown Rice and Dal",
        "diet": "Vegetarian",
        "calories": 550,
        "protein": 24,
        "carbs": 75,
        "fat": 10,
        "score": 94,
        "allergens": [],
    },
    {
        "meal": "Lunch",
        "name": "Paneer Roti Combo",
        "diet": "Vegetarian",
        "calories": 600,
        "protein": 28,
        "carbs": 55,
        "fat": 20,
        "score": 89,
        "allergens": ["Milk"],
    },
    {
        "meal": "Lunch",
        "name": "Grilled Chicken Rice Bowl",
        "diet": "Non-Vegetarian",
        "calories": 620,
        "protein": 42,
        "carbs": 60,
        "fat": 14,
        "score": 96,
        "allergens": [],
    },
    {
        "meal": "Lunch",
        "name": "Vegan Buddha Bowl",
        "diet": "Vegan",
        "calories": 540,
        "protein": 20,
        "carbs": 68,
        "fat": 16,
        "score": 95,
        "allergens": ["Soy"],
    },

    # DINNER
    {
        "meal": "Dinner",
        "name": "Vegetable Khichdi",
        "diet": "Vegetarian",
        "calories": 450,
        "protein": 16,
        "carbs": 65,
        "fat": 8,
        "score": 92,
        "allergens": [],
    },
    {
        "meal": "Dinner",
        "name": "Paneer Salad",
        "diet": "Vegetarian",
        "calories": 420,
        "protein": 25,
        "carbs": 18,
        "fat": 20,
        "score": 88,
        "allergens": ["Milk"],
    },
    {
        "meal": "Dinner",
        "name": "Fish and Vegetables",
        "diet": "Non-Vegetarian",
        "calories": 500,
        "protein": 38,
        "carbs": 22,
        "fat": 18,
        "score": 95,
        "allergens": ["Fish"],
    },
    {
        "meal": "Dinner",
        "name": "Vegan Lentil Soup",
        "diet": "Vegan",
        "calories": 430,
        "protein": 22,
        "carbs": 48,
        "fat": 9,
        "score": 93,
        "allergens": [],
    },

    # SNACKS
    {
        "meal": "Snack",
        "name": "Mixed Nuts",
        "diet": "Vegetarian",
        "calories": 200,
        "protein": 8,
        "carbs": 10,
        "fat": 15,
        "score": 90,
        "allergens": ["Nuts"],
    },
    {
        "meal": "Snack",
        "name": "Greek Yogurt",
        "diet": "Vegetarian",
        "calories": 180,
        "protein": 15,
        "carbs": 12,
        "fat": 6,
        "score": 92,
        "allergens": ["Milk"],
    },
    {
        "meal": "Snack",
        "name": "Protein Shake",
        "diet": "Non-Vegetarian",
        "calories": 220,
        "protein": 28,
        "carbs": 8,
        "fat": 4,
        "score": 94,
        "allergens": ["Milk"],
    },
    {
        "meal": "Snack",
        "name": "Roasted Chickpeas",
        "diet": "Vegan",
        "calories": 190,
        "protein": 11,
        "carbs": 25,
        "fat": 4,
        "score": 91,
        "allergens": [],
    },
]

FOOD_DF = pd.DataFrame(FOOD_DB)

MEAL_SLOTS = ["Breakfast", "Lunch", "Dinner", "Snack"]

# ============================================================
# DIET AND ALLERGEN CONSTRAINTS
# ============================================================

def diet_compatible(food, diet_type):
    """
    Non-Vegetarian: permits all catalogue diet categories.
    Vegetarian: permits vegetarian and vegan meals.
    Vegan: permits vegan meals only.
    """
    if diet_type == "Vegan":
        return food["diet"] == "Vegan"

    if diet_type == "Vegetarian":
        return food["diet"] in ("Vegetarian", "Vegan")

    return True


def csp_filter_foods(foods, diet_type, excluded_allergens):
    """
    Apply hard constraints before search.

    A food is removed if it violates the diet preference or contains
    any allergen selected by the user.
    """
    excluded = set(excluded_allergens)
    filtered = []

    for food in foods:
        if not diet_compatible(food, diet_type):
            continue

        if excluded.intersection(food["allergens"]):
            continue

        filtered.append(food.copy())

    return filtered


def build_domains(foods, diet_type, excluded_allergens):
    """
    Build a domain for each CSP variable (meal slot).
    Every candidate in a domain satisfies the dietary and allergy
    constraints before any search algorithm begins.
    """
    eligible_foods = csp_filter_foods(
        foods, diet_type, excluded_allergens
    )

    domains = {
        slot: [
            food for food in eligible_foods
            if food["meal"] == slot
        ]
        for slot in MEAL_SLOTS
    }

    return domains


# ============================================================
# TARGET CALORIE CONSTRAINT
# ============================================================

def calorie_bounds(target_calories, tolerance_percent):
    tolerance = target_calories * tolerance_percent / 100
    lower = max(0, target_calories - tolerance)
    upper = target_calories + tolerance
    return lower, upper


def selected_calories(assignment):
    return sum(food["calories"] for food in assignment.values())


def partial_calorie_feasible(
    assignment, domains, remaining_slots, lower, upper
):
    """
    Forward-checking bound.

    The partial assignment is potentially feasible only if its current
    calorie total plus the minimum possible remaining total does not
    exceed the upper bound, and its current total plus the maximum
    possible remaining total can still reach the lower bound.
    """
    current = selected_calories(assignment)

    min_remaining = 0
    max_remaining = 0

    for slot in remaining_slots:
        if not domains[slot]:
            return False

        min_remaining += min(
            food["calories"] for food in domains[slot]
        )
        max_remaining += max(
            food["calories"] for food in domains[slot]
        )

    return (
        current + min_remaining <= upper
        and current + max_remaining >= lower
    )


def complete_assignment_feasible(assignment, lower, upper):
    if len(assignment) != len(MEAL_SLOTS):
        return False

    calories = selected_calories(assignment)
    return lower <= calories <= upper


# ============================================================
# CONTENT-BASED RECOMMENDATION SCORING
# This ranks catalogue entries; it is not a clinical health score.
# ============================================================

def recommendation_score(food, goal):
    score = float(food["score"])

    if goal == "Weight Loss":
        score += food["protein"] * 0.8
        score -= food["calories"] * 0.015
        score -= food["fat"] * 0.2

    elif goal == "Maintenance":
        score += food["protein"] * 0.5

    elif goal == "Weight Gain":
        score += food["calories"] * 0.025
        score += food["protein"] * 0.5

    return score


def plan_preference_score(assignment, goal):
    return sum(
        recommendation_score(food, goal)
        for food in assignment.values()
    )


# ============================================================
# CSP OBJECTIVE AND HARD CONSTRAINTS
# ============================================================

def assignment_conflicts(assignment):
    """
    Return variables involved in duplicate food selections.
    In this catalogue each food belongs to one meal slot, but the
    constraint is kept explicit for extensibility.
    """
    conflicts = set()
    items = list(assignment.items())

    for (slot_a, food_a), (slot_b, food_b) in combinations(items, 2):
        if food_a["name"] == food_b["name"]:
            conflicts.add(slot_a)
            conflicts.add(slot_b)

    return conflicts


def assignment_is_valid(assignment, lower, upper):
    if set(assignment.keys()) != set(MEAL_SLOTS):
        return False

    if not complete_assignment_feasible(assignment, lower, upper):
        return False

    if assignment_conflicts(assignment):
        return False

    return True


# ============================================================
# ALGORITHM 1: BACKTRACKING SEARCH WITH FORWARD CHECKING
#
# Assign one meal slot at a time.
# Try candidates in preference-score order.
# Reject conflicts immediately.
# Prune partial assignments that cannot reach the calorie interval.
# ============================================================

def backtracking_search(domains, lower, upper, goal):
    assignment = {}
    nodes_visited = 0
    backtracks = 0
    trace = []

    # Minimum Remaining Values (MRV) variable ordering:
    # choose the unassigned slot with the smallest domain.
    # Ties use the original meal-slot order for reproducibility.
    slot_order_index = {
        slot: index for index, slot in enumerate(MEAL_SLOTS)
    }

    def search():
        nonlocal nodes_visited, backtracks

        if len(assignment) == len(MEAL_SLOTS):
            if assignment_is_valid(assignment, lower, upper):
                return dict(assignment)
            return None

        unassigned = [
            slot for slot in MEAL_SLOTS
            if slot not in assignment
        ]

        slot = min(
            unassigned,
            key=lambda s: (
                len(domains[s]),
                slot_order_index[s],
            ),
        )

        ordered_candidates = sorted(
            domains[slot],
            key=lambda food: recommendation_score(food, goal),
            reverse=True,
        )

        for food in ordered_candidates:
            nodes_visited += 1
            assignment[slot] = food

            duplicate_conflicts = assignment_conflicts(assignment)

            if duplicate_conflicts:
                trace.append(
                    f"Rejected {food['name']} for {slot}: "
                    "duplicate-food constraint violated."
                )
                del assignment[slot]
                backtracks += 1
                continue

            remaining = [
                s for s in MEAL_SLOTS
                if s not in assignment
            ]

            if not partial_calorie_feasible(
                assignment, domains, remaining, lower, upper
            ):
                trace.append(
                    f"Pruned {slot} = {food['name']}: "
                    "remaining domains cannot satisfy the calorie bounds."
                )
                del assignment[slot]
                backtracks += 1
                continue

            result = search()

            if result is not None:
                return result

            del assignment[slot]
            backtracks += 1

        return None

    solution = search()

    return {
        "solution": solution,
        "nodes_visited": nodes_visited,
        "backtracks": backtracks,
        "trace": trace,
        "feasible": solution is not None,
    }


# ============================================================
# ALGORITHM 2: INTELLIGENT BACKTRACKING
# Conflict-Directed Backjumping (CBJ)
#
# Instead of always returning to the immediately preceding
# variable, CBJ records which earlier variables caused a failure.
# It can jump directly to the deepest relevant conflicting variable.
#
# This implementation uses:
# - conflict sets
# - recursive backjumping
# - forward-checking bounds for the global calorie constraint
# ============================================================

def intelligent_backtracking_search(
    domains, lower, upper, goal
):
    assignment = {}
    nodes_visited = 0
    backjumps = 0
    trace = []

    slot_order_index = {
        slot: index for index, slot in enumerate(MEAL_SLOTS)
    }

    # Use deterministic variable order so conflict-set behavior
    # can be inspected and reproduced.
    ordered_slots = list(MEAL_SLOTS)

    def search(index):
        nonlocal nodes_visited, backjumps

        if index == len(ordered_slots):
            if assignment_is_valid(assignment, lower, upper):
                return dict(assignment), set()

            return None, set(assignment.keys())

        slot = ordered_slots[index]
        conflict_set = set()

        candidates = sorted(
            domains[slot],
            key=lambda food: recommendation_score(food, goal),
            reverse=True,
        )

        if not candidates:
            trace.append(f"Dead end: domain for {slot} is empty.")
            return None, set(assignment.keys())

        for food in candidates:
            nodes_visited += 1
            assignment[slot] = food

            # Binary constraints: duplicate selection.
            duplicate_conflicts = assignment_conflicts(assignment)

            if duplicate_conflicts:
                prior_conflicts = duplicate_conflicts - {slot}
                conflict_set.update(prior_conflicts)

                trace.append(
                    f"Conflict at {slot} = {food['name']}; "
                    f"conflicting earlier slots: {sorted(prior_conflicts)}."
                )

                del assignment[slot]
                continue

            remaining = ordered_slots[index + 1:]

            if not partial_calorie_feasible(
                assignment, domains, remaining, lower, upper
            ):
                # The global calorie constraint depends on the current
                # assignment, so all currently assigned variables are
                # conservatively included in the conflict set.
                relevant_conflicts = set(assignment.keys()) - {slot}
                conflict_set.update(relevant_conflicts)

                trace.append(
                    f"Calorie-bound conflict at {slot} = {food['name']}; "
                    f"conflict set: {sorted(relevant_conflicts)}."
                )

                del assignment[slot]
                continue

            solution, child_conflicts = search(index + 1)

            if solution is not None:
                return solution, set()

            # Conflict-directed backjumping:
            # if this variable is not involved in the child failure,
            # it cannot resolve that failure, so jump over it.
            if slot not in child_conflicts:
                backjumps += 1
                trace.append(
                    f"Backjump over {slot}; child conflict set is "
                    f"{sorted(child_conflicts)}."
                )
                del assignment[slot]
                return None, child_conflicts

            # The current variable is involved in the conflict.
            # Remove it from the propagated conflict set and merge
            # the remaining causes before trying another value.
            conflict_set.update(child_conflicts - {slot})
            del assignment[slot]

        trace.append(
            f"Exhausted domain for {slot}; propagated conflict set: "
            f"{sorted(conflict_set)}."
        )

        return None, conflict_set

    solution, final_conflicts = search(0)

    return {
        "solution": solution,
        "nodes_visited": nodes_visited,
        "backjumps": backjumps,
        "final_conflict_set": sorted(final_conflicts),
        "trace": trace,
        "feasible": solution is not None,
    }


# ============================================================
# ALGORITHM 3: LOCAL SEARCH FOR CSP
# Min-conflicts with randomized restarts
#
# Start with a complete assignment.
# Identify variables involved in violated constraints.
# Reassign one such variable to the value minimizing the objective.
# Use randomized restarts to reduce sensitivity to local minima.
#
# Hard constraints:
# - each variable must select from its eligible domain
# - no duplicate foods
#
# Calorie-band violations are minimized as a global constraint.
# The algorithm returns the best assignment found even if no exact
# solution exists, but clearly labels it infeasible.
# ============================================================

def calorie_violation(assignment, lower, upper):
    calories = selected_calories(assignment)

    if calories < lower:
        return lower - calories

    if calories > upper:
        return calories - upper

    return 0.0


def local_search_objective(
    assignment, lower, upper, goal
):
    # Large penalty for hard duplicate constraint violations.
    duplicate_count = len(assignment_conflicts(assignment))

    # Calorie deviation is measured in kcal outside the allowed band.
    deviation = calorie_violation(assignment, lower, upper)

    # Preference score breaks ties among similarly feasible plans.
    preference = plan_preference_score(assignment, goal)

    return (
        duplicate_count * 1_000_000
        + deviation * 100
        - preference * 0.01
    )


def local_search_csp(
    domains,
    lower,
    upper,
    goal,
    max_steps=500,
    restarts=25,
    random_seed=42,
):
    rng = random.Random(random_seed)

    if any(not domains[slot] for slot in MEAL_SLOTS):
        return {
            "solution": None,
            "feasible": False,
            "steps": 0,
            "restarts": 0,
            "objective": None,
            "trace": ["At least one meal slot has an empty domain."],
        }

    best_assignment = None
    best_objective = float("inf")
    best_steps = 0
    total_steps = 0
    trace = []

    for restart in range(restarts):
        current = {
            slot: rng.choice(domains[slot])
            for slot in MEAL_SLOTS
        }

        for step in range(max_steps + 1):
            total_steps += 1

            objective = local_search_objective(
                current, lower, upper, goal
            )

            if objective < best_objective:
                best_objective = objective
                best_assignment = dict(current)
                best_steps = total_steps

            if assignment_is_valid(current, lower, upper):
                return {
                    "solution": dict(current),
                    "feasible": True,
                    "steps": total_steps,
                    "restarts": restart + 1,
                    "objective": objective,
                    "trace": trace,
                }

            if step == max_steps:
                break

            involved = assignment_conflicts(current)

            # A calorie-band violation is a global constraint involving
            # all meal variables; any one can change the total.
            if calorie_violation(current, lower, upper) > 0:
                involved.update(MEAL_SLOTS)

            if not involved:
                involved = set(MEAL_SLOTS)

            variable = rng.choice(sorted(involved))

            candidate_results = []

            for candidate in domains[variable]:
                trial = dict(current)
                trial[variable] = candidate

                candidate_objective = local_search_objective(
                    trial, lower, upper, goal
                )

                candidate_results.append(
                    (candidate_objective, rng.random(), candidate)
                )

            candidate_results.sort(key=lambda item: (item[0], item[1]))

            if candidate_results:
                best_candidate_objective = candidate_results[0][0]

                # Min-conflicts usually chooses a minimum-conflict value.
                # A small random tie-break helps explore equal candidates.
                tied_candidates = [
                    item[2] for item in candidate_results
                    if abs(item[0] - best_candidate_objective) < 1e-9
                ]

                current[variable] = rng.choice(tied_candidates)

        trace.append(
            f"Restart {restart + 1}: best objective so far "
            f"{best_objective:.2f}."
        )

    feasible = (
        best_assignment is not None
        and assignment_is_valid(best_assignment, lower, upper)
    )

    return {
        "solution": best_assignment,
        "feasible": feasible,
        "steps": total_steps,
        "restarts": restarts,
        "objective": best_objective,
        "trace": trace,
    }


# ============================================================
# NUTRITION SUMMARY
# ============================================================

def nutrition_summary(assignment):
    meals = list(assignment.values())

    return {
        "calories": sum(food["calories"] for food in meals),
        "protein": sum(food["protein"] for food in meals),
        "carbs": sum(food["carbs"] for food in meals),
        "fat": sum(food["fat"] for food in meals),
        "catalogue_score": (
            sum(food["score"] for food in meals) / len(meals)
            if meals else 0
        ),
    }


def meal_plan_dataframe(assignment, goal):
    rows = []

    for slot in MEAL_SLOTS:
        food = assignment[slot]

        rows.append({
            "Meal": slot,
            "Food": food["name"],
            "Diet Category": food["diet"],
            "Calories (kcal)": food["calories"],
            "Protein (g)": food["protein"],
            "Carbs (g)": food["carbs"],
            "Fat (g)": food["fat"],
            "Catalogue Score": food["score"],
            "Preference Score": round(
                recommendation_score(food, goal), 2
            ),
            "Allergens": ", ".join(food["allergens"]) or "None listed",
        })

    return pd.DataFrame(rows)


# ============================================================
# USER INPUTS
# ============================================================

st.divider()
st.subheader("🧑 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=21,
    )

    sex_for_bmr = st.selectbox(
        "Sex used for BMR equation",
        ["Female", "Male"],
        help=(
            "The Mifflin-St Jeor equation uses a sex-specific constant. "
            "This estimate may not fit every individual."
        ),
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=70.0,
        step=0.5,
    )

with col2:
    height = st.number_input(
        "Height (cm)",
        min_value=100.0,
        max_value=250.0,
        value=170.0,
        step=0.5,
    )

    activity = st.selectbox(
        "Activity Level",
        [
            "Sedentary",
            "Lightly Active",
            "Moderately Active",
            "Very Active",
            "Athlete",
        ],
    )

    goal = st.selectbox(
        "Goal",
        ["Weight Loss", "Maintenance", "Weight Gain"],
    )

st.divider()
st.subheader("🥗 Dietary Constraints")

col3, col4 = st.columns(2)

with col3:
    diet_type = st.selectbox(
        "Diet Preference",
        ["Vegetarian", "Non-Vegetarian", "Vegan"],
    )

with col4:
    excluded_allergens = st.multiselect(
        "Allergens to exclude",
        ["Milk", "Egg", "Fish", "Soy", "Nuts"],
        help=(
            "Selected allergens are treated as hard constraints. "
            "Verify ingredients and cross-contamination with the food provider."
        ),
    )

st.subheader("🎯 Meal-Plan Constraints")

tolerance_percent = st.slider(
    "Allowed deviation from estimated daily calories (%)",
    min_value=5,
    max_value=50,
    value=25,
    step=5,
    help=(
        "The CSP tries to select four catalogue meals whose combined "
        "calories fall within this interval around the estimated target."
    ),
)

st.caption(
    "The current catalogue contains only four meal slots and a limited "
    "number of foods. A feasible combination may not exist for every "
    "calorie target or dietary restriction."
)

# ============================================================
# GENERATE MEAL PLAN
# ============================================================

if st.button(
    "🥗 Generate and Compare CSP Meal Plans",
    type="primary",
    use_container_width=True,
):
    # --------------------------------------------------------
    # BMR, TDEE AND TARGET CALORIES
    # --------------------------------------------------------

    bmi = weight / ((height / 100) ** 2)

    if sex_for_bmr == "Male":
        bmr = (
            10 * weight
            + 6.25 * height
            - 5 * age
            + 5
        )
    else:
        bmr = (
            10 * weight
            + 6.25 * height
            - 5 * age
            - 161
        )

    activity_factor = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Athlete": 1.9,
    }

    tdee = bmr * activity_factor[activity]

    if goal == "Weight Loss":
        target_calories = tdee - 500
    elif goal == "Weight Gain":
        target_calories = tdee + 500
    else:
        target_calories = tdee

    if target_calories <= 0:
        st.error(
            "The calculated calorie target is not valid. "
            "Review the inputs and seek qualified nutrition advice."
        )
        st.stop()

    # Illustrative macro split retained from the original module.
    protein_g = target_calories * 0.30 / 4
    carbs_g = target_calories * 0.40 / 4
    fat_g = target_calories * 0.30 / 9

    lower, upper = calorie_bounds(
        target_calories, tolerance_percent
    )

    # --------------------------------------------------------
    # BUILD CSP DOMAINS
    # --------------------------------------------------------

    domains = build_domains(
        FOOD_DB,
        diet_type,
        excluded_allergens,
    )

    st.session_state["food_plan_inputs"] = {
        "age": age,
        "sex_for_bmr": sex_for_bmr,
        "weight": weight,
        "height": height,
        "activity": activity,
        "goal": goal,
        "diet_type": diet_type,
        "excluded_allergens": excluded_allergens,
        "bmi": bmi,
        "bmr": bmr,
        "tdee": tdee,
        "target_calories": target_calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "lower": lower,
        "upper": upper,
    }

    empty_slots = [
        slot for slot in MEAL_SLOTS
        if not domains[slot]
    ]

    if empty_slots:
        st.session_state["food_plan_results"] = None
        st.error(
            "No eligible foods are available for these meal slots: "
            + ", ".join(empty_slots)
            + ". Relax a dietary constraint or expand the catalogue."
        )
        st.stop()

    # --------------------------------------------------------
    # RUN ALL THREE ALGORITHMS
    # --------------------------------------------------------

    with st.spinner("Solving the meal-planning CSP..."):
        backtracking_result = backtracking_search(
            domains, lower, upper, goal
        )

        intelligent_result = intelligent_backtracking_search(
            domains, lower, upper, goal
        )

        local_result = local_search_csp(
            domains,
            lower,
            upper,
            goal,
            max_steps=500,
            restarts=25,
            random_seed=42,
        )

    results = {
        "Backtracking Search": backtracking_result,
        "Intelligent Backtracking (CBJ)": intelligent_result,
        "Local Search (Min-Conflicts)": local_result,
    }

    st.session_state["food_plan_results"] = results

# ============================================================
# DISPLAY RESULTS
# ============================================================

results = st.session_state.get("food_plan_results")
inputs = st.session_state.get("food_plan_inputs")

if results and inputs:
    target_calories = inputs["target_calories"]
    lower = inputs["lower"]
    upper = inputs["upper"]
    goal = inputs["goal"]

    st.divider()
    st.subheader("📊 Nutrition Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("BMI", f"{inputs['bmi']:.1f}")
    c2.metric("Estimated BMR", f"{inputs['bmr']:.0f} kcal")
    c3.metric("Estimated TDEE", f"{inputs['tdee']:.0f} kcal")
    c4.metric("Estimated target", f"{target_calories:.0f} kcal")

    st.caption(
        "BMR, TDEE and calorie targets are estimates based on equations "
        "and activity multipliers. They are not clinical measurements."
    )

    st.markdown("**Illustrative daily macronutrient targets**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Protein", f"{inputs['protein_g']:.0f} g")
    m2.metric("Carbohydrates", f"{inputs['carbs_g']:.0f} g")
    m3.metric("Fat", f"{inputs['fat_g']:.0f} g")

    st.caption(
        "The 30% protein / 40% carbohydrate / 30% fat split is an "
        "illustrative default retained from the original code. "
        "It is not appropriate for every person."
    )

    # --------------------------------------------------------
    # ALGORITHM COMPARISON
    # --------------------------------------------------------

    st.divider()
    st.subheader("🧠 CSP Algorithm Comparison")

    comparison_rows = []

    for algorithm_name, result in results.items():
        solution = result.get("solution")
        feasible = bool(result.get("feasible"))

        if solution:
            total_calories = selected_calories(solution)
            calorie_error = abs(total_calories - target_calories)
        else:
            total_calories = None
            calorie_error = None

        if algorithm_name == "Backtracking Search":
            work_count = result["nodes_visited"]
            work_label = "Candidate assignments"
        elif algorithm_name == "Intelligent Backtracking (CBJ)":
            work_count = result["nodes_visited"]
            work_label = "Candidate assignments"
        else:
            work_count = result.get("steps", 0)
            work_label = "Search steps"

        comparison_rows.append({
            "Algorithm": algorithm_name,
            "Feasible plan found": "Yes" if feasible else "No",
            "Total calories": (
                round(total_calories, 1)
                if total_calories is not None else "—"
            ),
            "Absolute target deviation": (
                round(calorie_error, 1)
                if calorie_error is not None else "—"
            ),
            work_label: work_count,
        })

    st.dataframe(
        pd.DataFrame(comparison_rows),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # SELECT BEST FEASIBLE SOLUTION
    # Prefer a feasible exact-CSP solution, then higher preference score.
    # If no exact solution exists, use the best local-search assignment
    # only as a clearly labelled approximate result.
    # --------------------------------------------------------

    feasible_solutions = []

    for algorithm_name, result in results.items():
        solution = result.get("solution")

        if solution and assignment_is_valid(solution, lower, upper):
            feasible_solutions.append(
                (
                    plan_preference_score(solution, goal),
                    algorithm_name,
                    solution,
                )
            )

    if feasible_solutions:
        feasible_solutions.sort(reverse=True, key=lambda item: item[0])
        _, selected_algorithm, selected_plan = feasible_solutions[0]

        st.success(
            f"A feasible meal plan was found using "
            f"{selected_algorithm}."
        )
    else:
        local_solution = results["Local Search (Min-Conflicts)"].get(
            "solution"
        )

        if local_solution:
            selected_algorithm = "Local Search (Min-Conflicts)"
            selected_plan = local_solution

            st.warning(
                "No algorithm found a plan satisfying the full calorie "
                "interval and all hard constraints. The displayed local "
                "search result is an approximate plan, not a feasible CSP "
                "solution. Consider changing the calorie tolerance or "
                "expanding the food catalogue."
            )
        else:
            selected_algorithm = None
            selected_plan = None

            st.error(
                "No complete plan could be produced. Review the diet "
                "and allergy restrictions or expand the food catalogue."
            )

    # --------------------------------------------------------
    # DISPLAY SELECTED MEAL PLAN
    # --------------------------------------------------------

    if selected_plan:
        st.divider()
        st.subheader("🍽 Recommended Meal Plan")

        st.write(f"**Selected algorithm:** {selected_algorithm}")

        plan_df = meal_plan_dataframe(selected_plan, goal)

        st.dataframe(
            plan_df,
            use_container_width=True,
            hide_index=True,
        )

        totals = nutrition_summary(selected_plan)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Plan Calories", f"{totals['calories']} kcal")
        t2.metric("Protein", f"{totals['protein']} g")
        t3.metric("Carbohydrates", f"{totals['carbs']} g")
        t4.metric("Fat", f"{totals['fat']} g")

        calorie_difference = totals["calories"] - target_calories

        st.write(
            f"**Difference from estimated calorie target:** "
            f"{calorie_difference:+.0f} kcal"
        )

        if lower <= totals["calories"] <= upper:
            st.success(
                f"Calorie constraint satisfied: {lower:.0f}–{upper:.0f} kcal."
            )
        else:
            st.warning(
                f"Calorie constraint not satisfied. Required interval: "
                f"{lower:.0f}–{upper:.0f} kcal."
            )

        st.metric(
            "Average catalogue score",
            f"{totals['catalogue_score']:.1f}/100",
        )

        st.caption(
            "Catalogue scores are illustrative values from the food "
            "database, not clinically validated nutritional quality scores."
        )

        # ----------------------------------------------------
        # NUTRITION VISUALIZATION
        # ----------------------------------------------------

        st.subheader("Macronutrient Distribution")

        macro_df = pd.DataFrame({
            "Macronutrient": ["Protein", "Carbohydrates", "Fat"],
            "Grams": [
                totals["protein"],
                totals["carbs"],
                totals["fat"],
            ],
        })

        st.bar_chart(
            macro_df.set_index("Macronutrient")
        )

        # ----------------------------------------------------
        # ALGORITHM TRACES
        # ----------------------------------------------------

        st.divider()
        st.subheader("🔎 Algorithm Execution Details")

        with st.expander("Backtracking Search — Execution trace"):
            result = results["Backtracking Search"]

            st.write(
                f"Candidate assignments explored: "
                f"{result['nodes_visited']}"
            )
            st.write(f"Backtracks: {result['backtracks']}")
            st.write(
                "Feasible solution found: "
                + ("Yes" if result["feasible"] else "No")
            )

            for line in result["trace"][:100]:
                st.write("- " + line)

            if len(result["trace"]) > 100:
                st.caption("Trace truncated to the first 100 entries.")

        with st.expander(
            "Intelligent Backtracking — Conflict-directed backjumping"
        ):
            result = results["Intelligent Backtracking (CBJ)"]

            st.write(
                f"Candidate assignments explored: "
                f"{result['nodes_visited']}"
            )
            st.write(f"Backjumps: {result['backjumps']}")
            st.write(
                f"Final conflict set: "
                f"{result['final_conflict_set']}"
            )
            st.write(
                "Feasible solution found: "
                + ("Yes" if result["feasible"] else "No")
            )

            for line in result["trace"][:100]:
                st.write("- " + line)

            if len(result["trace"]) > 100:
                st.caption("Trace truncated to the first 100 entries.")

        with st.expander("Local Search — Min-conflicts"):
            result = results["Local Search (Min-Conflicts)"]

            st.write(f"Search steps: {result['steps']}")
            st.write(f"Restarts: {result['restarts']}")
            st.write(f"Final objective: {result['objective']}")
            st.write(
                "Feasible solution found: "
                + ("Yes" if result["feasible"] else "No")
            )

            for line in result["trace"][:50]:
                st.write("- " + line)

            if len(result["trace"]) > 50:
                st.caption("Trace truncated to the first 50 entries.")

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        report_lines = [
            "AI FOOD RECOMMENDATION REPORT",
            "=" * 40,
            f"Age: {inputs['age']}",
            f"Sex used for BMR equation: {inputs['sex_for_bmr']}",
            f"Weight: {inputs['weight']} kg",
            f"Height: {inputs['height']} cm",
            f"BMI: {inputs['bmi']:.2f}",
            f"Activity: {inputs['activity']}",
            f"Goal: {goal}",
            f"Diet preference: {inputs['diet_type']}",
            f"Excluded allergens: {', '.join(inputs['excluded_allergens']) or 'None selected'}",
            f"Estimated BMR: {inputs['bmr']:.0f} kcal",
            f"Estimated TDEE: {inputs['tdee']:.0f} kcal",
            f"Estimated target: {target_calories:.0f} kcal",
            f"Allowed calorie interval: {lower:.0f}–{upper:.0f} kcal",
            f"Selected algorithm: {selected_algorithm}",
            "",
            "MEAL PLAN",
            "-" * 40,
        ]

        for slot in MEAL_SLOTS:
            food = selected_plan[slot]
            report_lines.extend([
                f"{slot}: {food['name']}",
                f"  Calories: {food['calories']} kcal",
                f"  Protein: {food['protein']} g",
                f"  Carbohydrates: {food['carbs']} g",
                f"  Fat: {food['fat']} g",
                f"  Catalogue allergens: {', '.join(food['allergens']) or 'None listed'}",
                "",
            ])

        report_lines.extend([
            "TOTALS",
            "-" * 40,
            f"Calories: {totals['calories']} kcal",
            f"Protein: {totals['protein']} g",
            f"Carbohydrates: {totals['carbs']} g",
            f"Fat: {totals['fat']} g",
            f"Average catalogue score: {totals['catalogue_score']:.1f}/100",
            "",
            "LIMITATIONS",
            "-" * 40,
            "Food nutrition values are illustrative catalogue estimates.",
            "Allergens must be checked against actual product labels.",
            "Calorie and macro targets are estimates, not prescriptions.",
            "Consult a qualified dietitian or healthcare professional for individualized advice.",
        ])

        st.download_button(
            "📥 Download Food Recommendation Report",
            data="\n".join(report_lines),
            file_name="Food_Recommendation_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # ----------------------------------------------------
        # OPTIONAL GEMINI EXPLANATION
        # ----------------------------------------------------

        st.divider()
        st.subheader("🤖 Optional AI Nutrition Explanation")

        if st.button("Generate AI Meal-Plan Explanation"):
            if (
                not GEMINI_API_KEY
                or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
            ):
                st.warning(
                    "Add your Gemini API key to GEMINI_API_KEY near "
                    "the top of this file to enable the optional feature."
                )
            else:
                try:
                    import google.generativeai as genai

                    genai.configure(api_key=GEMINI_API_KEY)
                    model = genai.GenerativeModel(GEMINI_MODEL)

                    prompt = f"""
Explain this educational meal plan in clear, cautious language.

Age: {inputs['age']}
Goal: {goal}
Diet preference: {inputs['diet_type']}
Excluded allergens: {inputs['excluded_allergens']}
Estimated target calories: {target_calories:.0f}
Selected meals: {plan_df[['Meal', 'Food', 'Calories (kcal)', 'Protein (g)', 'Carbs (g)', 'Fat (g)']].to_dict(orient='records')}
Totals: {totals}

Requirements:
- Do not claim to be a certified dietitian.
- Do not diagnose or prescribe a medical diet.
- Do not claim the calorie or macro targets are suitable for everyone.
- Do not recommend foods containing the selected excluded allergens.
- State that the catalogue values are estimates and labels must be checked.
- Give practical general suggestions and recommend professional advice
  for medical conditions, pregnancy, eating disorders, or specialized diets.
"""

                    with st.spinner("Generating explanation..."):
                        response = model.generate_content(prompt)

                    st.write(response.text)

                except Exception as exc:
                    st.error(
                        "AI explanation could not be generated. Check "
                        "your API key, model availability, SDK setup, "
                        f"and network. Details: {exc}"
                    )

st.divider()
st.caption(
    "This application provides educational meal-planning support. "
    "It does not replace a registered dietitian or healthcare professional."
)
