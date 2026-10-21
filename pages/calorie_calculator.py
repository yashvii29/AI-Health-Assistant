
import random
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# 1. OPTIONAL GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

# Gemini is optional. The calculator and both algorithms work
# without an API key.


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Calorie Calculator",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 AI Calorie Calculator")
st.write(
    "Estimate your daily energy needs and compare two optimisation "
    "algorithms for finding a feasible macronutrient distribution."
)

st.caption(
    "For adults aged 18 and over. Estimates are educational and "
    "are not a substitute for individual medical or dietetic advice."
)


# ============================================================
# 3. CALORIE AND BMI CALCULATIONS
# ============================================================

ACTIVITY_MULTIPLIERS = {
    "Sedentary": 1.20,
    "Lightly Active": 1.375,
    "Moderately Active": 1.55,
    "Very Active": 1.725,
    "Extra Active": 1.90,
}


def calculate_bmr(weight_kg, height_cm, age, equation_sex):
    """
    Mifflin-St Jeor equation.

    Male equation:
        10W + 6.25H - 5A + 5

    Female equation:
        10W + 6.25H - 5A - 161

    W = weight in kg, H = height in cm, A = age in years.
    These equations are estimates, not direct measurements.
    """
    base = (
        10 * weight_kg
        + 6.25 * height_cm
        - 5 * age
    )

    if equation_sex == "Male equation":
        return base + 5

    return base - 161


def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def classify_adult_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal weight"
    if bmi < 30:
        return "Overweight"
    return "Obesity"


def calculate_target_calories(maintenance_calories, goal):
    """
    Use the requested goal adjustment as a transparent heuristic.
    The adjustment is not a guarantee of a particular weight change.
    """
    if goal == "Weight Loss":
        return maintenance_calories - 500.0

    if goal == "Weight Gain":
        return maintenance_calories + 300.0

    return maintenance_calories


# ============================================================
# 4. MACRONUTRIENT OPTIMISATION PROBLEM
# ============================================================
#
# A chromosome / candidate is:
#     [protein_percentage, carbohydrate_percentage, fat_percentage]
#
# Constraints:
#     Percentages sum to 100.
#     Protein: 10% to 35% of energy.
#     Carbohydrates: 45% to 65% of energy.
#     Fat: 20% to 35% of energy.
#
# These are general adult Acceptable Macronutrient Distribution
# Range boundaries, not individualised medical prescriptions.
#
# The objective is to find a feasible distribution close to the
# project's preferred 30/40/30 distribution. Because 40% carbs is
# below the stated 45% lower bound, the optimiser must compromise.
# ============================================================

MACRO_NAMES = ["Protein", "Carbohydrates", "Fat"]

MACRO_BOUNDS = {
    "Protein": (10.0, 35.0),
    "Carbohydrates": (45.0, 65.0),
    "Fat": (20.0, 35.0),
}

PREFERRED_MACROS = {
    "Protein": 30.0,
    "Carbohydrates": 40.0,
    "Fat": 30.0,
}


def is_feasible(candidate, tolerance=1e-7):
    """Return True only when all macro constraints are satisfied."""
    if len(candidate) != 3:
        return False

    if abs(sum(candidate) - 100.0) > tolerance:
        return False

    for index, name in enumerate(MACRO_NAMES):
        low, high = MACRO_BOUNDS[name]
        if candidate[index] < low - tolerance:
            return False
        if candidate[index] > high + tolerance:
            return False

    return True


def objective(candidate):
    """
    Lower is better.

    Minimise squared distance from the original preferred macro
    distribution, subject to the constraints above.
    """
    if not is_feasible(candidate):
        return float("inf")

    return sum(
        (candidate[i] - PREFERRED_MACROS[name]) ** 2
        for i, name in enumerate(MACRO_NAMES)
    )


def make_random_feasible_candidate(rng):
    """Sample a valid macro distribution."""
    for _ in range(10000):
        protein = rng.uniform(10.0, 35.0)
        carbs = rng.uniform(45.0, 65.0)
        fat = 100.0 - protein - carbs

        candidate = [protein, carbs, fat]

        if is_feasible(candidate):
            return candidate

    # A known feasible fallback.
    return [30.0, 45.0, 25.0]


def repair_candidate(candidate):
    """
    Project a candidate back into the feasible region.

    First clamp protein and carbohydrates, then derive fat as the
    remainder. If the resulting fat percentage is infeasible, search
    the feasible protein/carbohydrate space for the nearest candidate.
    """
    protein_guess = float(candidate[0])
    carbs_guess = float(candidate[1])

    protein_guess = max(10.0, min(35.0, protein_guess))
    carbs_guess = max(45.0, min(65.0, carbs_guess))

    best = None
    best_distance = float("inf")

    # A small grid projection is deterministic and ensures feasibility.
    for p_step in range(101):
        protein = 10.0 + p_step * 0.25

        for c_step in range(81):
            carbs = 45.0 + c_step * 0.25
            fat = 100.0 - protein - carbs
            candidate_fixed = [protein, carbs, fat]

            if not is_feasible(candidate_fixed):
                continue

            distance = (
                (protein - protein_guess) ** 2
                + (carbs - carbs_guess) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                best = candidate_fixed

    if best is None:
        return [30.0, 45.0, 25.0]

    return best


def percentages_to_grams(calories, candidate):
    """
    Convert energy percentages to grams using standard factors:
    protein = 4 kcal/g, carbohydrate = 4 kcal/g, fat = 9 kcal/g.
    """
    protein_pct, carb_pct, fat_pct = candidate

    return {
        "Protein": calories * protein_pct / 100.0 / 4.0,
        "Carbohydrates": calories * carb_pct / 100.0 / 4.0,
        "Fat": calories * fat_pct / 100.0 / 9.0,
    }


def macro_energy_breakdown(calories, candidate):
    return {
        "Protein": calories * candidate[0] / 100.0,
        "Carbohydrates": calories * candidate[1] / 100.0,
        "Fat": calories * candidate[2] / 100.0,
    }


# ============================================================
# 5. HILL CLIMBING
# ============================================================

def hill_climbing(
    initial_candidate=None,
    max_iterations=500,
    step_size=1.0,
    seed=42,
):
    """
    Steepest-ascent-style hill climbing for a minimisation problem.

    At each iteration, generate neighbouring feasible solutions by
    transferring a small percentage from one macro to another.
    Select the best strictly improving neighbour.

    Stop when no neighbour improves the objective or the iteration
    limit is reached. The result may be a local optimum.
    """
    rng = random.Random(seed)

    current = (
        repair_candidate(initial_candidate)
        if initial_candidate is not None
        else make_random_feasible_candidate(rng)
    )

    current_score = objective(current)

    history = [{
        "Iteration": 0,
        "Objective": current_score,
        "Protein %": current[0],
        "Carbohydrates %": current[1],
        "Fat %": current[2],
    }]

    for iteration in range(1, max_iterations + 1):
        neighbours = []

        for donor in range(3):
            for receiver in range(3):
                if donor == receiver:
                    continue

                candidate = current.copy()
                candidate[donor] -= step_size
                candidate[receiver] += step_size

                if is_feasible(candidate):
                    neighbours.append(candidate)

        if not neighbours:
            break

        best_neighbour = min(neighbours, key=objective)
        best_score = objective(best_neighbour)

        # Strict improvement prevents cycles.
        if best_score >= current_score - 1e-10:
            break

        current = best_neighbour
        current_score = best_score

        history.append({
            "Iteration": iteration,
            "Objective": current_score,
            "Protein %": current[0],
            "Carbohydrates %": current[1],
            "Fat %": current[2],
        })

    return {
        "solution": current,
        "objective": current_score,
        "history": history,
        "iterations": len(history) - 1,
        "feasible": is_feasible(current),
    }


# ============================================================
# 6. GENETIC ALGORITHM
# ============================================================

def tournament_selection(population, fitness_scores, rng, tournament_size=3):
    """Select the lowest-objective candidate from a random tournament."""
    indices = rng.sample(
        range(len(population)),
        k=min(tournament_size, len(population)),
    )

    winner_index = min(
        indices,
        key=lambda idx: fitness_scores[idx],
    )

    return population[winner_index].copy()


def crossover(parent1, parent2, rng):
    """
    Arithmetic crossover.

    A convex combination of feasible parents remains feasible because
    the constraints are linear.
    """
    alpha = rng.random()

    child = [
        alpha * parent1[i] + (1.0 - alpha) * parent2[i]
        for i in range(3)
    ]

    return child


def mutate(candidate, rng, mutation_rate=0.20, mutation_step=3.0):
    """
    Mutation transfers percentage points between two distinct macros,
    preserving the 100% sum. Invalid mutations are rejected.
    """
    child = candidate.copy()

    if rng.random() >= mutation_rate:
        return child

    for _ in range(30):
        donor, receiver = rng.sample(range(3), 2)
        amount = rng.uniform(0.5, mutation_step)

        proposed = child.copy()
        proposed[donor] -= amount
        proposed[receiver] += amount

        if is_feasible(proposed):
            return proposed

    return child


def genetic_algorithm(
    population_size=60,
    generations=100,
    mutation_rate=0.20,
    mutation_step=3.0,
    elite_count=2,
    seed=42,
):
    """
    Minimise the macro objective with a genetic algorithm.

    Operations:
    - Feasible random population initialisation
    - Tournament selection
    - Arithmetic crossover
    - Constraint-preserving mutation
    - Elitism
    - Best-fitness tracking by generation
    """
    if population_size < 4:
        raise ValueError("Population size must be at least 4.")

    if generations < 1:
        raise ValueError("Generations must be at least 1.")

    if not 1 <= elite_count < population_size:
        raise ValueError(
            "Elite count must be at least 1 and less than population size."
        )

    rng = random.Random(seed)

    population = [
        make_random_feasible_candidate(rng)
        for _ in range(population_size)
    ]

    history = []

    for generation in range(generations + 1):
        fitness_scores = [objective(individual) for individual in population]

        best_index = min(
            range(len(population)),
            key=lambda index: fitness_scores[index],
        )

        best = population[best_index].copy()
        best_score = fitness_scores[best_index]

        history.append({
            "Generation": generation,
            "Best objective": best_score,
            "Mean objective": sum(fitness_scores) / len(fitness_scores),
            "Protein %": best[0],
            "Carbohydrates %": best[1],
            "Fat %": best[2],
        })

        if generation == generations:
            break

        ranked_population = sorted(
            zip(population, fitness_scores),
            key=lambda pair: pair[1],
        )

        next_population = [
            individual.copy()
            for individual, _ in ranked_population[:elite_count]
        ]

        while len(next_population) < population_size:
            parent1 = tournament_selection(
                population, fitness_scores, rng
            )
            parent2 = tournament_selection(
                population, fitness_scores, rng
            )

            child = crossover(parent1, parent2, rng)
            child = mutate(
                child,
                rng,
                mutation_rate=mutation_rate,
                mutation_step=mutation_step,
            )

            if not is_feasible(child):
                child = repair_candidate(child)

            next_population.append(child)

        population = next_population

    final_fitness = [objective(individual) for individual in population]
    final_index = min(
        range(len(population)),
        key=lambda index: final_fitness[index],
    )

    final_solution = population[final_index].copy()

    return {
        "solution": final_solution,
        "objective": objective(final_solution),
        "history": history,
        "generations": generations,
        "feasible": is_feasible(final_solution),
    }


# ============================================================
# 7. DISPLAY HELPERS
# ============================================================

def build_macro_dataframe(calories, candidate):
    grams = percentages_to_grams(calories, candidate)
    energy = macro_energy_breakdown(calories, candidate)

    rows = []

    for index, name in enumerate(MACRO_NAMES):
        rows.append({
            "Macronutrient": name,
            "Energy percentage": candidate[index],
            "Energy (kcal)": energy[name],
            "Grams/day": grams[name],
        })

    return pd.DataFrame(rows)


def build_algorithm_comparison(hill_result, ga_result):
    rows = []

    for name, result, steps in [
        (
            "Hill Climbing",
            hill_result,
            hill_result["iterations"],
        ),
        (
            "Genetic Algorithm",
            ga_result,
            ga_result["generations"],
        ),
    ]:
        solution = result["solution"]

        rows.append({
            "Algorithm": name,
            "Objective (lower is better)": result["objective"],
            "Protein %": solution[0],
            "Carbohydrates %": solution[1],
            "Fat %": solution[2],
            "Feasible": result["feasible"],
            "Iterations / generations": steps,
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. USER INPUTS
# ============================================================

st.subheader("Your details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=120,
        value=25,
        step=1,
    )

    equation_sex = st.selectbox(
        "BMR equation",
        ["Male equation", "Female equation"],
        help=(
            "Choose the equation whose assumptions best fit you. "
            "These equations may not represent every individual."
        ),
    )

with col2:
    weight = st.number_input(
        "Weight (kg)",
        min_value=10.0,
        max_value=300.0,
        value=70.0,
        step=0.5,
    )

    height = st.number_input(
        "Height (cm)",
        min_value=50.0,
        max_value=250.0,
        value=170.0,
        step=0.5,
    )

with col3:
    activity = st.selectbox(
        "Activity level",
        list(ACTIVITY_MULTIPLIERS.keys()),
        index=2,
        help=(
            "Select the closest overall activity level, including "
            "routine movement and exercise."
        ),
    )

    goal = st.selectbox(
        "Goal",
        ["Weight Loss", "Maintenance", "Weight Gain"],
    )


st.subheader("Optimisation settings")

settings_col1, settings_col2, settings_col3 = st.columns(3)

with settings_col1:
    population_size = st.slider(
        "GA population size",
        min_value=20,
        max_value=150,
        value=60,
        step=10,
    )

with settings_col2:
    generations = st.slider(
        "GA generations",
        min_value=20,
        max_value=300,
        value=100,
        step=20,
    )

with settings_col3:
    mutation_rate = st.slider(
        "GA mutation probability",
        min_value=0.0,
        max_value=0.8,
        value=0.20,
        step=0.05,
    )

seed = st.number_input(
    "Random seed (reproducible results)",
    min_value=0,
    max_value=1_000_000,
    value=42,
    step=1,
)

run_button = st.button(
    "🔥 Calculate and Optimise",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 9. RUN CALCULATIONS AND ALGORITHMS
# ============================================================

if run_button:

    bmr = calculate_bmr(
        weight_kg=weight,
        height_cm=height,
        age=age,
        equation_sex=equation_sex,
    )

    tdee = bmr * ACTIVITY_MULTIPLIERS[activity]
    target_calories = calculate_target_calories(tdee, goal)
    bmi = calculate_bmi(weight, height)
    bmi_category = classify_adult_bmi(bmi)

    if bmr <= 0 or tdee <= 0 or target_calories <= 0:
        st.error(
            "The calculated calorie target is not positive. "
            "Check the inputs and goal."
        )
        st.stop()

    # Both algorithms solve the same macro optimisation problem.
    hill_result = hill_climbing(seed=int(seed))
    ga_result = genetic_algorithm(
        population_size=int(population_size),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        seed=int(seed),
    )

    # A deterministic feasible baseline is useful for comparison.
    baseline = [30.0, 45.0, 25.0]

    result_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "equation_sex": equation_sex,
        "weight": weight,
        "height": height,
        "activity": activity,
        "goal": goal,
        "bmr": bmr,
        "tdee": tdee,
        "target_calories": target_calories,
        "bmi": bmi,
        "bmi_category": bmi_category,
        "hill_result": hill_result,
        "ga_result": ga_result,
        "baseline": baseline,
        "population_size": population_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "seed": seed,
    }

    st.session_state["calorie_result"] = result_payload


# ============================================================
# 10. RESULTS
# ============================================================

if "calorie_result" in st.session_state:

    result = st.session_state["calorie_result"]

    bmr = result["bmr"]
    tdee = result["tdee"]
    target_calories = result["target_calories"]
    bmi = result["bmi"]
    bmi_category = result["bmi_category"]
    hill_result = result["hill_result"]
    ga_result = result["ga_result"]

    st.divider()
    st.header("📊 Energy Requirements")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("BMR", f"{bmr:,.0f} kcal/day")
    c2.metric("Estimated TDEE", f"{tdee:,.0f} kcal/day")
    c3.metric("Goal target", f"{target_calories:,.0f} kcal/day")
    c4.metric("BMI", f"{bmi:.1f}", help=bmi_category)

    st.caption(
        "BMR estimates energy use at rest. TDEE estimates daily energy "
        "use after applying the selected activity multiplier. Both are "
        "approximations and may differ from actual energy requirements."
    )

    if result["goal"] == "Weight Loss":
        st.info(
            "The weight-loss estimate uses a 500 kcal/day deficit from "
            "estimated maintenance. This may be unsuitable for some people."
        )
    elif result["goal"] == "Weight Gain":
        st.info(
            "The weight-gain estimate uses a 300 kcal/day surplus from "
            "estimated maintenance. Actual requirements vary."
        )
    else:
        st.info(
            "The maintenance estimate uses the selected activity multiplier."
        )

    st.subheader("BMI context")
    st.write(f"**Adult BMI category:** {bmi_category}")
    st.caption(
        "BMI is a screening measure, not a direct measurement of body fat "
        "or overall health. This calorie calculator does not diagnose disease."
    )

    st.divider()
    st.header("🧬 Algorithm Results")

    st.write(
        "Both algorithms optimise the same three-variable macro distribution "
        "under the same percentage constraints. The objective is the squared "
        "distance from the original preferred distribution of 30% protein, "
        "40% carbohydrates, and 30% fat."
    )

    st.markdown(
        """
        **Constraints used in this optimisation**

        - Protein: 10–35% of energy
        - Carbohydrates: 45–65% of energy
        - Fat: 20–35% of energy
        - Total: exactly 100%

        These broad adult ranges are general reference ranges. They are not
        appropriate for every individual or medical condition.
        """
    )

    st.caption(
        "The preferred 30/40/30 distribution conflicts with the carbohydrate "
        "lower bound of 45%. The objective therefore finds the closest "
        "feasible distribution rather than preserving 40% carbohydrates."
    )

    # --------------------------------------------------------
    # Algorithm comparison
    # --------------------------------------------------------

    comparison_df = build_algorithm_comparison(hill_result, ga_result)

    st.subheader("Algorithm comparison")
    st.dataframe(
        comparison_df.round(3),
        hide_index=True,
        use_container_width=True,
    )

    best_algorithm = (
        "Hill Climbing"
        if hill_result["objective"] <= ga_result["objective"]
        else "Genetic Algorithm"
    )

    st.success(
        f"Best result in this run: {best_algorithm}. "
        "A lower objective means a closer match to the preferred "
        "distribution while satisfying the constraints."
    )

    # --------------------------------------------------------
    # Show algorithm-specific outputs
    # --------------------------------------------------------

    alg_col1, alg_col2 = st.columns(2)

    with alg_col1:
        st.subheader("⛰️ Hill Climbing")

        st.metric(
            "Final objective",
            f"{hill_result['objective']:.4f}",
        )

        st.write(
            f"Accepted improvements: {hill_result['iterations']}"
        )

        hill_df = pd.DataFrame(hill_result["history"])
        st.line_chart(
            hill_df.set_index("Iteration")[["Objective"]]
        )

        with st.expander("View Hill Climbing trace"):
            st.dataframe(
                hill_df.round(4),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown(
            """
            **How it works**

            1. Start from a feasible macro distribution.
            2. Generate neighbours by transferring percentage points
               between two macronutrients.
            3. Evaluate each feasible neighbour with the objective function.
            4. Accept the best strict improvement.
            5. Stop when no neighbour improves the result or the iteration
               limit is reached.

            Hill Climbing is fast for this small problem, but it can stop at
            a local optimum.
            """
        )

    with alg_col2:
        st.subheader("🧬 Genetic Algorithm")

        st.metric(
            "Final objective",
            f"{ga_result['objective']:.4f}",
        )

        st.write(
            f"Generations: {ga_result['generations']}"
        )

        ga_df = pd.DataFrame(ga_result["history"])
        st.line_chart(
            ga_df.set_index("Generation")[
                ["Best objective", "Mean objective"]
            ]
        )

        with st.expander("View Genetic Algorithm trace"):
            st.dataframe(
                ga_df.round(4),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown(
            """
            **How it works**

            1. Create a population of random feasible distributions.
            2. Evaluate each candidate using the objective function.
            3. Select parents using tournament selection.
            4. Generate children using arithmetic crossover.
            5. Mutate candidates by transferring percentage points.
            6. Preserve elite candidates and repeat over generations.

            The GA explores a population rather than a single path. For this
            small convex optimisation problem, it is not guaranteed to
            outperform Hill Climbing.
            """
        )

    st.divider()
    st.header("🥗 Optimised Macronutrient Targets")

    # Show both outputs side by side; neither algorithm is assumed
    # to be universally superior.
    macro_col1, macro_col2 = st.columns(2)

    with macro_col1:
        st.subheader("Hill Climbing solution")
        hill_macro_df = build_macro_dataframe(
            target_calories, hill_result["solution"]
        )
        st.dataframe(
            hill_macro_df.round(2),
            hide_index=True,
            use_container_width=True,
        )

    with macro_col2:
        st.subheader("Genetic Algorithm solution")
        ga_macro_df = build_macro_dataframe(
            target_calories, ga_result["solution"]
        )
        st.dataframe(
            ga_macro_df.round(2),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Grams are calculated using 4 kcal/g for protein, 4 kcal/g for "
        "carbohydrates, and 9 kcal/g for fat. Displayed gram values are "
        "rounded; the underlying calculations use unrounded percentages."
    )

    # --------------------------------------------------------
    # Preferred baseline comparison
    # --------------------------------------------------------

    st.subheader("Preferred distribution vs optimised distribution")

    baseline_df = build_macro_dataframe(
        target_calories, result["baseline"]
    ).rename(columns={
        "Energy percentage": "Baseline %",
        "Energy (kcal)": "Baseline kcal",
        "Grams/day": "Baseline grams/day",
    })

    optimised_df = build_macro_dataframe(
        target_calories, ga_result["solution"]
    ).rename(columns={
        "Energy percentage": "GA %",
        "Energy (kcal)": "GA kcal",
        "Grams/day": "GA grams/day",
    })

    baseline_comparison = baseline_df.merge(
        optimised_df,
        on="Macronutrient",
    )

    st.dataframe(
        baseline_comparison.round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.divider()
    st.header("🤖 Optional AI Nutrition Explanation")

    st.caption(
        "AI explanations may be inaccurate. They do not replace a "
        "registered dietitian or qualified healthcare professional."
    )

    if st.button("Generate AI Nutrition Explanation"):

        if (
            not GEMINI_API_KEY
            or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
        ):
            st.warning(
                "Add your Gemini API key to GEMINI_API_KEY near the top "
                "of this file to enable this feature."
            )
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL)

                hill_solution = hill_result["solution"]
                ga_solution = ga_result["solution"]

                prompt = f"""
                Explain these estimated energy and macro results in plain,
                cautious, educational language.

                Age: {result['age']}
                Weight: {result['weight']} kg
                Height: {result['height']} cm
                BMR equation: {result['equation_sex']}
                Activity: {result['activity']}
                Goal: {result['goal']}
                BMI: {bmi:.2f} ({bmi_category})
                Estimated BMR: {bmr:.0f} kcal/day
                Estimated TDEE: {tdee:.0f} kcal/day
                Estimated goal calories: {target_calories:.0f} kcal/day

                Hill Climbing macro percentages:
                Protein {hill_solution[0]:.2f}%, carbohydrates
                {hill_solution[1]:.2f}%, fat {hill_solution[2]:.2f}%.

                Genetic Algorithm macro percentages:
                Protein {ga_solution[0]:.2f}%, carbohydrates
                {ga_solution[1]:.2f}%, fat {ga_solution[2]:.2f}%.

                Explain:
                1. The difference between BMR, TDEE, and goal calories.
                2. What the macro percentages mean.
                3. Why these values are estimates.
                4. General balanced eating habits.

                Do not diagnose, prescribe medical treatment, or claim
                that the macro optimisation is a clinical recommendation.
                """

                with st.spinner("Generating explanation..."):
                    response = model.generate_content(prompt)

                ai_text = getattr(response, "text", None)

                if ai_text:
                    st.session_state["calorie_ai_text"] = ai_text
                    st.markdown(ai_text)
                else:
                    st.warning("The AI service returned no text.")

            except Exception as exc:
                st.error(
                    "AI explanation could not be generated. Check the "
                    "API key, model availability, package version, and "
                    f"network connection. Details: {exc}"
                )

    if st.session_state.get("calorie_ai_text"):
        st.markdown(st.session_state["calorie_ai_text"])

    # --------------------------------------------------------
    # Downloadable report
    # --------------------------------------------------------

    st.divider()
    st.header("📄 Download Report")

    hill_solution = hill_result["solution"]
    ga_solution = ga_result["solution"]

    hill_grams = percentages_to_grams(
        target_calories, hill_solution
    )
    ga_grams = percentages_to_grams(
        target_calories, ga_solution
    )

    report_lines = [
        "AI CALORIE CALCULATOR REPORT",
        "=" * 42,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "INPUTS",
        f"Age: {result['age']}",
        f"Weight: {result['weight']} kg",
        f"Height: {result['height']} cm",
        f"BMR equation: {result['equation_sex']}",
        f"Activity level: {result['activity']}",
        f"Goal: {result['goal']}",
        "",
        "ENERGY ESTIMATES",
        f"BMR: {bmr:.0f} kcal/day",
        f"TDEE: {tdee:.0f} kcal/day",
        f"Goal calorie estimate: {target_calories:.0f} kcal/day",
        f"BMI: {bmi:.2f}",
        f"Adult BMI category: {bmi_category}",
        "",
        "HILL CLIMBING",
        f"Objective: {hill_result['objective']:.4f}",
        f"Protein: {hill_solution[0]:.2f}% / {hill_grams['Protein']:.1f} g",
        f"Carbohydrates: {hill_solution[1]:.2f}% / "
        f"{hill_grams['Carbohydrates']:.1f} g",
        f"Fat: {hill_solution[2]:.2f}% / {hill_grams['Fat']:.1f} g",
        f"Accepted improvements: {hill_result['iterations']}",
        "",
        "GENETIC ALGORITHM",
        f"Objective: {ga_result['objective']:.4f}",
        f"Protein: {ga_solution[0]:.2f}% / {ga_grams['Protein']:.1f} g",
        f"Carbohydrates: {ga_solution[1]:.2f}% / "
        f"{ga_grams['Carbohydrates']:.1f} g",
        f"Fat: {ga_solution[2]:.2f}% / {ga_grams['Fat']:.1f} g",
        f"Population size: {result['population_size']}",
        f"Generations: {result['generations']}",
        f"Mutation probability: {result['mutation_rate']}",
        f"Random seed: {result['seed']}",
        "",
        "OPTIMISATION CONSTRAINTS",
        "Protein: 10-35% of energy",
        "Carbohydrates: 45-65% of energy",
        "Fat: 20-35% of energy",
        "Macro percentages sum to 100%.",
        "",
        "LIMITATIONS",
        "BMR and TDEE are estimates, not measured values.",
        "Macro ranges are general adult reference ranges.",
        "The optimisation objective prioritises closeness to a preferred "
        "macro distribution; it does not optimise clinical outcomes.",
        "This report is educational and is not medical advice.",
    ]

    ai_report = st.session_state.get("calorie_ai_text", "")
    if ai_report:
        report_lines.extend([
            "",
            "OPTIONAL AI EXPLANATION",
            ai_report,
        ])

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Calorie Report (.txt)",
        data=report_text,
        file_name="Calorie_Report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# 11. REFERENCE GUIDE
# ============================================================

st.divider()
st.subheader("Activity Level Guide")

activity_df = pd.DataFrame({
    "Activity level": [
        "Sedentary",
        "Lightly Active",
        "Moderately Active",
        "Very Active",
        "Extra Active",
    ],
    "Multiplier": [
        1.20,
        1.375,
        1.55,
        1.725,
        1.90,
    ],
    "General description": [
        "Little routine exercise",
        "Light exercise or activity",
        "Moderate exercise or regular movement",
        "Hard exercise or substantial activity",
        "Very high activity or physically demanding work",
    ],
})

st.dataframe(
    activity_df,
    hide_index=True,
    use_container_width=True,
)

st.info(
    "If you are pregnant or breastfeeding, under 18, have a history of "
    "disordered eating, or have a medical condition that affects nutrition, "
    "seek individual advice rather than relying on this calculator."
)
