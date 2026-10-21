
import heapq
import re
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from collections import deque
from urllib.parse import quote_plus
from functools import lru_cache

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medicine Information Lookup",
    page_icon="💊",
    layout="wide"
)

# Optional Gemini integration. The lookup works without this key.
GEMINI_API_KEY = "AQ.Ab8RN6LDq8jPgJwvOqXjtYwkyiWbDgzO1YhoMA-uwQXCQ-85sQ"
GEMINI_MODEL = "gemini-3.8-flash"

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
MEDLINEPLUS_BASE = "https://medlineplus.gov"

REQUEST_TIMEOUT = 12

st.title("Medicine Information Lookup")
st.caption(
    "Search medicine names, access authoritative references, "
    "and explore five graph-search algorithms."
)

st.warning(
    "Educational information only. This application does not prescribe "
    "medicines, recommend a medicine for your symptoms, calculate doses, "
    "or determine whether a medicine is safe for you. Consult a doctor "
    "or pharmacist for personal medication decisions."
)

# ============================================================
# HTTP HELPERS
# ============================================================

def api_get(url, params=None):
    """Make a bounded-time GET request and validate the HTTP response."""
    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "MedicineInformationLookup/1.0"}
    )
    response.raise_for_status()
    return response.json()


def clean_text(value, max_length=500):
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_length]


# ============================================================
# RXNORM SEARCH
# RxNorm is a standardized drug terminology from the US NLM.
# Search results identify concepts; they are not prescribing advice.
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def search_rxnorm(medicine_name):
    url = f"{RXNORM_BASE}/approximateTerm.json"

    payload = api_get(
        url,
        params={
            "term": medicine_name,
            "maxEntries": 10
        }
    )

    candidates = (
        payload.get("approximateGroup", {})
        .get("candidate", [])
    )

    if isinstance(candidates, dict):
        candidates = [candidates]

    results = []
    seen = set()

    for candidate in candidates:
        rxcui = str(candidate.get("rxcui", "")).strip()

        if not rxcui or rxcui in seen:
            continue

        seen.add(rxcui)

        results.append({
            "rxcui": rxcui,
            "name": clean_text(candidate.get("name", ""), 250),
            "score": candidate.get("score", ""),
            "tty": clean_text(candidate.get("tty", ""), 80)
        })

    return results


@st.cache_data(ttl=3600, show_spinner=False)
def get_rxnorm_properties(rxcui):
    payload = api_get(f"{RXNORM_BASE}/rxcui/{rxcui}/properties.json")
    properties = payload.get("properties")

    return properties if isinstance(properties, dict) else {}


@st.cache_data(ttl=3600, show_spinner=False)
def get_rxnorm_related(rxcui):
    payload = api_get(f"{RXNORM_BASE}/rxcui/{rxcui}/allrelated.json")

    groups = (
        payload.get("allRelatedGroup", {})
        .get("conceptGroup", [])
    )

    related = []

    for group in groups:
        concepts = group.get("conceptProperties", [])

        if isinstance(concepts, dict):
            concepts = [concepts]

        for concept in concepts:
            related.append({
                "name": clean_text(concept.get("name", ""), 250),
                "rxcui": str(concept.get("rxcui", "")),
                "term_type": clean_text(concept.get("tty", ""), 80)
            })

    return related


# ============================================================
# DAILYMED LABEL SEARCH
# DailyMed provides current Structured Product Label information.
# Search results can include multiple products or manufacturers.
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def search_dailymed(medicine_name):
    payload = api_get(
        f"{DAILYMED_BASE}/drugnames.json",
        params={
            "drug_name": medicine_name,
            "name_type": "both",
            "pagesize": 10,
            "page": 1
        }
    )

    # API responses can vary by endpoint or response state.
    data = payload.get("data", [])

    if isinstance(data, dict):
        data = [data]

    results = []

    for item in data:
        results.append({
            "name": clean_text(
                item.get("drug_name")
                or item.get("name")
                or item.get("drugname")
                or "",
                250
            ),
            "setid": clean_text(
                item.get("setid") or item.get("set_id") or "",
                100
            ),
            "manufacturer": clean_text(
                item.get("manufacturer") or "",
                200
            )
        })

    return results


def dailymed_search_url(medicine_name):
    return (
        "https://dailymed.nlm.nih.gov/dailymed/search.cfm"
        f"?query={quote_plus(medicine_name)}"
    )


def medlineplus_search_url(medicine_name):
    return (
        "https://medlineplus.gov/druginformation.html"
    )


def rxnorm_search_url(medicine_name):
    return (
        "https://mor.nlm.nih.gov/RxNav/search"
        f"?searchBy=String&searchTerm={quote_plus(medicine_name)}"
    )


# ============================================================
# GRAPH MODEL FOR THE FIVE SEARCH ALGORITHMS
#
# Nodes represent navigation destinations in a medicine
# information workflow, not medical recommendations.
#
# Directed edges have positive costs.
# BFS/DFS ignore costs and find a path by graph traversal.
# UCS minimizes accumulated path cost.
# Greedy Best-First minimizes the heuristic only.
# A* minimizes accumulated cost + heuristic.
# ============================================================

INFO_GRAPH = {
    "Medicine Search": {
        "RxNorm Drug Identity": 1,
        "Official Drug Labels": 2,
        "Consumer Drug Information": 3
    },
    "RxNorm Drug Identity": {
        "Generic and Brand Concepts": 1,
        "Ingredients and Formulations": 2,
        "Medicine Search": 1
    },
    "Generic and Brand Concepts": {
        "Ingredients and Formulations": 1,
        "Official Drug Labels": 3
    },
    "Ingredients and Formulations": {
        "Official Drug Labels": 2,
        "Consumer Drug Information": 3
    },
    "Official Drug Labels": {
        "Warnings and Precautions": 1,
        "Ingredients and Formulations": 2,
        "Consumer Drug Information": 4
    },
    "Consumer Drug Information": {
        "Warnings and Precautions": 2,
        "Official Drug Labels": 4
    },
    "Warnings and Precautions": {}
}

# Heuristics estimate remaining graph-navigation effort.
# They are educational estimates, not clinical risk scores.
INFO_HEURISTICS = {
    "Medicine Search": {
        "Medicine Search": 0,
        "RxNorm Drug Identity": 1,
        "Generic and Brand Concepts": 2,
        "Ingredients and Formulations": 2,
        "Official Drug Labels": 1,
        "Consumer Drug Information": 1,
        "Warnings and Precautions": 0
    }
}


def validate_graph(graph):
    """Check graph endpoints and ensure edge costs are positive."""
    for node, edges in graph.items():
        if not isinstance(edges, dict):
            raise ValueError(f"Edges for {node} must be a dictionary.")

        for neighbor, cost in edges.items():
            if neighbor not in graph:
                raise ValueError(
                    f"Graph edge references unknown node: {neighbor}"
                )

            if not isinstance(cost, (int, float)) or cost <= 0:
                raise ValueError(
                    f"Edge {node} -> {neighbor} must have positive cost."
                )


validate_graph(INFO_GRAPH)


def reconstruct_path(parent, start, target):
    if target != start and target not in parent:
        return []

    path = [target]

    while path[-1] != start:
        path.append(parent[path[-1]])

    path.reverse()
    return path


def path_cost(graph, path):
    """Calculate total cost for a valid path."""
    if not path:
        return float("inf")

    total = 0

    for current_node, next_node in zip(path, path[1:]):
        if next_node not in graph[current_node]:
            raise ValueError("Path contains an invalid graph edge.")

        total += graph[current_node][next_node]

    return total


# ============================================================
# 1. BREADTH-FIRST SEARCH (BFS)
# Finds a path with the fewest edges in an unweighted view.
# It does not guarantee minimum weighted cost.
# ============================================================

def bfs(graph, start, target):
    if start not in graph or target not in graph:
        return {"path": [], "visited": [], "cost": float("inf")}

    queue = deque([start])
    visited = {start}
    parent = {}
    visit_order = []

    while queue:
        node = queue.popleft()
        visit_order.append(node)

        if node == target:
            break

        for neighbor in sorted(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    path = reconstruct_path(parent, start, target)

    return {
        "path": path,
        "visited": visit_order,
        "cost": path_cost(graph, path) if path else float("inf")
    }


# ============================================================
# 2. DEPTH-FIRST SEARCH (DFS)
# Finds a path by exploring deeply before backtracking.
# DFS does not guarantee the fewest edges or minimum cost.
# ============================================================

def dfs(graph, start, target):
    if start not in graph or target not in graph:
        return {"path": [], "visited": [], "cost": float("inf")}

    stack = [start]
    visited = set()
    parent = {}
    visit_order = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        visit_order.append(node)

        if node == target:
            break

        for neighbor in sorted(graph[node], reverse=True):
            if neighbor not in visited:
                if neighbor not in parent:
                    parent[neighbor] = node
                stack.append(neighbor)

    path = reconstruct_path(parent, start, target)

    return {
        "path": path,
        "visited": visit_order,
        "cost": path_cost(graph, path) if path else float("inf")
    }


# ============================================================
# 3. UNIFORM COST SEARCH (UCS)
# Expands the lowest accumulated-cost path first.
# With positive edge costs, UCS finds a minimum-cost path.
# ============================================================

def uniform_cost_search(graph, start, target):
    if start not in graph or target not in graph:
        return {"path": [], "visited": [], "cost": float("inf")}

    frontier = [(0, start)]
    best_cost = {start: 0}
    parent = {}
    settled = set()
    visit_order = []

    while frontier:
        cost_so_far, node = heapq.heappop(frontier)

        if node in settled:
            continue

        if cost_so_far != best_cost.get(node, float("inf")):
            continue

        settled.add(node)
        visit_order.append(node)

        if node == target:
            break

        for neighbor, edge_cost in sorted(graph[node].items()):
            new_cost = cost_so_far + edge_cost

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = node
                heapq.heappush(frontier, (new_cost, neighbor))

    path = reconstruct_path(parent, start, target)

    return {
        "path": path,
        "visited": visit_order,
        "cost": best_cost.get(target, float("inf"))
        if path else float("inf")
    }


# ============================================================
# 4. GREEDY BEST-FIRST SEARCH
# Chooses the frontier node with the smallest heuristic.
# It can be fast but is not guaranteed to find the cheapest path.
# ============================================================

def greedy_best_first_search(
    graph, start, target, heuristic
):
    if start not in graph or target not in graph:
        return {"path": [], "visited": [], "cost": float("inf")}

    frontier = [(heuristic.get(start, 0), start)]
    discovered = {start}
    parent = {}
    visit_order = []

    while frontier:
        _, node = heapq.heappop(frontier)
        visit_order.append(node)

        if node == target:
            break

        for neighbor in sorted(graph[node]):
            if neighbor not in discovered:
                discovered.add(neighbor)
                parent[neighbor] = node
                heapq.heappush(
                    frontier,
                    (heuristic.get(neighbor, 0), neighbor)
                )

    path = reconstruct_path(parent, start, target)

    return {
        "path": path,
        "visited": visit_order,
        "cost": path_cost(graph, path) if path else float("inf")
    }


# ============================================================
# 5. A* SEARCH
# Prioritizes f(n) = g(n) + h(n).
# Here g is accumulated path cost and h is the heuristic.
# Optimality requires an admissible heuristic; consistency
# is a convenient sufficient condition for graph-search A*.
# ============================================================

def a_star_search(graph, start, target, heuristic):
    if start not in graph or target not in graph:
        return {"path": [], "visited": [], "cost": float("inf")}

    frontier = [(heuristic.get(start, 0), 0, start)]
    best_cost = {start: 0}
    parent = {}
    closed = set()
    visit_order = []

    while frontier:
        _, cost_so_far, node = heapq.heappop(frontier)

        if cost_so_far != best_cost.get(node, float("inf")):
            continue

        if node in closed:
            continue

        closed.add(node)
        visit_order.append(node)

        if node == target:
            break

        for neighbor, edge_cost in sorted(graph[node].items()):
            new_cost = cost_so_far + edge_cost

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = node

                estimated_total = (
                    new_cost + heuristic.get(neighbor, 0)
                )

                heapq.heappush(
                    frontier,
                    (estimated_total, new_cost, neighbor)
                )

                # Reopen a node if a better path is found.
                closed.discard(neighbor)

    path = reconstruct_path(parent, start, target)

    return {
        "path": path,
        "visited": visit_order,
        "cost": best_cost.get(target, float("inf"))
        if path else float("inf")
    }


# ============================================================
# ALGORITHM LAB
# ============================================================

def format_path(path):
    return " → ".join(path) if path else "No path found"


def run_algorithm_lab(start, target):
    heuristic = INFO_HEURISTICS["Medicine Search"]

    algorithms = {
        "BFS": lambda: bfs(INFO_GRAPH, start, target),
        "DFS": lambda: dfs(INFO_GRAPH, start, target),
        "Uniform Cost Search": lambda: uniform_cost_search(
            INFO_GRAPH, start, target
        ),
        "Greedy Best-First Search": lambda: greedy_best_first_search(
            INFO_GRAPH, start, target, heuristic
        ),
        "A* Search": lambda: a_star_search(
            INFO_GRAPH, start, target, heuristic
        )
    }

    results = {}

    for name, algorithm in algorithms.items():
        results[name] = algorithm()

    return results


# ============================================================
# USER INTERFACE: MEDICINE SEARCH
# ============================================================

st.header("1. Search for a Medicine")

st.write(
    "Enter a generic or brand name. Search results come from "
    "RxNorm's medicine terminology service when available."
)

with st.form("medicine_search_form"):
    medicine_query = st.text_input(
        "Medicine name",
        placeholder="e.g. paracetamol, ibuprofen, metformin"
    )

    submitted = st.form_submit_button(
        "Search Medicine",
        type="primary",
        use_container_width=True
    )

if submitted:
    if not medicine_query.strip():
        st.error("Enter a medicine name.")
    else:
        try:
            with st.spinner("Searching RxNorm and DailyMed..."):
                rx_results = search_rxnorm(medicine_query.strip())

            st.session_state["medicine_query"] = medicine_query.strip()
            st.session_state["rx_results"] = rx_results
            st.session_state["selected_rxcui"] = None

            try:
                daily_results = search_dailymed(medicine_query.strip())
                st.session_state["daily_results"] = daily_results
                st.session_state["daily_error"] = None
            except Exception as exc:
                st.session_state["daily_results"] = []
                st.session_state["daily_error"] = str(exc)

        except requests.RequestException as exc:
            st.error(
                "RxNorm could not be reached. Check your internet "
                f"connection and try again. Details: {exc}"
            )
        except (ValueError, KeyError) as exc:
            st.error(f"Could not process the RxNorm response: {exc}")


query = st.session_state.get("medicine_query")
rx_results = st.session_state.get("rx_results", [])

if query:
    st.subheader(f"Search results for: {query}")

    if not rx_results:
        st.info(
            "No RxNorm match was returned. Check the spelling or search "
            "for a generic name. You can also try the official reference "
            "links below."
        )
    else:
        choices = {
            (
                f"{item['name']} "
                f"(RxCUI: {item['rxcui']}, type: {item['tty'] or 'unspecified'})"
            ): item["rxcui"]
            for item in rx_results
        }

        selected_label = st.selectbox(
            "Choose a matching medicine concept",
            list(choices.keys())
        )

        selected_rxcui = choices[selected_label]

        if st.button("View Selected Concept"):
            st.session_state["selected_rxcui"] = selected_rxcui

        if st.session_state.get("selected_rxcui") == selected_rxcui:
            try:
                properties = get_rxnorm_properties(selected_rxcui)
                related = get_rxnorm_related(selected_rxcui)

                st.markdown("### Standardized medicine concept")

                c1, c2 = st.columns(2)
                c1.metric("RxNorm Concept ID", selected_rxcui)
                c2.write(
                    f"**Concept name:** "
                    f"{properties.get('name', 'Not returned')}"
                )

                st.write(
                    f"**Term type:** "
                    f"{properties.get('tty', 'Not returned')}"
                )

                st.caption(
                    "RxNorm identifies and standardizes drug concepts. "
                    "A concept match alone does not confirm that a product "
                    "is marketed in your country or suitable for you."
                )

                if related:
                    st.markdown("### Related RxNorm concepts")
                    related_df = pd.DataFrame(related)

                    st.dataframe(
                        related_df,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No related concepts were returned.")

            except requests.RequestException as exc:
                st.error(f"Could not load concept details: {exc}")

    st.markdown("### Authoritative reference links")

    st.markdown(
        f"- [Search RxNorm / RxNav]({rxnorm_search_url(query)})\n"
        f"- [Search DailyMed official drug labels]({dailymed_search_url(query)})\n"
        f"- [MedlinePlus drug information]({medlineplus_search_url(query)})"
    )

    daily_error = st.session_state.get("daily_error")

    if daily_error:
        st.warning(
            "The DailyMed API search could not be completed. Use its "
            "official search page above. Technical detail: " + daily_error
        )
    else:
        daily_results = st.session_state.get("daily_results", [])

        if daily_results:
            st.markdown("### DailyMed API results")

            for index, item in enumerate(daily_results, start=1):
                st.write(
                    f"{index}. **{item['name'] or 'Unnamed result'}**"
                    + (
                        f" — {item['manufacturer']}"
                        if item["manufacturer"] else ""
                    )
                )

                if item["setid"]:
                    label_url = (
                        "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
                        f"?setid={quote_plus(item['setid'])}"
                    )
                    st.markdown(f"[View label]({label_url})")

        else:
            st.info(
                "No DailyMed API results were returned. Try the official "
                "DailyMed search link above."
            )

st.divider()

# ============================================================
# USER INTERFACE: SEARCH ALGORITHM LAB
# ============================================================

st.header("2. Search Algorithm Demonstration")

st.write(
    "Explore how each algorithm navigates a small, explicit graph of "
    "medicine-information resources. These paths represent information "
    "navigation, not treatment choices or medical recommendations."
)

nodes = list(INFO_GRAPH.keys())

c1, c2 = st.columns(2)

with c1:
    start_node = st.selectbox(
        "Starting node",
        nodes,
        index=nodes.index("Medicine Search"),
        key="algo_start"
    )

with c2:
    target_node = st.selectbox(
        "Destination node",
        nodes,
        index=nodes.index("Warnings and Precautions"),
        key="algo_target"
    )

if st.button("Run All Five Algorithms", type="primary"):
    results = run_algorithm_lab(start_node, target_node)
    st.session_state["algorithm_results"] = results

results = st.session_state.get("algorithm_results")

if results:
    st.markdown("### Results")

    summary_rows = []

    for name, result in results.items():
        summary_rows.append({
            "Algorithm": name,
            "Path found": "Yes" if result["path"] else "No",
            "Path length (edges)": (
                len(result["path"]) - 1 if result["path"] else None
            ),
            "Total edge cost": (
                result["cost"] if result["path"] else None
            ),
            "Nodes visited": len(result["visited"])
        })

    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True
    )

    for name, result in results.items():
        with st.expander(name, expanded=(name == "A* Search")):
            if result["path"]:
                st.write("**Path:**", format_path(result["path"]))
                st.write("**Total path cost:**", result["cost"])
            else:
                st.write("No path exists between the selected nodes.")

            st.write("**Visit order:**")
            st.write(" → ".join(result["visited"]))

    st.markdown("### Algorithm comparison")

    chart_df = pd.DataFrame(summary_rows)
    chart_df = chart_df.dropna(subset=["Nodes visited"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=chart_df["Algorithm"],
        y=chart_df["Nodes visited"],
        name="Nodes visited"
    ))

    fig.update_layout(
        title="Search effort by algorithm",
        xaxis_title="Algorithm",
        yaxis_title="Number of nodes visited",
        xaxis_tickangle=-20
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("### How the algorithms differ")

with st.expander("BFS — Breadth-First Search"):
    st.write(
        "Uses a FIFO queue and explores the graph level by level. "
        "It finds a path with the fewest edges in an unweighted graph, "
        "but does not guarantee the lowest total edge cost."
    )

with st.expander("DFS — Depth-First Search"):
    st.write(
        "Uses a stack to explore one branch before backtracking. "
        "It may find a path quickly, but the path is not necessarily "
        "shortest or cheapest."
    )

with st.expander("Uniform Cost Search"):
    st.write(
        "Uses a priority queue ordered by accumulated path cost g(n). "
        "With positive edge costs, it returns an optimal minimum-cost "
        "path when a path exists."
    )

with st.expander("Greedy Best-First Search"):
    st.write(
        "Prioritizes the node with the smallest estimated remaining "
        "distance h(n). It may reach a goal quickly but is not guaranteed "
        "to find a minimum-cost path."
    )

with st.expander("A* Search"):
    st.write(
        "Prioritizes f(n) = g(n) + h(n), combining accumulated cost "
        "and estimated remaining cost. Its optimality depends on the "
        "heuristic and graph-search conditions."
    )

st.divider()

# ============================================================
# GRAPH VISUALIZATION
# ============================================================

st.header("3. Medicine Information Knowledge Graph")

graph_edges = []

for source, neighbors in INFO_GRAPH.items():
    for destination, cost in neighbors.items():
        graph_edges.append({
            "Source": source,
            "Destination": destination,
            "Cost": cost
        })

st.dataframe(
    pd.DataFrame(graph_edges),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "The graph and costs are an educational navigation example. "
    "They do not represent medicine effectiveness, clinical priority, "
    "safety, or treatment preference."
)

st.divider()

# ============================================================
# OPTIONAL GEMINI SUMMARY
# ============================================================

st.header("4. Optional AI Explanation")

st.write(
    "Optionally use Gemini to explain the selected RxNorm concept in "
    "plain language. This feature is disabled until you provide a key."
)

if st.button("Explain Selected Medicine Concept"):
    if (
        not GEMINI_API_KEY
        or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE"
    ):
        st.warning(
            "Add your Gemini API key to GEMINI_API_KEY near the top "
            "of this file to enable this optional feature."
        )
    elif not query or not rx_results:
        st.info("Search for a medicine first.")
    else:
        selected_id = st.session_state.get("selected_rxcui")

        if not selected_id:
            st.info("Select a medicine concept and click View Selected Concept first.")
        else:
            try:
                import google.generativeai as genai

                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL)

                selected_item = next(
                    (
                        item for item in rx_results
                        if item["rxcui"] == selected_id
                    ),
                    None
                )

                concept_name = (
                    selected_item["name"]
                    if selected_item else query
                )

                prompt = f"""
Explain this standardized medicine concept for an educational
medicine-information application.

Medicine concept: {concept_name}
RxNorm identifier: {selected_id}

Rules:
- Do not recommend starting, stopping, or changing medication.
- Do not provide a personalized dose or treatment plan.
- Do not invent indications, side effects, contraindications,
  interactions, or clinical facts.
- Explain that the user should consult the official drug label
  and a qualified clinician or pharmacist.
- Distinguish identification of a medicine concept from advice
  about whether someone should take the medicine.
- Keep the explanation concise and clearly state limitations.
"""

                with st.spinner("Generating explanation..."):
                    response = model.generate_content(prompt)

                st.write(response.text)

            except Exception as exc:
                st.error(
                    "Gemini could not generate an explanation. Check the "
                    "API key, model availability, SDK configuration, "
                    f"and network. Details: {exc}"
                )

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Data sources: U.S. National Library of Medicine RxNorm/RxNav, "
    "DailyMed, and MedlinePlus. Availability and returned records can "
    "vary. Always check the original source for current product details."
)

st.caption(
    "NLM data-use notice: This product uses publicly available data "
    "from the U.S. National Library of Medicine (NLM), National "
    "Institutes of Health, Department of Health and Human Services. "
    "NLM is not responsible for this product and does not endorse "
    "or recommend it."
)
