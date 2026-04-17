"""
ui.py
-----
Streamlit dashboard for the Stem Agent Evolution system.

Run with:
    streamlit run stem_agent/ui.py
"""

import json

import streamlit as st
from graphviz import Digraph

import config
from compiler import GraphCompiler
from evolution import Evolution
from utils import summarize_spec


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stem Agent Evolution",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def draw_graph(spec: dict) -> Digraph:
    """Render a graph spec as a Graphviz Digraph."""
    dot = Digraph(graph_attr={"rankdir": "LR"})
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#e8f4f8")

    for node in spec.get("nodes", []):
        dot.node(node["id"], node["purpose"])

    for source, dest in spec.get("edges", []):
        dot.edge(source, dest)

    return dot


def render_agent_card(agent: dict, col, is_best: bool = False) -> None:
    """Render a single agent card inside a Streamlit column."""
    with col:
        label = f"🏆 Best — {agent['origin']}" if is_best else agent["origin"]
        if is_best:
            st.success(f"**{label}**")
        else:
            st.info(f"**{label}**")

        st.metric("Fitness", f"{agent['fitness']:.2f}")
        st.graphviz_chart(draw_graph(agent["spec"]))

        with st.expander("📜 Graph Spec (JSON)"):
            st.json(agent["spec"])

        if agent.get("parents"):
            with st.expander("🧬 Parent Lineage"):
                for idx, parent in enumerate(agent["parents"]):
                    st.caption(f"Parent {idx + 1}")
                    st.graphviz_chart(draw_graph(parent))

        st.markdown("#### 🧠 Node Outputs")
        raw = agent.get("output", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}

        outputs: dict = raw.get("data", {})
        if not outputs:
            st.warning("No output captured for this agent.")
        else:
            for node_id, text in outputs.items():
                with st.expander(f"🔹 {node_id}"):
                    st.write(text)


# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Controls")
task = st.sidebar.text_area("Task", height=140, placeholder="Describe what you want the agent to solve…")
generations = st.sidebar.slider("Generations", min_value=1, max_value=10, value=4)
run_btn = st.sidebar.button("🚀 Run Evolution", use_container_width=True)

# ── Title ──────────────────────────────────────────────────────────────────────

st.title("🧬 Stem Agent Evolution Visualizer")
st.caption("An evolutionary meta-agent that breeds and battles reasoning graphs.")

# ── Main logic ─────────────────────────────────────────────────────────────────

if run_btn and task.strip():
    config.reset_llm_calls()

    with st.spinner("Initialising population…"):
        evo = Evolution(task)

    best = None

    # Evolve generation by generation, showing a progress bar
    progress = st.progress(0, text="Starting evolution…")

    for gen in range(generations):
        progress.progress((gen + 1) / generations, text=f"Generation {gen + 1} / {generations}")
        evo._run_generation()  # noqa: SLF001

    progress.empty()
    best = evo.population[0]

    # ── Per-generation view ────────────────────────────────────────────────────
    for gen_idx, population in enumerate(evo.history):
        st.divider()
        st.header(f"🧬 Generation {gen_idx}")

        top_fitness = max(a["fitness"] for a in population)
        cols = st.columns(len(population))

        for i, agent in enumerate(population):
            render_agent_card(agent, cols[i], is_best=(agent["fitness"] == top_fitness))

    # ── Fitness trend ──────────────────────────────────────────────────────────
    st.divider()
    st.header("📈 Fitness Trend")

    fitness_trend = [
        max(a["fitness"] for a in gen_pop)
        for gen_pop in evo.history
    ]
    st.line_chart(fitness_trend)

    # ── Evolution summary ──────────────────────────────────────────────────────
    st.divider()
    st.header("🧬 Evolution Summary")

    col_init, col_final = st.columns(2)

    with col_init:
        st.subheader("Initial Agent")
        st.graphviz_chart(draw_graph(evo.initial))
        st.code("\n".join(summarize_spec(evo.initial)), language=None)

    with col_final:
        st.subheader("Final Agent")
        st.graphviz_chart(draw_graph(best.spec))
        st.code("\n".join(summarize_spec(best.spec)), language=None)

    initial_summary = summarize_spec(evo.initial)
    final_summary = summarize_spec(best.spec)

    if initial_summary != final_summary:
        st.success("🎉 Agent evolved successfully!")
    else:
        st.warning("⚠️ No significant structural evolution detected.")

    # ── Final output ───────────────────────────────────────────────────────────
    st.divider()
    st.header("🏁 Final Execution Output")

    final_output = best.last_output or {}
    if isinstance(final_output, str):
        try:
            final_output = json.loads(final_output)
        except json.JSONDecodeError:
            pass

    st.json(final_output)

    # ── Stats ──────────────────────────────────────────────────────────────────
    st.divider()
    st.header("📊 Run Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("LLM Calls", config.LLM_CALLS)
    c2.metric("Generations", generations)
    c3.metric("Final Fitness", f"{best.fitness:.2f}")

    st.success("✅ Evolution complete!")

elif run_btn:
    st.warning("Please enter a task before running.")
