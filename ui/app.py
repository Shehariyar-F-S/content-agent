"""
ui/app.py — Streamlit frontend.

Calls the FastAPI backend and displays results in a clean,
interview-ready UI. Run with:

    streamlit run ui/app.py

Make sure the FastAPI backend is running first:
    uvicorn src.api:app --reload
"""

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Content Intelligence Agent",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Content Intelligence Agent")
st.caption(
    "Multi-agent AI pipeline · Enrichment · Analysis · Sentiment · Generation · Evaluation"
)
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    title = st.text_input(
        "Show title",
        placeholder="e.g. Turkish for Beginners, The Masked Singer Germany",
        value=st.session_state.get("prefill_title", ""),
    )
    synopsis = st.text_area(
        "Synopsis (optional — helps with accuracy)",
        placeholder="Brief description of the show...",
        height=100,
    )

with col2:
    st.markdown("**Example shows to try**")
    for example in [
        "Turkish for Beginners",
        "Dark",
        "How I Met Your Mother",
        "The Masked Singer Germany",
        "Babylon Berlin",
    ]:
        if st.button(example, use_container_width=True):
            st.session_state["prefill_title"] = example
            st.rerun()

run = st.button("Run pipeline", type="primary", disabled=not title)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run and title:
    with st.spinner(f"Running 5-agent pipeline for '{title}'..."):
        try:
            response = httpx.post(
                f"{API_URL}/analyse",
                json={"title": title, "synopsis": synopsis or None},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError:
            st.error(
                "Cannot connect to the API. "
                "Make sure the backend is running: `uvicorn src.api:app --reload`"
            )
            st.stop()
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    # Pipeline metadata
    st.success(
        f"Run complete in {data['latency_seconds']}s  ·  "
        f"Agents: {', '.join(data['completed_agents'])}  ·  "
        f"Run ID: {data['run_id'][:8]}..."
    )

    if data["errors"]:
        for err in data["errors"]:
            st.warning(f"Non-fatal error: {err}")

    st.divider()

    # ---------------------------------------------------------------------------
    # Evaluation summary — shown first, most important
    # ---------------------------------------------------------------------------
    ev = data.get("evaluation")
    if ev:
        st.subheader("Pipeline evaluation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall confidence", f"{ev['overall_confidence']:.0%}")
        m2.metric("Hallucination flags", len(ev["hallucination_flags"]))
        m3.metric("Low confidence agents", len(ev["low_confidence_agents"]))
        m4.metric("Est. tokens used", ev["total_tokens_used"])

        if ev["hallucination_flags"]:
            for flag in ev["hallucination_flags"]:
                st.warning(f"Hallucination flag: {flag}")

        if ev["low_confidence_agents"]:
            st.info(f"Low confidence: {', '.join(ev['low_confidence_agents'])}")

    st.divider()

    # ---------------------------------------------------------------------------
    # Four agent result columns
    # ---------------------------------------------------------------------------
    col_e, col_a = st.columns(2)
    col_s, col_g = st.columns(2)

    # Enrichment
    with col_e:
        st.subheader("Enrichment")
        e = data.get("enrichment")
        if e:
            st.caption(f"Confidence: {e['confidence']:.0%} · {e['snippets_count']} web snippets")
            for key, value in e["facts"].items():
                if value and value != "unknown" and value != [] and value != "[]":
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.warning("Enrichment data unavailable.")

    # Analysis
    with col_a:
        st.subheader("Analysis")
        a = data.get("analysis")
        if a:
            st.caption(f"Confidence: {a['confidence']:.0%}")
            st.markdown("**Genre**")
            st.write(" · ".join(a["genre_tags"]))
            st.markdown("**Mood**")
            st.write(" · ".join(a["mood_tags"]))
            st.markdown("**Audience**")
            st.write(" · ".join(a["audience_tags"]))
            if a["content_warnings"] != ["None"]:
                st.markdown("**Content warnings**")
                st.write(" · ".join(a["content_warnings"]))
        else:
            st.warning("Analysis data unavailable.")

    # Sentiment
    with col_s:
        st.subheader("Sentiment")
        s = data.get("sentiment")
        if s:
            st.caption(f"Confidence: {s['confidence']:.0%} · {s['source_count']} sources")
            score = s["score"]
            color = "green" if score >= 70 else "orange" if score >= 50 else "red"
            st.markdown(f"**Score:** :{color}[{score}/100]")
            st.progress(score / 100)
            st.markdown(f"**Summary:** {s['summary']}")
            st.markdown(f"**Positives:** {s['positives']}")
            st.markdown(f"**Criticisms:** {s['criticisms']}")
        else:
            st.warning("Sentiment data unavailable.")

    # Generation
    with col_g:
        st.subheader("Generated copy")
        g = data.get("generation")
        if g:
            st.caption(f"Confidence: {g['confidence']:.0%}")
            st.markdown("**SEO title**")
            st.code(g["seo_title"], language=None)
            st.markdown(f"**Meta description** ({len(g['meta_description'])} chars)")
            st.code(g["meta_description"], language=None)
            st.markdown("**Instagram post**")
            st.text_area("", g["instagram_post"], height=100, label_visibility="collapsed")
            st.markdown("**Twitter / X post**")
            st.text_area(
                "",
                g["twitter_post"],
                height=80,
                label_visibility="collapsed",
            )
            st.caption(f"{len(g['twitter_post'])} / 280 chars")
        else:
            st.warning("Generation data unavailable.")