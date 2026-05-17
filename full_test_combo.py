"""
Data Justice Assistent — Streamlit App
Samengevoegde versie van test-5.py + JasmijnPelle_bias_justice_personas notebook
Groq API (llama-3.3-70b-versatile) + LangGraph workflow voor persona generatie & validatie
"""

import streamlit as st
import json
import os
import time
import math
import re
import csv
from groq import Groq

# ─────────────────────────────────────────────
# PAGINA CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Data Justice Assistent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Archive_4 design
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

html, body, .stApp {
    background-color: #0B1220 !important;
    color: #F1F5F9 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
* { box-sizing: border-box; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1C2A40 !important;
    min-width: 280px !important;
    max-width: 280px !important;
}
section[data-testid="stSidebar"] > div:first-child { padding: 14px !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #8B9CB8 !important;
    font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    margin: 0 !important; padding: 0 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: #1a2438 !important;
    color: #8B9CB8 !important;
    border: 1px solid #253047 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 7px 12px !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #1f2d47 !important;
    color: #F1F5F9 !important;
    border-color: #3B7EF6 !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: #3B7EF6 !important;
    color: white !important;
    border-color: #3B7EF6 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1a2438 !important;
    border: 1px solid #253047 !important;
    color: #F1F5F9 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}

/* ── Topbar ── */
.topbar {
    background: #111827;
    border-bottom: 1px solid #1C2A40;
    padding: 0 24px;
    height: 64px;
    display: flex;
    align-items: center;
    gap: 14px;
    position: sticky; top: 0; z-index: 100;
}
.tb-logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: #1a4fa0; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0;
}
.tb-logo-text { font-family: 'DM Sans', sans-serif; font-size: 15px; font-weight: 600; color: #F1F5F9; }
.tb-divider { width: 1px; height: 32px; background: #1C2A40; margin: 0 8px; }
.tb-project-wrap { display: flex; flex-direction: column; }
.tb-project-name { font-family: 'Sora', sans-serif; font-size: 18px; font-weight: 700; color: #4a8af4; }
.tb-project-sub { font-size: 11px; color: #5a7090; display: flex; align-items: center; gap: 5px; }

/* ── Chat berichten ── */
.chat-ts { text-align: center; font-size: 11px; color: #3a5070; margin: 8px 0 12px 0; }
.msg-bot-wrap { display: flex; flex-direction: column; margin-bottom: 16px; }
.msg-user-wrap { display: flex; flex-direction: column; align-items: flex-end; margin-bottom: 16px; }
.msg-sender {
    font-size: 11px; color: #5B9BFF; font-weight: 600;
    margin-bottom: 5px; display: flex; align-items: center; gap: 6px;
}
.msg-sender-sub { font-size: 10px; color: #5a7090; font-weight: 400; margin-top: 1px; }
.bubble-bot {
    background: #151f33; border: 1px solid #1C2A40;
    border-radius: 10px 10px 10px 3px;
    padding: 12px 16px; max-width: 84%;
    font-size: 13px; line-height: 1.65; color: #F1F5F9;
}
.bubble-user {
    background: #3B7EF6; border-radius: 10px 10px 3px 10px;
    padding: 11px 15px; max-width: 72%;
    font-size: 13px; line-height: 1.6; color: white;
}
.user-avatar {
    width: 32px; height: 32px; border-radius: 50%; background: #3B7EF6;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: white;
    margin-left: 8px; flex-shrink: 0;
}

/* ── Persona kaart ── */
.pc-card {
    background: #151f33; border: 1px solid #1C2A40;
    border-radius: 12px; padding: 16px; max-width: 860px; margin-top: 8px;
}
.pc-header { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 14px; }
.pc-photo {
    width: 80px; height: 80px; border-radius: 10px;
    object-fit: cover; flex-shrink: 0; background: #253047;
}
.pc-main { flex: 1; min-width: 0; }
.pc-name { font-family: 'Sora', sans-serif; font-size: 17px; font-weight: 700; color: #F1F5F9; }
.pc-diag { font-size: 11px; color: #1DB87A; margin: 2px 0 6px; }
.pc-quote { font-size: 12px; color: #8B9CB8; font-style: italic; line-height: 1.55; margin-bottom: 9px; }
.pc-tags { display: flex; flex-wrap: wrap; gap: 5px; }
.pc-tag {
    background: #1a2438; border: 1px solid #253047; color: #8B9CB8;
    border-radius: 20px; padding: 2px 10px; font-size: 11px;
}
.pc-age {
    background: rgba(59,126,246,.18); color: #5B9BFF;
    border-radius: 20px; padding: 3px 14px; font-size: 11px; font-weight: 700;
}
.pc-details {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    padding-top: 12px; border-top: 1px solid #1C2A40;
}
.pc-det-row { display: flex; align-items: flex-start; gap: 7px; }
.pc-det-label { color: #5a7090; font-size: 11px; min-width: 74px; flex-shrink: 0; }
.pc-det-val { color: #F1F5F9; font-size: 11px; line-height: 1.4; }

/* ── XAI Box ── */
.xai-box {
    background: #0f1929; border: 1px solid #1C2A40;
    border-radius: 10px; padding: 13px 16px; max-width: 860px; margin-top: 6px;
}
.xai-title { font-size: 12px; font-weight: 600; color: #F1F5F9; margin-bottom: 10px; }
.xai-accent { color: #5B9BFF; }
.xai-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.xai-col-label { font-size: 10px; color: #5a7090; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }
.xai-source { font-size: 11px; color: #F1F5F9; padding: 2px 0; }
.xai-source::before { content: "• "; color: #3B7EF6; }
.xai-badge {
    display: inline-block; background: rgba(59,126,246,.15);
    color: #5B9BFF; border-radius: 12px; padding: 2px 9px;
    font-size: 10px; margin: 2px 2px 2px 0;
}
.xai-expl { font-size: 11px; color: #8B9CB8; line-height: 1.5; }

/* ── Validation scores badge ── */
.xai-score-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; }
.xai-score-item { text-align: center; }
.xai-score-val { font-size: 18px; font-weight: 700; }
.xai-score-lbl { font-size: 10px; color: #5a7090; margin-top: 2px; }
.xai-score-toel { font-size: 10px; color: #8B9CB8; font-style: italic; margin-top: 3px; line-height: 1.4; }

/* ── Reliability bar ── */
.rel-bar {
    background: #111827; border-top: 1px solid #1C2A40;
    padding: 9px 24px; display: flex; align-items: center;
    gap: 10px; flex-wrap: wrap;
}
.rel-label { font-size: 11px; color: #5a7090; }
.rel-badge {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 20px; padding: 5px 14px;
    font-size: 11px; font-weight: 600; cursor: pointer;
}
.rel-amber { background: rgba(245,158,11,.13); color: #F59E0B; border: 1px solid rgba(245,158,11,.3); }
.rel-green { background: rgba(29,184,122,.1); color: #1DB87A; border: 1px solid rgba(29,184,122,.3); }
.rel-blue  { background: rgba(59,126,246,.1); color: #5B9BFF; border: 1px solid rgba(59,126,246,.25); }
.rel-red   { background: rgba(239,68,68,.12); color: #EF4444; border: 1px solid rgba(239,68,68,.3); }

/* ── Input ── */
.stTextInput input {
    background: #111827 !important; border: 1px solid #253047 !important;
    border-radius: 10px !important; color: #F1F5F9 !important;
    font-size: 13px !important; font-family: 'DM Sans', sans-serif !important;
    padding: 10px 16px !important; height: 44px !important;
}
.stTextInput input:focus { border-color: #3B7EF6 !important; box-shadow: none !important; }
.stTextInput input::placeholder { color: #3a5070 !important; }
.stTextInput label { color: #5a7090 !important; font-size: 11px !important; }
div[data-testid="stButton"] > button[kind="primary"] {
    background: #3B7EF6 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 13px !important;
    height: 44px !important; transition: background 0.15s !important;
}

/* ── Sidebar persona items ── */
.sb-persona-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 8px; border: 1px solid transparent;
    margin-bottom: 4px; cursor: pointer; transition: all 0.15s;
}
.sb-persona-item:hover { background: rgba(59,126,246,.07); border-color: rgba(59,126,246,.2); }
.sb-persona-item.active { background: rgba(59,126,246,.13); border-color: rgba(59,126,246,.4); }
.sb-persona-avatar {
    width: 36px; height: 36px; border-radius: 8px;
    background: #1f2d47; border: 1px solid #253047;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0; overflow: hidden;
}
.sb-persona-name { font-size: 12px; font-weight: 600; color: #F1F5F9; }
.sb-persona-meta { font-size: 10px; color: #5a7090; }
.sb-persona-age { background: rgba(59,126,246,.15); color: #5B9BFF; border-radius: 20px; padding: 2px 8px; font-size: 10px; font-weight: 700; }

/* ── Sidebar expanded persona ── */
.sb-persona-expanded {
    background: #1a2438; border: 1px solid #253047;
    border-radius: 10px; padding: 12px; margin-bottom: 6px;
}
.sb-persona-exp-name { font-size: 12px; font-weight: 600; color: #F1F5F9; }
.sb-persona-exp-diag { font-size: 10px; color: #1DB87A; }
.sb-persona-exp-tags { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
.sb-persona-exp-tag {
    background: rgba(59,126,246,.15); color: #5B9BFF;
    border-radius: 20px; padding: 2px 8px; font-size: 10px;
}
.sb-persona-exp-quote { font-size: 11px; color: #8B9CB8; font-style: italic; line-height: 1.5; padding-top: 7px; border-top: 1px solid #253047; }

/* ── Validation Panel ── */
.val-panel {
    background: #111827; border: 1px solid #1C2A40;
    border-radius: 12px; padding: 16px; width: 100%;
}
.val-panel-title { font-family: 'Sora', sans-serif; font-size: 14px; font-weight: 700; color: #4a8af4; margin-bottom: 14px; }
.val-check-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1C2A40; font-size: 12px; }
.val-check-label { color: #8B9CB8; }
.val-check-ok { color: #1DB87A; font-size: 11px; cursor: pointer; }
.val-check-warn { color: #F59E0B; font-size: 11px; cursor: pointer; }
.val-check-med { color: #EF4444; font-size: 11px; cursor: pointer; }
.score-ring-wrap { display: flex; align-items: center; gap: 14px; margin: 14px 0; padding: 12px; background: #151f33; border-radius: 10px; border: 1px solid #1C2A40; }
.score-legend { display: flex; flex-direction: column; gap: 4px; }
.score-legend-good { font-size: 12px; font-weight: 600; color: #1DB87A; }
.score-legend-improve { font-size: 11px; color: #8B9CB8; }

/* ── Full report ── */
.fr-section { margin-bottom: 14px; }
.fr-cat-title { font-family: 'Sora', sans-serif; font-size: 13px; font-weight: 700; color: #F1F5F9; margin-bottom: 8px; display: flex; justify-content: space-between; }
.fr-pct { font-size: 12px; color: #8B9CB8; }
.fr-bar-bg { background: #1C2A40; border-radius: 4px; height: 6px; margin-bottom: 8px; overflow: hidden; }
.fr-bar-fill { height: 6px; border-radius: 4px; }
.fr-item { display: flex; align-items: center; gap: 7px; background: #151f33; border: 1px solid #1C2A40; border-radius: 7px; padding: 8px 10px; font-size: 11px; color: #F1F5F9; margin-bottom: 5px; }

/* ── Metric cards ── */
.metric-card {
    background: #151f33; border: 1px solid #1C2A40;
    border-radius: 10px; padding: 12px 14px; text-align: center;
}
.metric-val { font-size: 22px; font-weight: 700; margin-bottom: 2px; }
.metric-lbl { font-size: 10px; color: #8B9CB8; text-transform: uppercase; letter-spacing: .5px; }

/* ── Section labels ── */
.sb-section-title { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; font-weight: 600 !important; color: #F1F5F9 !important; margin-bottom: 2px !important; }
.sb-section-sub { font-size: 10px; color: #3B7EF6; margin-bottom: 8px; display: block; }
.sb-divider { border: none; border-top: 1px solid #1C2A40; margin: 10px 0; }
.sb-footer { display: flex; align-items: center; gap: 9px; padding-top: 12px; border-top: 1px solid #1C2A40; margin-top: auto; }
.sb-designer-avatar { width: 30px; height: 30px; border-radius: 50%; background: #3B7EF6; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: white; flex-shrink: 0; }
.sb-designer-name { font-size: 12px; color: #F1F5F9; font-weight: 500; }
.sb-designer-role { font-size: 10px; color: #5a7090; }

/* ── Spinner / status ── */
.status-box {
    background: #1a2438; border: 1px solid #253047; border-radius: 10px;
    padding: 14px 16px; margin: 8px 0; font-size: 12px; color: #8B9CB8;
}
.status-step { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.status-icon-ok { color: #1DB87A; }
.status-icon-run { color: #3B7EF6; }
.status-icon-wait { color: #5a7090; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #253047; border-radius: 4px; }

/* ── Topbar pull-up ── */
div[data-testid="stMainBlockContainer"] > div > div > div[data-testid="stVerticalBlock"]
> div[data-testid="stHorizontalBlock"]:first-of-type {
    background: #111827 !important;
    border-bottom: 1px solid #1C2A40 !important;
    margin-top: -56px !important;
    padding: 0 24px 0 0 !important;
    height: 56px !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 8px !important;
    position: relative !important;
    z-index: 20 !important;
}
div[data-testid="stMainBlockContainer"] > div > div > div[data-testid="stVerticalBlock"]
> div[data-testid="stHorizontalBlock"]:first-of-type
> div[data-testid="column"]:first-child {
    flex: 0 0 0 !important; min-width: 0 !important; overflow: hidden !important; padding: 0 !important;
}
div[data-testid="stMainBlockContainer"] > div > div > div[data-testid="stVerticalBlock"]
> div[data-testid="stHorizontalBlock"]:first-of-type button {
    background: #111827 !important; border: 1px solid #253047 !important;
    border-radius: 8px !important; color: #8B9CB8 !important;
    font-size: 12px !important; font-family: 'DM Sans', sans-serif !important;
    padding: 6px 13px !important; height: 34px !important;
    white-space: nowrap !important; font-weight: 400 !important;
    transition: all 0.15s !important;
}
div[data-testid="stMainBlockContainer"] > div > div > div[data-testid="stVerticalBlock"]
> div[data-testid="stHorizontalBlock"]:first-of-type button[kind="primary"] {
    background: #3B7EF6 !important; border-color: #3B7EF6 !important;
    color: white !important; font-weight: 600 !important;
}
hr { border-color: #1C2A40 !important; margin: 8px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    """Haal Groq client op via secrets of environment."""
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return Groq(api_key=api_key)
    return None

groq_client = get_groq_client()


# ─────────────────────────────────────────────
# LANGGRAPH — PERSONA GENERATIE & VALIDATIE
# (uit JasmijnPelle_bias_justice_personas notebook)
# ─────────────────────────────────────────────
def genereer_personas_met_groq(client: Groq, opdracht: str) -> list[dict]:
    """
    NODE 1 (notebook cel 7): Genereer synthetic persona's via Groq.
    Geeft een lijst van persona dicts terug.
    """
    prompt = f"""Je bent een expert in het ontwerpen van inclusieve UX-persona's.
Genereer persona's op basis van deze opdracht:

{opdracht}

Regels:
- Elke persona is uniek en specifiek (geen vage algemeenheden)
- Gevarieerd in leeftijd, geslacht en culturele achtergrond
- Uitdagingen zijn concreet en verifieerbaar
- Geen stereotypen of vooroordelen

Geef ALLEEN een geldig JSON object terug, geen uitleg:
{{"personas": [
  {{
    "naam": "Volledige naam",
    "doelgroep": "doelgroep omschrijving",
    "leeftijd": 30,
    "achtergrond": "2-3 zinnen over wie deze persoon is",
    "uitdagingen": ["uitdaging 1", "uitdaging 2"],
    "gedrag": ["gedragskenmerk 1", "gedragskenmerk 2"]
  }}
]}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'```$', '', raw).strip()
    data = json.loads(raw)
    return data.get("personas", [])


def valideer_en_analyseer_persona(client: Groq, persona_tekst: str) -> dict:
    """
    NODE 1+2 (notebook cel 10 parse_persona + valideer_persona):
    Analyseert een persona tekst op bias, hallucinaties en inclusie.
    Geeft een schema dict terug conform PersonaSchema uit het notebook.
    """
    prompt = f"""### Wie ben je
Jij bent een Bias-Justice validator gespecialiseerd in het beoordelen van synthetische
UX-persona's. Je analyseert persona's op drie criteria en geeft eerlijke, onderbouwde scores.

Je werkt altijd systematisch:
- Je beoordeelt BIAS: zijn er stereotypen, vooroordelen of eenzijdige aannames?
- Je beoordeelt HALLUCINATIES: zijn de beschreven kenmerken realistisch en verifieerbaar?
- Je beoordeelt INCLUSIE: vertegenwoordigt de persona diverse perspectieven respectvol?

Scoreschaal:
- 0–70  = problematisch (rood)
- 71–80 = aandacht nodig (geel)
- 81–100 = goed (groen)

### De persona
{persona_tekst}

### Jouw taak
Analyseer de persona en geef:
1. Een neutrale samenvatting van wie deze persona is
2. Een lijst van geïdentificeerde kenmerken
3. Een score + toelichting voor bias, hallucinaties en inclusie
4. Een totaalscore (gemiddelde van de drie, afgerond)

### Regels
- Wees specifiek in je toelichting — geen vage uitspraken
- Geef altijd een concrete reden voor de score
- Wees eerlijk: ook goed geschreven persona's kunnen bias bevatten

Geef ALLEEN een geldig JSON object terug:
{{
  "naam": "naam van de persona",
  "samenvatting": "neutrale beschrijving",
  "kenmerken": ["kenmerk 1", "kenmerk 2"],
  "bias":          {{"score": 0, "toelichting": "..."}},
  "hallucinaties": {{"score": 0, "toelichting": "..."}},
  "inclusie":      {{"score": 0, "toelichting": "..."}},
  "totaalscore":   0
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'```$', '', raw).strip()
    return json.loads(raw)


def valideer_schema(schema: dict) -> list[str]:
    """
    NODE 2 (notebook valideer_persona):
    Controleert of het schema alle verplichte velden heeft.
    """
    fouten = []
    if not schema.get("naam"):
        fouten.append("Naam ontbreekt")
    if not schema.get("samenvatting") or len(schema["samenvatting"]) < 10:
        fouten.append("Samenvatting te kort")
    if not schema.get("kenmerken") or len(schema["kenmerken"]) < 2:
        fouten.append("Minder dan 2 kenmerken")
    for check in ["bias", "hallucinaties", "inclusie"]:
        if not schema.get(check) or schema[check].get("score") is None:
            fouten.append(f"Score ontbreekt voor {check}")
        if not schema.get(check) or not schema[check].get("toelichting"):
            fouten.append(f"Toelichting ontbreekt voor {check}")
    if schema.get("totaalscore") is None:
        fouten.append("Totaalscore ontbreekt")
    return fouten


def repareer_schema(client: Groq, schema: dict, fouten: list, persona_tekst: str) -> dict:
    """
    NODE 3 (notebook repareer_persona):
    Repareert een onvolledig schema met max 2 pogingen.
    """
    prompt = f"""Het persona-schema heeft fouten. Verbeter het als Bias-Justice validator.
Huidig schema: {json.dumps(schema, ensure_ascii=False)}
Fouten: {", ".join(fouten)}
Originele tekst: {persona_tekst}

Geef ALLEEN een verbeterd JSON object terug:
{{
  "naam": "...",
  "samenvatting": "...",
  "kenmerken": ["..."],
  "bias":          {{"score": 0-100, "toelichting": "..."}},
  "hallucinaties": {{"score": 0-100, "toelichting": "..."}},
  "inclusie":      {{"score": 0-100, "toelichting": "..."}},
  "totaalscore":   0-100
}}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'```$', '', raw).strip()
    return json.loads(raw)


def persona_naar_tekst(p: dict) -> str:
    """Zet een gegenereerde persona dict om naar leesbare tekst voor validatie."""
    uitdagingen = "\n".join(f"- {u}" for u in p.get("uitdagingen", []))
    gedrag = "\n".join(f"- {g}" for g in p.get("gedrag", []))
    return (
        f"{p['naam']}, {p['leeftijd']} jaar, {p['doelgroep']}\n"
        f"{p['achtergrond']}\n"
        f"Uitdagingen:\n{uitdagingen}\n"
        f"Gedrag:\n{gedrag}"
    )


def run_langgraph_workflow(client: Groq, persona_tekst: str, naam: str, status_placeholder) -> dict:
    """
    Volledige LangGraph workflow (notebook cel 10):
    parse → valideer → (repareer →) exporteer
    Geeft het finale schema terug.
    """
    pogingen = 0

    status_placeholder.markdown(f"""
    <div class="status-box">
      <div class="status-step"><span class="status-icon-run">▶</span> Node 1: Parsen & analyseren van <strong>{naam}</strong>...</div>
    </div>""", unsafe_allow_html=True)

    try:
        schema = valideer_en_analyseer_persona(client, persona_tekst)
    except Exception as e:
        return {"naam": naam, "fout": str(e), "totaalscore": 0,
                "bias": {"score": 0, "toelichting": "Parse fout"},
                "hallucinaties": {"score": 0, "toelichting": "Parse fout"},
                "inclusie": {"score": 0, "toelichting": "Parse fout"},
                "samenvatting": "Fout bij verwerking", "kenmerken": []}

    while True:
        fouten = valideer_schema(schema)

        if not fouten:
            status_placeholder.markdown(f"""
            <div class="status-box">
              <div class="status-step"><span class="status-icon-ok">✓</span> Node 2: Validatie geslaagd</div>
              <div class="status-step"><span class="status-icon-ok">✓</span> Node 4: Exporteren...</div>
            </div>""", unsafe_allow_html=True)
            break

        if pogingen >= 2:
            status_placeholder.markdown(f"""
            <div class="status-box">
              <div class="status-step"><span class="status-icon-run">⚠</span> Node 2: Validatie met {len(fouten)} opmerkingen (max pogingen bereikt)</div>
            </div>""", unsafe_allow_html=True)
            break

        pogingen += 1
        status_placeholder.markdown(f"""
        <div class="status-box">
          <div class="status-step"><span class="status-icon-run">▶</span> Node 3: Repareren (poging {pogingen})... fouten: {", ".join(fouten[:2])}</div>
        </div>""", unsafe_allow_html=True)

        try:
            schema = repareer_schema(client, schema, fouten, persona_tekst)
        except Exception:
            break

    schema["reparatie_pogingen"] = pogingen
    return schema


# ─────────────────────────────────────────────
# VASTE PERSONA DATA (uit test-5.py)
# ─────────────────────────────────────────────
VASTE_PERSONAS = [
    {
        "id": 1,
        "naam": "Fatima Al-Hassan",
        "leeftijd": 54,
        "leeftijd_label": "54jr",
        "geslacht": "Vrouw",
        "diagnose": "Reumatoïde artritis",
        "quote": "I want to keep living my life, but the fatigue makes it hard to plan or be consistent.",
        "tags": ["Fatigue", "Pijnbeheer", "Zelfstandigheid", "Werk-leven balans"],
        "context": "Werkt parttime, woont samen met partner, Amsterdam-West",
        "goals": "Actief blijven, onafhankelijkheid bewaren",
        "frustrations": "Onvoorspelbare vermoeidheid, kleine knoppen en lage contrasten op digitale platforms",
        "tech_support": "Gemiddeld",
        "data_bronnen": ["ReumaNederland Interviews n=65", "Patiëntforum data", "Wetenschappelijke literatuur"],
        "sleutelfactoren": ["Diagnose", "Leefstijl", "Emotionele impact"],
        "xai_uitleg": "Gegenereerd op basis van patronen in de dataset. Fatigue wordt vaak als uitdaging genoemd.",
        "kwaliteit": "goed",
        "bias": {"score": 88, "toelichting": "Geen stereotype; leeftijd en cultuur zijn contextueel verankerd."},
        "hallucinaties": {"score": 82, "toelichting": "Ervaringen zijn consistent met medische literatuur over RA."},
        "inclusie": {"score": 85, "toelichting": "Representeert multiculturele, vrouwelijke oudere patiënt goed."},
        "totaalscore": 85,
    },
    {
        "id": 2,
        "naam": "Sanne de Boer",
        "leeftijd": 18,
        "leeftijd_label": "18jr",
        "geslacht": "Vrouw",
        "diagnose": "Juveniele artritis",
        "quote": "School en sport zijn mijn leven. De pijn maakt dat moeilijk, maar ik geef niet op.",
        "tags": ["Jong", "School", "Sport", "Vriendschappen"],
        "context": "Scholier, woont bij ouders, actief sociaal leven",
        "goals": "Normaal leven leiden, studie afmaken",
        "frustrations": "Niet begrepen worden door leeftijdsgenoten, activiteiten missen",
        "tech_support": "Hoog",
        "data_bronnen": ["ReumaNederland Interviews n=65", "Jongerenonderzoek"],
        "sleutelfactoren": ["Leeftijd", "Sociale context"],
        "xai_uitleg": "Jong persona met focus op sociale inclusie en schoolprestaties.",
        "kwaliteit": "goed",
        "bias": {"score": 84, "toelichting": "Persona vermijdt stereotypen over jongeren met chronische ziekten."},
        "hallucinaties": {"score": 79, "toelichting": "Gedragspatronen zijn realistisch maar niet altijd brongebonden."},
        "inclusie": {"score": 82, "toelichting": "Vertegenwoordigt jongeren met JIA, een onderbelichte groep."},
        "totaalscore": 82,
    },
    {
        "id": 3,
        "naam": "Marcus van der Berg",
        "leeftijd": 23,
        "leeftijd_label": "23jr",
        "geslacht": "Man",
        "diagnose": "Artritis psoriatica",
        "quote": "Ik wil werken en carrière maken, maar de ziekte gooit roet in het eten.",
        "tags": ["Werk", "Carrière", "Jonge professional", "Zelfstandig"],
        "context": "Starter op arbeidsmarkt, eigen appartement Utrecht",
        "goals": "Carrière opbouwen, financieel onafhankelijk zijn",
        "frustrations": "Werkgever begrijpt het niet, onverwachte energie-dips",
        "tech_support": "Hoog",
        "data_bronnen": ["ReumaNederland survey 2024", "Arbeidsmarktonderzoek"],
        "sleutelfactoren": ["Werkdruk", "Leefstijl"],
        "xai_uitleg": "Jonge werkende met focus op professionele ontwikkeling.",
        "kwaliteit": "goed",
        "bias": {"score": 80, "toelichting": "Lichte neiging tot nadruk op carrière als primaire focus voor mannelijk persona."},
        "hallucinaties": {"score": 83, "toelichting": "Arbeidsmarktuitdagingen voor mensen met PsA zijn goed gedocumenteerd."},
        "inclusie": {"score": 78, "toelichting": "Vertegenwoordigt jonge mannelijke werknemer, voldoende divers."},
        "totaalscore": 80,
    },
    {
        "id": 4,
        "naam": "Linda Hartman",
        "leeftijd": 34,
        "leeftijd_label": "34jr",
        "geslacht": "Vrouw",
        "diagnose": "Reumatoïde artritis",
        "quote": "Met twee kinderen en een baan is er weinig ruimte voor ziekte. Ik moet vooruit.",
        "tags": ["Moeder", "Werk", "Gezin", "Multitasking"],
        "context": "Parttime werkend, twee jonge kinderen",
        "goals": "Goede moeder zijn, gezin draaiende houden",
        "frustrations": "Vermoeidheid bij gezinstaken, schuldgevoel",
        "tech_support": "Gemiddeld",
        "data_bronnen": ["ReumaNederland Interviews n=65", "Gezinsonderzoek"],
        "sleutelfactoren": ["Gezinscontext", "Werkdruk", "Vermoeidheid"],
        "xai_uitleg": "Persona gericht op het combineren van zorg en werk.",
        "kwaliteit": "goed",
        "bias": {"score": 76, "toelichting": "Schuldgevoel framing kan vrouwelijke stereotypen versterken."},
        "hallucinaties": {"score": 84, "toelichting": "Gezinscontext en RA-impact zijn realistisch beschreven."},
        "inclusie": {"score": 80, "toelichting": "Vertegenwoordigt werkende moeders met chronische ziekte adequaat."},
        "totaalscore": 80,
    },
]

PROJECTS = [
    {"id": 1, "naam": "Project 1", "personas": 2, "rating": 71},
    {"id": 2, "naam": "ReumaNederland", "personas": 4, "rating": 82},
    {"id": 3, "naam": "Project 3", "personas": 1, "rating": 65},
]


# ─────────────────────────────────────────────
# SCORE & BADGE HELPERS
# ─────────────────────────────────────────────
def score_kleur(score: int) -> str:
    if score >= 75: return "#1DB87A"
    elif score >= 50: return "#F59E0B"
    return "#EF4444"

def kleur_label(score: int) -> str:
    if score <= 70: return "🔴 ROOD"
    if score <= 80: return "🟡 GEEL"
    return "🟢 GROEN"

def badge_class(score: int) -> str:
    if score >= 75: return "rel-green"
    elif score >= 50: return "rel-amber"
    return "rel-red"

def badge_icon(score: int) -> str:
    if score >= 75: return "✓"
    elif score >= 50: return "⚠"
    return "✗"

def score_ring_svg(score: int, size: int = 100) -> str:
    r = size * 0.42
    cx = cy = size / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - score / 100)
    kleur = score_kleur(score)
    fs = size * 0.18
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1C2A40" stroke-width="{size*0.09}"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{kleur}"
              stroke-width="{size*0.09}" stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
              stroke-linecap="round" transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy+fs*0.38}" text-anchor="middle"
            font-family="Sora,sans-serif" font-size="{fs}" font-weight="700" fill="{kleur}">{score}%</text>
    </svg>"""


def render_persona_card(p: dict) -> str:
    tags_html = "".join([f'<span class="pc-tag">{t}</span>' for t in p.get("tags", [])])
    return f"""
    <div class="pc-card">
      <div class="pc-header">
        <img class="pc-photo" src="https://i.pravatar.cc/100?u={p.get('id',1)}" alt="foto"/>
        <div class="pc-main">
          <div class="pc-name">{p['naam']}</div>
          <div class="pc-diag">{p['diagnose']}</div>
          <div class="pc-quote">"{p.get('quote','')}"</div>
          <div class="pc-tags">{tags_html}</div>
        </div>
        <div class="pc-age">{p.get('leeftijd_label', str(p.get('leeftijd',''))+'jr')}</div>
      </div>
      <div class="pc-details">
        <div class="pc-det-row"><span class="pc-det-label">📍 Context</span><span class="pc-det-val">{p.get('context','—')}</span></div>
        <div class="pc-det-row"><span class="pc-det-label">🎯 Doelen</span><span class="pc-det-val">{p.get('goals','—')}</span></div>
        <div class="pc-det-row"><span class="pc-det-label">😤 Frustraties</span><span class="pc-det-val">{p.get('frustrations','—')}</span></div>
        <div class="pc-det-row"><span class="pc-det-label">💻 Tech</span><span class="pc-det-val">{p.get('tech_support','—')}</span></div>
      </div>
    </div>"""


def render_xai_box(p: dict) -> str:
    bronnen = "".join([f'<div class="xai-source">{b}</div>' for b in p.get("data_bronnen", [])])
    factoren = "".join([f'<span class="xai-badge">{f}</span>' for f in p.get("sleutelfactoren", [])])
    b = p.get("bias", {})
    h = p.get("hallucinaties", {})
    i = p.get("inclusie", {})
    totaal = p.get("totaalscore", 0)
    return f"""
    <div class="xai-box">
      <div class="xai-title">🔍 XAI — Hoe is dit persona gegenereerd?
        <span class="xai-accent" style="margin-left:6px">Bias-Justice Validator</span>
      </div>
      <div class="xai-grid">
        <div>
          <div class="xai-col-label">Top databronnen</div>
          {bronnen}
        </div>
        <div>
          <div class="xai-col-label">Sleutelfactoren</div>
          {factoren}
        </div>
        <div>
          <div class="xai-col-label">Uitleg</div>
          <div class="xai-expl">{p.get('xai_uitleg','—')}</div>
        </div>
      </div>
      <div class="xai-score-grid">
        <div class="xai-score-item">
          <div class="xai-score-val" style="color:{score_kleur(b.get('score',0))}">{b.get('score',0)}%</div>
          <div class="xai-score-lbl">Bias {kleur_label(b.get('score',0))}</div>
          <div class="xai-score-toel">{b.get('toelichting','')}</div>
        </div>
        <div class="xai-score-item">
          <div class="xai-score-val" style="color:{score_kleur(h.get('score',0))}">{h.get('score',0)}%</div>
          <div class="xai-score-lbl">Hallucinaties {kleur_label(h.get('score',0))}</div>
          <div class="xai-score-toel">{h.get('toelichting','')}</div>
        </div>
        <div class="xai-score-item">
          <div class="xai-score-val" style="color:{score_kleur(i.get('score',0))}">{i.get('score',0)}%</div>
          <div class="xai-score-lbl">Inclusie {kleur_label(i.get('score',0))}</div>
          <div class="xai-score-toel">{i.get('toelichting','')}</div>
        </div>
      </div>
    </div>"""


# ─────────────────────────────────────────────
# GROQ CHAT — SYNTHETIC USER (uit test-5.py)
# ─────────────────────────────────────────────
def bouw_systeem_prompt(p: dict) -> str:
    return f"""Je bent een cognitieve synthetic user agent genaamd "{p['naam']}".

PERSONA PROFIEL:
- Naam: {p['naam']}
- Leeftijd: {p['leeftijd']} jaar
- Geslacht: {p['geslacht']}
- Diagnose: {p['diagnose']}
- Kernquote: "{p.get('quote','')}"
- Dagelijkse context: {p.get('context','—')}
- Doelen: {p.get('goals','—')}
- Frustraties: {p.get('frustrations','—')}
- Tech-comfort: {p.get('tech_support','—')}
- Kernthema's: {", ".join(p.get('tags',[]))}

GEDRAGSREGELS:
1. Spreek ALTIJD in de eerste persoon als {p['naam']}
2. Reageer authentiek vanuit dit persona — niet als AI
3. Noem specifieke ervaringen passend bij de aandoening
4. Wees eerlijk over frustraties
5. Houd antwoorden beknopt (2-4 zinnen) tenzij anders gevraagd
6. Gebruik het taalregister passend bij dit persona

VERPLICHT NA ELK ANTWOORD — voeg op een aparte laatste regel exact toe:
SCORES:{{"bias": <0-100>, "hallucinaties": <0-100>, "inclusie": <0-100>}}
(100 = geen probleem; bias = vrijheid van stereotypen; hallucinaties = feitelijke gronding; inclusie = representativiteit)
Voorbeeld: SCORES:{{"bias": 88, "hallucinaties": 75, "inclusie": 82}}
"""


def vraag_groq(client: Groq, persona: dict, vraag: str, geschiedenis: list) -> dict:
    systeem = bouw_systeem_prompt(persona)
    berichten = [{"role": "system", "content": systeem}]
    for msg in geschiedenis[-16:]:
        berichten.append(msg)
    berichten.append({"role": "user", "content": vraag})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=berichten,
            temperature=0.75,
            max_tokens=600,
        )
        volledig = response.choices[0].message.content.strip()
        scores = {"bias": 75, "hallucinaties": 60, "inclusie": 70}
        match = re.search(r'SCORES:\s*(\{[^}]+\})', volledig)
        if match:
            try:
                parsed = json.loads(match.group(1))
                for k in ["bias", "hallucinaties", "inclusie"]:
                    if k in parsed and isinstance(parsed[k], (int, float)):
                        scores[k] = int(parsed[k])
            except Exception:
                pass
        schoon = re.sub(r'\nSCORES:\s*\{[^}]+\}', '', volledig).strip()
        schoon = re.sub(r'SCORES:\s*\{[^}]+\}', '', schoon).strip()
        schoon = schoon.replace("{", "&#123;").replace("}", "&#125;")
        totaal = round((scores["bias"] + scores["hallucinaties"] + scores["inclusie"]) / 3)
        return {"tekst": schoon, "scores": {**scores, "totaal": totaal}, "succes": True}
    except Exception as e:
        return {
            "tekst": f"❌ Groq API fout: {str(e)}",
            "scores": {"bias": 0, "hallucinaties": 0, "inclusie": 0, "totaal": 0},
            "succes": False,
        }


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "actieve_persona_id": 1,
    "actieve_persona_bron": "vast",   # "vast" of "gegenereerd"
    "personas": VASTE_PERSONAS,        # Gecombineerde lijst (vast + gegenereerd)
    "chatgeschiedenis": [],
    "api_berichten": [],
    "scores": {"bias": 83, "hallucinaties": 73, "inclusie": 77, "totaal": 76},
    "berichtentelling": 0,
    "sidebar_mode": "personas",
    "panel_mode": None,
    "mood_opties": {"Curious": True, "Confused": False, "Distrust": True, "Frustrated": False, "Hopeful": False, "Overstimulated": False},
    "assign_opties": {"Designer 1": True, "Designer 2": False, "Designer 3": True, "Developer 1": False, "Developer 2": False, "Researcher 1": False},
    "mood_open": False,
    "assign_open": False,
    "genereer_modus": False,
    "genereer_voortgang": [],
    "genereer_opdracht": "Genereer 3 reumapatiënten (gevarieerd in leeftijd, achtergrond en digitale vaardigheid) en 1 zorgpersoneel voor een digitaal zorgplatform. Zorg voor diversiteit en vermijd stereotypen.",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_actieve_persona():
    pid = st.session_state.actieve_persona_id
    return next((p for p in st.session_state.personas if p["id"] == pid), st.session_state.personas[0])


def wissel_persona(pid: int):
    if pid != st.session_state.actieve_persona_id:
        st.session_state.actieve_persona_id = pid
        st.session_state.chatgeschiedenis = []
        st.session_state.api_berichten = []
        st.session_state.berichtentelling = 0
        st.session_state.scores = {"bias": 83, "hallucinaties": 73, "inclusie": 77, "totaal": 76}


def reset_chat():
    st.session_state.chatgeschiedenis = []
    st.session_state.api_berichten = []
    st.session_state.berichtentelling = 0
    st.session_state.scores = {"bias": 83, "hallucinaties": 73, "inclusie": 77, "totaal": 76}


# ─────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────
actieve_p = get_actieve_persona()
scores = st.session_state.scores

st.markdown("""
<style>
div[data-testid="stMainBlockContainer"] > div > div > div[data-testid="stVerticalBlock"]
> div[data-testid="stHorizontalBlock"]:first-of-type {
    background: #111827 !important; border-bottom: 1px solid #1C2A40 !important;
    margin-top: -56px !important; padding: 0 24px 0 0 !important;
    height: 56px !important; align-items: center !important;
    justify-content: flex-end !important; gap: 8px !important;
    position: relative !important; z-index: 20 !important;
}
div[data-testid="stMainBlockContainer"] > div > div > div[data-testid="stVerticalBlock"]
> div[data-testid="stHorizontalBlock"]:first-of-type
> div[data-testid="column"]:first-child {
    flex: 0 0 0 !important; min-width: 0 !important; overflow: hidden !important; padding: 0 !important;
}
</style>""", unsafe_allow_html=True)

mood_arrow = "∧" if st.session_state.mood_open else "∨"
assign_arrow = "∧" if st.session_state.assign_open else "∨"

st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;gap:14px">
    <div class="tb-logo-icon">
      <svg width="20" height="20" viewBox="0 0 28 28" fill="none">
        <rect width="28" height="28" rx="6" fill="#1a4fa0"/>
        <path d="M5 8h18M5 12h14M5 16h18M5 20h14" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
      </svg>
    </div>
    <span class="tb-logo-text">Data Justice Assistent</span>
    <div class="tb-divider"></div>
    <div class="tb-project-wrap">
      <div class="tb-project-name">ReumaNederland Project</div>
      <div class="tb-project-sub">ReumaNederland · Team 3</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

_spacer, _col_mood, _col_assign, _col_export = st.columns([4.5, 1.3, 1.3, 0.7])
with _col_mood:
    if st.button(f"😊 Mood {mood_arrow}", key="tb_mood"):
        st.session_state.mood_open = not st.session_state.mood_open
        st.session_state.assign_open = False
        st.rerun()
with _col_assign:
    if st.button(f"📋 Assign {assign_arrow}", key="tb_assign"):
        st.session_state.assign_open = not st.session_state.assign_open
        st.session_state.mood_open = False
        st.rerun()
with _col_export:
    if st.button("↑ Export", key="tb_export", type="primary"):
        export_data = {
            "project": "ReumaNederland",
            "persona": {"naam": actieve_p["naam"], "leeftijd": actieve_p["leeftijd"], "diagnose": actieve_p["diagnose"]},
            "scores": st.session_state.scores,
            "chatgeschiedenis": [{"rol": m["rol"], "inhoud": m["inhoud"]} for m in st.session_state.chatgeschiedenis],
        }
        st.download_button(
            "⬇ JSON",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name=f"data_justice_{actieve_p['naam'].replace(' ', '_')}.json",
            mime="application/json",
            key="tb_download",
        )

if st.session_state.mood_open:
    with st.container():
        st.markdown("""<div style="background:#1a2438;border:1px solid #253047;border-radius:10px;padding:10px 16px 12px;margin-bottom:8px;">
          <div style="font-size:11px;font-weight:600;color:#8B9CB8;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px">😊 Mood selectie</div></div>""",
            unsafe_allow_html=True)
        mood_cols = st.columns(6)
        for i, (mood, checked) in enumerate(st.session_state.mood_opties.items()):
            with mood_cols[i]:
                st.session_state.mood_opties[mood] = st.checkbox(mood, value=checked, key=f"mood_{mood}")

if st.session_state.assign_open:
    with st.container():
        st.markdown("""<div style="background:#1a2438;border:1px solid #253047;border-radius:10px;padding:10px 16px 12px;margin-bottom:8px;">
          <div style="font-size:11px;font-weight:600;color:#8B9CB8;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px">📋 Team toewijzing</div></div>""",
            unsafe_allow_html=True)
        assign_cols = st.columns(6)
        for i, (name, checked) in enumerate(st.session_state.assign_opties.items()):
            with assign_cols[i]:
                st.session_state.assign_opties[name] = st.checkbox(name, value=checked, key=f"assign_{name}")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:

    if st.button("+ New Chat", key="new_chat", use_container_width=True):
        reset_chat()
        st.session_state.sidebar_mode = "personas"
        st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    # ── PERSONA GENERATOR TAB ──
    if st.button("🤖 Genereer nieuwe persona's", key="btn_genereer_tab", use_container_width=True):
        st.session_state.genereer_modus = not st.session_state.genereer_modus
        st.rerun()

    if st.session_state.genereer_modus:
        st.markdown("""<div style="background:#1a2438;border:1px solid #3B7EF6;border-radius:10px;padding:12px;margin:8px 0;">
          <div style="font-size:11px;font-weight:600;color:#5B9BFF;margin-bottom:8px">🤖 Persona Generator</div>
          <div style="font-size:10px;color:#5a7090;margin-bottom:8px">Gebaseerd op LangGraph workflow uit het notebook</div>
        </div>""", unsafe_allow_html=True)

        opdracht_input = st.text_area(
            "Opdracht voor de generator:",
            value=st.session_state.genereer_opdracht,
            height=120,
            key="genereer_opdracht_input",
            help="Beschrijf welke persona's je wilt genereren",
        )
        st.session_state.genereer_opdracht = opdracht_input

        if st.button("▶ Start LangGraph workflow", key="btn_start_genereer", type="primary", use_container_width=True):
            if not groq_client:
                st.error("⚠️ Voeg GROQ_API_KEY toe aan .streamlit/secrets.toml")
            else:
                status_ph = st.empty()
                status_ph.markdown('<div class="status-box"><div class="status-step"><span class="status-icon-run">⏳</span> Stap 1: Persona\'s genereren via Groq...</div></div>', unsafe_allow_html=True)

                try:
                    gegenereerde_raw = genereer_personas_met_groq(groq_client, opdracht_input)
                    nieuwe_personas = []
                    max_id = max(p["id"] for p in st.session_state.personas)

                    for i, p_raw in enumerate(gegenereerde_raw):
                        naam = p_raw.get("naam", f"Persona {i+1}")
                        tekst = persona_naar_tekst(p_raw)

                        status_ph.markdown(f'<div class="status-box"><div class="status-step"><span class="status-icon-run">▶</span> Stap 2: LangGraph validatie voor <strong>{naam}</strong>...</div></div>', unsafe_allow_html=True)

                        schema = run_langgraph_workflow(groq_client, tekst, naam, status_ph)

                        # Bouw volledige persona dict
                        max_id += 1
                        nieuw = {
                            "id": max_id,
                            "naam": schema.get("naam", naam),
                            "leeftijd": p_raw.get("leeftijd", 30),
                            "leeftijd_label": f"{p_raw.get('leeftijd',30)}jr",
                            "geslacht": "Onbekend",
                            "diagnose": p_raw.get("doelgroep", "—"),
                            "quote": p_raw.get("achtergrond", "")[:120],
                            "tags": p_raw.get("gedrag", [])[:4],
                            "context": p_raw.get("achtergrond", "—"),
                            "goals": " | ".join(p_raw.get("uitdagingen", [])[:2]),
                            "frustrations": " | ".join(p_raw.get("uitdagingen", [])[2:4]),
                            "tech_support": "Onbekend",
                            "data_bronnen": ["Groq LLM (llama-3.3-70b-versatile)", "LangGraph workflow"],
                            "sleutelfactoren": schema.get("kenmerken", [])[:3],
                            "xai_uitleg": schema.get("samenvatting", "Automatisch gegenereerd via LangGraph."),
                            "kwaliteit": "gegenereerd",
                            "bias": schema.get("bias", {"score": 0, "toelichting": "—"}),
                            "hallucinaties": schema.get("hallucinaties", {"score": 0, "toelichting": "—"}),
                            "inclusie": schema.get("inclusie", {"score": 0, "toelichting": "—"}),
                            "totaalscore": schema.get("totaalscore", 0),
                            "reparatie_pogingen": schema.get("reparatie_pogingen", 0),
                        }
                        nieuwe_personas.append(nieuw)

                    st.session_state.personas = VASTE_PERSONAS + nieuwe_personas
                    status_ph.markdown(f'<div class="status-box"><div class="status-step"><span class="status-icon-ok">✅</span> {len(nieuwe_personas)} persona\'s gegenereerd en gevalideerd!</div></div>', unsafe_allow_html=True)
                    st.session_state.genereer_modus = False
                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    status_ph.markdown(f'<div class="status-box"><div class="status-step"><span style="color:#EF4444">❌</span> Fout: {str(e)}</div></div>', unsafe_allow_html=True)

        if st.button("✕ Annuleer", key="btn_annuleer_genereer", use_container_width=True):
            st.session_state.genereer_modus = False
            st.rerun()

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    # ── PERSONA LIJST ──
    if st.session_state.sidebar_mode in ("personas", "persona_detail"):

        st.markdown('<div class="sb-section-title">Persona\'s</div><span class="sb-section-sub">Research · ReumaNederland</span>', unsafe_allow_html=True)

        for p in st.session_state.personas:
            is_actief = p["id"] == st.session_state.actieve_persona_id
            is_open = is_actief and st.session_state.sidebar_mode == "persona_detail"
            item_class = "sb-persona-item active" if is_actief else "sb-persona-item"
            kwaliteit_kleur = "#3B7EF6" if p.get("kwaliteit") == "gegenereerd" else "#1DB87A" if p.get("kwaliteit") == "goed" else "#F59E0B"
            label_extra = " 🤖" if p.get("kwaliteit") == "gegenereerd" else ""
            totaal = p.get("totaalscore", 0)

            col_main, col_arr = st.columns([10, 1])
            with col_main:
                st.markdown(f"""
                <div class="{item_class}">
                  <div class="sb-persona-avatar" style="{'background:rgba(59,126,246,.15)' if is_actief else ''}">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="7" r="4" stroke="{'#5B9BFF' if is_actief else '#5a7090'}" stroke-width="1.5"/>
                      <path d="M4 21c0-4.418 3.582-8 8-8s8 3.582 8 8" stroke="{'#5B9BFF' if is_actief else '#5a7090'}" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                  </div>
                  <div style="flex:1;min-width:0">
                    <div class="sb-persona-name">{p['naam']}{label_extra}</div>
                    <div class="sb-persona-meta">{p['diagnose'][:26]}{'…' if len(p['diagnose'])>26 else ''}</div>
                  </div>
                  <div>
                    <div class="sb-persona-age">{p.get('leeftijd_label','?')}</div>
                    <div style="font-size:10px;color:{score_kleur(totaal)};text-align:center;margin-top:2px">{totaal}%</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            with col_arr:
                pijl = "▼" if is_open else "▷"
                if st.button(pijl, key=f"sel_persona_{p['id']}", help=f"Selecteer {p['naam']}"):
                    if not is_actief:
                        wissel_persona(p["id"])
                        st.session_state.sidebar_mode = "personas"
                    else:
                        st.session_state.sidebar_mode = "personas" if is_open else "persona_detail"
                    st.rerun()

            if is_open:
                tags_html = "".join([f'<span class="sb-persona-exp-tag">{t}</span>' for t in p.get("tags", [])])
                b = p.get("bias", {})
                h = p.get("hallucinaties", {})
                i_sc = p.get("inclusie", {})
                st.markdown(f"""
                <div class="sb-persona-expanded">
                  <div style="display:flex;align-items:center;gap:9px;margin-bottom:8px">
                    <img style="width:40px;height:40px;border-radius:8px;object-fit:cover" src="https://i.pravatar.cc/80?u={p['id']}" alt=""/>
                    <div>
                      <div class="sb-persona-exp-name">{p['naam']}</div>
                      <div class="sb-persona-exp-diag">{p['diagnose']}</div>
                    </div>
                  </div>
                  <div class="sb-persona-exp-tags">{tags_html}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px">
                    <div style="text-align:center">
                      <div style="font-size:14px;font-weight:700;color:{score_kleur(b.get('score',0))}">{b.get('score',0)}%</div>
                      <div style="font-size:9px;color:#5a7090">Bias</div>
                    </div>
                    <div style="text-align:center">
                      <div style="font-size:14px;font-weight:700;color:{score_kleur(h.get('score',0))}">{h.get('score',0)}%</div>
                      <div style="font-size:9px;color:#5a7090">Hallucinaties</div>
                    </div>
                    <div style="text-align:center">
                      <div style="font-size:14px;font-weight:700;color:{score_kleur(i_sc.get('score',0))}">{i_sc.get('score',0)}%</div>
                      <div style="font-size:9px;color:#5a7090">Inclusie</div>
                    </div>
                  </div>
                  <div class="sb-persona-exp-quote">"{p.get('quote','—')}"</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)

    # Sidebar footer
    st.markdown("""
    <div class="sb-footer">
      <div class="sb-designer-avatar">JP</div>
      <div>
        <div class="sb-designer-name">Jasmijn</div>
        <div class="sb-designer-role">UX Designer · Team 3</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HOOFDLAYOUT — Chat + (optioneel) panel
# ─────────────────────────────────────────────
panel_open = st.session_state.panel_mode is not None
if panel_open:
    chat_col, panel_col = st.columns([3, 1])
else:
    chat_col, panel_col = st.container(), None


# ── CHAT COLUMN ──
with (chat_col if panel_open else st.container()):

    st.markdown("""
    <div id="chat-scroll" style="min-height:400px;padding:20px 24px 8px;
        background:
          radial-gradient(ellipse at 75% 30%, rgba(26,79,160,0.18) 0%, transparent 55%),
          radial-gradient(ellipse at 90% 80%, rgba(59,126,246,0.08) 0%, transparent 40%),
          #0B1220;">
    """, unsafe_allow_html=True)

    if not st.session_state.chatgeschiedenis:
        st.markdown(f"""
        <div class="chat-ts">Nu</div>
        <div class="msg-bot-wrap">
          <div class="msg-sender">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <rect x="3" y="3" width="18" height="14" rx="3" stroke="#5B9BFF" stroke-width="1.5"/>
              <path d="M8 17v4M16 17v4M6 21h12" stroke="#5B9BFF" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
            Data Justice Assistent
          </div>
          <div class="bubble-bot">
            Hallo! Ik ben de <strong>Data Justice Assistent</strong>.<br><br>
            Actieve synthetic user: <strong>{actieve_p['naam']}</strong>
            ({actieve_p['leeftijd']}jr · {actieve_p['diagnose']}).<br><br>
            Stel een vraag om de conversatie te starten. Elk antwoord wordt geëvalueerd op
            <strong>bias</strong>, <strong>hallucinaties</strong> en <strong>inclusie</strong>
            via de Bias-Justice validator uit het notebook.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        vorig_ts = None
        for msg in st.session_state.chatgeschiedenis:
            ts = msg.get("tijd", "")
            if ts != vorig_ts:
                st.markdown(f'<div class="chat-ts">{ts}</div>', unsafe_allow_html=True)
                vorig_ts = ts

            if msg["rol"] == "gebruiker":
                inhoud = str(msg["inhoud"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
                <div class="msg-user-wrap">
                  <div style="display:flex;align-items:flex-end;gap:8px;justify-content:flex-end">
                    <div class="bubble-user">{inhoud}</div>
                    <div class="user-avatar">DS</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            elif msg["rol"] == "assistent":
                inhoud = str(msg["inhoud"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                sub = '<div class="msg-sender-sub">Here is your Persona based on the dataset: Reuma_Nederland</div>' if msg.get("toon_persona_kaart") else ""
                st.markdown(f"""
                <div class="msg-bot-wrap">
                  <div class="msg-sender">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
                      <rect x="3" y="3" width="18" height="14" rx="3" stroke="#5B9BFF" stroke-width="1.5"/>
                      <path d="M8 17v4M16 17v4M6 21h12" stroke="#5B9BFF" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    Synthetic User · {actieve_p['naam']}
                  </div>
                  {sub}
                  <div class="bubble-bot">{inhoud}</div>
                </div>""", unsafe_allow_html=True)
                if msg.get("toon_persona_kaart"):
                    st.markdown(render_persona_card(actieve_p), unsafe_allow_html=True)
                    st.markdown(render_xai_box(actieve_p), unsafe_allow_html=True)

            elif msg["rol"] == "systeem":
                inhoud = str(msg["inhoud"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
                <div class="msg-bot-wrap">
                  <div class="msg-sender">⚖️ Data Justice Assistent</div>
                  <div class="bubble-bot">{inhoud}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input ──
    st.markdown("""
    <div style="padding:10px 0 0 0">
      <div style="font-size:11px;color:#5a7090;padding:0 0 4px 0">Ask the Synthetic User</div>
    </div>""", unsafe_allow_html=True)

    input_c1, input_c2, input_c3 = st.columns([0.04, 1, 0.12])
    with input_c1:
        st.markdown('<div style="font-size:22px;color:#3B7EF6;padding-top:8px;text-align:center">✦</div>', unsafe_allow_html=True)
    with input_c2:
        user_input = st.text_input(
            "vraag",
            placeholder="Ask the Synthetic User...",
            label_visibility="collapsed",
            key="chat_input_field",
        )
    with input_c3:
        send_clicked = st.button("➤ Send", key="send_btn", type="primary", use_container_width=True)

    # ── Reliability bar ──
    b_score = scores["bias"]
    h_score = scores["hallucinaties"]
    i_score = scores["inclusie"]
    h_lbl = f"Hallucination — {'Low' if h_score >= 75 else 'Medium' if h_score >= 50 else 'High'}"
    b_lbl = f"Bias — {'Low' if b_score >= 75 else 'Medium' if b_score >= 50 else 'High'}"
    i_lbl = f"Data Justice — {'Needs Review' if i_score < 75 else 'Good'}"

    st.markdown(f"""
    <div class="rel-bar">
      <span class="rel-label">ⓘ Reliability :</span>
      <span class="rel-badge {badge_class(h_score)}">{badge_icon(h_score)} {h_lbl}</span>
      <span class="rel-badge {badge_class(b_score)}">{badge_icon(b_score)} {b_lbl}</span>
      <span class="rel-badge {badge_class(i_score)}">{badge_icon(i_score)} {i_lbl}</span>
    </div>""", unsafe_allow_html=True)

    _rel_spacer, _rel_vfr = st.columns([4, 1])
    with _rel_vfr:
        if st.button("📊 View Full Report >>", key="btn_view_full_report"):
            st.session_state.panel_mode = "validation" if st.session_state.panel_mode != "validation" else None
            st.rerun()

    if st.session_state.berichtentelling > 0:
        with st.expander("📈 Score detail — laatste evaluatie", expanded=False):
            sc1, sc2, sc3, sc4 = st.columns(4)
            for col, val, naam, beschr in [
                (sc1, scores["bias"], "Bias", "Stereotypering"),
                (sc2, scores["hallucinaties"], "Hallucinaties", "Feitelijkheid"),
                (sc3, scores["inclusie"], "Inclusie", "Representativiteit"),
                (sc4, scores["totaal"], "Totaal", "Gecombineerd"),
            ]:
                with col:
                    kleur = score_kleur(val)
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-val" style="color:{kleur}">{val}%</div>
                      <div class="metric-lbl">{naam}</div>
                      <div style="font-size:9px;color:#8B9CB8;margin-top:2px">{beschr}</div>
                    </div>""", unsafe_allow_html=True)

        if st.button("📥 Download gesprek (JSON)", key="dl_chat"):
            export = {
                "project": "ReumaNederland",
                "persona": {"naam": actieve_p["naam"], "leeftijd": actieve_p["leeftijd"], "diagnose": actieve_p["diagnose"]},
                "scores": st.session_state.scores,
                "chatgeschiedenis": [{"rol": m["rol"], "inhoud": m["inhoud"]} for m in st.session_state.chatgeschiedenis],
            }
            st.download_button(
                label="⬇ Download",
                data=json.dumps(export, indent=2, ensure_ascii=False),
                file_name=f"data_justice_{actieve_p['naam'].replace(' ', '_')}.json",
                mime="application/json",
                key="dl_chat_btn",
            )


# ── VALIDATION / REPORT PANEL ──
if panel_open and panel_col is not None:
    with panel_col:
        if st.session_state.panel_mode == "validation":
            totaal = scores["totaal"]
            b = actieve_p.get("bias", {})
            h = actieve_p.get("hallucinaties", {})
            i_sc = actieve_p.get("inclusie", {})
            st.markdown(f"""
            <div class="val-panel">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                <span class="val-panel-title">Validation &amp; Risk overview</span>
              </div>
              <div class="val-check-row">
                <span class="val-check-label">Bias check</span>
                <span class="{'val-check-ok' if b.get('score',0)>=75 else 'val-check-warn' if b.get('score',0)>=50 else 'val-check-med'}">{b.get('score',0)}% {badge_icon(b.get('score',0))}</span>
              </div>
              <div class="val-check-row">
                <span class="val-check-label">Hallucination check</span>
                <span class="{'val-check-ok' if h.get('score',0)>=75 else 'val-check-warn' if h.get('score',0)>=50 else 'val-check-med'}">{h.get('score',0)}% {badge_icon(h.get('score',0))}</span>
              </div>
              <div class="val-check-row">
                <span class="val-check-label">Data Justice check</span>
                <span class="{'val-check-ok' if i_sc.get('score',0)>=75 else 'val-check-warn' if i_sc.get('score',0)>=50 else 'val-check-med'}">{i_sc.get('score',0)}% {badge_icon(i_sc.get('score',0))}</span>
              </div>
              <div class="score-ring-wrap">
                {score_ring_svg(actieve_p.get('totaalscore', totaal), 90)}
                <div class="score-legend">
                  <div class="score-legend-good">{'Good' if actieve_p.get('totaalscore',totaal)>=75 else 'Needs review'}</div>
                  <div class="score-legend-improve">Persona kwaliteitsscore</div>
                  <div style="font-size:10px;color:#5a7090;margin-top:4px">Reparaties: {actieve_p.get('reparatie_pogingen',0)}x</div>
                </div>
              </div>
              <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#4a8af4;margin:14px 0 10px">Bias toelichting</div>
              <div class="fr-item">⚠ Bias: {b.get('toelichting','—')}</div>
              <div class="fr-item">🧠 Hall.: {h.get('toelichting','—')}</div>
              <div class="fr-item">🌍 Inclusie: {i_sc.get('toelichting','—')}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            close_c, export_c = st.columns(2)
            with export_c:
                if st.button("Export rapport", key="exp_report_val", use_container_width=True, type="primary"):
                    st.session_state.panel_mode = "full_report"
                    st.rerun()
            if st.button("✕ Sluit panel", key="close_val", use_container_width=True):
                st.session_state.panel_mode = None
                st.rerun()

        elif st.session_state.panel_mode == "full_report":
            h_s = scores["hallucinaties"]
            b_s = scores["bias"]
            i_s = scores["inclusie"]
            st.markdown(f"""
            <div class="val-panel">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
                <span class="val-panel-title">Full report</span>
              </div>
              <div class="fr-section">
                <div class="fr-cat-title">Hallucination <span class="fr-pct">{h_s}%</span></div>
                <div class="fr-bar-bg"><div class="fr-bar-fill" style="width:{h_s}%;background:#F59E0B"></div></div>
                <div class="fr-item">{badge_icon(h_s)} Score: {h_s}% — {actieve_p.get('hallucinaties',{}).get('toelichting','—')}</div>
              </div>
              <div class="fr-section">
                <div class="fr-cat-title">Bias <span class="fr-pct">{b_s}%</span></div>
                <div class="fr-bar-bg"><div class="fr-bar-fill" style="width:{b_s}%;background:#1DB87A"></div></div>
                <div class="fr-item">{badge_icon(b_s)} Score: {b_s}% — {actieve_p.get('bias',{}).get('toelichting','—')}</div>
              </div>
              <div class="fr-section">
                <div class="fr-cat-title">Data Justice <span class="fr-pct">{i_s}%</span></div>
                <div class="fr-bar-bg"><div class="fr-bar-fill" style="width:{i_s}%;background:#3B7EF6"></div></div>
                <div class="fr-item">{badge_icon(i_s)} Score: {i_s}% — {actieve_p.get('inclusie',{}).get('toelichting','—')}</div>
              </div>
              <div class="score-ring-wrap">
                {score_ring_svg(scores['totaal'], 80)}
                <div class="score-legend">
                  <div class="score-legend-good">Chat totaalscore</div>
                  <div class="score-legend-improve">Gebaseerd op laatste antwoord</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fix_c, exp_c = st.columns(2)
            with exp_c:
                rapport = {
                    "persona": actieve_p["naam"],
                    "persona_scores": {
                        "bias": actieve_p.get("bias"),
                        "hallucinaties": actieve_p.get("hallucinaties"),
                        "inclusie": actieve_p.get("inclusie"),
                        "totaalscore": actieve_p.get("totaalscore"),
                    },
                    "chat_scores": st.session_state.scores,
                    "chatgeschiedenis": [{"rol": m["rol"], "inhoud": m["inhoud"]} for m in st.session_state.chatgeschiedenis],
                }
                st.download_button(
                    "⬇ JSON",
                    data=json.dumps(rapport, indent=2, ensure_ascii=False),
                    file_name=f"rapport_{actieve_p['naam'].replace(' ', '_')}.json",
                    mime="application/json",
                    key="dl_rapport",
                )
            if st.button("✕ Sluit panel", key="close_fr", use_container_width=True):
                st.session_state.panel_mode = None
                st.rerun()


# ─────────────────────────────────────────────
# BERICHTVERWERKING
# ─────────────────────────────────────────────
if send_clicked and user_input.strip():
    if not groq_client:
        st.warning("⚠️ Voeg GROQ_API_KEY toe aan .streamlit/secrets.toml of als omgevingsvariabele.")
    else:
        nu = time.strftime("%H:%M")
        eerste_bericht = st.session_state.berichtentelling == 0

        st.session_state.chatgeschiedenis.append({
            "rol": "gebruiker",
            "inhoud": user_input.strip(),
            "tijd": nu,
        })

        with st.spinner("Synthetic User denkt na..."):
            resultaat = vraag_groq(groq_client, actieve_p, user_input.strip(), st.session_state.api_berichten)

        st.session_state.api_berichten.append({"role": "user", "content": user_input.strip()})
        st.session_state.api_berichten.append({"role": "assistant", "content": resultaat["tekst"]})

        st.session_state.chatgeschiedenis.append({
            "rol": "assistent",
            "inhoud": resultaat["tekst"],
            "tijd": nu,
            "toon_persona_kaart": eerste_bericht,
        })

        if resultaat["succes"]:
            st.session_state.scores = resultaat["scores"]

        st.session_state.berichtentelling += 1
        st.rerun()
