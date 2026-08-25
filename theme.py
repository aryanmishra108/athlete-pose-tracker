"""
Visual identity for the app: a dark, instrument-panel aesthetic that treats
joint-angle data like readouts on a piece of lab equipment rather than a
generic SaaS dashboard. Signature device: corner-bracket "viewfinder"
framing around anything that's literally camera-tracked output (the
annotated video, the 3D pose viewer) -- a nod to the fact that this tool's
actual job is a camera looking at a body.

Palette:
  bg-void        #0F1316   page background
  bg-panel       #171D22   card/panel surface
  bg-panel-alt   #1E262C   nested surface (e.g. metric tiles)
  border-hair    #2B343B   hairline borders
  text-primary   #ECEFF2
  text-secondary #8A97A3
  amber (brand)  #FFB627   actions, brand accent
  cyan (data)    #47D6E0   tracked/measured data -- skeleton, live readouts
  red (critical) #FF5C5C
  info (cyan)    #47D6E0   reuses the data-cyan; info messages ARE data

Type:
  display  Big Shoulders Display   headings, big numbers
  body     IBM Plex Sans           prose, labels
  mono     IBM Plex Mono           every number: angles, timestamps, coords
"""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-void: #0F1316;
    --bg-panel: #171D22;
    --bg-panel-alt: #1E262C;
    --border-hair: #2B343B;
    --text-primary: #ECEFF2;
    --text-secondary: #8A97A3;
    --amber: #FFB627;
    --amber-dim: #C98F1B;
    --cyan: #47D6E0;
    --red: #FF5C5C;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: var(--bg-void); color: var(--text-primary); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border-hair);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Big Shoulders Display', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--text-primary);
}

/* ---------- Headings ---------- */
h1, h2, h3 {
    font-family: 'Big Shoulders Display', sans-serif;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.01em;
}

/* ---------- Buttons ---------- */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.8rem;
    font-weight: 500;
    border-radius: 2px;
    border: 1px solid var(--amber);
    background: transparent;
    color: var(--amber);
    transition: background 0.15s ease, color 0.15s ease;
}
.stButton > button:hover {
    background: var(--amber);
    color: #0F1316;
    border-color: var(--amber);
}
.stButton > button[kind="primary"] {
    background: var(--amber);
    color: #0F1316;
}
.stButton > button[kind="primary"]:hover {
    background: var(--amber-dim);
    border-color: var(--amber-dim);
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border-hair);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom: 2px solid var(--amber) !important;
}

/* ---------- Bordered containers -> viewfinder panels ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-panel);
    border: 1px solid var(--border-hair) !important;
    border-radius: 3px;
    position: relative;
    padding: 4px;
}
div[data-testid="stVerticalBlockBorderWrapper"]::before,
div[data-testid="stVerticalBlockBorderWrapper"]::after {
    content: "";
    position: absolute;
    width: 14px;
    height: 14px;
    border-color: var(--cyan);
    border-style: solid;
    pointer-events: none;
}
div[data-testid="stVerticalBlockBorderWrapper"]::before {
    top: -1px; left: -1px;
    border-width: 2px 0 0 2px;
}
div[data-testid="stVerticalBlockBorderWrapper"]::after {
    bottom: -1px; right: -1px;
    border-width: 0 2px 2px 0;
}

/* ---------- Metrics ---------- */
div[data-testid="stMetric"] {
    background: var(--bg-panel-alt);
    border: 1px solid var(--border-hair);
    border-radius: 2px;
    padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.7rem !important;
    color: var(--text-secondary) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--amber) !important;
}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] {
    border-radius: 2px;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ---------- Captions ---------- */
[data-testid="stCaptionContainer"], .stCaption {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-panel-alt);
    border: 1px dashed var(--border-hair);
    border-radius: 3px;
}

/* ---------- Sliders ---------- */
div[data-baseweb="slider"] [role="slider"] { background: var(--amber) !important; }

/* ---------- Custom components ---------- */
.eyebrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--cyan);
    margin: 1.1rem 0 0.4rem 0;
}
.eyebrow .bracket { color: var(--cyan); opacity: 0.7; font-weight: 600; }

.app-header {
    padding: 0.4rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border-hair);
    margin-bottom: 1.4rem;
}
.app-header .kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--amber);
    margin-bottom: 0.3rem;
}
.app-header h1 {
    font-size: 2.4rem;
    margin: 0;
    line-height: 1.05;
    text-transform: uppercase;
}
.app-header .subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-top: 0.5rem;
    max-width: 640px;
}

.flag-card {
    display: flex;
    gap: 0.7rem;
    align-items: baseline;
    background: var(--bg-panel-alt);
    border-left: 3px solid var(--border-hair);
    border-radius: 2px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.45rem;
}
.flag-card.critical { border-left-color: var(--red); }
.flag-card.warning { border-left-color: var(--amber); }
.flag-card.info { border-left-color: var(--cyan); }
.flag-card .ts {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    white-space: nowrap;
}
.flag-card .msg {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.9rem;
    color: var(--text-primary);
}
.flag-card .tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    white-space: nowrap;
}
.flag-card.critical .tag { background: rgba(255,92,92,0.15); color: var(--red); }
.flag-card.warning .tag { background: rgba(255,182,39,0.15); color: var(--amber); }
.flag-card.info .tag { background: rgba(71,214,224,0.15); color: var(--cyan); }
</style>
"""


def inject_theme():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(kicker: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="app-header">
            <div class="kicker">{kicker}</div>
            <h1>{title}</h1>
            <div class="subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eyebrow(text: str):
    st.markdown(
        f'<div class="eyebrow"><span class="bracket">\u250c</span>{text}<span class="bracket">\u2510</span></div>',
        unsafe_allow_html=True,
    )


def render_flag_card(severity: str, timestamp_sec: float, message: str):
    css_class = severity if severity in ("critical", "warning", "info") else "info"
    st.markdown(
        f"""
        <div class="flag-card {css_class}">
            <span class="tag">{severity}</span>
            <span class="ts">t={timestamp_sec:.2f}s</span>
            <span class="msg">{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
