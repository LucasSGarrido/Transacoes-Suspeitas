import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-soft: #162033;
            --line: rgba(148, 163, 184, 0.22);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --green: #22c55e;
            --amber: #f59e0b;
            --red: #ef4444;
            --cyan: #38bdf8;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 34rem),
                linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: var(--text);
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: #0b1220;
            border-right: 1px solid var(--line);
        }

        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
            animation: fadeIn 520ms ease both;
        }

        .risk-card {
            background: rgba(17, 24, 39, 0.88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            animation: fadeIn 560ms ease both;
        }

        .risk-card strong {
            display: block;
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .risk-card span {
            color: var(--text);
            font-size: 1.7rem;
            font-weight: 800;
        }

        .pulse-critical {
            color: var(--red);
            animation: pulse 1.8s infinite;
        }

        .hero-panel {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1.2rem;
            align-items: end;
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 42, 0.72));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.22);
        }

        .hero-panel h1 {
            margin: 0.25rem 0 0.5rem 0;
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.05;
        }

        .hero-panel p {
            max-width: 760px;
            margin: 0;
            color: var(--muted);
            font-size: 1rem;
        }

        .eyebrow {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-meta {
            display: grid;
            gap: 0.55rem;
            min-width: 240px;
        }

        .meta-pill {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--text);
            background: rgba(15, 23, 42, 0.68);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            font-size: 0.85rem;
        }

        .meta-pill span:first-child {
            color: var(--muted);
        }

        .note-strip {
            border-left: 3px solid var(--amber);
            background: rgba(245, 158, 11, 0.08);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            color: var(--text);
            margin: 1rem 0;
        }

        .section-title {
            margin-top: 1.4rem;
            margin-bottom: 0.6rem;
            font-size: 1.25rem;
            font-weight: 800;
        }

        .insight-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.75rem 0 1.2rem;
        }

        .insight-card {
            background: rgba(17, 24, 39, 0.86);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.15rem 1.2rem;
            min-height: 178px;
            animation: fadeIn 560ms ease both;
        }

        .insight-card h4 {
            margin: 0 0 0.75rem 0;
            color: var(--text);
            font-size: 1.08rem;
            line-height: 1.25;
        }

        .insight-card p {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0.5rem 0;
        }

        .insight-card b {
            color: #f8fafc;
        }

        .maturity-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 0.75rem;
        }

        .maturity-card {
            background: rgba(17, 24, 39, 0.86);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
        }

        .maturity-card strong {
            display: block;
            margin-bottom: 0.45rem;
            font-size: 1.02rem;
        }

        .maturity-card span {
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.5;
        }

        .maturity-card b {
            color: #f8fafc;
        }

        @media (max-width: 900px) {
            .hero-panel,
            .insight-grid,
            .maturity-grid {
                grid-template-columns: 1fr;
            }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulse {
            0% { opacity: 0.65; }
            50% { opacity: 1; }
            100% { opacity: 0.65; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
