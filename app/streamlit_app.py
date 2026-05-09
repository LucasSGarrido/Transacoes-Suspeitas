from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.components.charts import (  # noqa: E402
    amount_by_class_box,
    amount_distribution,
    animated_risk_scatter,
    class_profile_chart,
    confusion_heatmap,
    cost_curve_chart,
    cost_scenario_chart,
    data_quality_chart,
    feature_importance_chart,
    fraud_rate_chart,
    local_explanation_chart,
    model_comparison,
    risk_bucket_bar,
    risk_by_hour,
    risk_gauge,
    threshold_curve,
    transactions_over_time_chart,
)
from app.components.styles import inject_css  # noqa: E402
from src.config import (  # noqa: E402
    EXPLANATION_SUMMARY_PATH,
    FEATURE_IMPORTANCE_PATH,
    LATEST_EXPERIMENT_PATH,
    LOCAL_EXPLANATIONS_PATH,
    METRICS_PATH,
    SCORED_DATA_PATH,
    TEMPORAL_METRICS_PATH,
    THRESHOLD_PATH,
)
from src.evaluate import best_cost_threshold, cost_metrics  # noqa: E402
from src.features import add_risk_bucket, engineer_features  # noqa: E402


st.set_page_config(
    page_title="Fraud Risk Command Center",
    layout="wide",
)
inject_css()


@st.cache_data
def load_data(scored_mtime: float, metrics_mtime: float, config_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not SCORED_DATA_PATH.exists():
        st.error("Execute `python -m src.train` antes de abrir o dashboard.")
        st.stop()

    data = pd.read_csv(SCORED_DATA_PATH)
    data = engineer_features(data)
    data = add_risk_bucket(data)

    metrics = pd.read_csv(METRICS_PATH) if METRICS_PATH.exists() else pd.DataFrame()
    config = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8")) if THRESHOLD_PATH.exists() else {}
    return data, metrics, config


@st.cache_data
def load_optional_artifacts(
    temporal_mtime: float,
    importance_mtime: float,
    local_mtime: float,
    explanation_mtime: float,
    experiment_mtime: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, pd.DataFrame]:
    temporal = pd.read_csv(TEMPORAL_METRICS_PATH) if TEMPORAL_METRICS_PATH.exists() else pd.DataFrame()
    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH) if FEATURE_IMPORTANCE_PATH.exists() else pd.DataFrame()
    local = pd.read_csv(LOCAL_EXPLANATIONS_PATH) if LOCAL_EXPLANATIONS_PATH.exists() else pd.DataFrame()
    explanation = json.loads(EXPLANATION_SUMMARY_PATH.read_text(encoding="utf-8")) if EXPLANATION_SUMMARY_PATH.exists() else {}
    experiment = json.loads(LATEST_EXPERIMENT_PATH.read_text(encoding="utf-8")) if LATEST_EXPERIMENT_PATH.exists() else {}
    return temporal, importance, explanation, experiment, local


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def fmt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def fmt_money(value: float) -> str:
    return f"R$ {value:,.0f}".replace(",", ".")


def insight_card_html(title: str, question: str, reading: str, study: str) -> str:
    return (
        '<div class="insight-card">'
        f"<h4>{title}</h4>"
        f"<p><b>Pergunta:</b> {question}</p>"
        f"<p><b>Leitura:</b> {reading}</p>"
        f"<p><b>Estudo:</b> {study}</p>"
        "</div>"
    )


def insight_grid(items: list[tuple[str, str, str, str]]) -> None:
    cards = "".join(insight_card_html(*item) for item in items)
    st.markdown(f'<div class="insight-grid">{cards}</div>', unsafe_allow_html=True)


def analysis_columns(data: pd.DataFrame) -> list[str]:
    expected = ["Time", *[f"V{index}" for index in range(1, 29)], "Amount", "Class"]
    return [column for column in expected if column in data.columns]


def data_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    raw_data = data[analysis_columns(data)].copy()
    missing_cells = int(raw_data.isna().sum().sum())
    missing_columns = int((raw_data.isna().sum() > 0).sum())
    duplicate_rows = int(raw_data.duplicated().sum())
    rows = [
        ("Células ausentes", missing_cells),
        ("Colunas com ausentes", missing_columns),
        ("Linhas duplicadas", duplicate_rows),
    ]
    return pd.DataFrame(
        {
            "indicador": [row[0] for row in rows],
            "quantidade": [row[1] for row in rows],
            "status": ["atenção" if row[1] > 0 else "ok" for row in rows],
        }
    )


def build_cost_scenarios(data: pd.DataFrame) -> pd.DataFrame:
    scenario_config = [
        {
            "cenário": "Conservador",
            "falso_positivo": 60.0,
            "falso_negativo": 350.0,
            "revisão": 15.0,
            "leitura": "Menos alertas e maior cuidado com atrito em clientes legítimos.",
        },
        {
            "cenário": "Balanceado",
            "falso_positivo": 25.0,
            "falso_negativo": 500.0,
            "revisão": 10.0,
            "leitura": "Compromisso entre captura de fraude e capacidade operacional.",
        },
        {
            "cenário": "Agressivo",
            "falso_positivo": 10.0,
            "falso_negativo": 1000.0,
            "revisão": 10.0,
            "leitura": "Prioriza captura máxima de fraude mesmo gerando mais investigação.",
        },
    ]

    rows = []
    y_true = data["Class"].to_numpy()
    scores = data["risk_score"].to_numpy()
    for scenario in scenario_config:
        best_threshold, _ = best_cost_threshold(
            y_true,
            scores,
            false_positive_cost=scenario["falso_positivo"],
            false_negative_cost=scenario["falso_negativo"],
            true_positive_review_cost=scenario["revisão"],
        )
        result = cost_metrics(
            y_true,
            scores,
            best_threshold,
            false_positive_cost=scenario["falso_positivo"],
            false_negative_cost=scenario["falso_negativo"],
            true_positive_review_cost=scenario["revisão"],
        )
        rows.append(
            {
                "cenário": scenario["cenário"],
                "threshold": result["threshold"],
                "custo_total": result["total_cost"],
                "alertas": result["alerts"],
                "falsos_positivos": result["false_positives"],
                "falsos_negativos": result["false_negatives"],
                "fraudes_capturadas": result["true_positives"],
                "leitura": scenario["leitura"],
            }
        )
    return pd.DataFrame(rows)


def top_feature_text(importance: pd.DataFrame) -> str:
    if importance.empty:
        return "As variáveis mais importantes ainda não foram calculadas."

    top_features = importance.sort_values("importance", ascending=False)["feature"].head(3).tolist()
    if not top_features:
        return "As variáveis mais importantes ainda não foram calculadas."

    joined = ", ".join(top_features)
    return (
        f"As variáveis {joined} concentram a maior influência global. Como o dataset usa componentes anonimizados "
        "V1 a V28, a leitura correta é técnica: elas representam padrões latentes que separam fraude de comportamento "
        "legítimo, não atributos de negócio diretamente interpretáveis."
    )


df, metrics, config = load_data(
    file_mtime(SCORED_DATA_PATH),
    file_mtime(METRICS_PATH),
    file_mtime(THRESHOLD_PATH),
)
temporal_metrics, feature_importance, explanation_summary, experiment_log, local_explanations = load_optional_artifacts(
    file_mtime(TEMPORAL_METRICS_PATH),
    file_mtime(FEATURE_IMPORTANCE_PATH),
    file_mtime(LOCAL_EXPLANATIONS_PATH),
    file_mtime(EXPLANATION_SUMMARY_PATH),
    file_mtime(LATEST_EXPERIMENT_PATH),
)
default_threshold = float(config.get("threshold", 0.5))
dataset_source = config.get("dataset_source", "não informada")
best_model = config.get("model_name", "modelo não informado")
test_metrics = config.get("test", {})
base_fraud_rate = df["Class"].mean() if "Class" in df else 0

with st.sidebar:
    st.title("Controle de Risco")
    threshold = st.slider("Threshold", 0.01, 0.99, default_threshold, 0.01)
    min_score = st.slider("Score mínimo", 0.0, 1.0, 0.0, 0.05)
    selected_buckets = st.multiselect(
        "Faixas de risco",
        ["baixo", "moderado", "alto", "crítico"],
        default=["baixo", "moderado", "alto", "crítico"],
    )
    st.markdown(
        f"""
        <div class="note-strip">
            <b>Modelo:</b> {best_model}<br>
            <b>Fonte:</b> {dataset_source}<br>
            <b>Threshold treinado:</b> {default_threshold:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )

bucket_filter = ["critico" if bucket == "crítico" else bucket for bucket in selected_buckets]
working = df[df["risk_score"] >= min_score].copy()
working = working[working["risk_bucket"].isin(bucket_filter)]
working["prediction"] = (working["risk_score"] >= threshold).astype(int)
working = add_risk_bucket(working)

if working.empty:
    st.warning("Nenhuma transação encontrada com os filtros atuais.")
    st.stop()

alerts = int(working["prediction"].sum())
frauds = int(working["Class"].sum()) if "Class" in working else 0
captured = int(((working["Class"] == 1) & (working["prediction"] == 1)).sum()) if "Class" in working else 0
alert_rate = alerts / max(len(working), 1)
recall = captured / max(frauds, 1)
precision = captured / max(alerts, 1)
critical = int((working["risk_bucket"] == "critico").sum())
false_negatives = int(((working["Class"] == 1) & (working["prediction"] == 0)).sum()) if "Class" in working else 0
false_positives = int(((working["Class"] == 0) & (working["prediction"] == 1)).sum()) if "Class" in working else 0

st.markdown(
    f"""
    <div class="hero-panel">
        <div>
            <span class="eyebrow">Fraud Analytics MVP</span>
            <h1>Fraud Risk Command Center</h1>
            <p>
                Painel de investigação para priorizar transações por score de risco,
                comparar modelos e estudar o efeito do threshold na fila antifraude.
            </p>
        </div>
        <div class="hero-meta">
            <div class="meta-pill"><span>Modelo campeão</span><b>{best_model}</b></div>
            <div class="meta-pill"><span>Fonte dos dados</span><b>{dataset_source}</b></div>
            <div class="meta-pill"><span>Fraude na base</span><b>{base_fraud_rate:.3%}</b></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if min_score >= threshold:
    st.markdown(
        """
        <div class="note-strip">
            O score mínimo está maior ou igual ao threshold. Nesse recorte, os KPIs representam uma fila já
            pré-selecionada de transações suspeitas, não o universo completo da base.
        </div>
        """,
        unsafe_allow_html=True,
    )

if dataset_source == "demo_sintetico":
    st.markdown(
        """
        <div class="note-strip">
            Este dashboard está usando dados sintéticos de desenvolvimento. Para conclusões reais de portfólio,
            baixe o dataset do Kaggle e coloque o arquivo em <b>data/raw/creditcard.csv</b>, depois rode
            <b>python -m src.train</b> novamente.
        </div>
        """,
        unsafe_allow_html=True,
    )

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Base total", fmt_int(len(df)))
col2.metric("Recorte", fmt_int(len(working)))
col3.metric("Alertas", fmt_int(alerts))
col4.metric("Recall", f"{recall:.1%}")
col5.metric("Precisão", f"{precision:.1%}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Fraudes no recorte", fmt_int(frauds))
col2.metric("Fraudes capturadas", fmt_int(captured))
col3.metric("Falsos negativos", fmt_int(false_negatives))
col4.metric("Falsos positivos", fmt_int(false_positives))
col5.metric("Críticas", fmt_int(critical))

st.markdown('<div class="section-title">Resumo executivo automático</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="maturity-grid">
        <div class="maturity-card">
            <strong>Captura de fraude</strong>
            <span>O modelo capturou <b>{fmt_int(captured)}</b> de <b>{fmt_int(frauds)}</b> fraudes no recorte atual.</span>
        </div>
        <div class="maturity-card">
            <strong>Custo operacional</strong>
            <span>Foram gerados <b>{fmt_int(false_positives)}</b> falsos positivos, casos que consomem tempo da equipe.</span>
        </div>
        <div class="maturity-card">
            <strong>Recomendação</strong>
            <span>Usar como <b>fila de investigação</b>, não como bloqueio automático sem revisão humana.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, data_tab, models_tab, investigation_tab, cost_tab, explain_tab, temporal_tab, conclusion_tab = st.tabs(
    [
        "Visão Geral",
        "Análise dos Dados",
        "Modelos",
        "Investigação",
        "Custos",
        "Explicabilidade",
        "Mapa Temporal",
        "Conclusão",
    ]
)

with overview_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(risk_by_hour(working), width="stretch", key="overview_risk_by_hour")
    with right:
        st.plotly_chart(risk_gauge(alert_rate), width="stretch", key="overview_risk_gauge")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(confusion_heatmap(working, threshold), width="stretch", key="overview_confusion_heatmap")
    with right:
        st.plotly_chart(threshold_curve(working), width="stretch", key="overview_threshold_curve")

    st.markdown('<div class="section-title">Leitura dos gráficos</div>', unsafe_allow_html=True)
    insight_grid(
        [
            (
                "Volume e alertas por hora",
                "Em quais horários o modelo concentra mais alertas?",
                "Compare as barras de volume com a linha vermelha. Horários com muitos alertas e pouco volume merecem revisão.",
                "Padrão temporal, sazonalidade e risco de falso alarme por janela de tempo.",
            ),
            (
                "Gauge de taxa de alertas",
                "A fila gerada cabe na capacidade operacional?",
                "Uma taxa muito alta pode indicar que o threshold está agressivo demais para uma equipe pequena.",
                "Trade-off entre recall e volume de investigação.",
            ),
            (
                "Matriz de confusão",
                "Quais erros o modelo está cometendo?",
                "Falso negativo representa fraude perdida. Falso positivo representa investigação desnecessária.",
                "Custo de erro e função de custo para escolher threshold.",
            ),
            (
                "Precisão e recall por threshold",
                "Como a decisão muda quando o limite de risco muda?",
                "Recall tende a cair quando o threshold sobe; precisão tende a melhorar até certo ponto.",
                "F2-score, PR-AUC e cenários conservador, balanceado e agressivo.",
            ),
        ]
    )

with data_tab:
    quality = data_quality_summary(df)
    fraud_count = int(df["Class"].sum())
    legitimate_count = int((df["Class"] == 0).sum())
    duplicate_rows = int(quality.loc[quality["indicador"] == "Linhas duplicadas", "quantidade"].iloc[0])
    missing_cells = int(quality.loc[quality["indicador"] == "Células ausentes", "quantidade"].iloc[0])

    st.markdown('<div class="section-title">Análise exploratória real do dataset</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Transações", fmt_int(len(df)))
    col2.metric("Fraudes", fmt_int(fraud_count))
    col3.metric("Legítimas", fmt_int(legitimate_count))
    col4.metric("Taxa de fraude", f"{base_fraud_rate:.3%}")
    col5.metric("Duplicadas", fmt_int(duplicate_rows))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(fraud_rate_chart(df), width="stretch", key="data_fraud_rate")
    with right:
        st.plotly_chart(data_quality_chart(quality), width="stretch", key="data_quality")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(amount_distribution(df), width="stretch", key="data_amount_distribution")
    with right:
        st.plotly_chart(amount_by_class_box(df), width="stretch", key="data_amount_by_class")

    st.plotly_chart(transactions_over_time_chart(df), width="stretch", key="data_transactions_over_time")
    st.plotly_chart(class_profile_chart(df), width="stretch", key="data_class_profile")

    st.markdown(
        f"""
        <div class="note-strip">
            O dataset tem <b>{fmt_int(fraud_count)}</b> fraudes em <b>{fmt_int(len(df))}</b> transações,
            uma taxa de apenas <b>{base_fraud_rate:.3%}</b>. Isso confirma um problema altamente desbalanceado:
            acurácia isolada não é uma boa métrica, e por isso o projeto prioriza PR-AUC, recall, precisão e F2.
            A checagem encontrou <b>{fmt_int(missing_cells)}</b> células ausentes e <b>{fmt_int(duplicate_rows)}</b>
            linhas duplicadas no recorte bruto de colunas do dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    insight_grid(
        [
            (
                "Taxa de fraude",
                "O problema é balanceado?",
                "Não. A fraude é rara, então o modelo precisa ser avaliado por métricas sensíveis à classe minoritária.",
                "Desbalanceamento, PR-AUC, recall e F2-score.",
            ),
            (
                "Distribuição de Amount",
                "O valor da transação separa fraude de legítima?",
                "Amount ajuda a entender outliers, mas não resolve sozinho; as variáveis anonimizadas carregam muito sinal.",
                "Escala log, cauda longa e comparação por classe.",
            ),
            (
                "Transações por tempo",
                "O risco muda ao longo da janela temporal?",
                "A série por hora mostra concentração de fraudes e reforça a necessidade de validação temporal.",
                "Drift, sazonalidade e split por tempo.",
            ),
            (
                "Qualidade dos dados",
                "Há ausentes ou duplicados relevantes?",
                "A ausência de nulos simplifica o pipeline; duplicatas devem ser declaradas porque podem afetar avaliação.",
                "Auditoria de dados, leakage e reprodutibilidade.",
            ),
        ]
    )

with models_tab:
    if not metrics.empty:
        st.markdown('<div class="section-title">Validação estratificada</div>', unsafe_allow_html=True)
        st.plotly_chart(model_comparison(metrics), width="stretch", key="models_stratified_comparison")
        display_metrics = metrics[
            [
                "model",
                "validation_f2",
                "validation_pr_auc",
                "test_f2",
                "test_pr_auc",
                "test_recall",
                "test_precision",
                "threshold",
            ]
        ].copy()
        st.dataframe(
            display_metrics.style.format(
                {
                    "validation_f2": "{:.3f}",
                    "validation_pr_auc": "{:.3f}",
                    "test_f2": "{:.3f}",
                    "test_pr_auc": "{:.3f}",
                    "test_recall": "{:.3f}",
                    "test_precision": "{:.3f}",
                    "threshold": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    if not temporal_metrics.empty:
        st.markdown('<div class="section-title">Validação temporal</div>', unsafe_allow_html=True)
        st.plotly_chart(model_comparison(temporal_metrics), width="stretch", key="models_temporal_comparison")
        temporal_display = temporal_metrics[
            [
                "model",
                "validation_f2",
                "validation_pr_auc",
                "test_f2",
                "test_pr_auc",
                "test_recall",
                "test_precision",
                "threshold",
            ]
        ].copy()
        st.dataframe(
            temporal_display.style.format(
                {
                    "validation_f2": "{:.3f}",
                    "validation_pr_auc": "{:.3f}",
                    "test_f2": "{:.3f}",
                    "test_pr_auc": "{:.3f}",
                    "test_recall": "{:.3f}",
                    "test_precision": "{:.3f}",
                    "threshold": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(amount_distribution(working), width="stretch", key="models_amount_distribution")
    with right:
        st.plotly_chart(risk_bucket_bar(working), width="stretch", key="models_risk_bucket")

    st.markdown('<div class="section-title">Estudo dos modelos</div>', unsafe_allow_html=True)
    insight_grid(
        [
            (
                "Baseline contra modelos fortes",
                "O modelo complexo realmente melhora a decisão?",
                "Compare F2 e PR-AUC. No dataset real, a Random Forest venceu no F2, mas o XGBoost ficou próximo.",
                "Validação cruzada estratificada, validação temporal e estabilidade do ranking.",
            ),
            (
                "Isolation Forest",
                "Anomalia sem rótulo funciona bem neste caso?",
                "No dataset real, o Isolation Forest gerou muitos falsos positivos e precisão baixa.",
                "Quando detecção não supervisionada faz sentido e como calibrar contaminação.",
            ),
            (
                "Distribuição de valores",
                "Valor da transação ajuda a separar fraude?",
                "A escala log evita que poucos valores altos escondam o comportamento geral.",
                "Transformação log, outliers e variáveis derivadas de valor.",
            ),
            (
                "Faixas de risco",
                "O score está concentrado ou bem distribuído?",
                "Muitas transações em risco crítico podem indicar threshold agressivo ou amostra filtrada.",
                "Calibração de probabilidade e curvas de calibração.",
            ),
        ]
    )

with investigation_tab:
    top_risk = working.sort_values("risk_score", ascending=False).head(200)
    columns = [column for column in ["Time", "Amount", "Class", "risk_score", "risk_bucket", "prediction"] if column in top_risk]
    st.dataframe(
        top_risk[columns].style.format({"risk_score": "{:.3f}", "Amount": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
    st.markdown('<div class="section-title">Estudo da fila de investigação</div>', unsafe_allow_html=True)
    insight_grid(
        [
            (
                "Top transações por risco",
                "Quais casos devem ser revisados primeiro?",
                "A tabela ordena a fila operacional por score. Em um produto real, ela seria usada por analistas antifraude.",
                "Precision@K, Recall@K e capacidade diária da equipe.",
            ),
            (
                "Falsos positivos e falsos negativos",
                "O erro do modelo é aceitável para o negócio?",
                "Falso positivo custa tempo. Falso negativo pode custar perda financeira e dano ao cliente.",
                "Matriz de custo, perda esperada e bloqueio automático versus revisão humana.",
            ),
        ]
    )

with cost_tab:
    st.markdown('<div class="section-title">Simulador de custo por threshold</div>', unsafe_allow_html=True)
    left, middle, right = st.columns(3)
    with left:
        false_positive_cost = st.number_input("Custo por falso positivo", min_value=0.0, value=25.0, step=5.0)
    with middle:
        false_negative_cost = st.number_input("Custo por falso negativo", min_value=0.0, value=500.0, step=25.0)
    with right:
        true_positive_review_cost = st.number_input("Custo de revisar fraude capturada", min_value=0.0, value=10.0, step=5.0)

    best_cost_cutoff, cost_data = best_cost_threshold(
        working["Class"].to_numpy(),
        working["risk_score"].to_numpy(),
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        true_positive_review_cost=true_positive_review_cost,
    )
    current_cost = cost_metrics(
        working["Class"].to_numpy(),
        working["risk_score"].to_numpy(),
        threshold,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        true_positive_review_cost=true_positive_review_cost,
    )
    optimal_cost = cost_metrics(
        working["Class"].to_numpy(),
        working["risk_score"].to_numpy(),
        best_cost_cutoff,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        true_positive_review_cost=true_positive_review_cost,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Threshold atual", f"{threshold:.2f}")
    col2.metric("Threshold de menor custo", f"{best_cost_cutoff:.2f}")
    col3.metric("Custo atual", fmt_money(current_cost["total_cost"]))
    col4.metric("Custo mínimo", fmt_money(optimal_cost["total_cost"]))

    st.plotly_chart(cost_curve_chart(cost_data, threshold, best_cost_cutoff), width="stretch", key="cost_curve")
    st.dataframe(
        pd.DataFrame(
            [
                {"cenário": "Threshold atual", **current_cost},
                {"cenário": "Menor custo", **optimal_cost},
            ]
        )[
            [
                "cenário",
                "threshold",
                "total_cost",
                "alerts",
                "false_positives",
                "false_negatives",
                "true_positives",
            ]
        ].style.format({"threshold": "{:.2f}", "total_cost": "R$ {:.0f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown('<div class="section-title">Cenários de negócio</div>', unsafe_allow_html=True)
    scenario_table = build_cost_scenarios(working)
    st.plotly_chart(cost_scenario_chart(scenario_table), width="stretch", key="cost_scenarios")
    st.dataframe(
        scenario_table[
            [
                "cenário",
                "threshold",
                "custo_total",
                "alertas",
                "fraudes_capturadas",
                "falsos_positivos",
                "falsos_negativos",
                "leitura",
            ]
        ].style.format({"threshold": "{:.2f}", "custo_total": "R$ {:.0f}"}),
        width="stretch",
        hide_index=True,
    )

    insight_grid(
        [
            (
                "Conclusão de custo",
                "Qual threshold minimiza o custo estimado?",
                "O melhor threshold muda quando o custo de perder uma fraude aumenta ou quando a equipe tem pouco tempo para revisar alertas.",
                "Matriz de custo, análise de sensibilidade e priorização operacional.",
            ),
            (
                "Cenários conservador, balanceado e agressivo",
                "Como a política de risco muda a decisão?",
                "O cenário conservador preserva experiência do cliente; o agressivo aceita mais alertas para evitar perdas por fraude.",
                "Comparação de custo total, capacidade operacional e apetite a risco.",
            ),
            (
                "Uso prático",
                "O modelo deve bloquear ou priorizar?",
                "Mesmo com bom recall, a presença de falsos positivos favorece uso como fila de investigação antes de automação completa.",
                "Política de decisão, revisão humana e risco de impacto no cliente.",
            ),
        ]
    )

with explain_tab:
    st.markdown('<div class="section-title">Explicabilidade do modelo</div>', unsafe_allow_html=True)
    if feature_importance.empty:
        st.info("Execute `python -m src.explain` para gerar explicabilidade global e local.")
    else:
        st.caption(
            f"Método global: {explanation_summary.get('global_method', 'não informado')} | "
            f"Método local: {explanation_summary.get('local_method', 'não informado')}"
        )
        st.plotly_chart(feature_importance_chart(feature_importance), width="stretch", key="explain_global_importance")
        st.markdown(f'<div class="note-strip">{top_feature_text(feature_importance)}</div>', unsafe_allow_html=True)

        if not local_explanations.empty:
            selected_row = st.selectbox(
                "Transação para explicação local",
                sorted(local_explanations["row_index"].unique()),
            )
            selected_local = local_explanations[local_explanations["row_index"] == selected_row].copy()
            st.plotly_chart(local_explanation_chart(selected_local), width="stretch", key="explain_local_transaction")
            st.dataframe(
                selected_local[
                    ["risk_score", "class", "feature", "feature_value", "contribution"]
                ].style.format(
                    {
                        "risk_score": "{:.3f}",
                        "feature_value": "{:.3f}",
                        "contribution": "{:.6f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    insight_grid(
        [
            (
                "Importância global",
                "Quais variáveis mais pesam no modelo?",
                "As variáveis do Kaggle são anonimizadas, então a explicação é técnica, não semântica.",
                "SHAP, importância de árvore e limitação de features anonimizadas.",
            ),
            (
                "Explicação local",
                "Por que uma transação específica recebeu score alto?",
                "A explicação local mostra quais variáveis aumentaram ou reduziram o score daquele alerta.",
                "Auditoria de decisão e comunicação com pessoas não técnicas.",
            ),
        ]
    )

with temporal_tab:
    st.plotly_chart(animated_risk_scatter(working), width="stretch", key="temporal_risk_scatter")
    st.markdown('<div class="section-title">Estudo do mapa temporal</div>', unsafe_allow_html=True)
    insight_grid(
        [
            (
                "Risco por hora e valor",
                "O risco muda conforme o tempo e o valor transacionado?",
                "O mapa temporal permite observar concentrações de score alto ao longo das horas.",
                "Janelas temporais, comportamento sequencial e detecção de surtos.",
            ),
            (
                "Inspeção visual",
                "Existem grupos de transações parecidas?",
                "Pontos vermelhos ou amarelos concentrados sugerem segmentos que merecem explicação adicional.",
                "PCA, UMAP e SHAP para conectar visualização com explicabilidade.",
            ),
        ]
    )

with conclusion_tab:
    test_tp = int(test_metrics.get("true_positives", 0))
    test_fp = int(test_metrics.get("false_positives", 0))
    test_fn = int(test_metrics.get("false_negatives", 0))
    test_recall = float(test_metrics.get("recall", 0))
    test_precision = float(test_metrics.get("precision", 0))

    st.markdown('<div class="section-title">Conclusão executiva</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="maturity-grid">
            <div class="maturity-card">
                <strong>Problema</strong>
                <span>Detectar fraude em uma base altamente desbalanceada, com apenas <b>{base_fraud_rate:.3%}</b> de transações fraudulentas.</span>
            </div>
            <div class="maturity-card">
                <strong>Modelo escolhido</strong>
                <span>O pipeline selecionou <b>{best_model}</b> com threshold <b>{default_threshold:.2f}</b>, priorizando recall e F2.</span>
            </div>
            <div class="maturity-card">
                <strong>Resultado</strong>
                <span>No teste, capturou <b>{fmt_int(test_tp)}</b> fraudes, gerou <b>{fmt_int(test_fp)}</b> falsos positivos e deixou <b>{fmt_int(test_fn)}</b> fraudes passarem.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="note-strip">
            <b>Leitura final:</b> o modelo atingiu recall de <b>{test_recall:.1%}</b> e precisão de
            <b>{test_precision:.1%}</b> no teste. A recomendação é usá-lo como fila de investigação antifraude,
            com revisão humana para casos de maior score. O projeto não deve ser apresentado como bloqueio automático
            porque ainda existem falsos positivos, falsos negativos, variáveis anonimizadas e necessidade de validação
            contínua contra mudança de comportamento de fraude.
        </div>
        """,
        unsafe_allow_html=True,
    )

    insight_grid(
        [
            (
                "Limitações",
                "O que impede uso direto em produção?",
                "O dataset é anonimizado, não possui contexto de cliente, device, lojista ou chargeback financeiro real.",
                "Monitoramento de drift, calibração, custo real de erro e validação temporal contínua.",
            ),
            (
                "Recomendação final",
                "Como usar o modelo com segurança?",
                "Priorizar investigação humana dos maiores scores, medir Precision@K e ajustar threshold pela capacidade da equipe.",
                "Fila operacional, governança de decisão e explicabilidade local para auditoria.",
            ),
        ]
    )

    if experiment_log:
        st.markdown("#### Registro do último experimento")
        experiment_summary = {
            "Criado em UTC": experiment_log.get("created_at_utc"),
            "Fonte": experiment_log.get("dataset_source"),
            "Linhas": experiment_log.get("rows"),
            "Fraudes": experiment_log.get("frauds"),
            "Taxa de fraude": experiment_log.get("fraud_rate"),
            "Melhor modelo": experiment_log.get("best_model"),
            "Threshold": experiment_log.get("selected_threshold"),
        }
        st.dataframe(pd.DataFrame([experiment_summary]), width="stretch", hide_index=True)
