from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_score, recall_score


COLOR_MAP = {
    "baixo": "#22c55e",
    "moderado": "#38bdf8",
    "alto": "#f59e0b",
    "critico": "#ef4444",
}

CLASS_MAP = {0: "Legítima", 1: "Fraude"}


def plotly_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font_color="#e5e7eb",
        margin=dict(l=20, r=20, t=54 if title else 25, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zeroline=False)
    return fig


def risk_gauge(alert_rate: float, target_rate: float = 0.10) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=alert_rate * 100,
            number={"suffix": "%"},
            title={"text": "Taxa de alertas"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8"},
                "threshold": {
                    "line": {"color": "#f59e0b", "width": 4},
                    "thickness": 0.8,
                    "value": target_rate * 100,
                },
                "steps": [
                    {"range": [0, 5], "color": "rgba(34,197,94,0.22)"},
                    {"range": [5, 15], "color": "rgba(245,158,11,0.24)"},
                    {"range": [15, 100], "color": "rgba(239,68,68,0.26)"},
                ],
            },
        )
    )
    fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
    return fig


def amount_distribution(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(len(df), 6000), random_state=42)
    sample = sample.assign(classe=sample["Class"].map(CLASS_MAP))
    fig = px.histogram(
        sample,
        x="Amount",
        color="classe",
        nbins=70,
        barmode="overlay",
        color_discrete_map={"Legítima": "#38bdf8", "Fraude": "#ef4444"},
        log_y=True,
        labels={"Amount": "Valor", "classe": "Classe"},
    )
    fig.update_traces(opacity=0.72)
    return plotly_layout(fig, "Distribuição de valores por classe")


def fraud_rate_chart(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df["Class"]
        .map(CLASS_MAP)
        .value_counts()
        .reindex(["Legítima", "Fraude"], fill_value=0)
        .rename_axis("classe")
        .reset_index(name="transações")
    )
    total = max(int(grouped["transações"].sum()), 1)
    grouped["taxa"] = grouped["transações"] / total
    grouped["rótulo"] = grouped.apply(lambda row: f"{row['transações']:,.0f} ({row['taxa']:.3%})".replace(",", "."), axis=1)

    fig = px.bar(
        grouped,
        x="classe",
        y="transações",
        color="classe",
        text="rótulo",
        log_y=True,
        color_discrete_map={"Legítima": "#38bdf8", "Fraude": "#ef4444"},
        labels={"classe": "Classe", "transações": "Transações"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return plotly_layout(fig, "Taxa de fraude e desbalanceamento")


def transactions_over_time_chart(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    data["hora_base"] = (pd.to_numeric(data["Time"], errors="coerce").fillna(0) // 3600).astype(int)
    grouped = (
        data.groupby("hora_base", as_index=False)
        .agg(transações=("Class", "size"), fraudes=("Class", "sum"))
        .sort_values("hora_base")
    )
    grouped["taxa_fraude"] = grouped["fraudes"] / grouped["transações"].clip(lower=1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=grouped["hora_base"],
        y=grouped["transações"],
        name="transações",
        marker_color="#334155",
        secondary_y=False,
    )
    fig.add_scatter(
        x=grouped["hora_base"],
        y=grouped["taxa_fraude"] * 100,
        name="taxa de fraude",
        mode="lines+markers",
        line=dict(color="#ef4444", width=3),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Transações", secondary_y=False)
    fig.update_yaxes(title_text="Taxa de fraude (%)", secondary_y=True, gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Hora desde a primeira transação")
    return plotly_layout(fig, "Transações e taxa de fraude ao longo do tempo")


def amount_by_class_box(df: pd.DataFrame) -> go.Figure:
    sample = df[df["Amount"] > 0].sample(min((df["Amount"] > 0).sum(), 8000), random_state=42)
    sample = sample.assign(classe=sample["Class"].map(CLASS_MAP))
    fig = px.box(
        sample,
        x="classe",
        y="Amount",
        color="classe",
        points="outliers",
        log_y=True,
        color_discrete_map={"Legítima": "#38bdf8", "Fraude": "#ef4444"},
        labels={"classe": "Classe", "Amount": "Valor"},
    )
    return plotly_layout(fig, "Comparação de valores: fraude vs legítima")


def class_profile_chart(df: pd.DataFrame) -> go.Figure:
    data = (
        df.groupby("Class", as_index=False)
        .agg(
            valor_mediano=("Amount", "median"),
            valor_médio=("Amount", "mean"),
            score_médio=("risk_score", "mean"),
        )
        .assign(classe=lambda frame: frame["Class"].map(CLASS_MAP))
    )
    profile = data.melt(
        id_vars="classe",
        value_vars=["valor_mediano", "valor_médio", "score_médio"],
        var_name="indicador",
        value_name="valor",
    )
    profile["indicador"] = profile["indicador"].replace(
        {
            "valor_mediano": "Valor mediano",
            "valor_médio": "Valor médio",
            "score_médio": "Score médio",
        }
    )
    fig = px.bar(
        profile,
        x="classe",
        y="valor",
        color="classe",
        facet_col="indicador",
        facet_col_wrap=3,
        text="valor",
        color_discrete_map={"Legítima": "#38bdf8", "Fraude": "#ef4444"},
        labels={"valor": "Valor", "classe": "Classe"},
    )
    fig.update_yaxes(matches=None)
    fig.update_xaxes(title_text="")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
    fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
    return plotly_layout(fig, "Perfil comparativo por classe")


def data_quality_chart(summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        summary,
        x="indicador",
        y="quantidade",
        color="status",
        text="quantidade",
        color_discrete_map={"ok": "#22c55e", "atenção": "#f59e0b"},
        labels={"indicador": "Indicador", "quantidade": "Quantidade", "status": "Status"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return plotly_layout(fig, "Dados ausentes e duplicados")


def risk_by_hour(df: pd.DataFrame) -> go.Figure:
    data = df.copy()
    if "hour" not in data.columns:
        if "Time" in data.columns:
            data["hour"] = ((pd.to_numeric(data["Time"], errors="coerce").fillna(0) // 3600) % 24).astype(int)
        else:
            data["hour"] = 0

    grouped = (
        data.assign(alerta=(data["prediction"] == 1).astype(int))
        .groupby("hour", as_index=False)
        .agg(transações=("prediction", "size"), alertas=("alerta", "sum"), risco_médio=("risk_score", "mean"))
    )
    grouped["taxa_alerta"] = grouped["alertas"] / grouped["transações"].clip(lower=1)

    fig = go.Figure()
    fig.add_bar(x=grouped["hour"], y=grouped["transações"], name="transações", marker_color="#334155")
    fig.add_scatter(
        x=grouped["hour"],
        y=grouped["alertas"],
        name="alertas",
        mode="lines+markers",
        line=dict(color="#ef4444", width=3),
    )
    return plotly_layout(fig, "Volume e alertas por hora")


def risk_bucket_bar(df: pd.DataFrame) -> go.Figure:
    order = ["baixo", "moderado", "alto", "critico"]
    grouped = (
        df["risk_bucket"]
        .value_counts()
        .reindex(order, fill_value=0)
        .rename_axis("risk_bucket")
        .reset_index(name="transações")
    )
    grouped["faixa_de_risco"] = grouped["risk_bucket"].replace({"critico": "crítico"})
    fig = px.bar(
        grouped,
        x="faixa_de_risco",
        y="transações",
        color="faixa_de_risco",
        color_discrete_map={**COLOR_MAP, "crítico": COLOR_MAP["critico"]},
        text="transações",
        labels={"faixa_de_risco": "Faixa de risco", "transações": "Transações"},
    )
    fig.update_traces(textposition="outside")
    return plotly_layout(fig, "Distribuição por faixa de risco")


def confusion_heatmap(df: pd.DataFrame, threshold: float) -> go.Figure:
    predictions = (df["risk_score"] >= threshold).astype(int)
    matrix = confusion_matrix(df["Class"], predictions, labels=[0, 1])
    text = np.array(
        [
            ["TN<br>" + str(matrix[0, 0]), "FP<br>" + str(matrix[0, 1])],
            ["FN<br>" + str(matrix[1, 0]), "TP<br>" + str(matrix[1, 1])],
        ]
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Prevista legítima", "Prevista fraude"],
            y=["Real legítima", "Real fraude"],
            text=text,
            texttemplate="%{text}",
            colorscale=[[0, "#111827"], [0.5, "#f59e0b"], [1, "#ef4444"]],
            showscale=False,
        )
    )
    return plotly_layout(fig, "Matriz de confusão")


def threshold_curve(df: pd.DataFrame) -> go.Figure:
    rows = []
    y_true = df["Class"].to_numpy()
    scores = df["risk_score"].to_numpy()
    for threshold in np.linspace(0.01, 0.99, 80):
        pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "alertas": int(pred.sum()),
            }
        )
    curve = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_scatter(x=curve["threshold"], y=curve["precision"], mode="lines", name="precisão", line=dict(color="#38bdf8"))
    fig.add_scatter(x=curve["threshold"], y=curve["recall"], mode="lines", name="recall", line=dict(color="#ef4444"))
    return plotly_layout(fig, "Precisão e recall por threshold")


def model_comparison(metrics: pd.DataFrame) -> go.Figure:
    if metrics.empty:
        return plotly_layout(go.Figure(), "Comparação dos modelos")

    columns = ["test_f2", "test_pr_auc", "test_recall", "test_precision"]
    available = [column for column in columns if column in metrics.columns]
    data = metrics[["model", *available]].melt(
        id_vars="model",
        var_name="métrica",
        value_name="valor",
    )
    labels = {
        "test_f2": "F2",
        "test_pr_auc": "PR-AUC",
        "test_recall": "Recall",
        "test_precision": "Precisão",
    }
    data["métrica"] = data["métrica"].map(labels).fillna(data["métrica"])

    fig = px.bar(
        data,
        x="model",
        y="valor",
        color="métrica",
        barmode="group",
        color_discrete_sequence=["#38bdf8", "#22c55e", "#ef4444", "#f59e0b"],
        labels={"model": "Modelo", "valor": "Valor"},
    )
    fig.update_yaxes(range=[0, 1])
    return plotly_layout(fig, "Comparação dos modelos no teste")


def cost_curve_chart(curve: pd.DataFrame, selected_threshold: float, best_threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=curve["threshold"],
        y=curve["total_cost"],
        mode="lines",
        name="custo total",
        line=dict(color="#f59e0b", width=3),
    )
    fig.add_vline(
        x=selected_threshold,
        line_color="#38bdf8",
        line_dash="dash",
        annotation_text="threshold atual",
        annotation_position="top left",
    )
    fig.add_vline(
        x=best_threshold,
        line_color="#22c55e",
        line_dash="dot",
        annotation_text="menor custo",
        annotation_position="top right",
    )
    return plotly_layout(fig, "Custo estimado por threshold")


def cost_scenario_chart(scenarios: pd.DataFrame) -> go.Figure:
    data = scenarios.copy()
    data["custo_label"] = data["custo_total"].map(lambda value: f"R$ {value:,.0f}".replace(",", "."))
    fig = px.bar(
        data,
        x="cenário",
        y="custo_total",
        color="cenário",
        text="custo_label",
        color_discrete_sequence=["#38bdf8", "#22c55e", "#f59e0b"],
        labels={"cenário": "Cenário", "custo_total": "Custo total"},
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
    return plotly_layout(fig, "Comparação de cenários de negócio")


def feature_importance_chart(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = importance.sort_values("importance", ascending=False).head(top_n).copy()
    total = data["importance"].sum()
    data["participação"] = np.where(total > 0, data["importance"] / total, 0)
    data["rótulo"] = data["participação"].map(lambda value: f"{value:.1%}")
    data = data.sort_values("importance")

    fig = px.bar(
        data,
        x="importance",
        y="feature",
        orientation="h",
        color="participação",
        text="rótulo",
        color_continuous_scale=["#38bdf8", "#f59e0b", "#ef4444"],
        labels={"importance": "Importância média absoluta", "feature": "Variável"},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    return plotly_layout(fig, "SHAP global: variáveis mais influentes")


def local_explanation_chart(local_explanation: pd.DataFrame, top_n: int = 10) -> go.Figure:
    data = local_explanation.reindex(local_explanation["contribution"].abs().sort_values(ascending=False).index).head(top_n).copy()
    data["sinal"] = np.where(data["contribution"] >= 0, "aumenta risco", "reduz risco")
    data = data.sort_values("contribution")

    fig = px.bar(
        data,
        x="contribution",
        y="feature",
        orientation="h",
        color="sinal",
        color_discrete_map={"aumenta risco": "#ef4444", "reduz risco": "#38bdf8"},
        hover_data=["feature_value", "risk_score", "class"],
        labels={"contribution": "Contribuição SHAP/local", "feature": "Variável", "sinal": "Efeito"},
    )
    fig.add_vline(x=0, line_width=1, line_color="#e5e7eb")
    return plotly_layout(fig, "Explicação local da transação")


def animated_risk_scatter(df: pd.DataFrame) -> go.Figure:
    data = df.sample(min(len(df), 2500), random_state=42).copy()
    if "hour" not in data.columns:
        data["hour"] = ((pd.to_numeric(data.get("Time", 0), errors="coerce").fillna(0) // 3600) % 24).astype(int)

    data["risk_bucket"] = pd.Categorical(
        data["risk_bucket"],
        categories=["baixo", "moderado", "alto", "critico"],
        ordered=True,
    )
    data["faixa_de_risco"] = data["risk_bucket"].astype(str).replace({"critico": "crítico"})
    data["valor_plot"] = data["Amount"].clip(lower=0.01)

    fig = px.scatter(
        data.sort_values("hour"),
        x="risk_score",
        y="valor_plot",
        color="faixa_de_risco",
        size="risk_score",
        animation_frame="hour",
        hover_data=["Amount", "Class", "prediction"],
        color_discrete_map={**COLOR_MAP, "crítico": COLOR_MAP["critico"]},
        log_y=True,
        range_x=[0, 1],
        labels={"risk_score": "Score de risco", "valor_plot": "Valor", "faixa_de_risco": "Faixa de risco"},
    )
    return plotly_layout(fig, "Mapa temporal de risco por hora")
