import json
import logging
import re
import unicodedata
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

# Silencia warnings do Streamlit fora do runtime.
logging.getLogger("streamlit").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# NLTK - stopwords em portugues
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords as _sw

    _PT_STOPS = set(_sw.words("portuguese"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords as _sw

    _PT_STOPS = set(_sw.words("portuguese"))

# Termos extras do dominio (pouco informativos para wordcloud).
_PT_STOPS |= {
    "nagem", "loja", "produto", "produtos", "empresa", "site", "shopping",
    "cliente", "clientes", "dia", "dias", "reclame", "aqui", "editado",
    "aguardo", "falar", "resposta", "atendimento", "quero", "pra", "pro",
    "voces", "ainda", "apos", "fazer", "fiquei", "disse", "ficou", "ter",
}

# ---------------------------------------------------------------------------
# GeoJSON dos estados brasileiros (cache)
# ---------------------------------------------------------------------------
_GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforamerica/"
    "click_that_hood/master/public/data/brazil-states.geojson"
)


@st.cache_data(show_spinner="Carregando mapa do Brasil...")
def _load_geojson() -> dict:
    with urllib.request.urlopen(_GEOJSON_URL, timeout=15) as r:
        geo = json.loads(r.read().decode())
    for feat in geo["features"]:
        feat["id"] = feat["properties"]["sigla"]
    return geo


# ---------------------------------------------------------------------------
# Paleta e template
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Set2
TEMPLATE = "plotly_white"

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_ESTADOS_PESO = {
    "SP": 25, "RJ": 18, "MG": 12, "BA": 8, "PR": 7, "RS": 6, "PE": 10,
    "CE": 5, "PA": 3, "MA": 2, "GO": 2, "AM": 2, "SC": 3, "DF": 3,
    "ES": 2, "PB": 3, "RN": 2, "AL": 2, "PI": 1, "SE": 1,
}

_CIDADES = {
    "SP": ["Sao Paulo", "Campinas", "Santos"],
    "RJ": ["Rio de Janeiro", "Niteroi"],
    "MG": ["Belo Horizonte", "Uberlandia"],
    "BA": ["Salvador", "Feira de Santana"],
    "PR": ["Curitiba", "Londrina"],
    "RS": ["Porto Alegre", "Caxias do Sul"],
    "PE": ["Recife", "Olinda", "Jaboatao"],
    "CE": ["Fortaleza", "Juazeiro do Norte"],
    "PA": ["Belem"], "MA": ["Sao Luis"], "GO": ["Goiania"],
    "AM": ["Manaus"], "SC": ["Florianopolis", "Joinville"],
    "DF": ["Brasilia"], "ES": ["Vitoria", "Vila Velha"],
    "PB": ["Joao Pessoa"], "RN": ["Natal"], "AL": ["Maceio"],
    "PI": ["Teresina"], "SE": ["Aracaju"],
}

_TEMAS = [
    "Atraso na entrega", "Produto com defeito", "Troca e devolucao",
    "Atendimento", "Cobranca indevida", "Propaganda enganosa",
    "Garantia", "Cancelamento", "Reembolso", "Produto nao recebido",
]

_DESCRICOES = [
    "Comprei um produto e nao recebi no prazo informado. Ja se passaram mais de 30 dias.",
    "O produto chegou com defeito e a loja nao quer fazer a troca. Pessimo atendimento.",
    "Fui cobrado duas vezes no cartao. Quero o estorno imediato.",
    "Produto nao corresponde a descricao do site. Propaganda enganosa total.",
    "Solicitei cancelamento ha semanas e nao tive retorno algum.",
    "A garantia nao esta sendo honrada pela assistencia tecnica indicada.",
    "Pedi reembolso e ate agora nada. Estou completamente no prejuizo.",
    "Atendimento pessimo. Ninguem resolve meu problema, ficam jogando de um para outro.",
    "Produto parou de funcionar com menos de um mes de uso, absurdo.",
    "Entrega atrasada e sem previsao. Nenhuma comunicacao da loja, descaso total.",
]


def _generate_mock(n: int = 2_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ufs = list(_ESTADOS_PESO.keys())
    pesos = np.array(list(_ESTADOS_PESO.values()), dtype=float)
    pesos /= pesos.sum()

    estado = rng.choice(ufs, n, p=pesos)
    cidade = np.array([rng.choice(_CIDADES[e]) for e in estado])
    datas = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    tempo = rng.choice(datas, n)
    descs = rng.choice(_DESCRICOES, n)
    descs = np.array([" ".join([d] * rng.integers(1, 5)) for d in descs])

    return pd.DataFrame({
        "TEMPO": tempo,
        "LOCAL": [f"{c} - {e}" for c, e in zip(cidade, estado)],
        "TEMA": rng.choice(_TEMAS, n),
        "STATUS": rng.choice(
            ["Resolvido", "Nao resolvido", "Em analise", "Respondido"],
            n, p=[0.35, 0.30, 0.20, 0.15],
        ),
        "DESCRICAO": descs,
        "CATEGORIA": rng.choice(
            ["Eletronicos", "Eletrodomesticos", "Informatica", "Celulares", "Outros"], n,
        ),
        "CASOS": 1,
    })


# ---------------------------------------------------------------------------
# Carregamento e pre-processamento
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(uploaded=None):
    if uploaded is not None:
        if uploaded.name.lower().endswith(".parquet"):
            return pd.read_parquet(uploaded)
        return pd.read_csv(uploaded)

    for ext in ("csv", "parquet"):
        for folder in (".", "data"):
            p = Path(folder) / f"RECLAMEAQUI_NAGEM.{ext}"
            if p.exists():
                return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
    return None


@st.cache_data
def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().upper() for c in df.columns]

    for col in ("TEMA", "LOCAL", "STATUS", "DESCRICAO", "CATEGORIA"):
        if col in df.columns:
            df[col] = df[col].fillna("Nao informado").astype(str).str.strip()

    if "TEMPO" in df.columns:
        df["TEMPO"] = pd.to_datetime(df["TEMPO"], errors="coerce")
    elif {"ANO", "MES", "DIA"}.issubset(df.columns):
        df["TEMPO"] = pd.to_datetime(
            dict(year=df["ANO"], month=df["MES"], day=df["DIA"]), errors="coerce",
        )

    parts = df["LOCAL"].str.extract(r"^(?P<CIDADE>.*?)\s*-\s*(?P<UF>[A-Z]{2})$")
    df["CIDADE"] = parts["CIDADE"].fillna(df["LOCAL"]).str.strip().str.title()
    df["ESTADO"] = parts["UF"].fillna("NI").str.strip()
    df["CIDADE"] = df["CIDADE"].replace({"": "Nao informado", "--": "Nao informado"})

    for col in ("STATUS", "TEMA", "CATEGORIA"):
        df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

    df["TEXTO_LEN"] = df["DESCRICAO"].astype(str).str.len()
    df["MES_ANO"] = df["TEMPO"].dt.to_period("M").dt.to_timestamp()
    df["ANO"] = df["TEMPO"].dt.year.astype("Int64")
    df["CASOS"] = pd.to_numeric(df.get("CASOS", 1), errors="coerce").fillna(1)

    bins = [-1, 499, 999, 1_499, 10_000_000]
    labels = ["Ate 499", "500-999", "1000-1499", "1500+"]
    df["FAIXA_TEXTO"] = pd.cut(df["TEXTO_LEN"], bins=bins, labels=labels)

    return df


# ---------------------------------------------------------------------------
# Graficos
# ---------------------------------------------------------------------------

def chart_time_series(df: pd.DataFrame) -> go.Figure:
    ts = (
        df.groupby("MES_ANO", as_index=False)["CASOS"]
        .sum()
        .sort_values("MES_ANO")
    )
    ts["MM3"] = ts["CASOS"].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["MES_ANO"], y=ts["CASOS"],
        mode="lines+markers", name="Reclamacoes",
        line=dict(color=PALETTE[0], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=ts["MES_ANO"], y=ts["MM3"],
        mode="lines", name="Media Movel (3m)",
        line=dict(color=PALETTE[1], width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Evolucao Temporal das Reclamacoes",
        xaxis_title="Mes", yaxis_title="Quantidade",
        template=TEMPLATE, height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_choropleth(df: pd.DataFrame, ano: int, geojson: dict) -> go.Figure:
    subset = df[df["ANO"] == ano] if ano else df
    by_uf = (
        subset.groupby("ESTADO", as_index=False)["CASOS"]
        .sum()
        .query("ESTADO != 'NI'")
    )

    fig = px.choropleth(
        by_uf,
        geojson=geojson,
        locations="ESTADO",
        color="CASOS",
        hover_name="ESTADO",
        color_continuous_scale="YlOrRd",
        title=f"Distribuicao Geografica - {ano}",
    )
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        template=TEMPLATE, height=500,
        margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_colorbar=dict(title="Qtd"),
    )
    return fig


def chart_pareto(df: pd.DataFrame) -> go.Figure:
    p = (
        df.groupby("ESTADO", as_index=False)["CASOS"]
        .sum()
        .sort_values("CASOS", ascending=False)
    )
    p["PCT_ACUM"] = p["CASOS"].cumsum() / p["CASOS"].sum() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=p["ESTADO"], y=p["CASOS"], name="Reclamacoes",
        marker_color=PALETTE[0],
    ))
    fig.add_trace(go.Scatter(
        x=p["ESTADO"], y=p["PCT_ACUM"], name="% Acumulado",
        yaxis="y2", mode="lines+markers",
        line=dict(color=PALETTE[3], width=2),
    ))
    fig.update_layout(
        title="Pareto - Reclamacoes por Estado",
        xaxis_title="Estado", yaxis_title="Quantidade",
        yaxis2=dict(title="% Acumulado", overlaying="y", side="right", range=[0, 105]),
        template=TEMPLATE, height=420,
        margin=dict(l=20, r=40, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def chart_donut(df: pd.DataFrame) -> go.Figure:
    s = (
        df.groupby("STATUS", as_index=False)["CASOS"]
        .sum()
        .sort_values("CASOS", ascending=False)
    )
    fig = px.pie(
        s, names="STATUS", values="CASOS",
        hole=0.45, title="Proporcao por Status",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        template=TEMPLATE, height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


def chart_boxplot(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df, x="STATUS", y="TEXTO_LEN", points="outliers",
        title="Tamanho do Texto por Status",
        color="STATUS", color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        xaxis_title="Status", yaxis_title="Caracteres",
        template=TEMPLATE, height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    return fig


@st.cache_data
def _clean_text(series: pd.Series) -> str:
    def _norm(text: str) -> str:
        text = unicodedata.normalize("NFKD", text.lower())
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r"[^a-z\s]", " ", text)
        return " ".join(t for t in text.split() if len(t) > 2 and t not in _PT_STOPS)
    return " ".join(series.fillna("").astype(str).map(_norm))


def chart_wordcloud(df: pd.DataFrame):
    text = _clean_text(df["DESCRICAO"])
    if not text.strip():
        return None
    wc = WordCloud(
        width=1200, height=600, background_color="white",
        collocations=False, max_words=120, colormap="viridis",
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Dashboard ReclameAqui - Nagem",
        page_icon=":bar_chart:",
        layout="wide",
    )

    # CSS para destaque nas metricas
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: #f8f9fa;
            border-radius: .5rem;
            padding: 1rem;
            border-left: 4px solid #4e79a7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(":bar_chart: Dashboard ReclameAqui - Nagem")
    st.caption(
        "Analise exploratoria de reclamacoes: gargalos regionais, "
        "eficiencia de atendimento, sazonalidade e temas recorrentes."
    )

    # --- Sidebar ---------------------------------------------------------
    with st.sidebar:
        st.header(":file_folder: Base de dados")
        uploaded = st.file_uploader("CSV ou Parquet", type=["csv", "parquet"])

    df_raw = load_data(uploaded)
    use_mock = df_raw is None
    if use_mock:
        df_raw = _generate_mock()
        st.sidebar.info(
            "Usando dados de demonstracao. "
            "Envie seu arquivo para analise real."
        )

    df = preprocess(df_raw)

    with st.sidebar:
        st.header(":mag: Filtros")
        sel_est = st.multiselect(
            "Estado",
            sorted(df["ESTADO"].unique()),
            default=sorted(df["ESTADO"].unique()),
        )
        sel_sta = st.multiselect(
            "Status",
            sorted(df["STATUS"].unique()),
            default=sorted(df["STATUS"].unique()),
        )
        faixas_opts = df["FAIXA_TEXTO"].dropna().astype(str).unique().tolist()
        sel_faixa = st.multiselect(
            "Faixa de tamanho do texto",
            faixas_opts,
            default=faixas_opts,
        )

    df_f = df[
        df["ESTADO"].isin(sel_est)
        & df["STATUS"].isin(sel_sta)
        & df["FAIXA_TEXTO"].astype(str).isin(sel_faixa)
    ]

    # --- KPIs ------------------------------------------------------------
    total = int(df_f["CASOS"].sum())
    n_est = df_f["ESTADO"].nunique()
    n_cid = df_f["CIDADE"].nunique()
    pct_res = (df_f["STATUS"].eq("Resolvido").mean() * 100) if len(df_f) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reclamacoes", f"{total:,}".replace(",", "."))
    k2.metric("Estados", n_est)
    k3.metric("Cidades", n_cid)
    k4.metric("% Resolvido", f"{pct_res:.1f}%")

    st.markdown("---")

    # --- Serie temporal --------------------------------------------------
    st.plotly_chart(chart_time_series(df_f), width="stretch")

    # --- Mapa coropletico ------------------------------------------------
    geojson = _load_geojson()
    anos = sorted(df_f["ANO"].dropna().unique().tolist())
    if not anos:
        anos = sorted(df["ANO"].dropna().unique().tolist())
    ano = st.select_slider("Ano do mapa", options=anos, value=anos[-1])
    st.plotly_chart(chart_choropleth(df_f, ano, geojson), width="stretch")

    # --- Pareto + Donut --------------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_pareto(df_f), width="stretch")
    with c2:
        st.plotly_chart(chart_donut(df_f), width="stretch")

    # --- Boxplot + WordCloud ---------------------------------------------
    c3, c4 = st.columns([1.2, 1])
    with c3:
        st.plotly_chart(chart_boxplot(df_f), width="stretch")
    with c4:
        st.subheader("WordCloud das Descricoes")
        wc_fig = chart_wordcloud(df_f)
        if wc_fig:
            st.pyplot(wc_fig)
            plt.close(wc_fig)
        else:
            st.warning("Texto insuficiente para gerar a nuvem de palavras.")

    # --- Top temas e cidades ---------------------------------------------
    st.markdown("---")
    c5, c6 = st.columns(2)
    with c5:
        top_t = (
            df_f.groupby("TEMA", as_index=False)["CASOS"]
            .sum()
            .nlargest(10, "CASOS")
        )
        fig = px.bar(
            top_t, x="CASOS", y="TEMA", orientation="h",
            title="Top 10 Temas", color_discrete_sequence=[PALETTE[0]],
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            template=TEMPLATE, height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    with c6:
        top_c = (
            df_f.assign(LOC=df_f["CIDADE"] + " - " + df_f["ESTADO"])
            .groupby("LOC", as_index=False)["CASOS"]
            .sum()
            .nlargest(10, "CASOS")
        )
        fig = px.bar(
            top_c, x="CASOS", y="LOC", orientation="h",
            title="Top 10 Cidades", color_discrete_sequence=[PALETTE[2]],
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            template=TEMPLATE, height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    # --- Leitura executiva -----------------------------------------------
    st.markdown("### Leitura Executiva")
    if len(df_f):
        top_e = df_f.groupby("ESTADO")["CASOS"].sum().idxmax()
        top_loc = df_f.groupby(["CIDADE", "ESTADO"])["CASOS"].sum().idxmax()
        top_tema = df_f.groupby("TEMA")["CASOS"].sum().idxmax()
        st_long = df_f.groupby("STATUS")["TEXTO_LEN"].mean().idxmax()
        st.markdown(
            f"- Maior concentracao: **{top_e}**\n"
            f"- Cidade lider: **{top_loc[0]} - {top_loc[1]}**\n"
            f"- Tema mais frequente: **{top_tema}**\n"
            f"- Status com textos mais longos: **{st_long}** "
            f"- pode indicar casos mais complexos."
        )
    else:
        st.warning("Nenhum dado com os filtros atuais. Ajuste os filtros.")


if __name__ == "__main__":
    main()
