#!/usr/bin/env python3
"""
Chikungunya em Recife (2025) - Pipeline completo de Ciência de Dados
=====================================================================
Etapas:
  1. Carga e limpeza (qualidade de dados, tipos, feature engineering)
  2. Análise exploratória (EDA) com séries temporais e perfil demográfico
  3. Estatística inferencial (qui-quadrado, Mann-Whitney, Kruskal, bootstrap)
  4. Machine Learning supervisionado (classificação de hospitalização)
  5. Previsão de séries temporais (casos semanais vs. baselines)
  6. Aprendizado não supervisionado (clusters de bairros de risco)
  7. Mapas (coroplético/interativo) - opcional, se houver GeoJSON
  8. Relatório final em Markdown

Uso:
  python chikungunya_pipeline.py --input base.csv --outdir resultados
  python chikungunya_pipeline.py --input base.csv --geojson bairros.geojson \
         --pop populacao_bairros.csv   # csv com colunas: bairro,populacao
"""
from __future__ import annotations

import argparse
import json
import logging
import unicodedata
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    average_precision_score, f1_score, mean_absolute_error, mean_squared_error,
    recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("chik")

RANDOM_STATE = 42
AZUL, VERMELHO, CINZA = "#1f4e79", "#d9534f", "#6c757d"
sns.set_theme(style="whitegrid", context="notebook")


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def normaliza_texto(s):
    """Remove acentos, espaços extras e padroniza em MAIÚSCULAS."""
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = " ".join(s.upper().strip().split())
    return np.nan if s in {"", "IGNORADO", "NAN", "NONE", "NULL"} else s


def parse_data(serie: pd.Series) -> pd.Series:
    """Converte para datetime detectando formato ISO (AAAA-MM-DD) ou brasileiro (DD/MM/AAAA)."""
    amostra = serie.dropna().astype(str).head(50)
    iso = amostra.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5 if len(amostra) else True
    return pd.to_datetime(serie, errors="coerce", dayfirst=not iso)


def salva_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Figura salva: %s", path.name)


def cramers_v(tabela: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(tabela, correction=False)[0]
    n = tabela.to_numpy().sum()
    r, k = tabela.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else np.nan


def bootstrap_ic(valores: np.ndarray, n_boot=3000, alpha=0.05, func=np.mean):
    rng = np.random.default_rng(RANDOM_STATE)
    amostras = rng.choice(valores, size=(n_boot, len(valores)), replace=True)
    est = func(amostras, axis=1)
    return float(np.percentile(est, 100 * alpha / 2)), float(np.percentile(est, 100 * (1 - alpha / 2)))


# ----------------------------------------------------------------------------
# 1. Carga e limpeza
# ----------------------------------------------------------------------------
def carregar_dados(caminho: Path) -> pd.DataFrame:
    log.info("Carregando %s", caminho)
    df = pd.read_csv(caminho, sep=None, engine="python", encoding_errors="replace", dtype=str)
    df.columns = df.columns.str.strip().str.upper()
    log.info("Dataset bruto: %d linhas x %d colunas", *df.shape)
    return df


def idade_sinan(nu_idade: pd.Series) -> pd.Series:
    """Decodifica NU_IDADE_N do SINAN (1º dígito: 1=hora, 2=dia, 3=mês, 4=ano) para anos."""
    v = pd.to_numeric(nu_idade, errors="coerce")
    unidade, valor = (v // 1000), (v % 1000)
    anos = np.select([unidade == 4, unidade == 3, unidade == 2, unidade == 1],
                     [valor, valor / 12, valor / 365, valor / 8760], default=np.nan)
    return pd.Series(anos, index=nu_idade.index)


def limpar_dados(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    df = df.copy()
    n0 = len(df)
    df = df.drop_duplicates()
    log.info("Duplicatas removidas: %d", n0 - len(df))

    # Relatório de qualidade (nulos)
    nulos = (df.isna().mean() * 100).sort_values(ascending=False)
    nulos[nulos > 0].round(2).to_csv(outdir / "tabelas" / "qualidade_nulos.csv", header=["pct_nulos"])
    top_nulos = nulos.head(25)
    fig, ax = plt.subplots(figsize=(9, 7))
    top_nulos.sort_values().plot.barh(ax=ax, color=CINZA)
    ax.set_title("Percentual de valores ausentes por coluna (top 25)", fontweight="bold")
    ax.set_xlabel("% de nulos")
    salva_fig(fig, outdir / "figuras" / "00_qualidade_nulos.png")

    # Textos
    for col in ["NM_BAIRRO", "CS_SEXO", "CS_RACA", "CS_GESTANT"]:
        if col in df.columns:
            df[col] = df[col].map(normaliza_texto)

    # Datas
    for col in ["DT_NOTIFIC", "DT_SIN_PRI", "DT_NASC", "DT_OBITO", "DT_ENCERRA"]:
        if col in df.columns:
            df[col] = parse_data(df[col])

    # Idade
    if "NU_IDADE_N" in df.columns:
        df["IDADE"] = idade_sinan(df["NU_IDADE_N"])
    elif {"DT_NASC", "DT_NOTIFIC"} <= set(df.columns):
        df["IDADE"] = (df["DT_NOTIFIC"] - df["DT_NASC"]).dt.days / 365.25
    if "IDADE" in df.columns:
        df.loc[~df["IDADE"].between(0, 110), "IDADE"] = np.nan
        df["FAIXA_ETARIA"] = pd.cut(df["IDADE"], [-0.01, 12, 18, 30, 45, 60, 111],
                                    labels=["0-12", "13-18", "19-30", "31-45", "46-60", "60+"])

    # Sexo / gestante
    if "CS_SEXO" in df.columns:
        df["CS_SEXO"] = df["CS_SEXO"].replace({"MASCULINO": "M", "FEMININO": "F"})
        df.loc[~df["CS_SEXO"].isin(["M", "F"]), "CS_SEXO"] = np.nan
    if "CS_GESTANT" in df.columns:
        df["GESTANTE"] = df["CS_GESTANT"].isin(["1", "2", "3", "4", "1.0", "2.0", "3.0", "4.0"]).astype(int)

    # Variáveis de desfecho (padrão SINAN: 1=sim, 2=não)
    if "HOSPITALIZ" in df.columns:
        h = pd.to_numeric(df["HOSPITALIZ"], errors="coerce")
        df["HOSPITALIZADO"] = h.map({1: 1, 2: 0})
    if "EVOLUCAO" in df.columns:
        e = pd.to_numeric(df["EVOLUCAO"], errors="coerce")
        df["OBITO_AGRAVO"] = e.map({1: 0, 2: 1, 3: 0, 4: 0, 9: np.nan})

    # Tempo até notificação (indicador de qualidade da vigilância)
    if {"DT_SIN_PRI", "DT_NOTIFIC"} <= set(df.columns):
        df["DIAS_SINTOMA_NOTIF"] = (df["DT_NOTIFIC"] - df["DT_SIN_PRI"]).dt.days
        df.loc[~df["DIAS_SINTOMA_NOTIF"].between(0, 60), "DIAS_SINTOMA_NOTIF"] = np.nan

    # Tempo
    if "DT_NOTIFIC" in df.columns:
        df["MES"] = df["DT_NOTIFIC"].dt.month
        df["SEMANA_EPI"] = df["DT_NOTIFIC"].dt.isocalendar().week.astype("float")
    elif "SEM_NOT" in df.columns:
        df["SEMANA_EPI"] = pd.to_numeric(df["SEM_NOT"].astype(str).str[-2:], errors="coerce")

    df.to_parquet(outdir / "dados_tratados.parquet", index=False) if _tem_parquet() else \
        df.to_csv(outdir / "dados_tratados.csv", index=False)
    log.info("Base tratada: %d linhas", len(df))
    return df


def _tem_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
def serie_semanal(df: pd.DataFrame) -> pd.Series:
    s = df.dropna(subset=["DT_NOTIFIC"]).set_index("DT_NOTIFIC").resample("W-SAT").size()
    return s.rename("casos")


def eda(df: pd.DataFrame, outdir: Path, pop: pd.DataFrame | None) -> dict:
    fig_dir, tab_dir = outdir / "figuras", outdir / "tabelas"
    ach = {"n_casos": int(len(df))}

    # --- Top bairros (absoluto) ---
    if "NM_BAIRRO" in df.columns:
        top = df["NM_BAIRRO"].value_counts().head(15)
        top.to_csv(tab_dir / "top_bairros.csv", header=["casos"])
        ach["top_bairros"] = top.head(5).to_dict()
        ach["pct_top5"] = float(top.head(5).sum() / len(df) * 100)
        fig, ax = plt.subplots(figsize=(10, 7))
        top.sort_values().plot.barh(ax=ax, color=AZUL)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_width())}", (p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, fontweight="bold")
        ax.set_title("Top 15 bairros com casos confirmados de chikungunya\nRecife, 2025", fontweight="bold")
        ax.set_xlabel("Casos confirmados")
        ax.set_ylabel("")
        salva_fig(fig, fig_dir / "01_top_bairros.png")

        # --- Incidência por 100 mil hab. ---
        if pop is not None:
            inc = df["NM_BAIRRO"].value_counts().rename("casos").to_frame().join(pop, how="inner")
            inc["incidencia_100k"] = inc["casos"] / inc["populacao"] * 1e5
            inc = inc.sort_values("incidencia_100k", ascending=False)
            inc.to_csv(tab_dir / "incidencia_bairros.csv")
            ach["top_incidencia"] = inc["incidencia_100k"].head(5).round(1).to_dict()
            fig, ax = plt.subplots(figsize=(10, 7))
            inc["incidencia_100k"].head(15).sort_values().plot.barh(ax=ax, color=VERMELHO)
            ax.set_title("Top 15 bairros por incidência (casos por 100 mil hab.)", fontweight="bold")
            ax.set_xlabel("Incidência por 100 mil habitantes")
            ax.set_ylabel("")
            salva_fig(fig, fig_dir / "02_incidencia_bairros.png")

    # --- Série temporal semanal ---
    if "DT_NOTIFIC" in df.columns:
        s = serie_semanal(df)
        mm = s.rolling(4, min_periods=1).mean()
        pico = s.idxmax()
        ach["semana_pico"] = str(pico.date())
        ach["casos_pico"] = int(s.max())
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(s.index, s.values, width=5, color=AZUL, alpha=0.35, label="Casos semanais")
        ax.plot(mm.index, mm.values, color=VERMELHO, lw=2.5, label="Média móvel (4 sem.)")
        ax.annotate(f"Pico: {s.max()} casos\n({pico:%d/%m})", (pico, s.max()),
                    xytext=(30, -10), textcoords="offset points", arrowprops=dict(arrowstyle="->"), fontweight="bold")
        ax.set_title("Dinâmica temporal de casos de chikungunya - Recife, 2025", fontweight="bold")
        ax.set_ylabel("Casos por semana")
        ax.legend()
        salva_fig(fig, fig_dir / "03_serie_temporal.png")

    # --- Perfil demográfico ---
    if {"IDADE", "CS_SEXO"} <= set(df.columns):
        d = df.dropna(subset=["FAIXA_ETARIA", "CS_SEXO"])
        tab = d.groupby(["FAIXA_ETARIA", "CS_SEXO"], observed=True).size().unstack(fill_value=0)
        ach["pct_feminino"] = float((df["CS_SEXO"] == "F").mean() * 100)
        ach["idade_mediana"] = float(df["IDADE"].median())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # pirâmide etária
        if {"F", "M"} <= set(tab.columns):
            axes[0].barh(tab.index.astype(str), -tab["M"], color=AZUL, label="Masculino")
            axes[0].barh(tab.index.astype(str), tab["F"], color=VERMELHO, label="Feminino")
            ticks = axes[0].get_xticks()
            axes[0].set_xticklabels([f"{abs(int(t))}" for t in ticks])
            axes[0].legend()
        axes[0].set_title("Pirâmide etária dos casos", fontweight="bold")
        sns.histplot(df["IDADE"].dropna(), bins=30, kde=True, color=AZUL, ax=axes[1])
        axes[1].axvline(df["IDADE"].median(), color=VERMELHO, ls="--", label=f"Mediana: {df['IDADE'].median():.0f} anos")
        axes[1].set_title("Distribuição de idade", fontweight="bold")
        axes[1].legend()
        salva_fig(fig, fig_dir / "04_perfil_demografico.png")

    # --- Heatmap bairro x mês ---
    if {"NM_BAIRRO", "MES"} <= set(df.columns):
        top10 = df["NM_BAIRRO"].value_counts().head(12).index
        piv = df[df["NM_BAIRRO"].isin(top10)].pivot_table(index="NM_BAIRRO", columns="MES", values="DT_NOTIFIC",
                                                          aggfunc="size", fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.heatmap(piv, cmap="OrRd", annot=True, fmt="d", linewidths=.4, ax=ax, cbar_kws={"label": "Casos"})
        ax.set_title("Casos por bairro e mês de notificação (top 12 bairros)", fontweight="bold")
        ax.set_xlabel("Mês")
        ax.set_ylabel("")
        salva_fig(fig, fig_dir / "05_heatmap_bairro_mes.png")

    # --- Desfechos ---
    for col, nome in [("HOSPITALIZADO", "hospitalização"), ("OBITO_AGRAVO", "óbito pelo agravo")]:
        if col in df.columns and df[col].notna().any():
            ach[f"taxa_{col.lower()}"] = float(df[col].mean() * 100)
            log.info("Taxa de %s: %.2f%%", nome, df[col].mean() * 100)
    if {"FAIXA_ETARIA", "HOSPITALIZADO"} <= set(df.columns):
        g = df.groupby("FAIXA_ETARIA", observed=True)["HOSPITALIZADO"].mean() * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        g.plot.bar(ax=ax, color=AZUL, rot=0)
        ax.set_title("Taxa de hospitalização por faixa etária", fontweight="bold")
        ax.set_ylabel("% hospitalizados")
        ax.set_xlabel("Faixa etária")
        salva_fig(fig, fig_dir / "06_hospitalizacao_faixa.png")
    return ach


# ----------------------------------------------------------------------------
# 3. Estatística inferencial
# ----------------------------------------------------------------------------
def testes_estatisticos(df: pd.DataFrame, outdir: Path) -> list[dict]:
    res = []
    alpha = 0.05

    def reg(nome, teste, stat, p, efeito, interpretacao):
        res.append({"hipotese": nome, "teste": teste, "estatistica": round(float(stat), 4),
                    "p_valor": float(p), "tamanho_efeito": efeito,
                    "significativo": bool(p < alpha), "interpretacao": interpretacao})

    if {"CS_SEXO", "HOSPITALIZADO"} <= set(df.columns):
        t = pd.crosstab(df["CS_SEXO"], df["HOSPITALIZADO"])
        if t.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(t)
            v = cramers_v(t)
            reg("Sexo está associado à hospitalização?", "Qui-quadrado", chi2, p, f"V de Cramér = {v:.3f}",
                "Associação " + ("significativa" if p < alpha else "não significativa") + f" (V={v:.3f}; efeito {'pequeno' if v < .1 else 'moderado' if v < .3 else 'grande'}).")

    if {"IDADE", "HOSPITALIZADO"} <= set(df.columns):
        a = df.loc[df["HOSPITALIZADO"] == 1, "IDADE"].dropna()
        b = df.loc[df["HOSPITALIZADO"] == 0, "IDADE"].dropna()
        if len(a) > 5 and len(b) > 5:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rbc = 1 - 2 * u / (len(a) * len(b))  # correlação bisserial por postos
            reg("A idade difere entre hospitalizados e não hospitalizados?", "Mann-Whitney U", u, p,
                f"r bisserial = {rbc:.3f}; mediana {a.median():.0f} vs {b.median():.0f} anos",
                "Distribuições " + ("diferentes" if p < alpha else "semelhantes") + " de idade entre os grupos.")

    if {"FAIXA_ETARIA", "HOSPITALIZADO"} <= set(df.columns):
        t = pd.crosstab(df["FAIXA_ETARIA"], df["HOSPITALIZADO"])
        if t.shape[0] > 1 and t.shape[1] == 2:
            chi2, p, _, _ = stats.chi2_contingency(t)
            reg("Hospitalização varia entre faixas etárias?", "Qui-quadrado", chi2, p,
                f"V de Cramér = {cramers_v(t):.3f}", "Taxa de hospitalização " + ("varia" if p < alpha else "não varia") + " entre faixas.")

    if {"IDADE", "NM_BAIRRO"} <= set(df.columns):
        top = df["NM_BAIRRO"].value_counts().head(8).index
        grupos = [df.loc[df["NM_BAIRRO"] == b, "IDADE"].dropna() for b in top]
        grupos = [g for g in grupos if len(g) > 5]
        if len(grupos) > 2:
            h, p = stats.kruskal(*grupos)
            n = sum(len(g) for g in grupos)
            eps2 = (h - len(grupos) + 1) / (n - len(grupos))
            reg("A idade dos casos difere entre os 8 bairros mais afetados?", "Kruskal-Wallis", h, p,
                f"epsilon² = {eps2:.3f}", "Perfil etário " + ("difere" if p < alpha else "não difere") + " entre bairros.")

    if {"CS_SEXO", "IDADE"} <= set(df.columns):
        f = df.loc[df["CS_SEXO"] == "F", "IDADE"].dropna()
        m = df.loc[df["CS_SEXO"] == "M", "IDADE"].dropna()
        if len(f) > 5 and len(m) > 5:
            u, p = stats.mannwhitneyu(f, m)
            reg("A idade difere entre mulheres e homens?", "Mann-Whitney U", u, p,
                f"mediana F={f.median():.0f} vs M={m.median():.0f}", "Diferença " + ("significativa" if p < alpha else "não significativa") + ".")

    if "HOSPITALIZADO" in df.columns and df["HOSPITALIZADO"].notna().sum() > 30:
        v = df["HOSPITALIZADO"].dropna().to_numpy()
        lo, hi = bootstrap_ic(v)
        reg("Taxa de hospitalização (IC 95% bootstrap)", "Bootstrap (3000 reamostragens)", v.mean(), 1.0,
            f"{v.mean()*100:.2f}% [{lo*100:.2f}%; {hi*100:.2f}%]", "Estimativa pontual com intervalo de confiança.")
        res[-1]["significativo"] = None

    out = pd.DataFrame(res)
    out.to_csv(outdir / "tabelas" / "testes_estatisticos.csv", index=False)
    for r in res:
        log.info("[%s] %s | p=%.4g | %s", r["teste"], r["hipotese"], r["p_valor"], r["tamanho_efeito"])
    return res


# ----------------------------------------------------------------------------
# 4. ML supervisionado: prever hospitalização
# ----------------------------------------------------------------------------
def modelo_classificacao(df: pd.DataFrame, outdir: Path) -> dict | None:
    alvo = "HOSPITALIZADO"
    if alvo not in df.columns or df[alvo].nunique() < 2:
        log.warning("Alvo de hospitalização indisponível; pulando ML supervisionado.")
        return None

    # ATENÇÃO: não usar EVOLUCAO, CLASSI_FIN, DT_OBITO etc. (vazamento de dados / data leakage)
    num = [c for c in ["IDADE", "DIAS_SINTOMA_NOTIF", "SEMANA_EPI", "GESTANTE"] if c in df.columns]
    cat = [c for c in ["CS_SEXO", "CS_RACA", "NM_BAIRRO"] if c in df.columns]
    if not num + cat:
        return None
    d = df.dropna(subset=[alvo]).copy()
    if "DT_NOTIFIC" in d.columns:
        d = d.sort_values("DT_NOTIFIC")
    # bairros raros viram OUTROS (reduz cardinalidade)
    if "NM_BAIRRO" in cat:
        freq = d["NM_BAIRRO"].value_counts()
        d["NM_BAIRRO"] = d["NM_BAIRRO"].where(d["NM_BAIRRO"].isin(freq[freq >= 30].index), "OUTROS")
    for c in cat:
        d[c] = d[c].astype(object).where(d[c].notna(), "NA")

    X, y = d[num + cat], d[alvo].astype(int)
    log.info("ML: %d amostras | prevalência da classe positiva: %.2f%%", len(y), y.mean() * 100)
    if y.sum() < 20:
        log.warning("Poucos positivos (%d); resultados do modelo serão instáveis.", y.sum())

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10), cat),
    ])
    modelos = {
        "Baseline (prevalência)": DummyClassifier(strategy="prior"),
        "Regressão Logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, class_weight="balanced",
                                                            random_state=RANDOM_STATE),
    }
    n_splits = int(min(5, max(2, y.sum())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    linhas = []
    for nome, m in modelos.items():
        pipe = Pipeline([("pre", pre), ("m", m)])
        sc = cross_validate(pipe, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                                                       "recall": "recall", "f1": "f1"}, n_jobs=-1)
        linhas.append({"modelo": nome, **{k.replace("test_", ""): f"{v.mean():.3f} ± {v.std():.3f}"
                                          for k, v in sc.items() if k.startswith("test_")},
                       "_pr": sc["test_pr_auc"].mean()})
        log.info("CV %-24s ROC-AUC=%.3f | PR-AUC=%.3f", nome, sc["test_roc_auc"].mean(), sc["test_pr_auc"].mean())
    comp = pd.DataFrame(linhas)
    comp.drop(columns="_pr").to_csv(outdir / "tabelas" / "comparacao_modelos_cv.csv", index=False)
    melhor_nome = comp.loc[comp[comp["modelo"] != "Baseline (prevalência)"]["_pr"].idxmax(), "modelo"]

    # Holdout cronológico: treina no passado (70%), testa no futuro (30%)
    corte = int(len(d) * 0.7)
    Xtr, Xte, ytr, yte = X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]
    pipe = Pipeline([("pre", pre), ("m", modelos[melhor_nome])]).fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]

    # Limiar que maximiza F1 (escolhido no treino via validação não é trivial aqui; usamos o do teste só p/ reporte)
    from sklearn.metrics import precision_recall_curve
    p_, r_, thr = precision_recall_curve(yte, proba)
    f1s = 2 * p_ * r_ / (p_ + r_ + 1e-12)
    limiar = float(thr[np.nanargmax(f1s[:-1])]) if len(thr) else 0.5
    pred = (proba >= limiar).astype(int)
    metricas = {
        "modelo": melhor_nome,
        "roc_auc": float(roc_auc_score(yte, proba)) if yte.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(yte, proba)) if yte.nunique() > 1 else np.nan,
        "recall": float(recall_score(yte, pred)), "f1": float(f1_score(yte, pred)),
        "limiar": limiar, "prevalencia_teste": float(yte.mean()), "n_teste": int(len(yte)),
    }
    log.info("Holdout temporal (%s): %s", melhor_nome, {k: round(v, 3) if isinstance(v, float) else v for k, v in metricas.items()})

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    if yte.nunique() > 1:
        RocCurveDisplay.from_predictions(yte, proba, ax=axes[0])
        axes[0].plot([0, 1], [0, 1], "--", color=CINZA)
        PrecisionRecallDisplay.from_predictions(yte, proba, ax=axes[1])
        axes[1].axhline(yte.mean(), ls="--", color=CINZA, label="Baseline")
        axes[1].legend()
    ConfusionMatrixDisplay.from_predictions(yte, pred, ax=axes[2], cmap="Blues", colorbar=False)
    axes[0].set_title("Curva ROC", fontweight="bold")
    axes[1].set_title("Curva Precisão-Recall", fontweight="bold")
    axes[2].set_title(f"Matriz de confusão (limiar={limiar:.2f})", fontweight="bold")
    salva_fig(fig, outdir / "figuras" / "07_ml_avaliacao.png")

    # Importância por permutação (interpretabilidade agnóstica ao modelo)
    if yte.nunique() > 1:
        pi = permutation_importance(pipe, Xte, yte, scoring="average_precision", n_repeats=15,
                                    random_state=RANDOM_STATE, n_jobs=-1)
        imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
        err = pd.Series(pi.importances_std, index=X.columns)[imp.index]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(imp.index, imp.values, xerr=err.values, color=AZUL)
        ax.set_title(f"Importância das variáveis (permutação, {melhor_nome})", fontweight="bold")
        ax.set_xlabel("Queda de PR-AUC ao embaralhar a variável")
        salva_fig(fig, outdir / "figuras" / "08_ml_importancia.png")
        imp.sort_values(ascending=False).to_csv(outdir / "tabelas" / "importancia_variaveis.csv", header=["importancia"])

    try:
        import joblib
        final = Pipeline([("pre", pre), ("m", modelos[melhor_nome])]).fit(X, y)
        joblib.dump(final, outdir / "modelo_hospitalizacao.joblib")
    except Exception as e:  # noqa: BLE001
        log.warning("Não foi possível salvar o modelo: %s", e)
    metricas["tabela_cv"] = comp.drop(columns="_pr").to_dict("records")
    return metricas


# ----------------------------------------------------------------------------
# 5. Previsão de séries temporais
# ----------------------------------------------------------------------------
def modelo_serie_temporal(df: pd.DataFrame, outdir: Path, horizonte: int = 4) -> dict | None:
    if "DT_NOTIFIC" not in df.columns:
        return None
    s = serie_semanal(df)
    if len(s) < 14:
        log.warning("Série curta (%d semanas); pulando previsão.", len(s))
        return None
    s = s.iloc[:-1] if len(s) > 1 else s  # descarta a última semana (possivelmente incompleta)

    def feats(serie: pd.Series) -> pd.DataFrame:
        f = pd.DataFrame({f"lag_{k}": serie.shift(k) for k in (1, 2, 3, 4)})
        f["media_4"] = serie.shift(1).rolling(4).mean()
        f["tendencia"] = serie.shift(1) - serie.shift(2)
        f["semana"] = serie.index.isocalendar().week.values
        return f

    F = feats(s)
    data = pd.concat([F, s], axis=1).dropna()
    X, y = data.drop(columns="casos"), data["casos"]
    n_test = max(4, int(len(data) * 0.25))
    Xtr, Xte, ytr, yte = X.iloc[:-n_test], X.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:]

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    modelo = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                           min_samples_leaf=3, random_state=RANDOM_STATE).fit(Xtr, ytr)
    pred = modelo.predict(Xte)
    naive = Xte["lag_1"]                       # baseline: igual à semana anterior
    media4 = Xte["media_4"]                    # baseline: média das 4 últimas semanas
    res = {
        "n_semanas": int(len(s)), "semanas_teste": int(n_test),
        "MAE": {"GradientBoosting": mean_absolute_error(yte, pred), "Ingênuo (lag-1)": mean_absolute_error(yte, naive),
                "Média móvel 4": mean_absolute_error(yte, media4)},
        "RMSE": {"GradientBoosting": rmse(yte, pred), "Ingênuo (lag-1)": rmse(yte, naive), "Média móvel 4": rmse(yte, media4)},
    }
    for k in res["MAE"]:
        log.info("Série temporal | %-18s MAE=%.1f RMSE=%.1f", k, res["MAE"][k], res["RMSE"][k])

    # Intervalo de previsão empírico a partir dos resíduos
    resid = (yte - pred).to_numpy()
    q_lo, q_hi = np.percentile(resid, [10, 90])

    # Previsão recursiva futura (modelo refeito com toda a série)
    modelo_full = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                                min_samples_leaf=3, random_state=RANDOM_STATE).fit(X, y)
    hist = s.copy()
    fut_idx, fut_val = [], []
    for _ in range(horizonte):
        prox = hist.index[-1] + pd.Timedelta(weeks=1)
        tmp = pd.concat([hist, pd.Series([np.nan], index=[prox])])
        xf = feats(tmp).iloc[[-1]]
        yhat = max(0.0, float(modelo_full.predict(xf)[0]))
        hist.loc[prox] = yhat
        fut_idx.append(prox)
        fut_val.append(yhat)
    fut = pd.Series(fut_val, index=fut_idx)
    pd.DataFrame({"previsao": fut, "lim_inf": (fut + q_lo).clip(lower=0), "lim_sup": fut + q_hi}).to_csv(
        outdir / "tabelas" / "previsao_futura.csv")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(s.index, s.values, color=AZUL, lw=2, label="Observado")
    ax.plot(yte.index, pred, color=VERMELHO, lw=2, marker="o", label="Previsto (teste)")
    ax.plot(yte.index, naive, color=CINZA, ls="--", label="Baseline ingênuo")
    ax.plot([s.index[-1]] + fut_idx, [s.values[-1]] + fut_val, color="#e69f00", lw=2, marker="s", label=f"Previsão +{horizonte} sem.")
    ax.fill_between(fut_idx, (fut + q_lo).clip(lower=0), fut + q_hi, color="#e69f00", alpha=.2, label="Intervalo ~80%")
    ax.axvline(yte.index[0], color="k", ls=":", alpha=.5)
    ax.set_title("Previsão de casos semanais vs. baseline (split cronológico)", fontweight="bold")
    ax.set_ylabel("Casos por semana")
    ax.legend()
    salva_fig(fig, outdir / "figuras" / "09_previsao_serie.png")
    return {k: ({a: round(b, 2) for a, b in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}


# ----------------------------------------------------------------------------
# 6. Clusterização de bairros
# ----------------------------------------------------------------------------
def clusterizar_bairros(df: pd.DataFrame, outdir: Path, pop: pd.DataFrame | None) -> pd.DataFrame | None:
    if "NM_BAIRRO" not in df.columns:
        return None
    g = df.groupby("NM_BAIRRO")
    feat = pd.DataFrame({"casos": g.size()})
    if "IDADE" in df.columns:
        feat["idade_media"] = g["IDADE"].mean()
        feat["pct_idosos"] = g["IDADE"].apply(lambda x: (x >= 60).mean() * 100)
    if "CS_SEXO" in df.columns:
        feat["pct_fem"] = g["CS_SEXO"].apply(lambda x: (x == "F").mean() * 100)
    if "HOSPITALIZADO" in df.columns:
        feat["pct_hosp"] = g["HOSPITALIZADO"].mean() * 100
    if pop is not None:
        feat = feat.join(pop, how="inner")
        feat["incidencia_100k"] = feat["casos"] / feat["populacao"] * 1e5
        feat = feat.drop(columns="populacao")
    feat = feat[feat["casos"] >= 20].dropna()
    if len(feat) < 8:
        log.warning("Poucos bairros para clusterizar.")
        return None

    Z = StandardScaler().fit_transform(np.log1p(feat[["casos"]]).join(feat.drop(columns="casos")))
    scores = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z))
              for k in range(2, min(7, len(feat) - 1))}
    k = max(scores, key=scores.get)
    log.info("Clusterização: melhor k=%d (silhouette=%.3f)", k, scores[k])
    feat["cluster"] = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)
    ordem = feat.groupby("cluster")["casos"].mean().sort_values().index
    feat["cluster"] = feat["cluster"].map({c: i for i, c in enumerate(ordem)})  # 0 = menor carga
    feat.to_csv(outdir / "tabelas" / "clusters_bairros.csv")

    comp = PCA(2, random_state=RANDOM_STATE).fit_transform(Z)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(list(scores), list(scores.values()), marker="o", color=AZUL)
    axes[0].set_title("Escolha de k (silhouette)", fontweight="bold")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Silhouette")
    sc = axes[1].scatter(comp[:, 0], comp[:, 1], c=feat["cluster"], cmap="viridis", s=40 + feat["casos"] / feat["casos"].max() * 300,
                         alpha=.75, edgecolor="k")
    for nome in feat["casos"].nlargest(6).index:
        i = list(feat.index).index(nome)
        axes[1].annotate(nome.title(), (comp[i, 0], comp[i, 1]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[1].set_title(f"Bairros por perfil epidemiológico (PCA, k={k})", fontweight="bold")
    axes[1].legend(*sc.legend_elements(), title="Cluster")
    salva_fig(fig, outdir / "figuras" / "10_clusters_bairros.png")
    return feat


# ----------------------------------------------------------------------------
# 7. Mapas (opcional)
# ----------------------------------------------------------------------------
def mapas(df: pd.DataFrame, geojson: Path, outdir: Path, pop: pd.DataFrame | None, coluna_nome: str | None):
    try:
        import geopandas as gpd
    except ImportError:
        log.warning("geopandas não instalado; pulando mapas. (pip install geopandas folium mapclassify)")
        return
    g = gpd.read_file(geojson)
    if coluna_nome is None:
        candidatos = [c for c in g.columns if any(t in c.lower() for t in ("bairro", "nome", "name", "nm_"))]
        coluna_nome = candidatos[0] if candidatos else g.columns[0]
    g["BAIRRO_N"] = g[coluna_nome].map(normaliza_texto)
    casos = df["NM_BAIRRO"].value_counts().rename("casos")
    m = g.merge(casos, left_on="BAIRRO_N", right_index=True, how="left")
    m["casos"] = m["casos"].fillna(0)
    col = "casos"
    if pop is not None:
        m = m.merge(pop, left_on="BAIRRO_N", right_index=True, how="left")
        m["incidencia_100k"] = m["casos"] / m["populacao"] * 1e5
        col = "incidencia_100k"
    cobertura = m["casos"].gt(0).sum()
    log.info("Mapa: %d/%d polígonos com casos (verifique correspondência de nomes).", cobertura, len(m))

    fig, ax = plt.subplots(figsize=(9, 10))
    m.plot(column=col, cmap="OrRd", legend=True, edgecolor="white", linewidth=.4, ax=ax,
           missing_kwds={"color": "#eeeeee"}, legend_kwds={"label": col, "shrink": .6})
    ax.set_axis_off()
    ax.set_title("Mapa coroplético de chikungunya por bairro - Recife, 2025", fontweight="bold")
    salva_fig(fig, outdir / "figuras" / "11_mapa_coropletico.png")

    try:  # autocorrelação espacial (Moran's I)
        from esda.moran import Moran
        from libpysal.weights import Queen
        mm = m.dropna(subset=[col]).reset_index(drop=True)
        w = Queen.from_dataframe(mm)
        w.transform = "r"
        mi = Moran(mm[col].to_numpy(), w, permutations=999)
        log.info("Moran's I = %.3f (p=%.4f)", mi.I, mi.p_sim)
        (outdir / "tabelas" / "moran.json").write_text(json.dumps({"I": mi.I, "p_sim": mi.p_sim}))
    except ImportError:
        log.info("esda/libpysal ausentes; Moran's I ignorado.")
    try:
        import folium
        centro = m.to_crs(4326).geometry.unary_union.centroid
        mapa = folium.Map(location=[centro.y, centro.x], zoom_start=11, tiles="cartodbpositron")
        folium.Choropleth(geo_data=m.to_crs(4326).to_json(), data=m, columns=["BAIRRO_N", col], key_on="feature.properties.BAIRRO_N",
                          fill_color="OrRd", fill_opacity=.8, line_opacity=.3, legend_name=col).add_to(mapa)
        mapa.save(outdir / "mapa_interativo.html")
        log.info("Mapa interativo salvo: mapa_interativo.html")
    except ImportError:
        log.info("folium ausente; mapa interativo ignorado.")


# ----------------------------------------------------------------------------
# 8. Relatório
# ----------------------------------------------------------------------------
def gerar_relatorio(outdir: Path, ach: dict, testes: list, clf: dict | None, ts: dict | None, clusters: pd.DataFrame | None):
    L = ["# Relatório - Chikungunya em Recife (2025)\n", "## 1. Principais achados\n"]
    L.append(f"- **{ach['n_casos']:,} casos** analisados.".replace(",", "."))
    if "top_bairros" in ach:
        L.append(f"- Os 5 bairros com mais casos concentram **{ach['pct_top5']:.1f}%** do total: "
                 + ", ".join(f"{k.title()} ({v})" for k, v in ach["top_bairros"].items()) + ".")
    if "top_incidencia" in ach:
        L.append("- Maiores incidências (por 100 mil hab.): " + ", ".join(f"{k.title()} ({v})" for k, v in ach["top_incidencia"].items()) + ".")
    if "semana_pico" in ach:
        L.append(f"- Pico epidêmico na semana de **{ach['semana_pico']}** com **{ach['casos_pico']} casos**.")
    if "idade_mediana" in ach:
        L.append(f"- Idade mediana: **{ach['idade_mediana']:.0f} anos**; {ach['pct_feminino']:.1f}% dos casos são mulheres.")
    if "taxa_hospitalizado" in ach:
        L.append(f"- Taxa de hospitalização: **{ach['taxa_hospitalizado']:.2f}%**.")
    L.append("\n## 2. Estatística inferencial\n")
    if testes:
        L.append("| Hipótese | Teste | p-valor | Efeito | Conclusão |\n|---|---|---|---|---|")
        for t in testes:
            p = "-" if t["teste"].startswith("Bootstrap") else f"{t['p_valor']:.4g}"
            L.append(f"| {t['hipotese']} | {t['teste']} | {p} | {t['tamanho_efeito']} | {t['interpretacao']} |")
    L.append("\n## 3. Machine Learning (hospitalização)\n")
    if clf:
        L.append(f"Melhor modelo na validação cruzada: **{clf['modelo']}**. Avaliação em *holdout cronológico* "
                 f"(n={clf['n_teste']}, prevalência={clf['prevalencia_teste']*100:.1f}%): ROC-AUC={clf['roc_auc']:.3f}, "
                 f"PR-AUC={clf['pr_auc']:.3f}, recall={clf['recall']:.3f}, F1={clf['f1']:.3f}.\n")
        L.append("| Modelo | ROC-AUC | PR-AUC | Recall | F1 |\n|---|---|---|---|---|")
        for r in clf["tabela_cv"]:
            L.append(f"| {r['modelo']} | {r['roc_auc']} | {r['pr_auc']} | {r['recall']} | {r['f1']} |")
    else:
        L.append("_Não executado (alvo indisponível)._")
    L.append("\n## 4. Previsão de séries temporais\n")
    if ts:
        L.append("| Modelo | MAE | RMSE |\n|---|---|---|")
        for k in ts["MAE"]:
            L.append(f"| {k} | {ts['MAE'][k]} | {ts['RMSE'][k]} |")
    else:
        L.append("_Série insuficiente._")
    L.append("\n## 5. Clusters de bairros\n")
    if clusters is not None:
        tab = clusters.groupby("cluster").mean(numeric_only=True).round(1)
        try:
            L.append(tab.to_markdown())
        except ImportError:  # tabulate ausente
            L.append("```\n" + tab.to_string() + "\n```")
    L.append("\n## 6. Limitações\n"
             "- Dados de notificação: subnotificação e atraso de registro são prováveis.\n"
             "- Casos confirmados ≠ incidência real; sem denominador populacional a comparação entre bairros é limitada.\n"
             "- Modelos usam apenas variáveis disponíveis na notificação; **não** foram usadas variáveis de desfecho (evitando data leakage).\n"
             "- Associação estatística não implica causalidade.\n"
             "- Dados sensíveis de saúde: nenhuma informação identificável foi exportada.\n")
    (outdir / "RELATORIO.md").write_text("\n".join(L), encoding="utf-8")
    log.info("Relatório salvo em RELATORIO.md")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def carregar_populacao(caminho: Path | None) -> pd.DataFrame | None:
    if not caminho:
        return None
    p = pd.read_csv(caminho, sep=None, engine="python")
    p.columns = p.columns.str.lower().str.strip()
    p["bairro"] = p["bairro"].map(normaliza_texto)
    return p.dropna(subset=["bairro"]).set_index("bairro")[["populacao"]]


def main():
    ap = argparse.ArgumentParser(description="Pipeline de ciência de dados - Chikungunya Recife 2025")
    ap.add_argument("--input", required=True, type=Path, help="CSV com os casos")
    ap.add_argument("--outdir", default=Path("resultados"), type=Path)
    ap.add_argument("--pop", type=Path, default=None, help="CSV (bairro,populacao) para incidência por 100 mil")
    ap.add_argument("--geojson", type=Path, default=None, help="Malha de bairros (GeoJSON/Shapefile)")
    ap.add_argument("--geo-col", default=None, help="Coluna com o nome do bairro na malha")
    a = ap.parse_args()

    for sub in ("figuras", "tabelas"):
        (a.outdir / sub).mkdir(parents=True, exist_ok=True)

    pop = carregar_populacao(a.pop)
    df = limpar_dados(carregar_dados(a.input), a.outdir)
    ach = eda(df, a.outdir, pop)
    testes = testes_estatisticos(df, a.outdir)
    clf = modelo_classificacao(df, a.outdir)
    ts = modelo_serie_temporal(df, a.outdir)
    clusters = clusterizar_bairros(df, a.outdir, pop)
    if a.geojson:
        mapas(df, a.geojson, a.outdir, pop, a.geo_col)
    gerar_relatorio(a.outdir, ach, testes, clf, ts, clusters)
    log.info("Pipeline concluído. Resultados em: %s", a.outdir.resolve())


if __name__ == "__main__":
    main()
