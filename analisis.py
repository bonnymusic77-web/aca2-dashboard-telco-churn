"""
analisis.py
-----------
Pruebas estadísticas y resúmenes automáticos sobre el dataset ya limpio.

Todas las funciones reciben un DataFrame, así que sirven igual para la base
completa (notebook) que para el subconjunto filtrado en el dashboard.
"""

import numpy as np
import pandas as pd
from scipy import stats

from procesamiento import COLS_CATEGORICAS, COLS_NUMERICAS, etiqueta

ALFA = 0.05


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def intervalo_wilson(exitos, n, confianza=0.95):
    """Intervalo de confianza de Wilson para una proporción.
    Se comporta mejor que el intervalo normal cuando n es pequeño."""
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - confianza) / 2)
    p = exitos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margen = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centro - margen, centro + margen


def magnitud_efecto(valor):
    """Interpretación de Cohen (1988) para V de Cramér con gl* = 1 y r biserial."""
    valor = abs(valor)
    if valor < 0.1:
        return "Despreciable"
    if valor < 0.3:
        return "Pequeño"
    if valor < 0.5:
        return "Mediano"
    return "Grande"


def fmt_num(valor, decimales=0):
    """Número con formato colombiano: punto de miles y coma decimal."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_pct(proporcion, decimales=1):
    return f"{fmt_num(100 * proporcion, decimales)} %"


def formato_p(p):
    # espacio no separable para que "p < 0,001" no se parta en dos líneas
    return "<\u00a00,001" if p < 0.001 else f"=\u00a0{fmt_num(p, 3)}"


# ---------------------------------------------------------------------------
# Descriptivos
# ---------------------------------------------------------------------------
def resumen_descriptivo(df, cols=None):
    """Estadísticos de tendencia, dispersión y forma para variables numéricas."""
    cols = cols or COLS_NUMERICAS
    tabla = df[cols].describe().T
    tabla["mediana"] = df[cols].median()
    tabla["asimetria"] = df[cols].skew()
    tabla["curtosis"] = df[cols].kurt()
    tabla["coef_variacion_%"] = 100 * tabla["std"] / tabla["mean"]
    tabla = tabla.rename(columns={"count": "n", "mean": "media", "std": "desv_est"})
    return tabla[["n", "media", "mediana", "desv_est", "min", "25%", "75%", "max",
                  "asimetria", "curtosis", "coef_variacion_%"]].round(2)


def tasa_abandono_por(df, col, confianza=0.95):
    """Clientes, abandonos y tasa de abandono por categoría, con IC de Wilson."""
    tabla = (
        df.groupby(col, observed=True)["abandono"]
        .agg(clientes="count", abandonos="sum")
        .reset_index()
    )
    tabla["tasa"] = tabla["abandonos"] / tabla["clientes"]
    ic = [intervalo_wilson(a, n, confianza) for a, n in zip(tabla["abandonos"], tabla["clientes"])]
    tabla["ic_inf"] = [i[0] for i in ic]
    tabla["ic_sup"] = [i[1] for i in ic]
    tabla["participacion"] = tabla["clientes"] / tabla["clientes"].sum()
    return tabla


# ---------------------------------------------------------------------------
# Pruebas de hipótesis
# ---------------------------------------------------------------------------
def prueba_normalidad(df, cols=None):
    """Prueba de D'Agostino-Pearson. Con n grande casi cualquier desviación
    resulta significativa, por eso se reporta junto a asimetría y curtosis."""
    cols = cols or COLS_NUMERICAS
    filas = []
    for col in cols:
        estadistico, p = stats.normaltest(df[col])
        filas.append({
            "variable": col,
            "asimetria": round(df[col].skew(), 3),
            "curtosis": round(df[col].kurt(), 3),
            "estadistico_K2": round(estadistico, 1),
            "p_valor": p,
            "normal": "Sí" if p >= ALFA else "No",
        })
    return pd.DataFrame(filas)


def cramers_v(tabla_contingencia):
    """V de Cramér a partir de una tabla de contingencia."""
    chi2 = stats.chi2_contingency(tabla_contingencia, correction=False)[0]
    n = tabla_contingencia.to_numpy().sum()
    k = min(tabla_contingencia.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else np.nan


def pruebas_chi_cuadrado(df, cols=None, objetivo="Churn"):
    """Chi-cuadrado de independencia de cada categórica frente al abandono.

    Incluye V de Cramér como tamaño del efecto, el porcentaje de frecuencias
    esperadas >= 5 (supuesto de la prueba) y la corrección de Bonferroni por
    realizar varias pruebas a la vez."""
    cols = cols or COLS_CATEGORICAS
    filas = []
    for col in cols:
        tabla = pd.crosstab(df[col], df[objetivo])
        if tabla.shape[0] < 2 or tabla.shape[1] < 2:
            continue
        chi2, p, gl, esperadas = stats.chi2_contingency(tabla, correction=False)
        filas.append({
            "variable": col,
            "nombre": etiqueta(col),
            "chi2": round(chi2, 2),
            "gl": gl,
            "p_valor": p,
            "v_cramer": round(cramers_v(tabla), 3),
            "esperadas_>=5_%": round(100 * (esperadas >= 5).mean(), 1),
        })
    if not filas:
        return pd.DataFrame()
    resultado = pd.DataFrame(filas)
    m = len(resultado)
    resultado["p_bonferroni"] = (resultado["p_valor"] * m).clip(upper=1)
    resultado["significativa"] = np.where(resultado["p_bonferroni"] < ALFA, "Sí", "No")
    resultado["efecto"] = resultado["v_cramer"].apply(magnitud_efecto)
    return resultado.sort_values("v_cramer", ascending=False).reset_index(drop=True)


def comparar_numericas(df, cols=None):
    """U de Mann-Whitney entre clientes que abandonan y los que se quedan.

    Se usa una prueba no paramétrica porque las variables no son normales.
    El tamaño del efecto es la correlación biserial por rangos:
    r = 1 - 2U / (n1·n2); r > 0 indica valores mayores en quienes abandonan."""
    cols = cols or COLS_NUMERICAS
    filas = []
    si = df[df["abandono"] == 1]
    no = df[df["abandono"] == 0]
    if si.empty or no.empty:
        return pd.DataFrame()
    for col in cols:
        u, p = stats.mannwhitneyu(no[col], si[col], alternative="two-sided")
        r = 1 - 2 * u / (len(si) * len(no))
        filas.append({
            "variable": col,
            "nombre": etiqueta(col),
            "mediana_se_queda": round(no[col].median(), 2),
            "mediana_abandona": round(si[col].median(), 2),
            "U": round(u, 0),
            "p_valor": p,
            "r_biserial": round(r, 3),
            "efecto": magnitud_efecto(r),
        })
    return pd.DataFrame(filas)


def prueba_kruskal(df, numerica, categorica):
    """Kruskal-Wallis: ¿la distribución de una numérica cambia entre grupos?"""
    grupos = [g[numerica].to_numpy() for _, g in df.groupby(categorica, observed=True)]
    h, p = stats.kruskal(*grupos)
    n, k = len(df), len(grupos)
    eta2_h = (h - k + 1) / (n - k)  # tamaño del efecto η²H (Tomczak y Tomczak, 2014)
    return {"H": round(float(h), 2), "gl": k - 1, "p_valor": float(p), "eta2_H": round(float(eta2_h), 3)}


def prueba_tasa_vs_global(abandonos, clientes, tasa_global):
    """Prueba binomial: ¿la tasa de un segmento difiere de la tasa global?"""
    if clientes == 0:
        return np.nan
    return stats.binomtest(int(abandonos), int(clientes), tasa_global).pvalue


# ---------------------------------------------------------------------------
# Correlaciones
# ---------------------------------------------------------------------------
def matriz_correlacion(df, cols=None, metodo="spearman"):
    cols = cols or COLS_NUMERICAS + ["num_servicios", "abandono"]
    return df[cols].corr(method=metodo)


def matriz_cramers_v(df, cols):
    """Matriz simétrica de V de Cramér entre variables categóricas."""
    matriz = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = cramers_v(pd.crosstab(df[a], df[b]))
            matriz.loc[a, b] = matriz.loc[b, a] = v
    return matriz


# ---------------------------------------------------------------------------
# Análisis automático (lo consume el dashboard)
# ---------------------------------------------------------------------------
def kpis(df):
    clientes = len(df)
    abandonos = int(df["abandono"].sum())
    return {
        "clientes": clientes,
        "abandonos": abandonos,
        "tasa": abandonos / clientes if clientes else np.nan,
        "ingreso_mensual": float(df["MonthlyCharges"].sum()),
        "ingreso_en_riesgo": float(df.loc[df["abandono"] == 1, "MonthlyCharges"].sum()),
        "antiguedad_media": float(df["tenure"].mean()) if clientes else np.nan,
    }


def segmentos_criticos(df, cols=None, min_clientes=100, top=5):
    """Busca las categorías con mayor tasa de abandono (con tamaño mínimo)."""
    cols = cols or COLS_CATEGORICAS + ["grupo_antiguedad", "pago_automatico"]
    tablas = []
    for col in cols:
        if df[col].nunique() < 2:  # variable ya fijada por un filtro
            continue
        t = tasa_abandono_por(df, col)
        t = t[t["clientes"] >= min_clientes]
        t.insert(0, "variable", etiqueta(col))
        tablas.append(t.rename(columns={col: "categoria"}))
    if not tablas:
        return pd.DataFrame()
    todo = pd.concat(tablas, ignore_index=True)
    return todo.sort_values("tasa", ascending=False).head(top).reset_index(drop=True)


def analisis_automatico(df_filtrado, df_total):
    """Genera hallazgos en texto a partir del subconjunto filtrado.

    Devuelve un diccionario con los hallazgos, las tablas de pruebas y un
    indicador de si hubo datos suficientes para aplicar las pruebas."""
    resultado = {"hallazgos": [], "suficiente": True}
    n = len(df_filtrado)
    abandonos = int(df_filtrado["abandono"].sum())
    clase_menor = min(abandonos, n - abandonos)
    if n < 30 or clase_menor < 10:
        resultado["suficiente"] = False
        resultado["hallazgos"].append(
            f"La selección tiene {fmt_num(n)} clientes y {fmt_num(abandonos)} "
            f"{'abandono' if abandonos == 1 else 'abandonos'}. "
            "Se necesitan al menos 30 clientes y 10 casos en cada grupo (abandona / "
            "permanece) para que las pruebas sean confiables; amplíe los filtros."
        )
        return resultado

    k_seg, k_tot = kpis(df_filtrado), kpis(df_total)
    diferencia = 100 * (k_seg["tasa"] - k_tot["tasa"])
    p_seg = prueba_tasa_vs_global(k_seg["abandonos"], n, k_tot["tasa"])
    comparacion = "por encima" if diferencia > 0 else "por debajo"
    if n == len(df_total):
        texto = (f"La base completa tiene {fmt_num(n)} clientes y una tasa de abandono "
                 f"de {fmt_pct(k_tot['tasa'])}.")
    elif p_seg < ALFA:
        texto = (f"La selección ({fmt_num(n)} clientes) tiene una tasa de abandono de "
                 f"{fmt_pct(k_seg['tasa'])}, {fmt_num(abs(diferencia), 1)} puntos {comparacion} "
                 f"del total. La diferencia es estadísticamente significativa "
                 f"(prueba binomial, p {formato_p(p_seg)}).")
    else:
        texto = (f"La selección ({fmt_num(n)} clientes) tiene una tasa de abandono de "
                 f"{fmt_pct(k_seg['tasa'])}, sin diferencia significativa frente al total "
                 f"({fmt_pct(k_tot['tasa'])}; p {formato_p(p_seg)}).")
    resultado["hallazgos"].append(texto)

    participacion = k_seg["ingreso_en_riesgo"] / k_seg["ingreso_mensual"]
    resultado["hallazgos"].append(
        f"Los clientes que abandonaron representaban USD {fmt_num(k_seg['ingreso_en_riesgo'])} "
        f"de facturación mensual, el {fmt_pct(participacion)} del ingreso de la selección."
    )

    chi = pruebas_chi_cuadrado(df_filtrado)
    resultado["chi_cuadrado"] = chi
    validas = chi[(chi["significativa"] == "Sí") & (chi["esperadas_>=5_%"] >= 80)] if not chi.empty else chi
    if not validas.empty:
        top = validas.iloc[0]
        tabla_top = tasa_abandono_por(df_filtrado, top["variable"]).sort_values("tasa")
        alta, baja = tabla_top.iloc[-1], tabla_top.iloc[0]
        resultado["hallazgos"].append(
            f"La variable más asociada al abandono es «{top['nombre']}» "
            f"(V de Cramér = {fmt_num(top['v_cramer'], 2)}, efecto {top['efecto'].lower()}): "
            f"la tasa va de {fmt_pct(baja['tasa'])} en «{baja[top['variable']]}» a "
            f"{fmt_pct(alta['tasa'])} en «{alta[top['variable']]}»."
        )

    mw = comparar_numericas(df_filtrado)
    resultado["mann_whitney"] = mw
    if not mw.empty:
        fila = mw.loc[mw["r_biserial"].abs().idxmax()]
        if fila["p_valor"] < ALFA:
            sentido = "mayor" if fila["r_biserial"] > 0 else "menor"
            resultado["hallazgos"].append(
                f"Quienes abandonan tienen {sentido} «{fila['nombre']}»: mediana de "
                f"{fmt_num(fila['mediana_abandona'], 1)} frente a {fmt_num(fila['mediana_se_queda'], 1)} "
                f"en quienes permanecen (Mann-Whitney, p {formato_p(fila['p_valor'])}; "
                f"r\u00a0=\u00a0{fmt_num(fila['r_biserial'], 2)})."
            )

    seg = segmentos_criticos(df_filtrado, min_clientes=max(30, int(0.03 * n)))
    resultado["segmentos"] = seg
    if not seg.empty:
        s = seg.iloc[0]
        resultado["hallazgos"].append(
            f"El segmento de mayor riesgo es «{s['variable']}: {s['categoria']}», con "
            f"{fmt_pct(s['tasa'])} de abandono entre {fmt_num(s['clientes'])} clientes "
            f"(IC 95 %: {fmt_pct(s['ic_inf'])} a {fmt_pct(s['ic_sup'])})."
        )

    resultado["descriptivos"] = resumen_descriptivo(df_filtrado)
    return resultado
