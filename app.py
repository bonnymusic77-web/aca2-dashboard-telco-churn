"""
app.py
------
Dashboard analítico de abandono de clientes (Telco Customer Churn).

Ejecución local:   streamlit run app.py
Publicación:       Streamlit Community Cloud (ver README.md)
"""

import pandas as pd
import streamlit as st

import analisis as an
import graficos as gr
import procesamiento as pr
from analisis import fmt_num, fmt_pct

st.set_page_config(
    page_title="Abandono de clientes · Telco",
    page_icon=":material/insights:",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.7rem;}
      .equipo {color: #52514e; font-size: .92rem; margin: -.6rem 0 .2rem 0; line-height: 1.55;}
      .hallazgo {padding: .75rem 1rem; border-left: 3px solid #2a78d6; background: #f3f2ee;
                 border-radius: 4px; margin-bottom: .6rem; line-height: 1.5;}
    </style>
    """,
    unsafe_allow_html=True,
)

CONFIG_GRAFICO = {"displaylogo": False, "modeBarButtonsToRemove": ["toImage"]}
INTEGRANTES = ["Bonny Darhyll Brayan Galindo Santos", "Andrés Alberto López Herrera", "Camilo José Rodríguez Ayubi"]
FILTROS_CATEGORICOS = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]
VARIABLES_CATEGORICAS = pr.COLS_CATEGORICAS + ["grupo_antiguedad", "pago_automatico", "num_servicios"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando y limpiando los datos…")
def obtener_datos():
    crudo = pr.cargar_datos()
    limpio, bitacora = pr.preparar_datos(crudo)
    return crudo, limpio, bitacora


def opciones(df, col):
    valores = df[col].unique().tolist()
    if col in pr.ORDEN_CATEGORIAS:
        return [v for v in pr.ORDEN_CATEGORIAS[col] if v in valores]
    return sorted(valores)


crudo, df, bitacora = obtener_datos()
CARGO_MIN, CARGO_MAX = int(df["MonthlyCharges"].min()), int(df["MonthlyCharges"].max()) + 1
TASA_GLOBAL = df["abandono"].mean()


# ---------------------------------------------------------------------------
# Filtros (barra lateral)
# ---------------------------------------------------------------------------
def valores_por_defecto():
    for col in FILTROS_CATEGORICOS:
        st.session_state[f"f_{col}"] = []
    st.session_state["f_tenure"] = (0, 72)
    st.session_state["f_cargo"] = (CARGO_MIN, CARGO_MAX)


if "f_tenure" not in st.session_state:
    valores_por_defecto()

with st.sidebar:
    st.markdown("### :material/filter_alt: Filtros")
    st.caption("Todos los gráficos se recalculan con la selección. Un filtro vacío incluye todas las opciones.")
    for col in FILTROS_CATEGORICOS:
        st.multiselect(pr.etiqueta(col), opciones(df, col), key=f"f_{col}", placeholder="Todas")
    st.slider("Antigüedad (meses)", 0, 72, key="f_tenure")
    st.slider("Cargo mensual (USD)", CARGO_MIN, CARGO_MAX, key="f_cargo")
    st.button("Restablecer filtros", icon=":material/restart_alt:", on_click=valores_por_defecto, width="stretch")
    st.divider()
    st.caption(
        "**Fuente:** IBM Telco Customer Churn (Kaggle).  \n"
        "**Docente:** Hamilton Rivera"
    )


def aplicar_filtros(datos):
    mascara = pd.Series(True, index=datos.index)
    for col in FILTROS_CATEGORICOS:
        seleccion = st.session_state[f"f_{col}"]
        if seleccion:
            mascara &= datos[col].isin(seleccion)
    t_min, t_max = st.session_state["f_tenure"]
    c_min, c_max = st.session_state["f_cargo"]
    mascara &= datos["tenure"].between(t_min, t_max)
    mascara &= datos["MonthlyCharges"].between(c_min, c_max)
    return datos.loc[mascara]


dff = aplicar_filtros(df)


# ---------------------------------------------------------------------------
# Encabezado e indicadores
# ---------------------------------------------------------------------------
st.title("Abandono de clientes en telecomunicaciones")
st.markdown(
    "<div class='equipo'>Análisis del abandono (<i>churn</i>) de clientes con filtros, gráficos interactivos "
    "y pruebas estadísticas.<br>"
    f"<b>Integrantes:</b> {' · '.join(INTEGRANTES)}<br>"
    "Especialización en Inteligencia Artificial · Corporación Unificada Nacional de Educación Superior (CUN) · "
    "Programación para Analítica de Datos e IA</div>",
    unsafe_allow_html=True,
)
st.caption(
    f"Mostrando **{fmt_num(len(dff))}** de {fmt_num(len(df))} clientes "
    f"({fmt_pct(len(dff) / len(df))} de la base)."
)

if dff.empty:
    st.warning("Ningún cliente cumple con los filtros seleccionados. Amplíe la selección o restablezca los filtros.",
               icon=":material/search_off:")
    st.stop()

k_sel, k_tot = an.kpis(dff), an.kpis(df)
HAY_FILTRO = len(dff) < len(df)


def delta_puntos(valor, referencia="total"):
    if not HAY_FILTRO and referencia == "total":
        return None
    return f"{valor:+.1f}".replace(".", ",") + f" pp vs. {referencia}"


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clientes", fmt_num(k_sel["clientes"]), border=True,
          help="Clientes que cumplen con los filtros activos.")
c2.metric("Tasa de abandono", fmt_pct(k_sel["tasa"]), delta_puntos(100 * (k_sel["tasa"] - k_tot["tasa"])),
          delta_color="inverse", border=True, help="Porcentaje de clientes que se retiraron en el último mes.")
c3.metric("Clientes perdidos", fmt_num(k_sel["abandonos"]), border=True)
c4.metric("Ingreso en riesgo", f"USD {fmt_num(k_sel['ingreso_en_riesgo'])}",
          f"{fmt_pct(k_sel['ingreso_en_riesgo'] / k_sel['ingreso_mensual'])} del ingreso",
          delta_color="off", delta_arrow="off", border=True,
          help="Facturación mensual de los clientes que abandonaron y su peso sobre el ingreso de la vista.")
c5.metric("Antigüedad media", f"{fmt_num(k_sel['antiguedad_media'], 1)} meses",
          f"{k_sel['antiguedad_media'] - k_tot['antiguedad_media']:+.1f}".replace(".", ",") + " meses vs. total"
          if HAY_FILTRO else None,
          border=True)

tab_resumen, tab_explorar, tab_segmentos, tab_auto, tab_vista, tab_calidad = st.tabs([
    ":material/dashboard: Resumen",
    ":material/query_stats: Exploración",
    ":material/grid_view: Segmentos",
    ":material/auto_awesome: Análisis automático",
    ":material/table_view: Vista previa",
    ":material/cleaning_services: Calidad de datos",
])


def mostrar(fig, key, **kwargs):
    """Muestra una figura Plotly con el estilo propio (sin el tema de Streamlit)."""
    return st.plotly_chart(fig, key=key, theme=None, config=CONFIG_GRAFICO, **kwargs)


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
with tab_resumen:
    izq, der = st.columns(2)
    with izq, st.container(border=True):
        mostrar(gr.tasa_por_categoria(dff, "Contract", TASA_GLOBAL, alto=360), "res_contrato")
    with der, st.container(border=True):
        mostrar(gr.tasa_por_antiguedad(dff, paso=6, tasa_global=TASA_GLOBAL), "res_antiguedad")
    izq, der = st.columns(2)
    with izq, st.container(border=True):
        mostrar(gr.tasa_por_categoria(dff, "PaymentMethod", TASA_GLOBAL, alto=340), "res_pago")
    with der, st.container(border=True):
        mostrar(gr.ingreso_en_riesgo(dff, "InternetService", alto=340), "res_ingreso")
    st.caption("Las barras de error muestran el intervalo de confianza del 95 % (Wilson). "
               "La línea gris marca la tasa de abandono de toda la base.")


# ---------------------------------------------------------------------------
# Exploración
# ---------------------------------------------------------------------------
with tab_explorar:
    variables = {pr.etiqueta(c): c for c in pr.COLS_NUMERICAS + VARIABLES_CATEGORICAS}
    nombre = st.selectbox("Variable a explorar", list(variables), index=0)
    var = variables[nombre]

    grafico, detalle = st.columns([3, 2])
    if var in pr.COLS_NUMERICAS:
        with grafico, st.container(border=True):
            mostrar(gr.distribucion_numerica(dff, var), "exp_numerica")
        with detalle:
            st.markdown(f"**{nombre}: permanece vs. abandona**")
            resumen = (dff.groupby("Churn")[var].agg(["count", "mean", "median", "std"])
                       .rename(index={"No": "Permanece", "Sí": "Abandona"}).rename_axis("Grupo")
                       .rename(columns={"count": "Clientes", "mean": "Media", "median": "Mediana", "std": "Desv. est."}))
            st.dataframe(resumen.round(2), width="stretch")
            if dff["abandono"].nunique() == 2 and min(dff["abandono"].value_counts()) >= 10:
                mw = an.comparar_numericas(dff, [var]).iloc[0]
                conclusion = "hay" if mw["p_valor"] < an.ALFA else "no hay"
                st.info(f"Prueba U de Mann-Whitney: p {an.formato_p(mw['p_valor'])}, "
                        f"r = {fmt_num(mw['r_biserial'], 2)} (efecto {mw['efecto'].lower()}). "
                        f"Con 95 % de confianza, {conclusion} diferencia entre ambos grupos.",
                        icon=":material/functions:")
    else:
        with grafico, st.container(border=True):
            mostrar(gr.tasa_por_categoria(dff, var, TASA_GLOBAL), "exp_categorica")
        with detalle:
            st.markdown(f"**{nombre}: detalle por categoría**")
            tabla = an.tasa_abandono_por(dff, var)
            st.dataframe(
                tabla[[var, "clientes", "abandonos", "tasa", "participacion"]],
                hide_index=True, width="stretch",
                column_config={
                    var: st.column_config.TextColumn(nombre),
                    "clientes": st.column_config.NumberColumn("Clientes"),
                    "abandonos": st.column_config.NumberColumn("Abandonos"),
                    "tasa": st.column_config.ProgressColumn("Tasa de abandono", format="percent", min_value=0, max_value=1),
                    "participacion": st.column_config.NumberColumn("% de clientes", format="percent"),
                },
            )
            chi = an.pruebas_chi_cuadrado(dff, [var])
            if not chi.empty:
                fila = chi.iloc[0]
                conclusion = "depende" if fila["p_valor"] < an.ALFA else "no depende"
                st.info(f"Chi-cuadrado: χ² = {fmt_num(fila['chi2'], 1)}, gl = {fila['gl']}, "
                        f"p {an.formato_p(fila['p_valor'])}, V de Cramér = {fmt_num(fila['v_cramer'], 2)}. "
                        f"El abandono {conclusion} de esta variable (α = 0,05).",
                        icon=":material/functions:")

    st.markdown("#### Selección interactiva de clientes")
    st.caption("Arrastre sobre el gráfico (caja o lazo) para analizar un grupo; doble clic para quitar la selección.")
    with st.container(border=True):
        evento = mostrar(gr.dispersion_antiguedad_cargo(dff), "exp_dispersion",
                         on_select="rerun", selection_mode=("box", "lasso"))
    puntos = evento.selection.points if evento and evento.selection else []
    ids = {p["customdata"][0] for p in puntos if p.get("customdata")}
    if ids:
        grupo = dff[dff["customerID"].isin(ids)]
        k_grupo = an.kpis(grupo)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Clientes seleccionados", fmt_num(k_grupo["clientes"]))
        s2.metric("Tasa de abandono", fmt_pct(k_grupo["tasa"]),
                  f"{100 * (k_grupo['tasa'] - k_sel['tasa']):+.1f}".replace(".", ",") + " pp vs. filtro actual",
                  delta_color="inverse")
        s3.metric("Cargo mensual medio", f"USD {fmt_num(grupo['MonthlyCharges'].mean(), 2)}")
        s4.metric("Contrato más común", grupo["Contract"].mode().iloc[0])
    else:
        st.caption(":material/touch_app: Aún no hay clientes seleccionados.")


# ---------------------------------------------------------------------------
# Segmentos
# ---------------------------------------------------------------------------
with tab_segmentos:
    categoricas = {pr.etiqueta(c): c for c in VARIABLES_CATEGORICAS}
    nombres = list(categoricas)
    a, b = st.columns(2)
    fila = a.selectbox("Variable en filas", nombres, index=nombres.index(pr.etiqueta("Contract")))
    columna = b.selectbox("Variable en columnas", nombres, index=nombres.index(pr.etiqueta("InternetService")))
    with st.container(border=True):
        if fila == columna:
            st.info("Elija dos variables distintas para cruzarlas.")
        else:
            mostrar(gr.mapa_calor_segmentos(dff, categoricas[fila], categoricas[columna]), "seg_mapa")

    st.markdown("#### Segmentos con mayor tasa de abandono")
    minimo = st.slider("Tamaño mínimo del segmento (clientes)", 10, 500, 100, step=10,
                       help="Evita que grupos muy pequeños aparezcan arriba solo por azar.")
    top = an.segmentos_criticos(dff, min_clientes=minimo, top=10)
    if top.empty:
        st.info("No hay segmentos con ese tamaño mínimo en la selección actual.")
    else:
        st.dataframe(
            top[["variable", "categoria", "clientes", "abandonos", "tasa", "ic_inf", "ic_sup"]],
            hide_index=True, width="stretch",
            column_config={
                "variable": "Variable", "categoria": "Categoría",
                "clientes": st.column_config.NumberColumn("Clientes"),
                "abandonos": st.column_config.NumberColumn("Abandonos"),
                "tasa": st.column_config.ProgressColumn("Tasa de abandono", format="percent", min_value=0, max_value=1),
                "ic_inf": st.column_config.NumberColumn("IC 95 % inferior", format="percent"),
                "ic_sup": st.column_config.NumberColumn("IC 95 % superior", format="percent"),
            },
        )


# ---------------------------------------------------------------------------
# Análisis automático
# ---------------------------------------------------------------------------
with tab_auto:
    resultado = an.analisis_automatico(dff, df)
    st.markdown("#### :material/auto_awesome: Hallazgos de la selección actual")
    for texto in resultado["hallazgos"]:
        st.markdown(f"<div class='hallazgo'>{texto}</div>", unsafe_allow_html=True)

    if resultado["suficiente"]:
        izq, der = st.columns(2)
        with izq, st.container(border=True):
            mostrar(gr.ranking_asociacion(resultado["chi_cuadrado"]), "auto_ranking")
        with der, st.container(border=True):
            mostrar(gr.mapa_correlacion(an.matriz_correlacion(dff), "Correlación de Spearman"), "auto_correlacion")

        with st.expander("Estadísticos descriptivos", icon=":material/table_chart:"):
            desc = resultado["descriptivos"].rename(index=pr.etiqueta)
            st.dataframe(desc, width="stretch")
        with st.expander("Pruebas chi-cuadrado (variables categóricas)", icon=":material/functions:"):
            st.dataframe(
                resultado["chi_cuadrado"].drop(columns="variable"), hide_index=True, width="stretch",
                column_config={"nombre": "Variable", "p_valor": st.column_config.NumberColumn("p", format="%.2e"),
                               "p_bonferroni": st.column_config.NumberColumn("p Bonferroni", format="%.2e"),
                               "v_cramer": "V de Cramér", "esperadas_>=5_%": "% esperadas ≥ 5"},
            )
        with st.expander("Pruebas U de Mann-Whitney (variables numéricas)", icon=":material/functions:"):
            st.dataframe(
                resultado["mann_whitney"].drop(columns="variable"), hide_index=True, width="stretch",
                column_config={"nombre": "Variable", "p_valor": st.column_config.NumberColumn("p", format="%.2e"),
                               "mediana_se_queda": "Mediana (permanece)", "mediana_abandona": "Mediana (abandona)",
                               "r_biserial": "r biserial"},
            )
        st.caption("Nivel de significancia α = 0,05. En chi-cuadrado se aplica corrección de Bonferroni por "
                   "comparaciones múltiples. Tamaños de efecto según Cohen (1988): 0,1 pequeño, 0,3 mediano, 0,5 grande.")


# ---------------------------------------------------------------------------
# Vista previa de los datos
# ---------------------------------------------------------------------------
with tab_vista:
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Registros en la selección", fmt_num(len(dff)))
    d2.metric("Columnas del dataset limpio", fmt_num(dff.shape[1]))
    d3.metric("Variables numéricas", fmt_num(len(pr.COLS_NUMERICAS) + 1))
    d4.metric("Variables categóricas", fmt_num(len(VARIABLES_CATEGORICAS) - 1))

    st.markdown("#### Primeros registros")
    vista = dff.drop(columns=["abandono"]).rename(columns=pr.etiqueta)
    st.dataframe(vista.head(10), hide_index=True, width="stretch")

    izq, der = st.columns([3, 2])
    with izq:
        st.markdown("#### Estructura del dataset")
        estructura = pd.DataFrame({
            "Columna": vista.columns,
            "Nombre original": dff.drop(columns=["abandono"]).columns,
            "Tipo": ["Numérica" if pd.api.types.is_numeric_dtype(dff[c]) else "Categórica"
                     for c in dff.drop(columns=["abandono"]).columns],
            "No nulos": dff.drop(columns=["abandono"]).notna().sum().to_numpy(),
            "Valores únicos": dff.drop(columns=["abandono"]).nunique().to_numpy(),
        })
        st.dataframe(estructura, hide_index=True, width="stretch", height=390)
    with der:
        st.markdown("#### Resumen estadístico")
        nombres_estadisticos = {"n": "Registros", "media": "Media", "mediana": "Mediana", "desv_est": "Desv. estándar",
                                "min": "Mínimo", "25%": "Percentil 25", "75%": "Percentil 75", "max": "Máximo",
                                "asimetria": "Asimetría", "curtosis": "Curtosis", "coef_variacion_%": "Coef. variación %"}
        resumen = an.resumen_descriptivo(dff).rename(index=pr.etiqueta).T.rename(index=nombres_estadisticos)
        st.dataframe(resumen, width="stretch", height=390)

    st.download_button(
        "Descargar selección completa (CSV)", vista.to_csv(index=False).encode("utf-8-sig"),
        file_name="telco_churn_filtrado.csv", mime="text/csv", icon=":material/download:",
    )


# ---------------------------------------------------------------------------
# Calidad de datos
# ---------------------------------------------------------------------------
with tab_calidad:
    st.markdown("#### Bitácora del proceso de limpieza")
    st.dataframe(bitacora, hide_index=True, width="stretch",
                 column_config={"paso": "Paso", "detalle": "Qué se hizo", "filas": "Filas al final"})
    izq, der = st.columns(2)
    with izq:
        st.markdown("#### Diagnóstico del archivo crudo")
        diag = pr.diagnosticar_calidad(crudo)
        st.dataframe(diag[["columna", "tipo", "nulos", "vacios", "unicos"]], hide_index=True, width="stretch",
                     height=400, column_config={"columna": "Columna", "tipo": "Tipo", "nulos": "Nulos",
                                                "vacios": "Textos vacíos", "unicos": "Valores únicos"})
    with der:
        st.markdown("#### Valores atípicos (regla IQR)")
        out = pr.detectar_outliers_iqr(df).round(1)
        out["variable"] = out["variable"].map(pr.etiqueta)
        st.dataframe(out[["variable", "limite_inferior", "limite_superior", "min", "max", "atipicos"]],
                     hide_index=True, width="stretch",
                     column_config={"variable": "Variable", "limite_inferior": "Límite inferior",
                                    "limite_superior": "Límite superior", "min": "Mínimo", "max": "Máximo",
                                    "atipicos": "Atípicos"})
        consistencia = pr.verificar_consistencia_cargos(df)
        vacios_tc = int(diag.loc[diag["columna"] == "TotalCharges", "vacios"].iloc[0])
        st.markdown(
            f"- **TotalCharges** llegó como texto con {vacios_tc} valores vacíos; corresponden a clientes con "
            "0 meses de antigüedad, por lo que se imputaron con 0.\n"
            f"- Duplicados eliminados: {len(crudo) - len(df)}. No hay categorías fuera de dominio.\n"
            f"- Atípicos por IQR: {int(out['atipicos'].sum())}. Los extremos son tarifas y antigüedades "
            "reales, así que se conservan.\n"
            f"- El cargo total difiere en mediana {fmt_num(consistencia['desviacion_mediana_pct'], 1)} % de "
            f"antigüedad × cargo mensual (cambios de tarifa); solo {consistencia['clientes_fuera_tolerancia']} "
            "clientes superan el 25 %."
        )
