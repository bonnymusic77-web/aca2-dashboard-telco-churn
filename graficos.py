"""
graficos.py
-----------
Figuras interactivas (Plotly) del dashboard. Todas usan el mismo estilo:
fondo claro, líneas de guía tenues y dos colores fijos para identificar
a los clientes que permanecen (azul) y a los que abandonan (naranja).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analisis import fmt_num, fmt_pct, intervalo_wilson, tasa_abandono_por
from procesamiento import ORDEN_CATEGORIAS, etiqueta

# Paleta validada para daltonismo (azul / naranja) y tonos neutros
AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS = "#c3c2b7"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
TINTA_TENUE = "#898781"
GUIA = "#e1e0d9"
FONDO = "#fcfcfb"
COLORES_CHURN = {"No": AZUL, "Sí": NARANJA}
NOMBRES_CHURN = {"No": "Permanece", "Sí": "Abandona"}
ESCALA_SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ESCALA_DIVERGENTE = [[0, "#1c5cab"], [0.5, "#f0efec"], [1, "#c23a39"]]


def _estilo(fig, titulo=None, alto=360, leyenda=True):
    """Aplica el formato común a todas las figuras."""
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left", y=0.98, yanchor="top", yref="container",
                   font=dict(size=15, color=TINTA)) if titulo else None,
        height=alto,
        template="plotly_white",
        paper_bgcolor=FONDO,
        plot_bgcolor=FONDO,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=TINTA_SECUNDARIA),
        margin=dict(l=10, r=20, t=(88 if leyenda else 60) if titulo else 30, b=10),
        separators=",.",
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="white", bordercolor=GUIA, font=dict(color=TINTA, size=12)),
        barcornerradius=4,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GUIA, gridwidth=1, zeroline=False,
                     linecolor=GRIS, tickfont=dict(color=TINTA_TENUE))
    fig.update_yaxes(showgrid=True, gridcolor=GUIA, gridwidth=1, zeroline=False,
                     linecolor=GRIS, tickfont=dict(color=TINTA_TENUE))
    return fig


def _orden(col, valores):
    """Respeta el orden lógico si existe; si no, deja el orden recibido."""
    if col in ORDEN_CATEGORIAS:
        return [v for v in ORDEN_CATEGORIAS[col] if v in set(valores)]
    return list(valores)


def _espacio_barras(alto, n_barras, grosor=24):
    """Calcula el bargap para que cada barra quede de ~24 px sin importar la altura."""
    espacio_por_barra = (alto - 100) / max(n_barras, 1)
    return float(min(0.8, max(0.3, 1 - grosor / espacio_por_barra)))


def figura_vacia(mensaje="No hay datos para los filtros seleccionados"):
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(size=14, color=TINTA_TENUE))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _estilo(fig, alto=260, leyenda=False)


# ---------------------------------------------------------------------------
# Tasa de abandono
# ---------------------------------------------------------------------------
def tasa_por_categoria(df, col, tasa_global=None, titulo=None, alto=None):
    """Barras horizontales con la tasa de abandono por categoría e IC 95 %."""
    if df.empty:
        return figura_vacia()
    tabla = tasa_abandono_por(df, col)
    if col in ORDEN_CATEGORIAS:
        tabla[col] = pd.Categorical(tabla[col], _orden(col, tabla[col]), ordered=True)
        tabla = tabla.sort_values(col, ascending=False)
    elif pd.api.types.is_numeric_dtype(tabla[col]):
        tabla = tabla.sort_values(col, ascending=False)
    else:
        tabla = tabla.sort_values("tasa")
    tabla[col] = tabla[col].astype(str)

    fig = go.Figure(go.Bar(
        x=tabla["tasa"], y=tabla[col], orientation="h",
        marker=dict(color=AZUL),
        error_x=dict(type="data", symmetric=False,
                     array=tabla["ic_sup"] - tabla["tasa"],
                     arrayminus=tabla["tasa"] - tabla["ic_inf"],
                     color=TINTA_TENUE, thickness=1, width=4),
        customdata=np.column_stack([tabla["clientes"], tabla["abandonos"], tabla["ic_inf"], tabla["ic_sup"]]),
        hovertemplate=(
            "<b>%{y}</b><br>Tasa de abandono: %{x:.1%}"
            "<br>IC 95 %: %{customdata[2]:.1%} – %{customdata[3]:.1%}"
            "<br>Clientes: %{customdata[0]:,.0f}<br>Abandonos: %{customdata[1]:,.0f}<extra></extra>"
        ),
    ))
    # Etiqueta de valor después del intervalo, para que no se monte sobre la barra de error
    fig.add_trace(go.Scatter(
        x=tabla["ic_sup"], y=tabla[col], mode="text", text=[fmt_pct(t) for t in tabla["tasa"]],
        textposition="middle right", textfont=dict(color=TINTA_SECUNDARIA, size=12),
        hoverinfo="skip", cliponaxis=False,
    ))
    if tasa_global is not None:
        fig.add_vline(x=tasa_global, line_width=1, line_color=TINTA_TENUE,
                      annotation_text=f"Promedio {fmt_pct(tasa_global)}",
                      annotation_position="top", annotation_font=dict(color=TINTA_TENUE, size=11))
    alto = alto or max(240, 110 + 46 * len(tabla))
    _estilo(fig, titulo or f"Tasa de abandono por {etiqueta(col).lower()}", alto=alto, leyenda=False)
    fig.update_layout(bargap=_espacio_barras(alto, len(tabla)))
    fig.update_xaxes(tickformat=".0%", rangemode="tozero",
                     range=[0, min(1, float(tabla["ic_sup"].max()) * 1.25 + 0.02)])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TINTA_SECUNDARIA))
    return fig


def tasa_por_antiguedad(df, paso=6, tasa_global=None):
    """Línea de la tasa de abandono según los meses de permanencia, con banda IC 95 %."""
    if df.empty:
        return figura_vacia()
    bordes = np.arange(0, 72 + paso, paso)
    grupos = pd.cut(df["tenure"], bins=bordes, include_lowest=True)
    tabla = df.groupby(grupos, observed=True)["abandono"].agg(clientes="count", abandonos="sum").reset_index()
    tabla = tabla[tabla["clientes"] > 0]
    tabla["tasa"] = tabla["abandonos"] / tabla["clientes"]
    ic = np.array([intervalo_wilson(a, n) for a, n in zip(tabla["abandonos"], tabla["clientes"])])
    tabla["mes"] = [int(i.right) for i in tabla["tenure"]]
    tabla["rango"] = [f"{max(int(np.ceil(i.left)), 0)}–{int(i.right)} meses" for i in tabla["tenure"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([tabla["mes"], tabla["mes"][::-1]]),
        y=np.concatenate([ic[:, 1], ic[:, 0][::-1]]),
        fill="toself", fillcolor="rgba(42,120,214,0.10)", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=tabla["mes"], y=tabla["tasa"], mode="lines+markers",
        line=dict(color=AZUL, width=2),
        marker=dict(size=8, color=AZUL, line=dict(color=FONDO, width=2)),
        customdata=np.column_stack([tabla["rango"], tabla["clientes"], ic[:, 0], ic[:, 1]]),
        hovertemplate=("<b>%{customdata[0]}</b><br>Tasa de abandono: %{y:.1%}"
                       "<br>Clientes: %{customdata[1]}<extra></extra>"),
        showlegend=False,
    ))
    if tasa_global is not None:
        fig.add_hline(y=tasa_global, line_width=1, line_color=TINTA_TENUE,
                      annotation_text=f"Promedio {fmt_pct(tasa_global)}",
                      annotation_position="top right", annotation_font=dict(color=TINTA_TENUE, size=11))
    _estilo(fig, "Tasa de abandono según la antigüedad del cliente", alto=360, leyenda=False)
    fig.update_xaxes(title="Meses de permanencia", dtick=12, range=[0, 74])
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(hovermode="x")
    return fig


def ingreso_en_riesgo(df, col, alto=None):
    """Facturación mensual de los clientes que abandonaron, por categoría."""
    perdidos = df[df["abandono"] == 1]
    if perdidos.empty:
        return figura_vacia("No hay abandonos en la selección")
    tabla = perdidos.groupby(col, observed=True)["MonthlyCharges"].agg(["sum", "count"]).reset_index()
    tabla = tabla.sort_values("sum")
    tabla[col] = tabla[col].astype(str)
    fig = go.Figure(go.Bar(
        x=tabla["sum"], y=tabla[col], orientation="h", marker=dict(color=NARANJA),
        text=[f"USD {fmt_num(v)}" for v in tabla["sum"]],
        textposition="outside", cliponaxis=False, textfont=dict(color=TINTA_SECUNDARIA),
        customdata=tabla["count"],
        hovertemplate="<b>%{y}</b><br>Ingreso mensual perdido: USD %{x:,.0f}<br>Clientes perdidos: %{customdata}<extra></extra>",
    ))
    _estilo(fig, f"Ingreso mensual perdido por {etiqueta(col).lower()}",
            alto=alto or max(240, 110 + 46 * len(tabla)), leyenda=False)
    fig.update_layout(bargap=_espacio_barras(alto or max(240, 110 + 46 * len(tabla)), len(tabla)))
    fig.update_xaxes(tickprefix="USD ", range=[0, tabla["sum"].max() * 1.3])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=TINTA_SECUNDARIA))
    return fig


# ---------------------------------------------------------------------------
# Distribuciones y relaciones
# ---------------------------------------------------------------------------
def distribucion_numerica(df, col):
    """Histograma superpuesto (porcentaje dentro de cada grupo) con caja marginal."""
    if df.empty:
        return figura_vacia()
    datos = df.assign(Estado=df["Churn"].map(NOMBRES_CHURN))
    fig = px.histogram(
        datos, x=col, color="Estado", marginal="box", barmode="overlay",
        histnorm="percent", nbins=36, opacity=0.6,
        color_discrete_map={"Permanece": AZUL, "Abandona": NARANJA},
        category_orders={"Estado": ["Permanece", "Abandona"]},
        labels={col: etiqueta(col)},
    )
    fig.update_traces(hovertemplate="%{x}<br>%{y:.1f} % del grupo<extra></extra>", selector=dict(type="histogram"))
    _estilo(fig, f"Distribución de {etiqueta(col).lower()} según abandono", alto=420)
    fig.update_yaxes(title="% de clientes del grupo", row=1, col=1)
    fig.update_layout(bargap=0.04)
    return fig


def dispersion_antiguedad_cargo(df):
    """Dispersión antigüedad vs. cargo mensual; admite selección por caja o lazo."""
    if df.empty:
        return figura_vacia()
    fig = go.Figure()
    for valor in ["No", "Sí"]:
        parte = df[df["Churn"] == valor]
        fig.add_trace(go.Scattergl(
            x=parte["tenure"], y=parte["MonthlyCharges"], mode="markers",
            name=NOMBRES_CHURN[valor],
            marker=dict(size=7, color=COLORES_CHURN[valor], opacity=0.55, line=dict(width=0.5, color=FONDO)),
            customdata=np.column_stack([parte["customerID"], parte["Contract"], parte["InternetService"]]),
            hovertemplate=("<b>%{customdata[0]}</b><br>Antigüedad: %{x} meses"
                           "<br>Cargo mensual: USD %{y:.2f}<br>Contrato: %{customdata[1]}"
                           "<br>Internet: %{customdata[2]}<extra></extra>"),
        ))
    _estilo(fig, "Antigüedad frente a cargo mensual (seleccione una zona con caja o lazo)", alto=440)
    fig.update_xaxes(title="Antigüedad (meses)")
    fig.update_yaxes(title="Cargo mensual (USD)")
    fig.update_layout(dragmode="select")
    return fig


def mapa_calor_segmentos(df, fila, columna):
    """Tasa de abandono cruzando dos variables categóricas."""
    if df.empty:
        return figura_vacia()
    tabla = df.groupby([fila, columna], observed=True)["abandono"].agg(["mean", "count"]).reset_index()
    tasas = tabla.pivot(index=fila, columns=columna, values="mean")
    conteos = tabla.pivot(index=fila, columns=columna, values="count")
    tasas = tasas.reindex(index=_orden(fila, tasas.index), columns=_orden(columna, tasas.columns))
    conteos = conteos.reindex(index=tasas.index, columns=tasas.columns)
    texto = [[f"{fmt_pct(t)}<br>n = {fmt_num(c)}" if pd.notna(c) else "" for t, c in zip(ft, fc)]
             for ft, fc in zip(tasas.to_numpy(), conteos.to_numpy())]
    fig = go.Figure(go.Heatmap(
        z=tasas.to_numpy(), x=[str(c) for c in tasas.columns], y=[str(i) for i in tasas.index],
        text=texto, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=ESCALA_SECUENCIAL, zmin=0, zmax=max(0.6, float(np.nanmax(tasas.to_numpy()))),
        xgap=2, ygap=2, customdata=conteos.to_numpy(),
        hovertemplate=(f"{etiqueta(fila)}: %{{y}}<br>{etiqueta(columna)}: %{{x}}"
                       "<br>Tasa de abandono: %{z:.1%}<br>Clientes: %{customdata}<extra></extra>"),
        colorbar=dict(title=dict(text="Tasa", side="top"), tickformat=".0%", thickness=12, outlinewidth=0),
    ))
    _estilo(fig, f"Tasa de abandono: {etiqueta(fila).lower()} × {etiqueta(columna).lower()}",
            alto=max(320, 140 + 70 * len(tasas.index)), leyenda=False)
    fig.update_xaxes(showgrid=False, title=etiqueta(columna), side="bottom")
    fig.update_yaxes(showgrid=False, title=etiqueta(fila), autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Resultados estadísticos
# ---------------------------------------------------------------------------
def ranking_asociacion(tabla_chi):
    """V de Cramér por variable; en gris las asociaciones no significativas."""
    if tabla_chi is None or tabla_chi.empty:
        return figura_vacia("No fue posible calcular las pruebas")
    tabla = tabla_chi.sort_values("v_cramer")
    fig = go.Figure()
    for estado, color, nombre in [("Sí", AZUL, "Significativa (Bonferroni)"), ("No", GRIS, "No significativa")]:
        parte = tabla[tabla["significativa"] == estado]
        if parte.empty:
            continue
        fig.add_trace(go.Bar(
            x=parte["v_cramer"], y=parte["nombre"], orientation="h", name=nombre,
            marker=dict(color=color),
            customdata=np.column_stack([parte["chi2"], parte["p_bonferroni"], parte["efecto"]]),
            hovertemplate=("<b>%{y}</b><br>V de Cramér: %{x:.3f}<br>χ²: %{customdata[0]}"
                           "<br>p ajustado: %{customdata[1]:.2e}<br>Efecto: %{customdata[2]}<extra></extra>"),
        ))
    _estilo(fig, "Fuerza de asociación con el abandono (V de Cramér)", alto=max(300, 90 + 26 * len(tabla)))
    fig.update_layout(bargap=0.35, barmode="overlay")
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=tabla["nombre"].tolist(),
                     tickfont=dict(color=TINTA_SECUNDARIA))
    for x, texto in [(0.1, "pequeño"), (0.3, "mediano")]:
        fig.add_vline(x=x, line_width=1, line_color=GRIS, layer="below", annotation_text=texto,
                      annotation_position="bottom right", annotation_font=dict(color=TINTA_TENUE, size=10))
    return fig


ETIQUETAS_CORTAS = {"tenure": "Antigüedad", "MonthlyCharges": "Cargo mensual", "TotalCharges": "Cargo total",
                    "num_servicios": "N.º servicios", "abandono": "Abandono"}


def mapa_correlacion(matriz, titulo="Correlación de Spearman"):
    etiquetas = [ETIQUETAS_CORTAS.get(c, etiqueta(c)) for c in matriz.columns]
    fig = go.Figure(go.Heatmap(
        z=matriz.to_numpy(), x=etiquetas, y=etiquetas,
        text=np.vectorize(lambda v: f"{v:.2f}".replace(".", ","))(matriz.to_numpy()),
        texttemplate="%{text}", colorscale=ESCALA_DIVERGENTE, zmin=-1, zmax=1, xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(thickness=12, outlinewidth=0),
    ))
    _estilo(fig, titulo, alto=430, leyenda=False)
    fig.update_xaxes(showgrid=False, tickangle=-30, side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig
