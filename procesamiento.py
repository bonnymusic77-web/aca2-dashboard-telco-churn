"""
procesamiento.py
----------------
Carga, limpieza y transformación del dataset Telco Customer Churn (IBM).

El notebook y el dashboard importan este mismo módulo, de modo que el EDA
y la aplicación trabajan siempre sobre datos preparados con las mismas reglas.
"""

from pathlib import Path

import numpy as np
import pandas as pd

URL_DATOS = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RUTA_LOCAL = Path(__file__).parent / "data" / "Telco-Customer-Churn.csv"

# ---------------------------------------------------------------------------
# Definición de columnas
# ---------------------------------------------------------------------------
COL_ID = "customerID"
COL_OBJETIVO = "Churn"

COLS_NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges"]

COLS_CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

SERVICIOS_ADICIONALES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Valores válidos de cada variable categórica en el archivo original.
# Cualquier valor por fuera de estos conjuntos se trata como dato inválido.
_SI_NO = {"Yes", "No"}
_SERVICIO_INTERNET = {"Yes", "No", "No internet service"}
DOMINIOS = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": _SI_NO,
    "Dependents": _SI_NO,
    "PhoneService": _SI_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{col: _SERVICIO_INTERNET for col in SERVICIOS_ADICIONALES},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": _SI_NO,
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    "Churn": _SI_NO,
}

# Traducción de categorías para que el EDA y el dashboard queden en español
TRADUCCIONES = {
    "Yes": "Sí",
    "No": "No",
    "Female": "Femenino",
    "Male": "Masculino",
    "No phone service": "Sin servicio telefónico",
    "No internet service": "Sin servicio de internet",
    "Fiber optic": "Fibra óptica",
    "Month-to-month": "Mes a mes",
    "One year": "Un año",
    "Two year": "Dos años",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia (automático)",
    "Credit card (automatic)": "Tarjeta de crédito (automático)",
}

# Nombres legibles para gráficos, tablas y filtros
ETIQUETAS = {
    "customerID": "ID cliente",
    "gender": "Género",
    "SeniorCitizen": "Adulto mayor",
    "Partner": "Tiene pareja",
    "Dependents": "Tiene dependientes",
    "tenure": "Antigüedad (meses)",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Múltiples líneas",
    "InternetService": "Servicio de internet",
    "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección de equipos",
    "TechSupport": "Soporte técnico",
    "StreamingTV": "TV por streaming",
    "StreamingMovies": "Películas por streaming",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Factura electrónica",
    "PaymentMethod": "Método de pago",
    "MonthlyCharges": "Cargo mensual (USD)",
    "TotalCharges": "Cargo total (USD)",
    "Churn": "Abandono",
    "abandono": "Abandono (0/1)",
    "grupo_antiguedad": "Grupo de antigüedad",
    "num_servicios": "N.º de servicios adicionales",
    "pago_automatico": "Pago automático",
}

# Orden lógico de las categorías (para que los gráficos no queden alfabéticos)
GRUPOS_ANTIGUEDAD = ["0-12 meses", "13-24 meses", "25-48 meses", "49-72 meses"]
ORDEN_CATEGORIAS = {
    "Contract": ["Mes a mes", "Un año", "Dos años"],
    "InternetService": ["DSL", "Fibra óptica", "No"],
    "grupo_antiguedad": GRUPOS_ANTIGUEDAD,
    "Churn": ["No", "Sí"],
}


def etiqueta(col):
    """Devuelve el nombre en español de una columna."""
    return ETIQUETAS.get(col, col)


# ---------------------------------------------------------------------------
# 1. Carga y diagnóstico
# ---------------------------------------------------------------------------
def cargar_datos(ruta=None):
    """Lee el CSV crudo. Usa la ruta indicada, luego la copia local y,
    si no existe, descarga el archivo desde el repositorio de IBM."""
    for candidata in (ruta, RUTA_LOCAL, "Telco-Customer-Churn.csv"):
        if candidata is not None and Path(candidata).exists():
            return pd.read_csv(candidata)
    return pd.read_csv(URL_DATOS)


def diagnosticar_calidad(df):
    """Resumen de calidad por columna: tipo, nulos, textos vacíos y únicos."""
    filas = []
    for col in df.columns:
        serie = df[col]
        vacios = 0
        if not pd.api.types.is_numeric_dtype(serie):
            vacios = int(serie.astype(str).str.strip().eq("").sum())
        filas.append({
            "columna": col,
            "tipo": str(serie.dtype),
            "nulos": int(serie.isna().sum()),
            "vacios": vacios,
            "unicos": int(serie.nunique(dropna=True)),
            "ejemplo": serie.dropna().iloc[0] if serie.notna().any() else None,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
def estandarizar_texto(df):
    """Quita espacios sobrantes y convierte textos vacíos en NaN."""
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].str.strip().replace("", np.nan)
    return df


def corregir_tipos(df):
    """Convierte a número las columnas que llegan como texto (TotalCharges)."""
    df = df.copy()
    for col in COLS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    return df


def eliminar_duplicados(df):
    """Elimina filas repetidas y clientes con ID duplicado (conserva el primero)."""
    n_inicial = len(df)
    df = df.drop_duplicates().drop_duplicates(subset=COL_ID, keep="first")
    return df.reset_index(drop=True), n_inicial - len(df)


def validar_dominios(df):
    """Marca como NaN los valores imposibles: categorías no esperadas,
    antigüedad o cargos negativos. Devuelve el conteo por columna."""
    df = df.copy()
    invalidos = {}
    for col, validos in DOMINIOS.items():
        mascara = df[col].notna() & ~df[col].isin(validos)
        invalidos[col] = int(mascara.sum())
        df.loc[mascara, col] = np.nan
    for col in COLS_NUMERICAS:
        mascara = df[col] < 0
        invalidos[col] = int(mascara.sum())
        df.loc[mascara, col] = np.nan
    return df, {k: v for k, v in invalidos.items() if v > 0}


def imputar_nulos(df):
    """Imputa faltantes con reglas de negocio y, en último caso, estadísticas.

    - TotalCharges nulo con tenure = 0: cliente recién vinculado que aún no
      ha recibido factura, por lo que su cargo acumulado es 0.
    - TotalCharges nulo con tenure > 0: tenure × MonthlyCharges.
    - Resto de numéricas: mediana (robusta ante asimetría).
    - Categóricas: moda.
    - Filas sin customerID o sin Churn se eliminan (no se pueden analizar).
    """
    df = df.copy()
    registro = {}

    sin_clave = df[COL_ID].isna() | df[COL_OBJETIVO].isna()
    registro["filas_sin_id_o_churn"] = int(sin_clave.sum())
    df = df.loc[~sin_clave].copy()

    nuevos = df["TotalCharges"].isna() & df["tenure"].eq(0)
    registro["TotalCharges_cliente_nuevo"] = int(nuevos.sum())
    df.loc[nuevos, "TotalCharges"] = 0.0

    estimables = df["TotalCharges"].isna() & df["tenure"].gt(0) & df["MonthlyCharges"].notna()
    registro["TotalCharges_estimado"] = int(estimables.sum())
    df.loc[estimables, "TotalCharges"] = df.loc[estimables, "tenure"] * df.loc[estimables, "MonthlyCharges"]

    for col in COLS_NUMERICAS:
        n = int(df[col].isna().sum())
        if n:
            df[col] = df[col].fillna(df[col].median())
            registro[f"{col}_mediana"] = n

    for col in COLS_CATEGORICAS:
        n = int(df[col].isna().sum())
        if n:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
            registro[f"{col}_moda"] = n

    return df.reset_index(drop=True), {k: v for k, v in registro.items() if v > 0}


def detectar_outliers_iqr(df, cols=None, k=1.5):
    """Tabla de límites de Tukey (Q1 - k·IQR, Q3 + k·IQR) y atípicos por variable."""
    cols = cols or COLS_NUMERICAS
    filas = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - k * iqr, q3 + k * iqr
        n_out = int(((df[col] < lim_inf) | (df[col] > lim_sup)).sum())
        filas.append({
            "variable": col, "Q1": q1, "Q3": q3, "IQR": iqr,
            "limite_inferior": lim_inf, "limite_superior": lim_sup,
            "min": df[col].min(), "max": df[col].max(),
            "atipicos": n_out, "pct_atipicos": 100 * n_out / len(df),
        })
    return pd.DataFrame(filas)


def tratar_outliers(df, cols=None, metodo="conservar", k=1.5):
    """Aplica el tratamiento elegido a los atípicos por IQR.

    metodo = "conservar"  -> no modifica (valores reales del negocio)
    metodo = "winsorizar" -> recorta a los límites de Tukey
    metodo = "eliminar"   -> quita las filas con algún atípico
    """
    cols = cols or COLS_NUMERICAS
    if metodo == "conservar":
        return df.copy()
    limites = detectar_outliers_iqr(df, cols, k).set_index("variable")
    df = df.copy()
    if metodo == "winsorizar":
        for col in cols:
            df[col] = df[col].clip(limites.at[col, "limite_inferior"], limites.at[col, "limite_superior"])
        return df
    if metodo == "eliminar":
        mascara = pd.Series(False, index=df.index)
        for col in cols:
            mascara |= (df[col] < limites.at[col, "limite_inferior"]) | (df[col] > limites.at[col, "limite_superior"])
        return df.loc[~mascara].reset_index(drop=True)
    raise ValueError("metodo debe ser 'conservar', 'winsorizar' o 'eliminar'")


def verificar_consistencia_cargos(df, tolerancia=0.25):
    """Compara TotalCharges con tenure × MonthlyCharges.

    Una diferencia moderada es normal (cambios de tarifa en el tiempo); la
    función cuenta cuántos clientes se alejan más de la tolerancia indicada."""
    esperado = df["tenure"] * df["MonthlyCharges"]
    base = esperado.where(esperado > 0)
    desviacion = (df["TotalCharges"] - esperado).abs() / base
    return {
        "desviacion_mediana_pct": round(float(100 * desviacion.median()), 2),
        "clientes_fuera_tolerancia": int((desviacion > tolerancia).sum()),
        "pct_fuera_tolerancia": round(float(100 * (desviacion > tolerancia).mean()), 2),
    }


# ---------------------------------------------------------------------------
# 3. Transformación
# ---------------------------------------------------------------------------
def transformar_variables(df):
    """Traduce categorías y crea variables derivadas para el análisis."""
    df = df.copy()

    # Variables nuevas (se calculan antes de traducir, sobre los valores originales)
    df["abandono"] = (df[COL_OBJETIVO] == "Yes").astype(int)
    df["num_servicios"] = (df[SERVICIOS_ADICIONALES] == "Yes").sum(axis=1).astype(int)
    df["pago_automatico"] = np.where(
        df["PaymentMethod"].astype(str).str.contains("automatic"), "Sí", "No"
    )
    df["grupo_antiguedad"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72], labels=GRUPOS_ANTIGUEDAD
    ).astype(str)

    # SeniorCitizen viene como 0/1: se pasa a texto para tratarla como categoría
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Sí", 0: "No"})

    # Traducción al español de las categorías
    for col in COLS_CATEGORICAS + [COL_OBJETIVO]:
        df[col] = df[col].replace(TRADUCCIONES)

    df["tenure"] = df["tenure"].astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. Pipeline completo
# ---------------------------------------------------------------------------
def _describir_conteos(conteos, vacio="ninguno"):
    """{'Contract': 1, 'MonthlyCharges': 2} -> 'Contract (1), MonthlyCharges (2)'."""
    if not conteos:
        return vacio
    return ", ".join(f"{clave} ({valor})" for clave, valor in conteos.items())


def preparar_datos(df_crudo, metodo_outliers="conservar"):
    """Ejecuta toda la limpieza en orden y devuelve (df_limpio, bitacora).

    La bitácora deja registro de cuántos registros afectó cada paso, lo que
    permite justificar la limpieza en el reporte y detectar cambios si el
    archivo de origen se actualiza."""
    bitacora = []

    def anotar(paso, detalle, filas):
        bitacora.append({"paso": paso, "detalle": detalle, "filas": filas})

    anotar("Carga", "Archivo crudo", len(df_crudo))

    nulos_antes = int(df_crudo.isna().sum().sum())
    df = estandarizar_texto(df_crudo)
    vacios = int(df.isna().sum().sum()) - nulos_antes
    anotar("Texto", f"Espacios recortados; {vacios} textos vacíos pasan a NaN", len(df))

    nulos_antes = int(df.isna().sum().sum())
    df = corregir_tipos(df)
    no_convertibles = int(df.isna().sum().sum()) - nulos_antes
    anotar("Tipos", f"Columnas numéricas convertidas ({no_convertibles} valores no convertibles)", len(df))

    df, n_dup = eliminar_duplicados(df)
    anotar("Duplicados", f"{n_dup} registros eliminados", len(df))

    df, invalidos = validar_dominios(df)
    anotar("Dominios", f"Valores fuera de dominio: {_describir_conteos(invalidos)}", len(df))

    df, imputados = imputar_nulos(df)
    anotar("Nulos", f"Imputaciones: {_describir_conteos(imputados, 'ninguna')}", len(df))

    atipicos = detectar_outliers_iqr(df)
    total_atipicos = int(atipicos["atipicos"].sum())
    df = tratar_outliers(df, metodo=metodo_outliers)
    anotar("Atípicos", f"{total_atipicos} atípicos por IQR; tratamiento: {metodo_outliers}", len(df))

    df = transformar_variables(df)
    anotar("Transformación", "Categorías en español + 4 variables derivadas", len(df))

    # Validaciones finales: si algo falla, el pipeline se detiene
    assert df.isna().sum().sum() == 0, "Quedaron valores nulos"
    assert df[COL_ID].is_unique, "Hay clientes duplicados"
    assert (df["TotalCharges"] >= 0).all(), "Cargos totales negativos"

    return df, pd.DataFrame(bitacora)
