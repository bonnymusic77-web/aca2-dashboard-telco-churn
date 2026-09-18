# Dashboard de abandono de clientes · Telco Customer Churn

Proyecto **ACA 2** de la asignatura *Programación para Analítica de Datos e IA*
(Especialización en Inteligencia Artificial, CUN · Docente: Hamilton Rivera).

Aplicación web en Streamlit para explorar el abandono (*churn*) de 7.043 clientes de una empresa
de telecomunicaciones, con filtros dinámicos, gráficos interactivos y pruebas estadísticas.
Continúa el EDA del ACA 1 con el mismo dataset.

**Integrantes:** Bonny Darhyll Brayan Galindo Santos · Andrés Alberto López Herrera · Camilo José Rodríguez Ayubi

## Enlaces del proyecto

| Recurso | Enlace |
|---|---|
| Aplicación publicada | *(pendiente)* |
| Notebook en Google Colab | [`ACA2_Dashboard_EDA_Telco_Churn.ipynb`](ACA2_Dashboard_EDA_Telco_Churn.ipynb) |
| Reporte técnico (PDF) | [`Reporte_ACA2_Telco_Churn.pdf`](Reporte_ACA2_Telco_Churn.pdf) |
| Video de sustentación | *(pendiente)* |

## Contenido del repositorio

```
├── app.py                              # Interfaz del dashboard (Streamlit)
├── procesamiento.py                    # Carga, limpieza y transformación de datos
├── analisis.py                         # Pruebas estadísticas y análisis automático
├── graficos.py                         # Figuras interactivas (Plotly)
├── data/Telco-Customer-Churn.csv       # Dataset original (IBM)
├── requirements.txt
├── .streamlit/config.toml              # Tema de la aplicación
├── ACA2_Dashboard_EDA_Telco_Churn.ipynb  # Notebook con el EDA y el código del dashboard
├── Reporte_ACA2_Telco_Churn.pdf        # Reporte técnico en normas APA
└── Guion_Video_Sustentacion_ACA2.docx  # Guion del video
```

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Funcionalidades

- Filtros por contrato, servicio de internet, método de pago, soporte técnico, adulto mayor,
  antigüedad y cargo mensual.
- Indicadores: clientes, tasa de abandono, clientes perdidos, ingreso en riesgo y antigüedad media.
- Gráficos con zoom, *hover* y selección por caja o lazo.
- **Análisis automático:** hallazgos redactados, chi-cuadrado con V de Cramér, U de Mann-Whitney,
  correlación de Spearman y descriptivos, recalculados para cada filtro.
- Mapa de calor de segmentos, vista previa del dataset, descarga en CSV y bitácora de limpieza.

## Fuente de datos

IBM. *Telco Customer Churn* [Conjunto de datos]. Kaggle.
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
