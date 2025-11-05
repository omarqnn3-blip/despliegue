# Creamos el archivo de la APP en el interprete principal (Python)
### se corre con: streamlit run prueba.py
#####################################################
# Importamos librerias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
######################################################

# Definimos la instancia
@st.cache_resource
def load_data():
    # Lectura del archivo csv
    dfb = pd.read_csv("barcelona_super_limpio.csv")
    dfc = pd.read_csv("cambridgelimpio.csv")
    dfbo = pd.read_csv("limpiosB.csv")  # Boston
    dfh = pd.read_csv("hawaii_limpio.csv")
    dfbu = pd.read_csv("Budapest_Limpio.csv")
    return dfb, dfc, dfbo, dfh, dfbu


# limpieza de datos
def clean_data(dfb, dfc, dfbo, dfh, dfbu):
    # Barcelona
    if "price" in dfb.columns:
        dfb["price"] = (
            dfb["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Cambridge
    if "price" in dfc.columns:
        dfc["price"] = (
            dfc["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Boston
    if "price" in dfbo.columns:
        dfbo["price"] = (
            dfbo["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Hawái
    if "price" in dfh.columns:
        dfh["price"] = (
            dfh["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    # Budapest
    if "price" in dfbu.columns:
        dfbu["price"] = (
            dfbu["price"].astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )

    return dfb, dfc, dfbo, dfh, dfbu


###############################################################################
# CREACIÓN DEL DASHBOARD
###############################################################################
# Configuración de página (debe ser la primera llamada de Streamlit)
st.set_page_config(
    page_title="Dashboard Airbnb",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS (arreglado el cierre)
st.markdown("""
<style>
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] > div:first-child { background: #734042; }
</style>
""", unsafe_allow_html=True)

# Sidebar
logo_sidebar = 'airbnb.png'
col1, col2, col3 = st.sidebar.columns([1, 3, 1])
with col2:
    st.image(logo_sidebar)
    st.write("---")
st.sidebar.title("Análisis de Datos Airbnb")

# Widget 1: Selectbox (vista)
View = st.sidebar.selectbox(
    label="Tipo de Análisis",
    options=["Extracción de Características", "Tablas comparativas", "Regresión Lineal"],
)

# Widget 2: Checkbox (ver datos)
show_data = st.sidebar.checkbox(label="Mostrar Datos")

# Cargamos datos
dfb, dfc, dfbo, dfh, dfbu = load_data()
dfb, dfc, dfbo, dfh, dfbu = clean_data(dfb, dfc, dfbo, dfh, dfbu)

# ---- Mapa ciudad -> DataFrame (USARÁS ESTO EN TODAS LAS SECCIONES) ----
dfs_ciudades = {
    "Barcelona": dfb,
    "Cambridge": dfc,
    "Boston": dfbo,
    "Hawái": dfh,
    "Budapest": dfbu,
}

# Mostrar datos
if show_data:
    st.subheader("Datos de Airbnb en Barcelona")
    st.dataframe(dfb.head(10))
    st.subheader("Datos de Airbnb en Cambridge")
    st.dataframe(dfc.head(10))
    st.subheader("Datos de Airbnb en Boston")
    st.dataframe(dfbo.head(10))
    st.subheader("Datos de Airbnb en Hawái")
    st.dataframe(dfh.head(10))
    st.subheader("Datos de Airbnb en Budapest")
    st.dataframe(dfbu.head(10))

# Multiselect para extracción (puede elegir varias)
ciudades_multiselect = st.sidebar.multiselect(
    label="Selecciona las Ciudades (Extracción)",
    options=list(dfs_ciudades.keys()),
    default=["Barcelona", "Cambridge"],
    max_selections=4
)

# Radio para regresión (una por default)
ciudad_regresion = st.sidebar.radio(
    label="Ciudad para regresión (default)",
    options=list(dfs_ciudades.keys()),
    index=0,
)

# Multiselect para regresión (comparativa hasta 4)
ciudades_reg_sel = st.sidebar.multiselect(
    label="Ciudades para regresión (comparar hasta 4)",
    options=list(dfs_ciudades.keys()),
    default=[ciudad_regresion],
    max_selections=4
)

# Helper de hallazgos (se usa al final de cada sección)
def generar_hallazgos(ciudades):
    lines = []
    resumen = []
    for c in ciudades:
        df = dfs_ciudades[c]
        n = len(df)
        med_price = (df["price"].astype(float).median()
                     if "price" in df.columns else np.nan)
        corr_ap = (df[["accommodates", "price"]].corr().iloc[0, 1]
                   if set(["accommodates", "price"]).issubset(df.columns) else np.nan)
        resumen.append({"c": c, "n": n, "med_price": med_price, "corr": corr_ap})

    # mediana de price
    val = [r for r in resumen if not np.isnan(r["med_price"])]
    if val:
        top = max(val, key=lambda r: r["med_price"])
        low = min(val, key=lambda r: r["med_price"])
        if top["c"] != low["c"]:
            lines.append(
                f"• **{top['c']}** tiene la **mediana de precio** más alta (≈ {top['med_price']:.0f}); "
                f"**{low['c']}** la más baja (≈ {low['med_price']:.0f})."
            )

    # correlación accommodates–price
    valc = [r for r in resumen if not np.isnan(r["corr"])]
    if valc:
        strong = max(valc, key=lambda r: abs(r["corr"]))
        signo = "positiva" if strong["corr"] >= 0 else "negativa"
        lines.append(
            f"• La relación **accommodates–price** más marcada está en **{strong['c']}** "
            f"({signo}, r≈{strong['corr']:.2f})."
        )

    # tamaño de muestra
    topn = max(resumen, key=lambda r: r["n"])
    lines.append(f"• **{topn['c']}** cuenta con el **mayor número de anuncios** (n={topn['n']}).")
    return lines


###############################################################################
# 1) EXTRACCIÓN DE CARACTERÍSTICAS
###############################################################################
if View == "Extracción de Características":
    st.title("Extracción de Características")
    st.write("Análisis de características clave en los datos de Airbnb.")

    if not ciudades_multiselect:
        st.warning("Selecciona al menos una ciudad en la barra lateral 👈")
    else:
        n = len(ciudades_multiselect)
        cols = st.columns(min(4, n))

        # Script para las 4 columnas
        for i, ciudad in enumerate(ciudades_multiselect):

            if i > 0 and i % 4 == 0:
                cols = st.columns(min(4, n - i))

            df_ciudad = dfs_ciudades[ciudad]
            num_cols = df_ciudad.select_dtypes(include="number")

            with cols[i % 4]:
                st.subheader(ciudad)

                # gráfico de barrios si existe
                if "neighbourhood_cleansed" in df_ciudad.columns:
                    st.caption("Top barrios por número de alojamientos")
                    barrio_counts = df_ciudad["neighbourhood_cleansed"].value_counts().head(20)
                    st.bar_chart(barrio_counts)

                # multiselect de variables numéricas para esta ciudad
                if not num_cols.empty:
                    opciones = list(num_cols.columns)
                    default_vars = [c for c in opciones if c != "price"][:2] or [opciones[0]]
                    vars_sel = st.multiselect(
                        f"Variables numéricas para graficar ({ciudad})",
                        options=opciones,
                        default=default_vars,
                        key=f"vars_{ciudad}",
                    )

                    # 1. Histograma por variable
                    for v in vars_sel:
                        fig_hist = px.histogram(
                            df_ciudad.dropna(subset=[v]),
                            x=v,
                            nbins=30,
                            title=f"Distribución de {v} en {ciudad}",
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # 2. Scatter vs price si existe
                        if "price" in df_ciudad.columns and v != "price":
                            tmp = df_ciudad[[v, "price"]].dropna()
                            if not tmp.empty:
                                fig_scatter = px.scatter(
                                    tmp, x=v, y="price", trendline="ols",
                                    title=f"{v} vs Price en {ciudad}",
                                    labels={v: v, "price": "Price"},
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)

                    # 3. Boxplot de price (si existe)
                    if "price" in df_ciudad.columns:
                        tmp_price = df_ciudad[["price"]].dropna()
                        if not tmp_price.empty:
                            fig_box = px.box(tmp_price, y="price", title=f"Distribución de Price en {ciudad}")
                            st.plotly_chart(fig_box, use_container_width=True)

        # ---- Hallazgos ----
        st.markdown("### Hallazgos")
        for l in generar_hallazgos(ciudades_multiselect):
            st.markdown(l)

###############################################################################
# 2) TABLAS COMPARATIVAS
###############################################################################
elif View == "Tablas comparativas":
    st.title("Tablas Comparativas")
    st.write("Una tabla por ciudad; se muestran en filas de hasta 4.")

    ciudades_sel = st.multiselect(
        "Selecciona las ciudades a comparar",
        options=list(dfs_ciudades.keys()),
        default=["Barcelona", "Cambridge"],
        key="cmp_ciudades",
        max_selections=4
    )

    if not ciudades_sel:
        st.warning("Selecciona al menos una ciudad para comparar.")
    else:
        n = len(ciudades_sel)
        cols = st.columns(min(4, n))

        # Script para las 4 columnas
        for i, ciudad in enumerate(ciudades_sel):

            if i > 0 and i % 4 == 0:
                cols = st.columns(min(4, n - i))

            df_ciudad = dfs_ciudades[ciudad]
            num_cols = df_ciudad.select_dtypes(include="number")

            with cols[i % 4]:
                st.subheader(ciudad)
                if num_cols.empty:
                    st.info("Sin columnas numéricas.")
                else:
                    tabla = num_cols.agg(["mean", "median", "std"]).T.rename(
                        columns={"mean": "Media", "median": "Mediana", "std": "DesvEst"}
                    )
                    st.dataframe(tabla, use_container_width=True)

        # ---- Hallazgos ----
        st.markdown("### Hallazgos")
        for l in generar_hallazgos(ciudades_sel):
            st.markdown(l)

###############################################################################
# 3) REGRESIÓN LINEAL
###############################################################################
elif View == "Regresión Lineal":
    st.title("Regresión Lineal")
    st.write("Comparación de modelos por ciudad (hasta 4 simultáneas).")

    if not ciudades_reg_sel:
        st.warning("Selecciona al menos una ciudad para analizar.")
    else:
        n = len(ciudades_reg_sel)
        cols = st.columns(min(4, n))

        # Script para las 4 columnas
        for i, ciudad in enumerate(ciudades_reg_sel):

            if i > 0 and i % 4 == 0:
                cols = st.columns(min(4, n - i))

            df_ciudad = dfs_ciudades[ciudad]
            num_cols = df_ciudad.select_dtypes(include="number")

            with cols[i % 4]:
                st.subheader(ciudad)

                # Validación
                if not set(["accommodates", "price"]).issubset(df_ciudad.columns):
                    st.info("Faltan columnas 'accommodates' y/o 'price'.")
                    continue

                tmp = df_ciudad[["accommodates", "price"]].astype(float).dropna()
                if len(tmp) < 3:
                    st.info("Datos insuficientes para ajustar el modelo.")
                    continue

                x = tmp["accommodates"].to_numpy()
                y = tmp["price"].to_numpy()

                # Ajuste lineal simple
                a, b = np.polyfit(x, y, 1)
                y_pred = a * x + b

                # Métrica R² (in-sample)
                r2 = r2_score(y, y_pred)

                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{r2:.3f}")
                m2.metric("Pendiente (β1)", f"{a:.3f}")
                m3.metric("Intersección (β0)", f"{b:.2f}")

                # Gráfica con recta
                fig = px.scatter(
                    tmp, x="accommodates", y="price",
                    labels={"accommodates": "Accommodates", "price": "Price"},
                    title="Price ~ Accommodates"
                )
                x_line = np.linspace(x.min(), x.max(), 50)
                fig.add_trace(go.Scatter(x=x_line, y=a * x_line + b,
                                         mode="lines", name="Predicción"))
                st.plotly_chart(fig, use_container_width=True)

        # ---- Hallazgos ----
        st.markdown("### Hallazgos")
        for l in generar_hallazgos(ciudades_reg_sel):
            st.markdown(l)