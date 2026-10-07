"""
Panel Catastral Inteligente — Demostración de Streamlit
Curso: Machine Learning Aplicado al Espacio Geográfico — CNR El Salvador
Módulo V: Implementación Práctica y Herramientas
Facilitadora: Jessica Martínez · doulus.jefis@gmail.com

Todos los datos son SINTÉTICOS (generados al iniciar la app). No se usa
información real del CNR. La app es un solo archivo para que sea fácil de
leer, copiar y modificar por los participantes.
"""
import numpy as np
import pandas as pd
import folium
import plotly.express as px
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Panel Catastral Inteligente · CNR",
    page_icon="🗺️",
    layout="wide",
)

NAVY, TEAL, GOLD = "#1E2761", "#1B7F8C", "#C9A227"
COLORES_SEMAFORO = {"🟢 Verde": "#2E9E5B", "🟡 Amarillo": "#E8B923",
                    "🔴 Rojo": "#D64545", "⚪ Gris": "#8A94A6"}

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 17px; }
      h1, h2, h3 { color: #1E2761; }
      div[data-testid="stMetricValue"] { color: #1B7F8C; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. DATOS SINTÉTICOS (municipios de El Salvador, valores ficticios)
# ---------------------------------------------------------------------------
MUNICIPIOS = {
    # nombre: (lat, lon, factor de valor, elevación base m, efecto no observado)
    "Antiguo Cuscatlán": (13.6700, -89.2400, 1.80, 900, 0.10),
    "San Salvador":      (13.6929, -89.2182, 1.50, 680, 0.05),
    "Santa Tecla":       (13.6769, -89.2797, 1.35, 920, -0.04),
    "La Libertad":       (13.4883, -89.3225, 1.00, 15, 0.14),
    "Santa Ana":         (13.9942, -89.5597, 0.95, 660, -0.08),
    "Soyapango":         (13.7100, -89.1400, 0.90, 650, 0.00),
    "San Miguel":        (13.4833, -88.1833, 0.85, 110, 0.07),
    "Sonsonate":         (13.7189, -89.7242, 0.80, 220, -0.12),
    "Zacatecoluca":      (13.5000, -88.8700, 0.65, 120, 0.03),
    "Usulután":          (13.3500, -88.4500, 0.60, 80, -0.06),
}
USOS = ["Residencial", "Comercial", "Industrial", "Agrícola"]
MULT_USO = {"Residencial": 1.0, "Comercial": 2.0, "Industrial": 1.25, "Agrícola": 0.28}
FEATURES_NUM = ["latitud", "longitud", "area_m2", "dist_vial_m",
                "ndvi", "ndbi", "elevacion_m", "pendiente_pct"]
COLUMNAS_PLANTILLA = FEATURES_NUM + ["uso_suelo"]


@st.cache_data(show_spinner=False)
def generar_datos(n_por_municipio: int = 150, semilla: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    filas = []
    for muni, (lat0, lon0, factor, elev0, efecto) in MUNICIPIOS.items():
        n = n_por_municipio
        lat = lat0 + rng.normal(0, 0.022, n)
        lon = lon0 + rng.normal(0, 0.022, n)
        dist_centro_km = np.hypot(lat - lat0, lon - lon0) * 111
        probs = {"Antiguo Cuscatlán": [.55, .30, .10, .05], "San Salvador": [.50, .38, .10, .02],
                 "Santa Tecla": [.60, .25, .10, .05], "Soyapango": [.55, .15, .27, .03]}
        p = probs.get(muni, [.45, .15, .05, .35])
        uso = rng.choice(USOS, n, p=p)
        area = np.exp(rng.normal(5.9, 0.75, n))
        area = np.where(uso == "Agrícola", area * 8, area)
        dist_vial = rng.exponential(220, n)
        ndvi = np.clip(rng.normal(0.42, 0.18, n) - 0.25 * (dist_centro_km < 1.5)
                       + 0.25 * (uso == "Agrícola"), -0.1, 0.9)
        ndbi = np.clip(0.30 - 0.55 * ndvi + rng.normal(0, 0.05, n), -0.5, 0.6)
        elev = elev0 + rng.normal(0, 25, n) + 8 * dist_centro_km
        pend = np.abs(rng.normal(8, 6, n))
        ln_valor = (np.log(120 * factor) + np.log([MULT_USO[u] for u in uso])
                    - 0.0009 * dist_vial - 0.06 * dist_centro_km - 0.012 * pend
                    - 0.00004 * area + 0.25 * (ndvi < 0.25) * (uso != "Agrícola")
                    + efecto + rng.normal(0, 0.12, n))
        valor = np.exp(ln_valor)
        # Valor declarado: casi igual al "real", con ~7 % de parcelas irregulares
        declarado = valor * np.exp(rng.normal(0, 0.08, n))
        irregular = rng.random(n) < 0.07
        declarado = np.where(irregular, declarado * rng.choice([0.45, 0.5, 1.9, 2.2], n), declarado)
        filas.append(pd.DataFrame({
            "municipio": muni, "latitud": lat, "longitud": lon, "area_m2": area,
            "dist_vial_m": dist_vial, "ndvi": ndvi, "ndbi": ndbi, "elevacion_m": elev,
            "pendiente_pct": pend, "uso_suelo": uso,
            "valor_m2_real": valor, "valor_declarado_m2": declarado,
        }))
    df = pd.concat(filas, ignore_index=True)
    df.insert(0, "id_parcela", [f"P-{i:04d}" for i in range(1, len(df) + 1)])
    return df


def construir_X(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de variables del modelo (misma función para entrenar y predecir)."""
    X = df[FEATURES_NUM].copy().astype(float)
    X["area_m2"] = np.log(X["area_m2"].clip(lower=1))
    for u in USOS:
        X[f"uso_{u}"] = (df["uso_suelo"] == u).astype(float)
    return X


# ---------------------------------------------------------------------------
# 3. MODELO (se entrena una sola vez gracias a st.cache_resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando el modelo (solo la primera vez)…")
def entrenar_modelo():
    df = generar_datos()
    X, y = construir_X(df), np.log(df["valor_m2_real"])
    modelo = RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                   n_jobs=-1, random_state=42).fit(X.values, y)

    # Validación: aleatoria vs espacial (se deja fuera un municipio completo)
    rf_cv = RandomForestRegressor(n_estimators=60, min_samples_leaf=3,
                                  n_jobs=-1, random_state=42)
    pred_rand = cross_val_predict(rf_cv, X.values, y, cv=KFold(5, shuffle=True, random_state=1))
    pred_esp = cross_val_predict(rf_cv, X.values, y, cv=GroupKFold(5), groups=df["municipio"])
    val = pd.DataFrame({
        "Estrategia": ["K-Fold aleatorio", "Validación espacial (por municipio)"],
        "R²": [r2_score(y, pred_rand), r2_score(y, pred_esp)],
        "Error medio (US$/m²)": [mean_absolute_error(df["valor_m2_real"], np.exp(pred_rand)),
                                 mean_absolute_error(df["valor_m2_real"], np.exp(pred_esp))],
    })
    df = df.assign(valor_pred_m2=np.exp(pred_rand))  # predicción fuera de muestra (cada parcela se predice sin haberla visto)

    # Dominio de aplicabilidad: distancia de Mahalanobis
    Z = X[["area_m2", "dist_vial_m", "ndvi", "ndbi", "elevacion_m", "pendiente_pct"]].values
    mu, cov_inv = Z.mean(0), np.linalg.pinv(np.cov(Z, rowvar=False))
    d = np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, cov_inv, Z - mu))
    dominio = {"mu": mu, "cov_inv": cov_inv, "p99": float(np.percentile(d, 99))}

    imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values()
    return modelo, df, val, dominio, imp


def mahalanobis(df_nuevo, dominio):
    Z = construir_X(df_nuevo)[["area_m2", "dist_vial_m", "ndvi", "ndbi",
                               "elevacion_m", "pendiente_pct"]].values
    D = Z - dominio["mu"]
    return np.sqrt(np.einsum("ij,jk,ik->i", D, dominio["cov_inv"], D))


def predecir_con_incertidumbre(modelo, df_nuevo):
    """Predicción + intervalo del 90 % usando la dispersión de los 150 árboles."""
    X = construir_X(df_nuevo).values
    arboles = np.exp(np.array([t.predict(X) for t in modelo.estimators_]))
    return (np.median(arboles, 0), np.percentile(arboles, 5, 0), np.percentile(arboles, 95, 0))


def semaforo(razon, fuera_dominio, tol_verde, tol_amarillo):
    desv = abs(razon - 1)
    if fuera_dominio:
        return "⚪ Gris"
    return "🟢 Verde" if desv <= tol_verde else ("🟡 Amarillo" if desv <= tol_amarillo else "🔴 Rojo")



# Contorno simplificado de El Salvador (longitud, latitud) para dibujarlo aunque
# el mapa base de calles no cargue. Fuente: Natural Earth / datasets/geo-countries.
CONTORNO_SV = [[-90.0983,13.7314],[-90.1143,13.7963],[-90.107,13.8441],[-90.0871,13.8705],[-90.0474,13.8943],[-90.023,13.937],[-89.8908,14.0359],[-89.8359,14.0591],[-89.803,14.0555],[-89.7623,14.03],[-89.7476,14.0376],[-89.7526,14.0752],[-89.7099,14.1487],[-89.6898,14.17],[-89.6383,14.2006],[-89.5309,14.2256],[-89.5245,14.2318],[-89.5247,14.2474],[-89.5449,14.2618],[-89.5586,14.3028],[-89.5922,14.3138],[-89.5984,14.328],[-89.5761,14.3531],[-89.5906,14.386],[-89.5841,14.4018],[-89.5699,14.412],[-89.5552,14.4108],[-89.5442,14.4001],[-89.5414,14.3815],[-89.503,14.4201],[-89.4853,14.4274],[-89.4468,14.4155],[-89.4311,14.4187],[-89.3982,14.4454],[-89.3969,14.426],[-89.3906,14.436],[-89.3618,14.4155],[-89.3076,14.4071],[-89.2765,14.3927],[-89.2255,14.3821],[-89.2002,14.3647],[-89.154,14.3513],[-89.1222,14.3947],[-89.1075,14.396],[-89.0918,14.3748],[-89.0945,14.3449],[-89.0831,14.3353],[-89.033,14.3237],[-89.0181,14.2717],[-88.9835,14.2421],[-88.966,14.1914],[-88.9572,14.1852],[-88.9216,14.1999],[-88.906,14.1996],[-88.8813,14.1842],[-88.8603,14.1585],[-88.8388,14.1004],[-88.7773,14.0997],[-88.7464,14.1113],[-88.7392,14.0999],[-88.7466,14.0525],[-88.731,14.0425],[-88.7049,14.038],[-88.6927,14.0253],[-88.6909,14.01],[-88.6339,14.0145],[-88.5581,13.9909],[-88.5044,13.9837],[-88.4954,13.9751],[-88.5023,13.9689],[-88.4961,13.9557],[-88.5023,13.9171],[-88.4887,13.8813],[-88.488,13.8653],[-88.4968,13.8512],[-88.4702,13.8525],[-88.4332,13.8711],[-88.3929,13.8798],[-88.3601,13.8724],[-88.3551,13.891],[-88.3262,13.8851],[-88.3193,13.8976],[-88.2854,13.906],[-88.2636,13.9331],[-88.2364,13.9381],[-88.2316,13.9518],[-88.2444,13.9661],[-88.2449,13.9893],[-88.2351,13.9929],[-88.1775,13.9852],[-88.1254,13.9913],[-88.099,13.9898],[-88.0875,13.9805],[-88.0729,13.9434],[-88.0236,13.8912],[-88.0153,13.8664],[-87.9548,13.8925],[-87.9253,13.8796],[-87.9001,13.8941],[-87.8609,13.8902],[-87.8329,13.9153],[-87.8179,13.9159],[-87.8085,13.9086],[-87.7906,13.8705],[-87.7036,13.815],[-87.7378,13.7388],[-87.7314,13.7222],[-87.755,13.6899],[-87.7601,13.6169],[-87.7899,13.5328],[-87.7795,13.51],[-87.7692,13.5064],[-87.748,13.5132],[-87.7334,13.5077],[-87.7206,13.487],[-87.7212,13.4608],[-87.7384,13.4417],[-87.8172,13.4066],[-87.8257,13.4113],[-87.8383,13.4409],[-87.8664,13.3933],[-87.8738,13.3653],[-87.8436,13.3485],[-87.7915,13.305],[-87.7899,13.287],[-87.7981,13.2657],[-87.818,13.2504],[-87.8923,13.2139],[-87.9125,13.1982],[-87.916,13.1822],[-87.8929,13.1666],[-87.9282,13.1586],[-88.0984,13.174],[-88.2083,13.1604],[-88.3009,13.1742],[-88.3249,13.1666],[-88.3523,13.174],[-88.3523,13.1802],[-88.3297,13.1817],[-88.3281,13.1971],[-88.3402,13.2161],[-88.3584,13.2287],[-88.3379,13.1945],[-88.3918,13.1939],[-88.4069,13.1877],[-88.3972,13.1836],[-88.3659,13.1877],[-88.3659,13.1802],[-88.3892,13.1708],[-88.4148,13.1716],[-88.4348,13.1837],[-88.441,13.2082],[-88.4274,13.2014],[-88.4137,13.2212],[-88.4308,13.2288],[-88.4496,13.2546],[-88.4621,13.2622],[-88.4502,13.2254],[-88.4547,13.215],[-88.5298,13.2429],[-88.554,13.2764],[-88.5987,13.2839],[-88.7228,13.2696],[-88.7087,13.2605],[-88.6666,13.2558],[-88.5774,13.2628],[-88.5366,13.2287],[-88.5366,13.2212],[-88.5987,13.2355],[-88.5987,13.2287],[-88.4739,13.1986],[-88.4547,13.1802],[-88.4617,13.1683],[-88.4796,13.1712],[-88.5275,13.1984],[-88.7857,13.2453],[-89.0648,13.372],[-89.1835,13.4431],[-89.2515,13.4723],[-89.3249,13.4893],[-89.385,13.4953],[-89.5271,13.4942],[-89.6798,13.5347],[-89.7131,13.528],[-89.808,13.5269],[-89.8233,13.5371],[-89.8371,13.5968],[-89.8497,13.6069],[-89.9513,13.6641],[-90.0983,13.7314]]


def _tooltip_html(fila, campos):
    partes = [f"<b>{fila['id_parcela']}</b>"] if "id_parcela" in fila else []
    for col, (etiqueta, fmt) in campos.items():
        if col in fila:
            valor = fila[col]
            partes.append(f"{etiqueta}: {valor:{fmt}}" if fmt else f"{etiqueta}: {valor}")
    return "<br>".join(partes)


def mapa_folium(df, color_col, campos, colores=None, etiqueta_color="", alto=520):
    """Mapa interactivo con Folium (Leaflet): capas base, contorno del país,
    puntos coloreados y leyenda. Los puntos y el contorno se dibujan aunque los
    mosaicos del mapa base no se puedan descargar."""
    m = folium.Map(location=[13.75, -88.9], zoom_start=8, tiles=None,
                   control_scale=True, prefer_canvas=True)
    folium.TileLayer("OpenStreetMap", name="Calles (OpenStreetMap)").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, HERE, Garmin", name="Gris claro (Esri)", show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Imagen satelital (Esri)", show=False).add_to(m)

    folium.GeoJson(
        {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [CONTORNO_SV]}},
        name="Contorno de El Salvador",
        style_function=lambda _: {"color": NAVY, "weight": 2.5, "fillColor": "#1B7F8C", "fillOpacity": 0.06},
    ).add_to(m)

    if colores is None:  # variable continua → escala de color
        lo, hi = np.percentile(df[color_col], [2, 98])
        escala = LinearColormap(["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
                                vmin=float(lo), vmax=float(hi))
        color_de = lambda v: escala(min(max(v, lo), hi))
        # Leyenda propia en HTML/CSS (la de branca descarga d3 desde internet y puede fallar)
        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;'
            'padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);'
            f'font:13px sans-serif;color:#1E2761"><b>{etiqueta_color}</b>'
            '<div style="width:190px;height:12px;margin:6px 0 2px;border-radius:3px;'
            'background:linear-gradient(to right,#440154,#3b528b,#21918c,#5ec962,#fde725)"></div>'
            f'<div style="display:flex;justify-content:space-between"><span>{lo:,.0f}</span>'
            f'<span>{(lo + hi) / 2:,.0f}</span><span>{hi:,.0f}</span></div></div>'))
    else:                # variable categórica → colores fijos + leyenda HTML
        color_de = lambda v: colores.get(v, "#8A94A6")
        items = "".join(f'<div><span style="background:{c};display:inline-block;width:12px;'
                        f'height:12px;border-radius:50%;margin-right:6px"></span>{n.split(" ", 1)[-1]}</div>'
                        for n, c in colores.items())
        m.get_root().html.add_child(folium.Element(
            '<div style="position:fixed;bottom:24px;left:12px;z-index:9999;background:white;'
            'padding:8px 12px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);'
            f'font:13px sans-serif;color:#1E2761">{items}</div>'))

    capa = folium.FeatureGroup(name="Parcelas")
    for fila in df.to_dict("records"):
        c = color_de(fila[color_col])
        folium.CircleMarker(
            location=[fila["latitud"], fila["longitud"]], radius=5, color=c, weight=0.6,
            fill=True, fill_color=c, fill_opacity=0.85,
            tooltip=folium.Tooltip(_tooltip_html(fila, campos), sticky=True)).add_to(capa)
    capa.add_to(m)

    if "municipio" in df.columns:  # etiquetas de municipio
        etiquetas = folium.FeatureGroup(name="Nombres de municipio", show=True)
        for muni in df["municipio"].unique():
            lat0, lon0 = MUNICIPIOS[muni][:2]
            folium.Marker([lat0 + 0.06, lon0], icon=folium.DivIcon(html=(
                f'<div style="font:600 11px sans-serif;color:#1E2761;text-shadow:0 0 3px #fff,'
                f'0 0 3px #fff;white-space:nowrap;transform:translateX(-50%)">{muni}</div>'))
            ).add_to(etiquetas)
        etiquetas.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, height=alto, use_container_width=True, returned_objects=[])


def mapa_plano(df, color_col, campos, colores=None, etiqueta_color="", alto=520):
    """Alternativa sin mapas en línea: dispersión lon/lat con el contorno del país."""
    if colores is None:
        fig = px.scatter(df, x="longitud", y="latitud", color=color_col, height=alto,
                         color_continuous_scale="Viridis", labels={color_col: etiqueta_color},
                         hover_data={c: True for c in campos if c in df.columns})
    else:
        fig = px.scatter(df, x="longitud", y="latitud", color=color_col, height=alto,
                         color_discrete_map=colores, category_orders={color_col: list(colores)},
                         hover_data={c: True for c in campos if c in df.columns})
    lon, lat = zip(*CONTORNO_SV)
    fig.add_scatter(x=lon, y=lat, mode="lines", line=dict(color=NAVY, width=2),
                    hoverinfo="skip", showlegend=False)
    fig.update_traces(marker=dict(size=7, opacity=0.85), selector=dict(mode="markers"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0 / np.cos(np.radians(13.6)))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#F4F7FB",
                      xaxis_title="Longitud", yaxis_title="Latitud", legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def mostrar_mapa(df, color_col, campos, colores=None, etiqueta_color="", alto=520):
    (mapa_folium if usar_folium else mapa_plano)(df, color_col, campos, colores, etiqueta_color, alto)


modelo, datos, validacion, dominio, importancias = entrenar_modelo()

# ---------------------------------------------------------------------------
# 4. ENCABEZADO Y BARRA LATERAL
# ---------------------------------------------------------------------------
st.title("🗺️ Panel Catastral Inteligente")
st.caption("Demostración de Streamlit · Módulo V: Implementación Práctica y Herramientas · "
           "Curso *Machine Learning Aplicado al Espacio Geográfico* — CNR El Salvador")

with st.sidebar:
    st.header("Filtros del panel")
    sel_muni = st.multiselect("Municipios", list(MUNICIPIOS), default=list(MUNICIPIOS))
    sel_uso = st.multiselect("Uso de suelo", USOS, default=USOS)
    usar_folium = st.toggle("Mapa interactivo (Folium)", value=True,
                            help="Mapa de El Salvador con Folium/Leaflet. Si en tu red no se ve, "
                                 "apágalo para usar un plano de coordenadas que funciona en cualquier red.")
    st.divider()
    st.subheader("Reglas del semáforo")
    tol_verde = st.slider("Verde si el valor declarado difiere ≤ (%)", 5, 40, 25, step=5) / 100
    tol_amar = st.slider("Amarillo si difiere ≤ (%)", 20, 80, 50, step=5) / 100
    st.info("Cambia un control y toda la app se recalcula. "
            "Eso es Streamlit: **tu script se vuelve a ejecutar de arriba abajo**.")
    st.caption("Datos 100 % sintéticos con fines didácticos.")

filtro = datos["municipio"].isin(sel_muni) & datos["uso_suelo"].isin(sel_uso)
vista = datos[filtro].copy()

tab_inicio, tab_val, tab_alertas, tab_modelo, tab_csv, tab_como = st.tabs([
    "🏠 Resumen", "💰 Valuador", "🚦 Alertas", "📈 Calidad del modelo",
    "📂 Sube tus datos", "🧩 ¿Cómo está hecho?"])

# ---------------------------------------------------------------------------
# 5. PESTAÑA: RESUMEN
# ---------------------------------------------------------------------------
with tab_inicio:
    if vista.empty:
        st.warning("Selecciona al menos un municipio y un uso de suelo en la barra lateral.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parcelas", f"{len(vista):,}")
        c2.metric("Valor mediano", f"US$ {vista['valor_m2_real'].median():,.0f}/m²")
        c3.metric("Área mediana", f"{vista['area_m2'].median():,.0f} m²")
        c4.metric("NDVI medio", f"{vista['ndvi'].mean():.2f}")

        izq, der = st.columns([3, 2])
        with izq:
            st.subheader("Mapa de valor por parcela")
            mostrar_mapa(vista, "valor_m2_real",
                         {"municipio": ("Municipio", ""), "uso_suelo": ("Uso", ""),
                          "valor_m2_real": ("Valor real (US$/m²)", ",.0f")},
                         etiqueta_color="US$/m²")
        with der:
            st.subheader("Valor por municipio")
            orden = vista.groupby("municipio")["valor_m2_real"].median().sort_values().index
            fig2 = px.box(vista, x="valor_m2_real", y="municipio", color_discrete_sequence=[TEAL],
                          category_orders={"municipio": list(orden)},
                          labels={"valor_m2_real": "US$/m²", "municipio": ""}, height=520)
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, width="stretch")
        with st.expander("Ver tabla de datos"):
            st.dataframe(vista.drop(columns=["valor_pred_m2"]).round(2), width="stretch")

# ---------------------------------------------------------------------------
# 6. PESTAÑA: VALUADOR (modelo en vivo + incertidumbre)
# ---------------------------------------------------------------------------
with tab_val:
    st.subheader("Valuador masivo con intervalo de confianza")
    st.write("Describe una parcela y el modelo (Random Forest) estima su valor por m², "
             "con un **intervalo del 90 %** que mide cuánta confianza debes tener.")
    a, b, c = st.columns(3)
    with a:
        muni_v = st.selectbox("Municipio de referencia", list(MUNICIPIOS), index=2)
        uso_v = st.selectbox("Uso de suelo", USOS)
        area_v = st.number_input("Área (m²)", 50, 100000, 350, step=50)
    with b:
        dist_v = st.slider("Distancia a vía principal (m)", 0, 1500, 150, step=10)
        ndvi_v = st.slider("NDVI (vegetación)", -0.1, 0.9, 0.35, step=0.05)
        ndbi_v = st.slider("NDBI (construcción)", -0.5, 0.6, 0.10, step=0.05)
    with c:
        elev_v = st.slider("Elevación (m s. n. m.)", 0, 1300, int(MUNICIPIOS[muni_v][3]), step=10)
        pend_v = st.slider("Pendiente (%)", 0, 40, 6)
        desv_lat = st.slider("Desplazamiento N–S desde el centro (km)", -5.0, 5.0, 0.0, step=0.5)

    lat0, lon0 = MUNICIPIOS[muni_v][:2]
    una = pd.DataFrame([{"latitud": lat0 + desv_lat / 111, "longitud": lon0, "area_m2": area_v,
                         "dist_vial_m": dist_v, "ndvi": ndvi_v, "ndbi": ndbi_v,
                         "elevacion_m": elev_v, "pendiente_pct": pend_v, "uso_suelo": uso_v}])
    med, p5, p95 = (x[0] for x in predecir_con_incertidumbre(modelo, una))
    ancho_rel = (p95 - p5) / med
    fuera = bool(mahalanobis(una, dominio)[0] > dominio["p99"])
    estado = ("⚪ Gris" if fuera else "🟢 Verde" if ancho_rel < 0.30
              else "🟡 Amarillo" if ancho_rel < 0.55 else "🔴 Rojo")
    mensaje = {"🟢 Verde": "Predicción confiable: puede usarse para valoración masiva.",
               "🟡 Amarillo": "Confianza media: conviene una revisión de campo por muestreo.",
               "🔴 Rojo": "Incertidumbre alta: requiere verificación por un perito.",
               "⚪ Gris": "Fuera del dominio de entrenamiento: el modelo NO debería usarse aquí."}[estado]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valor estimado", f"US$ {med:,.0f}/m²")
    k2.metric("Intervalo 90 %", f"{p5:,.0f} – {p95:,.0f}")
    k3.metric("Valor total estimado", f"US$ {med * area_v:,.0f}")
    k4.metric("Semáforo", estado)
    {"🟢 Verde": st.success, "🟡 Amarillo": st.warning,
     "🔴 Rojo": st.error, "⚪ Gris": st.info}[estado](mensaje)

    comparables = datos[datos["municipio"] == muni_v]
    fig = px.histogram(comparables, x="valor_m2_real", nbins=30, color_discrete_sequence=["#B8C4D6"],
                       labels={"valor_m2_real": "US$/m² (parcelas del municipio)"}, height=300)
    fig.add_vline(x=med, line_color=TEAL, line_width=3, annotation_text="Tu parcela")
    fig.add_vrect(x0=p5, x1=p95, fillcolor=TEAL, opacity=0.15, line_width=0)
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), showlegend=False)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 7. PESTAÑA: ALERTAS (semáforo catastral)
# ---------------------------------------------------------------------------
with tab_alertas:
    st.subheader("Semáforo de alertas catastrales")
    st.write("Compara el **valor declarado** de cada parcela con el **valor que predice el modelo** "
             "(estimado sin que el modelo haya visto esa parcela). "
             "Las reglas se ajustan en la barra lateral.")
    if vista.empty:
        st.warning("No hay parcelas con los filtros actuales.")
    else:
        v = vista.copy()
        v["razon"] = v["valor_declarado_m2"] / v["valor_pred_m2"]
        v["fuera_dominio"] = mahalanobis(v, dominio) > dominio["p99"]
        v["semaforo"] = [semaforo(r, f, tol_verde, tol_amar)
                         for r, f in zip(v["razon"], v["fuera_dominio"])]
        conteo = v["semaforo"].value_counts().reindex(list(COLORES_SEMAFORO), fill_value=0)
        cols = st.columns(4)
        for col, (nombre, n) in zip(cols, conteo.items()):
            col.metric(nombre, f"{n:,}  ({n / len(v):.0%})")

        mostrar_mapa(v, "semaforo",
                     {"municipio": ("Municipio", ""), "semaforo": ("Semáforo", ""),
                      "valor_declarado_m2": ("Declarado (US$/m²)", ",.0f"),
                      "valor_pred_m2": ("Predicho (US$/m²)", ",.0f")},
                     colores=COLORES_SEMAFORO)

        urgentes = (v[v["semaforo"].isin(["🔴 Rojo", "⚪ Gris"])]
                    [["id_parcela", "municipio", "uso_suelo", "area_m2", "valor_declarado_m2",
                      "valor_pred_m2", "razon", "semaforo"]]
                    .sort_values("razon", key=lambda s: (s - 1).abs(), ascending=False).round(2))
        st.markdown(f"**{len(urgentes)} parcelas para revisión prioritaria**")
        st.dataframe(urgentes, width="stretch", hide_index=True)
        st.download_button("⬇️ Descargar lista de revisión (CSV)",
                           urgentes.to_csv(index=False).encode("utf-8-sig"),
                           "parcelas_para_revision.csv", "text/csv")

# ---------------------------------------------------------------------------
# 8. PESTAÑA: CALIDAD DEL MODELO (Módulo IV dentro de la app)
# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("¿Qué tan bueno es el modelo? — validación aleatoria vs. espacial")
    r_a, r_e = validacion["R²"]
    c1, c2 = st.columns(2)
    c1.metric("R² con K-Fold aleatorio", f"{r_a:.2f}")
    c2.metric("R² con validación espacial", f"{r_e:.2f}", f"{r_e - r_a:+.2f}", delta_color="inverse")
    fig = px.bar(validacion, x="Estrategia", y="R²", color="Estrategia", text_auto=".2f",
                 color_discrete_sequence=[GOLD, TEAL], height=340)
    fig.update_layout(showlegend=False, yaxis_range=[0, 1], margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.info("**Lección del Módulo IV:** si mezclas parcelas vecinas entre entrenamiento y prueba, "
            "el modelo parece mejor de lo que realmente es (*vecino copión*). "
            "Al dejar fuera un municipio completo se ve el desempeño honesto.")
    st.subheader("Variables que más influyen")
    fig2 = px.bar(importancias, orientation="h", color_discrete_sequence=[NAVY],
                  labels={"value": "Importancia", "index": ""}, height=380)
    fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# 9. PESTAÑA: SUBE TUS DATOS (carga de CSV + predicción por lotes)
# ---------------------------------------------------------------------------
with tab_csv:
    st.subheader("Sube tu propio CSV y obtén predicciones")
    st.write("Descarga la plantilla, reemplaza las filas con tus parcelas y vuelve a subirla. "
             "Columnas requeridas: " + ", ".join(f"`{c}`" for c in COLUMNAS_PLANTILLA) + ".")
    plantilla = datos[COLUMNAS_PLANTILLA + ["valor_declarado_m2"]].sample(15, random_state=3).round(3)
    st.download_button("⬇️ Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8-sig"),
                       "plantilla_parcelas.csv", "text/csv")
    archivo = st.file_uploader("Sube tu archivo CSV", type="csv")
    if archivo is not None:
        try:
            nuevo = pd.read_csv(archivo)
            faltan = [c for c in COLUMNAS_PLANTILLA if c not in nuevo.columns]
            if faltan:
                st.error(f"Faltan columnas: {', '.join(faltan)}")
            elif not set(nuevo["uso_suelo"]).issubset(USOS):
                st.error(f"`uso_suelo` solo admite: {', '.join(USOS)}")
            else:
                med, p5, p95 = predecir_con_incertidumbre(modelo, nuevo)
                nuevo["valor_pred_m2"] = med.round(1)
                nuevo["ic90_inferior"], nuevo["ic90_superior"] = p5.round(1), p95.round(1)
                nuevo["fuera_de_dominio"] = mahalanobis(nuevo, dominio) > dominio["p99"]
                if "valor_declarado_m2" in nuevo.columns:
                    nuevo["razon_declarado_vs_pred"] = (nuevo["valor_declarado_m2"] / med).round(2)
                    nuevo["semaforo"] = [semaforo(r, f, tol_verde, tol_amar) for r, f in
                                         zip(nuevo["razon_declarado_vs_pred"], nuevo["fuera_de_dominio"])]
                st.success(f"Se procesaron {len(nuevo):,} parcelas.")
                st.dataframe(nuevo, width="stretch", hide_index=True)
                nuevo["id_parcela"] = [f"Fila {i + 1}" for i in range(len(nuevo))]
                mostrar_mapa(nuevo, "valor_pred_m2",
                             {"valor_pred_m2": ("Valor predicho (US$/m²)", ",.0f"),
                              "ic90_inferior": ("IC 90 % inferior", ",.0f"),
                              "ic90_superior": ("IC 90 % superior", ",.0f")},
                             etiqueta_color="US$/m² predicho", alto=460)
                st.download_button("⬇️ Descargar resultados", nuevo.to_csv(index=False).encode("utf-8-sig"),
                                   "resultados_prediccion.csv", "text/csv")
        except Exception as e:  # noqa: BLE001 — mensaje amable para el usuario
            st.error(f"No se pudo leer el archivo: {e}")

# ---------------------------------------------------------------------------
# 10. PESTAÑA: ¿CÓMO ESTÁ HECHO? (para enseñar Streamlit)
# ---------------------------------------------------------------------------
with tab_como:
    st.subheader("Cada elemento de esta app, en una línea de Python")
    st.write("Todo este panel está en **un solo archivo (`app.py`)**. Estas son las piezas que "
             "usamos; todas las puedes reutilizar en tu propia aplicación:")
    tabla = pd.DataFrame([
        ("Título y texto", "st.title(...), st.write(...), st.caption(...)", "Encabezados y explicaciones"),
        ("Controles", "st.slider(...), st.selectbox(...), st.multiselect(...)", "Filtros y entradas del usuario"),
        ("Métricas", "st.metric('Parcelas', 1500)", "Indicadores (KPI) grandes"),
        ("Pestañas y columnas", "st.tabs([...]), st.columns(3)", "Organizar la pantalla"),
        ("Gráficos y mapas", "px.scatter_map(df, lat=..., lon=...)", "Mapas interactivos con Plotly"),
        ("Tablas", "st.dataframe(df)", "Tablas ordenables y filtrables"),
        ("Carga de archivos", "st.file_uploader('CSV', type='csv')", "El usuario sube sus datos"),
        ("Descarga", "st.download_button(...)", "Exportar resultados"),
        ("Mensajes de estado", "st.success(...), st.warning(...), st.error(...)", "Semáforo y alertas"),
        ("Memoria (caché)", "@st.cache_data / @st.cache_resource", "Entrenar el modelo una sola vez"),
    ], columns=["Qué quiero", "Código", "Para qué sirve"])
    st.dataframe(tabla, width="stretch", hide_index=True)
    st.markdown("**Ejemplo mínimo — una app completa de 6 líneas:**")
    st.code('''import streamlit as st
import pandas as pd

st.title("Mi primera app catastral")
archivo = st.file_uploader("Sube un CSV con latitud y longitud", type="csv")
if archivo:
    st.map(pd.read_csv(archivo))''', language="python")
    st.markdown("**Idea clave:** cada vez que el usuario mueve un control, Streamlit vuelve a "
                "ejecutar el script de arriba a abajo. Por eso se usa la caché para no "
                "re-entrenar el modelo cada vez.")
    st.markdown("---")
    st.caption("Facilitadora: Jessica Martínez · doulus.jefis@gmail.com · Datos sintéticos, "
               "solo con fines educativos.")
