# app_projet.py — Transposition 1:1 de ton notebook en Streamlit (une seule page)
# Nécessite: pip install streamlit folium streamlit-folium

import time
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Carto (Folium)
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Projet Padel/Tennis — Notebook → Streamlit", page_icon="🎾", layout="wide")
st.title("🎾 Projet Padel/Tennis — Transposition Notebook → Streamlit")

# =========================
# Sidebar : chemins CSV
# =========================
with st.sidebar:
    st.header("📂 Chemins CSV")
    data_es_path = st.text_input("Équipements (clean)", "data/data_es_clean.csv")
    dvf_idf_path = st.text_input("DVF communes IDF", "data/dvf_communes_2024_idf.csv")
    lic_path     = st.text_input("Licences Tennis IDF", "data/lic_2022_tennis_idf.csv")
    geo_path     = st.text_input("Géographie FR", "data/geographie.csv")
    com_path     = st.text_input("Communes France (clean)", "data/communes_france_clean.csv")
    chiffres_path= st.text_input("Chiffres Tennis/Padel FR", "data/ChiffresPadelTennis.csv")
    st.caption("💡 Adapte si besoin. Les colonnes attendues sont celles de ton notebook.")

def load_csv(path, sep=";"):
    p = Path(path)
    if not p.exists():
        st.error(f"Fichier introuvable : {p}")
        return None
    try:
        return pd.read_csv(p, sep=sep, low_memory=False)
    except Exception as e:
        st.error(f"Erreur lecture {p.name} : {e}")
        return None

# =========================
# Chargements (identique notebook, mêmes noms de variables)
# =========================
data_es = load_csv(data_es_path, sep=';')
dvf_idf = load_csv(dvf_idf_path, sep=';')
lic_tennis = load_csv(lic_path, sep=';')
chiffres_padel_tennis = load_csv(chiffres_path, sep=';')  # utilisé dans plusieurs cellules
geo_file = load_csv(geo_path, sep=',')  # le fichier géographie.csv semble être en CSV classique (,)
communes = load_csv(com_path, sep=';')

# =========================
# Cellule 1 — Préparation & agrégations communes_idf / dep_df
# =========================
if (data_es is not None) and (dvf_idf is not None) and (lic_tennis is not None):
    IDF_DEPS = ['75','77','78','91','92','93','94','95']

    # data_es
    data_es.columns = data_es.columns.str.strip()
    data_es['Département Code'] = data_es['Département Code'].astype(str)
    data_es['Commune INSEE'] = data_es['Commune INSEE'].astype(str).str.zfill(5)
    data_es_idf = data_es[data_es['Département Code'].isin(IDF_DEPS)].copy()

    # dvf
    dvf_idf.columns = dvf_idf.columns.str.strip().str.lower()
    dvf_idf['insee_com'] = dvf_idf['insee_com'].astype(str).str.zfill(5)
    dvf_idf['dep_code'] = dvf_idf['dep_code'].astype(str)

    # licences tennis
    lic_tennis.columns = lic_tennis.columns.str.strip().str.lower()
    lic_tennis['code_insee'] = lic_tennis['code_insee'].astype(str).str.zfill(5)
    if 'dep_code' not in lic_tennis.columns:
        lic_tennis['dep_code'] = lic_tennis['code_insee'].str[:2]
    lic_tennis['dep_code'] = lic_tennis['dep_code'].astype(str)

    # Comptes Padel (communes)
    padel_by_com = (
        data_es_idf[data_es_idf["Type d'équipement sportif"].str.contains("padel", case=False, na=False)]
        .groupby("Commune INSEE").size().rename("nb_padel").reset_index()
    )
    padel_by_com["Commune INSEE"] = padel_by_com["Commune INSEE"].astype(str).str.zfill(5)

    # Licences Tennis (communes)
    lic_by_com = lic_tennis.groupby("code_insee")["total"].sum().rename("licencies").reset_index()

    # Table communes_idf : prix + dep + indicateurs tennis/padel
    communes_idf = (dvf_idf
        .merge(padel_by_com, left_on="insee_com", right_on="Commune INSEE", how="left")
        .merge(lic_by_com, left_on="insee_com", right_on="code_insee", how="left")
        .drop(columns=["Commune INSEE","code_insee"])
        .copy()
    )
    communes_idf["nb_padel"] = communes_idf["nb_padel"].fillna(0).astype(int)
    communes_idf["licencies"] = communes_idf["licencies"].fillna(0).astype(int)

    # Comptes Padel/Tennis (département)
    padel_by_dep  = (data_es_idf[data_es_idf["Type d'équipement sportif"].str.contains("padel", case=False, na=False)]
                     .groupby("Département Code").size().rename("nb_padel_dep"))
    tennis_by_dep = lic_tennis.groupby("dep_code")["total"].sum().rename("licencies_dep")
    # prix pondéré par nb_mutations
    if "nb_mutations" not in dvf_idf.columns:
        dvf_idf["nb_mutations"] = 1
    prix_by_dep = dvf_idf.groupby("dep_code")["prix_m2_moyen"].apply(
        lambda s: np.average(s.dropna(), weights=dvf_idf.loc[s.index, "nb_mutations"].fillna(1))
    ).rename("prix_m2_pond_dep")

    dep_df = (
        pd.DataFrame({"dep_code": IDF_DEPS})
        .merge(padel_by_dep, left_on="dep_code", right_index=True, how="left")
        .merge(tennis_by_dep, left_on="dep_code", right_index=True, how="left")
        .merge(prix_by_dep, left_on="dep_code", right_index=True, how="left")
        .fillna(0)
    )
    dep_df["nb_padel_dep"]  = dep_df["nb_padel_dep"].astype(int)
    dep_df["licencies_dep"] = dep_df["licencies_dep"].astype(int)
else:
    communes_idf = None
    dep_df = None

# =========================
# Cellule 2 — Évolution Licences/Terrains (log à gauche)
# =========================
st.markdown("## 📈 Évolution France — Licences & Terrains (Tennis vs Padel)")

if chiffres_padel_tennis is not None:
    df = chiffres_padel_tennis.copy()

    # 1) Nettoyage colonnes parasites éventuelles ("Unnamed: x")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # 2) Normalisation douce des noms de colonnes (garde tes noms si déjà OK)
    ren = {
        "Année": "Annee",
        "Licences Padel (FR)": "LicencesPadel_FR",
        "Licences Tennis (FR)": "LicenceTennis_FR",
        "Terrains Tennis (FR)": "TerrainTennisFR",
        "Terrains Padel (FR)": "TerrainPadelFR",
    }
    df = df.rename(columns=ren)

    # 3) Conversion robuste des nombres ("1 018 721", "5 000" → float)
    def to_number(s):
        return (
            pd.Series(s, dtype="string")
              .str.replace("\u202f", "", regex=False)  # fine space
              .str.replace(" ", "", regex=False)
              .str.replace(",", ".", regex=False)
              .pipe(pd.to_numeric, errors="coerce")
        )

    need = ["Annee", "LicenceTennis_FR", "LicencesPadel_FR", "TerrainTennisFR", "TerrainPadelFR"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans chiffres_padel_tennis : {missing}")
    else:
        # conversions
        df["Annee"]            = to_number(df["Annee"]).astype("Int64")
        df["LicenceTennis_FR"] = to_number(df["LicenceTennis_FR"])
        df["LicencesPadel_FR"] = to_number(df["LicencesPadel_FR"])
        df["TerrainTennisFR"]  = to_number(df["TerrainTennisFR"])
        df["TerrainPadelFR"]   = to_number(df["TerrainPadelFR"])

        # drop lignes non valides et trier
        df = df.dropna(subset=["Annee"]).copy()
        df["Annee"] = df["Annee"].astype(int)
        df = df.sort_values("Annee")

        x = df["Annee"].values
        y_tennis_lic = df["LicenceTennis_FR"].values
        y_padel_lic  = df["LicencesPadel_FR"].values
        y_tennis_trn = df["TerrainTennisFR"].values
        y_padel_trn  = df["TerrainPadelFR"].values

        yticks = np.array([1e4, 5e4, 5e5, 1e6], dtype=float)
        ytick_labels = ["10 000", "50 000", "500 000", "1 000 000"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)

        # Licences (log)
        ax1.plot(x, y_tennis_lic, marker="o", linewidth=2, label="Licences Tennis (FR)")
        ax1.plot(x, y_padel_lic,  marker="o", linewidth=2, label="Licences Padel (FR)")
        ax1.set_yscale("log")
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(ytick_labels)
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Nombre de licences")
        ax1.set_title("Évolution des licences — Tennis vs Padel")
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        ax1.legend()

        def annotate_line(ax, xvals, yvals):
            for xi, yi in zip(xvals, yvals):
                if pd.notna(yi) and yi > 0:
                    ax.annotate(f"{int(yi):,}".replace(",", " "),
                                xy=(xi, yi), xytext=(0, 6),
                                textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)

        annotate_line(ax1, x, y_tennis_lic)
        annotate_line(ax1, x, y_padel_lic)

        # Terrains (lin)
        ax2.plot(x, y_tennis_trn, marker="o", linewidth=2, label="Terrains Tennis (FR)")
        ax2.plot(x, y_padel_trn,  marker="o", linewidth=2, label="Terrains Padel (FR)")
        ax2.set_xlabel("Année")
        ax2.set_ylabel("Nombre de terrains")
        ax2.set_title("Évolution des terrains — Tennis vs Padel")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()

        for ax in (ax1, ax2):
            ax.set_xticks(x)
            ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("Ajoute 'ChiffresPadelTennis.csv' pour cette section.")


# =========================
# Cellule 3 — Ratios Licences/Terrain + graphe des ratios
# =========================
st.markdown("## 🧮 Ratios — Licences par terrain (Tennis vs Padel)")
if chiffres_padel_tennis is not None:
    df = chiffres_padel_tennis.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.rename(columns={
        "Année": "Annee",
        "Licences Padel (FR)": "LicencesPadel_FR",
        "Licences Tennis (FR)": "LicenceTennis_FR",
        "Terrains Tennis (FR)": "TerrainTennisFR",
        "Terrains Padel (FR)": "TerrainPadelFR",
    })

    def to_number(s):
        return (
            pd.Series(s, dtype="string")
              .str.replace("\u202f", "", regex=False)
              .str.replace(" ", "", regex=False)
              .str.replace(",", ".", regex=False)
              .pipe(pd.to_numeric, errors="coerce")
        )

    need = ["Annee", "LicenceTennis_FR", "LicencesPadel_FR", "TerrainTennisFR", "TerrainPadelFR"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"Colonnes manquantes : {miss}")
    else:
        for c in need:
            df[c] = to_number(df[c])
        df = df.dropna(subset=["Annee"]).copy()
        df["Annee"] = df["Annee"].astype(int)
        df = df.sort_values("Annee")

        # éviter /0
        df["TerrainTennisFR"] = df["TerrainTennisFR"].replace({0: np.nan})
        df["TerrainPadelFR"]  = df["TerrainPadelFR"].replace({0: np.nan})
        df["LicencesParTerrain_Tennis"] = df["LicenceTennis_FR"] / df["TerrainTennisFR"]
        df["LicencesParTerrain_Padel"]  = df["LicencesPadel_FR"]  / df["TerrainPadelFR"]

        # tableau stylé
        table = (
            df[[
                "Annee",
                "LicenceTennis_FR", "TerrainTennisFR", "LicencesParTerrain_Tennis",
                "LicencesPadel_FR",  "TerrainPadelFR",  "LicencesParTerrain_Padel"
            ]]
            .rename(columns={
                "Annee": "Année",
                "LicenceTennis_FR": "Licences Tennis (FR)",
                "TerrainTennisFR": "Terrains Tennis (FR)",
                "LicencesParTerrain_Tennis": "Licences / Terrain (Tennis)",
                "LicencesPadel_FR": "Licences Padel (FR)",
                "TerrainPadelFR": "Terrains Padel (FR)",
                "LicencesParTerrain_Padel": "Licences / Terrain (Padel)"
            })
        )

        fmt_int = lambda x: "" if pd.isna(x) else f"{int(x):,}".replace(",", " ")
        fmt_1d  = lambda x: "" if pd.isna(x) else f"{x:,.1f}".replace(",", " ")

        st.dataframe(
            table.style.format({
                "Licences Tennis (FR)": fmt_int,
                "Terrains Tennis (FR)": fmt_int,
                "Licences / Terrain (Tennis)": fmt_1d,
                "Licences Padel (FR)": fmt_int,
                "Terrains Padel (FR)": fmt_int,
                "Licences / Terrain (Padel)": fmt_1d,
            }).set_caption("Comparatif — Licenciés par terrain (Tennis vs Padel)"),
            use_container_width=True
        )

        # graphe des ratios
        x = df["Annee"].values
        y_tennis_ratio = df["LicencesParTerrain_Tennis"].values
        y_padel_ratio  = df["LicencesParTerrain_Padel"].values

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y_tennis_ratio, marker="o", linewidth=2, label="Licences / Terrain (Tennis)")
        ax.plot(x, y_padel_ratio,  marker="o", linewidth=2, label="Licences / Terrain (Padel)")
        ax.set_xlabel("Année")
        ax.set_ylabel("Licences par terrain")
        ax.set_title("Tennis vs Padel — Licences par terrain")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        fmt1d = lambda v: "" if pd.isna(v) else f"{v:,.1f}".replace(",", " ")
        for xi, yt, yp in zip(x, y_tennis_ratio, y_padel_ratio):
            if pd.notna(yt):
                ax.annotate(fmt1d(yt), (xi, yt), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)
            if pd.notna(yp):
                ax.annotate(fmt1d(yp), (xi, yp), textcoords="offset points", xytext=(0,-10), ha="center", fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

# =========================
# Cellule 4 — Progression Padel_FR jusqu’à l’année choisie (slider + lecture auto)
# =========================
st.markdown("## ▶️ Progression des pratiquants PADEL (FR) — lecture par année")
if chiffres_padel_tennis is not None:
    dfp = chiffres_padel_tennis.copy()
    dfp = dfp.loc[:, ~dfp.columns.str.startswith("Unnamed")]
    dfp = dfp.rename(columns={"Année": "Annee"})
    if {"Annee","Padel_FR"}.issubset(dfp.columns):
        def to_number(s):
            return (
                pd.Series(s, dtype="string")
                  .str.replace("\u202f", "", regex=False)
                  .str.replace(" ", "", regex=False)
                  .str.replace(",", ".", regex=False)
                  .pipe(pd.to_numeric, errors="coerce")
            )

        dfp["Annee"]    = to_number(dfp["Annee"]).dropna().astype(int)
        dfp["Padel_FR"] = to_number(dfp["Padel_FR"])
        dfp = dfp.dropna(subset=["Annee","Padel_FR"]).sort_values("Annee")

        years = dfp["Annee"].tolist()
        if years:
            c1, c2 = st.columns([3,1])
            with c1:
                year = st.slider("Année", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[0]), step=1)
            with c2:
                autoplay = st.checkbox("Lecture auto", value=False)

            def draw_until(cutoff):
                d = dfp[dfp["Annee"] <= cutoff]
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(d["Annee"], d["Padel_FR"], marker="o", linewidth=2, label="PADEL Pratiquants")
                ax.fill_between(d["Annee"], d["Padel_FR"], step="pre", alpha=0.2)
                ax.set_xlabel("Année")
                ax.set_ylabel("Nombre de pratiquants Padel (FR)")
                ax.set_title(f"PADEL — progression des pratiquants jusqu’à {cutoff}")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend()
                if len(d) > 0:
                    ax.annotate(f"{int(d['Padel_FR'].iloc[-1]):,}".replace(",", " "),
                                xy=(d["Annee"].iloc[-1], d["Padel_FR"].iloc[-1]),
                                xytext=(0, 8), textcoords="offset points",
                                ha="center", va="bottom", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

            if autoplay:
                for y in years:
                    draw_until(y)
                    time.sleep(0.6)
                    if y != years[-1]:
                        st.experimental_rerun()
            else:
                draw_until(year)
    else:
        st.info("Colonnes 'Annee' et/ou 'Padel_FR' manquantes dans le CSV.")

# =========================
# Cellule 5 — FR métropolitaine : Heatmap Padel/Tennis (Folium)
# =========================
st.markdown("## 🗺️ France — Heatmap (Padel/Tennis)")
if geo_file is not None:
    geo = geo_file.copy()
    geo.columns = geo.columns.str.strip().str.lower()
    if "sport" in geo.columns:
        geo["sport"] = geo["sport"].str.lower().replace("paddle","padel")
    for c in ["latitude","longitude"]:
        if c in geo.columns:
            geo[c] = pd.to_numeric(geo[c], errors="coerce")
    geo = geo.dropna(subset=["latitude","longitude"])

    # bbox France métro
    FR_BBOX = (40.0, -6.5, 52.5, 11.0)
    geo = geo[
        (geo["latitude"].between(FR_BBOX[0], FR_BBOX[2])) &
        (geo["longitude"].between(FR_BBOX[1], FR_BBOX[3]))
    ].copy()

    sport = st.selectbox("Sport", [("Padel","padel"), ("Tennis","tennis")], index=0, format_func=lambda x: x[0])[1]

    def make_heatmap(sport: str):
        df = geo[geo["sport"]==sport].copy() if "sport" in geo.columns else geo.copy()
        if df.empty:
            m = folium.Map(location=[46.5, 2.4], zoom_start=6, tiles="CartoDB positron")
            folium.Marker([46.5, 2.4], tooltip="Aucun point trouvé").add_to(m)
            return m
        lat_c, lon_c = df["latitude"].median(), df["longitude"].median()
        m = folium.Map(location=[lat_c, lon_c], zoom_start=6, tiles="CartoDB positron")
        weight = 5 if sport == "padel" else 1
        radius = 18 if sport == "padel" else 10
        hm_data = [[r.latitude, r.longitude, weight] for r in df.itertuples()]
        HeatMap(hm_data, radius=radius, blur=int(radius*1.2), min_opacity=0.25, max_zoom=10).add_to(m)
        m.fit_bounds([[FR_BBOX[0], FR_BBOX[1]], [FR_BBOX[2], FR_BBOX[3]]])
        return m

    m = make_heatmap(sport)
    st_folium(m, width=900, height=600)

# =========================
# Cellule 6 — Île-de-France : points / heatmap (Folium)
# =========================
st.markdown("## 🗺️ Île-de-France — Points / Heatmap (Padel/Tennis)")
if geo_file is not None:
    geo = geo_file.copy()
    geo.columns = geo.columns.str.strip().str.lower()
    if "sport" in geo.columns:
        geo["sport"] = geo["sport"].str.lower().replace("paddle","padel")
    for c in ["latitude","longitude"]:
        if c in geo.columns:
            geo[c] = pd.to_numeric(geo[c], errors="coerce")
    geo = geo.dropna(subset=["latitude","longitude"])

    IDF_BBOX = (48.15, 1.45, 49.23, 3.50)
    geo_idf = geo[
        geo["latitude"].between(IDF_BBOX[0], IDF_BBOX[2])
        & geo["longitude"].between(IDF_BBOX[1], IDF_BBOX[3])
    ].copy()

    col1, col2 = st.columns(2)
    with col1:
        sport_idf = st.selectbox("Sport (IDF)", [("Padel","padel"), ("Tennis","tennis")], index=0, format_func=lambda x: x[0])[1]
    with col2:
        vue = st.selectbox("Vue", [("Points","points"), ("Heatmap","heatmap")], index=0, format_func=lambda x: x[0])[1]

    def make_map_idf(sport: str, vue: str):
        df = geo_idf[geo_idf["sport"]==sport].copy() if "sport" in geo_idf.columns else geo_idf.copy()
        m = folium.Map(location=[48.86, 2.35], zoom_start=9, tiles="CartoDB positron")
        if df.empty:
            folium.Marker([48.86, 2.35], tooltip="Aucun point pour ce filtre").add_to(m)
            return m
        if vue == "points":
            radius = 6 if sport == "padel" else 3
            color  = "#d73027" if sport == "padel" else "#2b8cbe"
            cluster = MarkerCluster(name=f"{sport.capitalize()}").add_to(m)
            for r in df.itertuples():
                folium.CircleMarker(
                    location=[r.latitude, r.longitude],
                    radius=radius, color=color, fill=True, fill_opacity=0.85, weight=0.6
                ).add_to(cluster)
        else:
            weight = 5 if sport == "padel" else 1
            radius = 16 if sport == "padel" else 10
            hm_data = [[r.latitude, r.longitude, weight] for r in df.itertuples()]
            HeatMap(hm_data, radius=radius, blur=int(radius*1.2), min_opacity=0.25).add_to(m)
        m.fit_bounds([[IDF_BBOX[0], IDF_BBOX[1]], [IDF_BBOX[2], IDF_BBOX[3]]])
        folium.LayerControl(collapsed=True).add_to(m)
        return m

    m_idf = make_map_idf(sport_idf, vue)
    st_folium(m_idf, width=900, height=600)

# =========================
# Cellule 7 — Barres nb points Padel/Tennis par département (via jointure communes)
# =========================
st.markdown("## 🧱 IDF — Nombre de points par département (Padel/Tennis)")
def normalize_name(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    s = s.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8"))
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

if (geo_file is not None) and (communes is not None):
    IDF_DEPS = ["75","77","78","91","92","93","94","95"]
    geo = geo_file.copy()
    geo.columns = geo.columns.str.strip().str.lower()
    communes = communes.copy()
    communes.columns = communes.columns.str.strip().str.lower()
    if "sport" in geo.columns:
        geo["sport"] = geo["sport"].str.lower().replace("paddle","padel")
    for c in ["latitude","longitude"]:
        if c in geo.columns:
            geo[c] = pd.to_numeric(geo[c], errors="coerce")

    geo["commune_norm"] = normalize_name(geo["commune"])
    geo["commune_norm"] = geo["commune_norm"].str.replace(r"^paris\s*\d{1,2}(er|e)?(\s*arrondissement)?$", "paris", regex=True)
    if "nom_standard" in communes.columns:
        communes["commune_norm"] = normalize_name(communes["nom_standard"])
    else:
        # fallback: "commune" si pas de nom_standard
        communes["commune_norm"] = normalize_name(communes.get("commune", pd.Series(dtype=str)))

    geo_dep = geo.merge(communes[["commune_norm", "dep_code"]], on="commune_norm", how="left")
    geo_dep = geo_dep[geo_dep["dep_code"].astype(str).isin(IDF_DEPS)].copy()

    geo_dep["lat_r"] = geo_dep["latitude"].round(4)
    geo_dep["lon_r"] = geo_dep["longitude"].round(4)
    geo_dep = geo_dep.drop_duplicates(subset=["dep_code","sport","lat_r","lon_r"])

    padel_counts  = (geo_dep[geo_dep["sport"]=="padel"].groupby("dep_code").size().reindex(IDF_DEPS, fill_value=0))
    tennis_counts = (geo_dep[geo_dep["sport"]=="tennis"].groupby("dep_code").size().reindex(IDF_DEPS, fill_value=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)

    axes[0].bar(IDF_DEPS, padel_counts.values, alpha=0.85)
    axes[0].set_title("Nombre de points PADEL par département (IDF)")
    axes[0].set_xlabel("Département")
    axes[0].set_ylabel("Nombre de points")
    for x, v in zip(IDF_DEPS, padel_counts.values):
        axes[0].text(x, v + max(1, v*0.02), str(int(v)), ha="center", va="bottom", fontsize=9)

    axes[1].bar(IDF_DEPS, tennis_counts.values, alpha=0.85)
    axes[1].set_title("Nombre de points TENNIS par département (IDF)")
    axes[1].set_xlabel("Département")
    for x, v in zip(IDF_DEPS, tennis_counts.values):
        axes[1].text(x, v + max(1, v*0.02), str(int(v)), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

# =========================
# Cellule 8 — Scatter opportunité départementale (nb_tennis vs score)
# =========================
st.markdown("## 🧭 IDF — Opportunités départementales (Tennis vs Padel)")
if (geo_file is not None) and (communes is not None):
    IDF_DEPS = ["75","77","78","91","92","93","94","95"]
    DEP_LABELS = {"75":"Paris","77":"Seine-et-Marne","78":"Yvelines","91":"Essonne",
                  "92":"Hauts-de-Seine","93":"Seine-Saint-Denis","94":"Val-de-Marne","95":"Val-d'Oise"}

    geo = geo_file.copy()
    geo.columns = geo.columns.str.strip().str.lower()
    communes2 = communes.copy()
    communes2.columns = communes2.columns.str.strip().str.lower()
    if "sport" in geo.columns:
        geo["sport"] = geo["sport"].str.lower().replace("paddle","padel")
    for c in ["latitude","longitude"]:
        if c in geo.columns:
            geo[c] = pd.to_numeric(geo[c], errors="coerce")

    def normalize_name2(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip().str.lower()
        s = s.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii","ignore").decode("utf-8"))
        return s.str.replace(r"\s+", " ", regex=True)

    geo["commune_norm"] = normalize_name2(geo["commune"])
    geo["commune_norm"] = geo["commune_norm"].str.replace(r"^paris\s*\d{1,2}(er|e)?(\s*arrondissement)?$", "paris", regex=True)
    if "nom_standard" in communes2.columns:
        communes2["commune_norm"] = normalize_name2(communes2["nom_standard"])
    else:
        communes2["commune_norm"] = normalize_name2(communes2.get("commune", pd.Series(dtype=str)))

    geo_dep = geo.merge(communes2[["commune_norm","dep_code"]], on="commune_norm", how="left")
    geo_dep = geo_dep[geo_dep["dep_code"].astype(str).isin(IDF_DEPS)].copy()
    geo_dep["lat_r"] = geo_dep["latitude"].round(4)
    geo_dep["lon_r"] = geo_dep["longitude"].round(4)
    geo_dep = geo_dep.drop_duplicates(subset=["dep_code","sport","lat_r","lon_r"])

    padel_counts  = geo_dep[geo_dep["sport"]=="padel"].groupby("dep_code").size()
    tennis_counts = geo_dep[geo_dep["sport"]=="tennis"].groupby("dep_code").size()

    df = pd.DataFrame({"dep_code": IDF_DEPS})
    df["nb_padel"]  = df["dep_code"].map(padel_counts).fillna(0).astype(int)
    df["nb_tennis"] = df["dep_code"].map(tennis_counts).fillna(0).astype(int)
    df["dep_nom"]   = df["dep_code"].map(DEP_LABELS)

    df["score_opportunite"] = df["nb_tennis"] / (df["nb_padel"] + 1)

    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.scatter(df["nb_tennis"], df["score_opportunite"], s=110, alpha=0.9, edgecolors="k", linewidths=0.5)
    for _, r in df.iterrows():
        ax.annotate(r["dep_code"], (r["nb_tennis"], r["score_opportunite"]),
                    textcoords="offset points", xytext=(0,7), ha="center", fontsize=9)
    ax.set_xlabel("Nombre de points TENNIS (par département)")
    ax.set_ylabel("Score d'opportunité = TENNIS / (PADEL + 1)")
    ax.set_title("Île-de-France — Opportunités départementales (Tennis vs Padel)")
    ax.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# Cellule 9 — Barh “départements sous-équipés en padel” (+ exclure Paris)
# =========================
st.markdown("## 📉 IDF — Départements sous-équipés en Padel")
if (data_es is not None) and (lic_tennis is not None):
    IDF_DEPS = ['75','77','78','91','92','93','94','95']
    data_es2 = data_es.copy()
    data_es2.columns = data_es2.columns.str.strip()
    data_es2['Département Code'] = data_es2['Département Code'].astype(str)
    data_es_idf = data_es2[data_es2['Département Code'].isin(IDF_DEPS)].copy()

    lic2 = lic_tennis.copy()
    lic2.columns = lic2.columns.str.strip().str.lower()
    lic2['code_insee'] = lic2['code_insee'].astype(str).str.zfill(5)
    if 'dep_code' not in lic2.columns:
        lic2['dep_code'] = lic2['code_insee'].str[:2]
    lic2['dep_code'] = lic2['dep_code'].astype(str)

    padel_by_dep = (data_es_idf[data_es_idf["Type d'équipement sportif"].str.contains("padel", case=False, na=False)]
                    .groupby("Département Code").size().rename("nb_padel"))
    tennis_by_dep = lic2.groupby("dep_code")["total"].sum().rename("lic_tennis")

    df_gap = (pd.DataFrame({"dep_code": IDF_DEPS})
          .merge(padel_by_dep, left_on="dep_code", right_index=True, how="left")
          .merge(tennis_by_dep, left_on="dep_code", right_index=True, how="left")
          .fillna(0))
    df_gap["nb_padel"] = df_gap["nb_padel"].astype(int)
    df_gap["lic_tennis"] = df_gap["lic_tennis"].astype(int)
    df_gap["gap"] = df_gap["lic_tennis"] / (df_gap["nb_padel"] + 1)

    excl_75 = st.checkbox("Exclure Paris (75)", value=True)
    dplot = df_gap.copy()
    if excl_75:
        dplot = dplot[dplot["dep_code"]!="75"]
    dplot = dplot.sort_values("gap", ascending=True)

    fig, ax = plt.subplots(figsize=(8,5.2))
    ax.barh(dplot["dep_code"], dplot["gap"], alpha=0.9)
    for _, r in dplot.iterrows():
        ax.text(r["gap"], r["dep_code"], f"{r['gap']:.1f}", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Sous-équipement = Licenciés Tennis / (Padel + 1)")
    ax.set_title("IDF — Départements les plus sous-équipés en padel")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# Cellule 10 — Histogramme des prix DVF par département (sélecteur)
# =========================
st.markdown("## 💶 DVF — Distribution des prix au m² par département (IDF)")
if dvf_idf is not None:
    dvf2 = dvf_idf.copy()
    dvf2.columns = dvf2.columns.str.strip().str.lower()
    dvf2["dep_code"] = dvf2["dep_code"].astype(str)
    IDF_DEPS = ['75','77','78','91','92','93','94','95']
    dep = st.selectbox("Département", IDF_DEPS, index=1)  # "77" par défaut
    s = dvf2.loc[dvf2["dep_code"]==dep, "prix_m2_moyen"].dropna()
    if s.empty:
        st.warning("Pas de données de prix pour ce département.")
    else:
        med = s.median()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(s, bins=20, alpha=0.85)
        ax.axvline(med, color="red", ls="--", label=f"Médiane = {med:.0f} € / m²")
        ax.set_xlabel("Prix au m² (€)")
        ax.set_ylabel("Nombre de communes")
        ax.set_title(f"Distribution des prix au m² — Département {dep}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

# =========================
# Cellule 11 — Comparaison départements IDF (critère: prix pondéré OU score)
# =========================
st.markdown("## 🏆 Comparaison départements IDF — prix pondéré / score global")
if dep_df is not None:
    dfc = dep_df.copy()
    # Score global du notebook
    dfc["score"] = (dfc["licencies_dep"] / (dfc["nb_padel_dep"] + 1)) * (10000 / dfc["prix_m2_pond_dep"])
    crit_label = {
        "prix_m2_pond_dep": "Prix moyen du m² (pondéré)",
        "score": "Score global d'opportunité",
    }
    crit = st.selectbox("Critère", list(crit_label.keys()), index=0, format_func=lambda k: crit_label[k])
    order_asc = True if crit == "prix_m2_pond_dep" else False
    top = dfc.sort_values(crit, ascending=order_asc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["dep_code"], top[crit], alpha=0.85)
    ax.set_xlabel(crit_label[crit]); ax.set_ylabel("Département")
    ax.set_title(f"Comparaison des départements IDF — {crit_label[crit]}")
    ax.invert_yaxis(); ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**📊 Indicateurs par département :**")
    for r in top.itertuples():
        st.write(f"- Dépt {r.dep_code} : prix_m²={getattr(r,'prix_m2_pond_dep'):.0f}€ | score={r.score:.1f}")

# =========================
# Cellule 12 — Top communes pour implanter un complexe (92, 77 ou combiné)
# =========================
st.markdown("## 🏗️ Top communes pour implanter un complexe Padel — 92 / 77 / combiné")
if communes_idf is not None and data_es is not None:
    OPTIONS = {
        "92 — Hauts-de-Seine": ["92"],
        "77 — Seine-et-Marne": ["77"],
        "92 + 77 (combinés)": ["92", "77"]
    }
    choix = st.selectbox("Zone", list(OPTIONS.keys()), index=2)
    prix_max = st.slider("Prix m² max", min_value=3000, max_value=15000, step=500, value=12000)

    df = communes_idf.copy()
    dep_codes = OPTIONS[choix]
    df = df[df["dep_code"].isin(dep_codes)]
    df = df[df["prix_m2_moyen"] <= prix_max].copy()

    if df.empty:
        st.warning("⚠ Aucune commune ne correspond aux critères.")
    else:
        df["score"] = (df["licencies"] / (df["nb_padel"] + 1)) * (10000 / df["prix_m2_moyen"])
        top = df.nlargest(min(20, len(df)), "score").copy()

        # Nom de commune (si dispo dans data_es_idf)
        data_es_idf_local = data_es.copy()
        data_es_idf_local['Département Code'] = data_es_idf_local['Département Code'].astype(str)
        data_es_idf_local = data_es_idf_local[data_es_idf_local['Département Code'].isin(['75','77','78','91','92','93','94','95'])].copy()
        nom_map = (
            data_es_idf_local[["Commune INSEE", "Commune nom"]]
            .drop_duplicates()
        )
        nom_map["Commune INSEE"] = nom_map["Commune INSEE"].astype(str).str.zfill(5)
        top = top.merge(nom_map, left_on="insee_com", right_on="Commune INSEE", how="left")
        top["label"] = top["Commune nom"].fillna(top["insee_com"])

        # Couleurs selon nb padel existants
        colors = ["green" if x == 0 else "orange" if x < 3 else "red" for x in top["nb_padel"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["label"], top["score"], color=colors, alpha=0.85)
        ax.set_xlabel("Score d'opportunité")
        ax.set_title(f"Top communes pour implanter un complexe Padel — {choix}")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"### 🥇 Top {min(5, len(top))} — {choix}")
        for i, r in enumerate(top.head(5).itertuples(), 1):
            st.write(
                f"{i}. {getattr(r,'label')} (dep {r.dep_code}) — "
                f"Prix/m²={r.prix_m2_moyen:.0f}€, Lic.={r.licencies}, "
                f"Padel={r.nb_padel}, Score={r.score:.1f}"
            )
