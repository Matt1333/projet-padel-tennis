import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

st.set_page_config(page_title="Nettoyage des données", page_icon="🧹", layout="wide")

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True, parents=True)

IDF_DEPS = {"75","77","78","91","92","93","94","95"}

st.title("🧹 Pipeline de nettoyage – Export des CSV propres")
st.write("Cette app rejoue les grandes étapes du notebook **Nettoyage.ipynb** et génère des CSV *clean* pour l’app projet.")

with st.sidebar:
    st.header("📂 Fichiers d'entrée (data/)")
    data_es_path = st.text_input("Fichier équipements sportifs (brut)", str(DATA_DIR / "data-es.csv"))
    communes_path = st.text_input("Fichier communes France 2025 (brut)", str(DATA_DIR / "communes-france-2025.csv"))
    lic_path = st.text_input("Fichier licences 2022 (brut)", str(DATA_DIR / "lic-data-2022.csv"))
    rna_path = st.text_input("Fichier associations RNA (optionnel)", str(DATA_DIR / "rna_idf.csv"))

    st.header("📁 Dossier de sortie")
    out_dir_str = st.text_input("Dossier output", str(OUT_DIR))
    if out_dir_str:
        OUT_DIR = Path(out_dir_str)
        OUT_DIR.mkdir(exist_ok=True, parents=True)

st.markdown("---")

def clean_data_es(src, out):
    """Nettoyage du fichier équipements sportifs -> data_es_clean.csv"""
    st.subheader("1) Nettoyage des équipements sportifs")
    st.write(f"Lecture : **{src}**")
    chunksize = 200_000
    cols_keep = [
        "Numéro de l'équipement sportif", "Numéro de l'installation sportive",
        "Nom de l'équipement sportif", "Type d'équipement sportif",
        "Commune nom", "Commune INSEE", "Département Code",
        "Latitude", "Longitude"
    ]
    total_kept = 0
    first = True
    try:
        for df in pd.read_csv(src, sep=";", chunksize=chunksize, low_memory=False):
            df = df[[c for c in cols_keep if c in df.columns]].copy()
            # dédoublonnage et géos valides
            subset = [c for c in ["Numéro de l'équipement sportif","Type d'équipement sportif","Commune nom","Latitude","Longitude"] if c in df.columns]
            if subset:
                df = df.drop_duplicates(subset=subset)
            if {"Latitude","Longitude"}.issubset(df.columns):
                df = df.dropna(subset=["Latitude","Longitude"])
            df.to_csv(out, sep=";", index=False, mode="w" if first else "a", header=first)
            first = False
            total_kept += len(df)
        st.success(f"✅ Export : {out} — {total_kept:,} lignes")
        head = pd.read_csv(out, sep=";", nrows=5, low_memory=False)
        st.dataframe(head)
    except Exception as e:
        st.error(f"Erreur nettoyage équipements : {e}")

def clean_communes(src, out_all, out_idf):
    """Nettoyage des communes -> communes_france_clean.csv + communes_france_idf.csv"""
    st.subheader("2) Communes de France (codes INSEE, geo, pop)")
    st.write(f"Lecture : **{src}**")
    try:
        df = pd.read_csv(src, sep=";", low_memory=False)
        # Harmonisation colonnes possibles (adaptation minimale)
        rename = {
            "code_departement":"dep_code",
            "code_insee":"insee_com",
            "nom_commune":"commune",
            "longitude":"lon",
            "latitude":"lat",
            "population":"pop",
            "densite":"densite"
        }
        for k,v in list(rename.items()):
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k:v})

        df.to_csv(out_all, sep=";", index=False)
        idf = df[df.get("dep_code","").astype(str).isin(IDF_DEPS)].copy()
        idf.to_csv(out_idf, sep=";", index=False)

        st.success(f"✅ Export national : {out_all} — {len(df):,} lignes")
        st.success(f"✅ Export IDF     : {out_idf} — {len(idf):,} lignes")
        st.dataframe(idf.head())
    except Exception as e:
        st.error(f"Erreur nettoyage communes : {e}")

def clean_licences(src, out_all, out_tennis_idf):
    """Licences sportives 2022 -> lic_2022_clean.csv + lic_2022_tennis_idf.csv"""
    st.subheader("3) Licences sportives 2022")
    st.write(f"Lecture : **{src}**")
    try:
        df = pd.read_csv(src, sep=";", quotechar='"', low_memory=False)
        rename_map = {
            "Code Commune": "code_insee",
            "Commune": "commune",
            "Département": "dep_code",
            "Région": "region_nom",
            "Fédération": "federation",
            "Total": "total"
        }
        df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
        df["dep_code"] = df["dep_code"].astype(str).str.zfill(2)
        df.to_csv(out_all, sep=";", index=False)

        tennis = df[df["federation"].str.contains("tennis", case=False, na=False)].copy() if "federation" in df.columns else df.iloc[0:0].copy()
        idf   = tennis[tennis["dep_code"].isin(IDF_DEPS)].copy() if "dep_code" in tennis.columns else tennis
        idf.to_csv(out_tennis_idf, sep=";", index=False)

        st.success(f"✅ Export licences (all) : {out_all} — {len(df):,}")
        st.success(f"✅ Export tennis IDF     : {out_tennis_idf} — {len(idf):,}")
        st.dataframe(idf.head())
    except Exception as e:
        st.error(f"Erreur nettoyage licences : {e}")

def clean_rna(src, out):
    """Associations RNA IDF -> rna_idf_clean.csv (si le fichier est fourni)"""
    st.subheader("4) Associations (RNA) – optionnel")
    if not Path(src).exists():
        st.info("Aucun fichier RNA fourni — étape ignorée.")
        return
    st.write(f"Lecture : **{src}**")
    try:
        df = pd.read_csv(src, sep=";", low_memory=False)
        # Exemple de parsing coords si présent dans une colonne 'coordonnees' / 'adresse'
        for candidate in ["coordonnees","adresse","geo","position"]:
            if candidate in df.columns:
                lon = df[candidate].str.extract(r"[-+]?\d+\.\d+").astype(float)
                lat = df[candidate].str.extract(r".*?([-+]?\d+\.\d+)$").astype(float)
                df["longitude"] = pd.to_numeric(lon[0], errors="coerce")
                df["latitude"]  = pd.to_numeric(lat[0], errors="coerce")
                break

        # Compteurs thématiques si champ 'objet'
        if "objet" in df.columns:
            rna_sport  = df["objet"].str.contains("sport",  case=False, na=False)
            rna_tennis = df["objet"].str.contains("tennis", case=False, na=False)
            rna_padel  = df["objet"].str.contains("padel",  case=False, na=False)
            st.write(f"Associations sportives IDF : {int(rna_sport.sum()):,}")
            st.write(f"Tennis : {int(rna_tennis.sum()):,} — Padel : {int(rna_padel.sum()):,}")

        df.to_csv(out, sep=";", index=False)
        st.success(f"✅ Export : {out} — {len(df):,} lignes")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Erreur nettoyage RNA : {e}")

# --- UI d’exécution ---
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Lancer le pipeline de nettoyage"):
        clean_data_es(data_es_path, OUT_DIR / "data_es_clean.csv")
        clean_communes(communes_path, OUT_DIR / "communes_france_clean.csv", OUT_DIR / "communes_france_idf.csv")
        clean_licences(lic_path, OUT_DIR / "lic_2022_clean.csv", OUT_DIR / "lic_2022_tennis_idf.csv")
        clean_rna(rna_path, OUT_DIR / "rna_idf_clean.csv")

with col2:
    st.info("💾 Les fichiers nettoyés seront disponibles dans le dossier **outputs/**.\n"
            "Tu pourras ensuite les utiliser dans l’app **app_projet.py**.")
