# app.py
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Churn Telco", layout="wide")
st.title("Prediksi Churn Pelanggan Telekomunikasi")
st.caption("Upload CSV atau input manual (dropdown/slider) + autofill dari baris dataset + tampil label asli dataset (jika ada)")

MODEL_PATH = "best_churn_model.joblib"
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# =========================
# Load model & data
# =========================
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_data
def load_data_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalisasi TotalCharges (kadang kebaca string/spasi)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

expected_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None

# =========================
# Opsi kategori sesuai Telco churn dataset
# =========================
YES_NO = ["Yes", "No"]

OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

NUMERIC_DEFAULTS = {
    "tenure": (0, 72, 12),               # min, max, default
    "MonthlyCharges": (0.0, 200.0, 70.0),
    "TotalCharges": (0.0, 10000.0, 1000.0),
}

telco_cols = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling",
    "PaymentMethod","MonthlyCharges","TotalCharges"
]

needed = expected_cols if expected_cols is not None else telco_cols

# =========================
# Load dataset raw (untuk label asli) + fitur (untuk autofill/prediksi)
# =========================
try:
    df_telco_raw = load_data_raw(DATA_PATH)

    # label asli untuk pembanding (jika ada)
    if "Churn" in df_telco_raw.columns:
        y_true_all = df_telco_raw["Churn"].map({"Yes": 1, "No": 0})
    else:
        y_true_all = None

    df_telco = df_telco_raw.copy()
    if "Churn" in df_telco.columns:
        df_telco = df_telco.drop(columns=["Churn"])
    if "customerID" in df_telco.columns:
        df_telco = df_telco.drop(columns=["customerID"])

    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in df_telco.columns]
        if missing:
            st.error(f"Dataset untuk autofill tidak sesuai kolom model. Kolom hilang: {missing}")
            st.stop()
        df_telco = df_telco[expected_cols]
    else:
        miss_std = [c for c in telco_cols if c not in df_telco.columns]
        if miss_std:
            st.error(f"Dataset untuk autofill kurang kolom standar Telco: {miss_std}")
            st.stop()
        df_telco = df_telco[telco_cols]

except Exception as e:
    st.error(
        f"Gagal load dataset dari '{DATA_PATH}'. Pastikan file CSV ada di folder yang sama dengan app.py.\n\n"
        f"Detail: {e}"
    )
    st.stop()

# =========================
# Helper: sync session state from a selected dataset row
# =========================
def apply_row_to_session(row: pd.Series, columns: list[str]):
    """Simpan nilai dari 1 baris dataset ke st.session_state agar form terisi otomatis."""
    for c in columns:
        if c not in row.index:
            continue

        v = row[c]

        if pd.isna(v):
            if c in OPTIONS:
                st.session_state[c] = OPTIONS[c][0]
            elif c in NUMERIC_DEFAULTS:
                st.session_state[c] = NUMERIC_DEFAULTS[c][2]
            else:
                st.session_state[c] = ""
            continue

        if c in OPTIONS:
            # SeniorCitizen harus int
            if c == "SeniorCitizen":
                try:
                    v = int(float(v))
                except Exception:
                    v = 0
                st.session_state[c] = v if v in OPTIONS[c] else OPTIONS[c][0]
            else:
                v = str(v)
                st.session_state[c] = v if v in OPTIONS[c] else OPTIONS[c][0]
        else:
            st.session_state[c] = v

    if st.session_state.get("PhoneService") == "No":
        st.session_state["MultipleLines"] = "No phone service"

    if st.session_state.get("InternetService") == "No":
        for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]:
            st.session_state[c] = "No internet service"

# =========================
# UI Tabs
# =========================
tab1, tab2 = st.tabs(["Upload CSV", "Input Manual (Pilihan + Autofill)"])

# =========================
# TAB 1: Upload CSV
# =========================
with tab1:
    st.subheader("Prediksi dari CSV")
    st.write("Upload CSV tanpa kolom target `Churn`. Jika ada `customerID`, akan otomatis dibuang. Jika `Churn` ada, akan ditampilkan sebagai label asli (pembanding).")

    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded is not None:
        df_in_raw = pd.read_csv(uploaded)

        # ambil label asli kalau ada (untuk pembanding)
        if "Churn" in df_in_raw.columns:
            y_true_csv = df_in_raw["Churn"].map({"Yes": 1, "No": 0})
        else:
            y_true_csv = None

        df_in = df_in_raw.copy()
        if "Churn" in df_in.columns:
            df_in = df_in.drop(columns=["Churn"])
        if "customerID" in df_in.columns:
            df_in = df_in.drop(columns=["customerID"])
        if "TotalCharges" in df_in.columns:
            df_in["TotalCharges"] = pd.to_numeric(df_in["TotalCharges"], errors="coerce")

        if expected_cols is not None:
            missing = [c for c in expected_cols if c not in df_in.columns]
            extra = [c for c in df_in.columns if c not in expected_cols]
            if missing:
                st.error(f"Kolom CSV kurang: {missing}")
                st.stop()
            if extra:
                st.warning(f"Ada kolom ekstra, akan diabaikan: {extra}")
            df_in = df_in[expected_cols]
        else:
            # fallback ke kolom standar kalau model tidak punya feature_names_in_
            missing_std = [c for c in telco_cols if c not in df_in.columns]
            if missing_std:
                st.error(f"Kolom CSV kurang (standar Telco): {missing_std}")
                st.stop()
            df_in = df_in[telco_cols]

        st.write("Preview data input:")
        st.dataframe(df_in.head(20), use_container_width=True)

        if st.button("Prediksi (CSV)"):
            try:
                # prediksi model (kelas & probabilitas)
                proba = model.predict_proba(df_in)[:, 1]
                pred = model.predict(df_in)

                out = df_in.copy()
                out["Prediksi_Churn"] = pred
                out["Label_Prediksi"] = out["Prediksi_Churn"].map({0: "Tidak Churn", 1: "Churn"})
                out["Probabilitas_Churn"] = proba

                # kalau ada label asli, tampilkan sebagai pembanding
                if y_true_csv is not None:
                    out["Label_Asli"] = y_true_csv.values
                    out["Label_Asli_Teks"] = out["Label_Asli"].map({0: "Tidak Churn", 1: "Churn"})
                    out["Cocok_Label_Asli"] = (out["Prediksi_Churn"].values == out["Label_Asli"].values)

                st.success("Selesai!")
                st.dataframe(out, use_container_width=True)

                st.download_button(
                    "Download hasil prediksi (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="hasil_prediksi_churn.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Gagal melakukan prediksi. Detail error: {e}")

# =========================
# TAB 2: Input Manual + Autofill + tampil label asli
# =========================
with tab2:
    st.subheader("Input Manual (1 pelanggan) — Dropdown/Slider + Autofill")

    n = len(df_telco)

    if "selected_no" not in st.session_state:
        st.session_state["selected_no"] = 1

    c_pick, c_prev, c_next, c_fill = st.columns([3, 1, 1, 2])

    with c_pick:
        selected_no = st.number_input(
            "Pilih nomor data (1 = baris pertama)",
            min_value=1,
            max_value=n,
            value=int(st.session_state["selected_no"]),
            step=1,
        )
        st.session_state["selected_no"] = int(selected_no)

    with c_prev:
        st.write("")
        st.write("")
        if st.button("Prev"):
            st.session_state["selected_no"] = max(1, st.session_state["selected_no"] - 1)

    with c_next:
        st.write("")
        st.write("")
        if st.button("Next"):
            st.session_state["selected_no"] = min(n, st.session_state["selected_no"] + 1)

    with c_fill:
        st.write("")
        st.write("")
        if st.button("Isi Otomatis"):
            row0 = df_telco.iloc[st.session_state["selected_no"] - 1]
            apply_row_to_session(row0, needed)
            st.success(f"Form terisi dari dataset nomor {st.session_state['selected_no']}")

    # tampilkan label asli dataset sebagai info pembanding (jika tersedia)
    if y_true_all is not None:
        true_val = int(y_true_all.iloc[st.session_state["selected_no"] - 1])
        true_label = "Churn" if true_val == 1 else "Tidak Churn"
        st.info(f"Label asli dataset (baris ini): **{true_label}**")
    else:
        st.warning("Kolom 'Churn' tidak ditemukan di dataset, jadi label asli tidak bisa ditampilkan.")

    st.write("Form ini dikunci agar konsisten dengan aturan dataset Telco.")

    with st.form("manual_form"):
        c1, c2, c3 = st.columns(3)
        values = {}

        def input_field(colname: str, container):
            if colname in OPTIONS:
                opts = OPTIONS[colname]
                default_val = st.session_state.get(colname, opts[0])

                # SeniorCitizen harus int
                if colname == "SeniorCitizen":
                    try:
                        default_val = int(float(default_val))
                    except Exception:
                        default_val = 0

                if default_val not in opts:
                    default_val = opts[0]
                default_idx = opts.index(default_val)
                return container.selectbox(colname, opts, index=default_idx, key=colname)

            if colname in NUMERIC_DEFAULTS:
                mn, mx, dfv = NUMERIC_DEFAULTS[colname]
                current = st.session_state.get(colname, dfv)

                if isinstance(mn, int) and isinstance(mx, int):
                    try:
                        current = int(float(current))
                    except Exception:
                        current = int(dfv)
                    return container.slider(colname, mn, mx, current, key=colname)

                try:
                    current = float(current)
                except Exception:
                    current = float(dfv)

                return container.number_input(
                    colname,
                    min_value=float(mn),
                    max_value=float(mx),
                    value=current,
                    step=0.1,
                    key=colname,
                )

            default_text = st.session_state.get(colname, "")
            return container.text_input(colname, value=str(default_text), key=colname)

        for i, colname in enumerate(needed):
            container = c1 if i % 3 == 0 else (c2 if i % 3 == 1 else c3)
            values[colname] = input_field(colname, container)

        submitted = st.form_submit_button("Prediksi (Manual)")

    if submitted:
        if values.get("PhoneService") == "No":
            values["MultipleLines"] = "No phone service"

        if values.get("InternetService") == "No":
            for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]:
                values[c] = "No internet service"

        df_one = pd.DataFrame([values])

        for num_col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
            if num_col in df_one.columns:
                df_one[num_col] = pd.to_numeric(df_one[num_col], errors="coerce")

        if expected_cols is not None:
            missing = [c for c in expected_cols if c not in df_one.columns]
            if missing:
                st.error(f"Input manual kurang kolom: {missing}")
                st.stop()
            df_one = df_one[expected_cols]
        else:
            df_one = df_one[telco_cols]

        try:
            proba = float(model.predict_proba(df_one)[0][1])
            pred = int(model.predict(df_one)[0])

            label = "Churn" if pred == 1 else "Tidak Churn"
            st.success(f"Hasil prediksi model: **{label}**")
            st.write(f"Probabilitas churn (model): **{proba:.4f}**")

            # pembanding label asli dataset untuk baris terpilih (jika ada)
            if y_true_all is not None:
                true_val = int(y_true_all.iloc[st.session_state["selected_no"] - 1])
                true_label = "Churn" if true_val == 1 else "Tidak Churn"
                st.write(f"Label asli dataset (baris terpilih): **{true_label}**")
                st.write(f"Cocok dengan label asli? **{pred == true_val}**")

            st.write("Data input (setelah dikunci konsisten):")
            st.dataframe(df_one, use_container_width=True)

        except Exception as e:
            st.error("Gagal prediksi.\n\n" f"Detail: {e}")

st.divider()
st.caption("Catatan: Prediksi kelas berasal dari model (sesuai notebook). Jika label asli dataset tersedia, ditampilkan sebagai pembanding.")
