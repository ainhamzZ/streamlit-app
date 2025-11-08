import streamlit as st
import pandas as pd
import ta
from io import BytesIO
from pycaret.regression import setup, compare_models, predict_model, pull, save_model
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Binance Data Analytics Dashboard", layout="wide")
st.title("🪙 Binance Data Analytics Dashboard")
st.caption("Clean, model, export, and analyze Binance data seamlessly.")

tabs = st.tabs([
    "🔧 Data Cleaning",
    "📈 Predictive Modeling",
    "📊 Tableau Export",
    "🧠 AI Insights"
])

# ===================== TAB 1: DATA CLEANING =====================
with tabs[0]:
    st.header("Step 1: Upload and Clean Binance Data")
    uploaded_file = st.file_uploader(
        "Upload Binance Data",
        type=["csv", "parquet", "json", "h5"],
        key="clean_tab"
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file, orient='records', convert_dates=["open_time","close_time"])
            elif uploaded_file.name.endswith(".h5"):
                df = pd.read_hdf(uploaded_file, key='df')
            else:
                st.error("Unsupported file format")
                st.stop()
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

        st.success(f"✅ Loaded {uploaded_file.name} successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Parse datetime
        for col in ["open_time", "close_time"]:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], unit="ms" if df[col].mean() > 1e11 else "s")
                else:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

        # Candle duration in seconds
        if "open_time" in df.columns and "close_time" in df.columns:
            df["candle_duration"] = ((df["close_time"] - df["open_time"]).dt.total_seconds()).round(2)

        # Drop duplicates and sort
        df = df.drop_duplicates(subset="open_time").set_index("open_time").sort_index()

        # Ensure numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(method="ffill")

        # Add indicators
        if "close" in df.columns:
            df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi().round(2)
            df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator().round(2)
            df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator().round(2)
        df = df.fillna(method="ffill")

        st.subheader("Cleaned Data Preview")
        st.dataframe(df.head(20))
        st.subheader("Summary Statistics")
        st.write(df.describe())

        st.subheader("Download Cleaned Data")
        export_format = st.selectbox("Choose export format", ["CSV","Parquet","JSON","HDF5 (.h5)"], key="clean_export_format")
        if st.button("📥 Download Cleaned Data"):
            if export_format in ["CSV","Parquet","JSON"]:
                buffer = BytesIO()
                if export_format == "CSV":
                    df.to_csv(buffer, index=True)
                    mime = "text/csv"; filename="binance_cleaned.csv"
                elif export_format == "Parquet":
                    df.to_parquet(buffer, index=True)
                    mime = "application/octet-stream"; filename="binance_cleaned.parquet"
                elif export_format == "JSON":
                    df.to_json(buffer, orient="records", date_format="iso")
                    mime = "application/json"; filename="binance_cleaned.json"
                buffer.seek(0)
                st.download_button(f"Download {export_format}", buffer, file_name=filename, mime=mime)
            else:  # HDF5
                temp_file="binance_cleaned.h5"
                df.to_hdf(temp_file, key='df', mode='w')
                with open(temp_file,"rb") as f:
                    st.download_button("Download HDF5 (.h5)", f, file_name=temp_file, mime="application/octet-stream")

        st.session_state["cleaned_data"] = df
        st.success("✅ Cleaning complete! Proceed to 'Predictive Modeling' tab.")
    else:
        st.info("Please upload a Binance CSV, Parquet, JSON, or HDF5 file to begin.")

# ===================== TAB 2: PREDICTIVE MODEL =====================
import plotly.graph_objects as go
from pycaret.regression import setup, compare_models, predict_model, pull, save_model

with tabs[1]:
    st.header("Step 2: Predictive Modeling with PyCaret")

    # Upload cleaned dataset
    model_file = st.file_uploader(
        "Upload Cleaned Data",
        type=["csv", "parquet", "json", "h5"],
        key="model_tab"
    )

    # Load dataset
    if "cleaned_data" in st.session_state and model_file is None:
        df = st.session_state["cleaned_data"]
        st.info("Using cleaned data from Step 1")
    elif model_file:
        try:
            if model_file.name.endswith(".csv"):
                df = pd.read_csv(model_file)
            elif model_file.name.endswith(".parquet"):
                df = pd.read_parquet(model_file)
            elif model_file.name.endswith(".json"):
                df = pd.read_json(model_file)
            elif model_file.name.endswith(".h5"):
                df = pd.read_hdf(model_file, key='df')
            else:
                st.error("Unsupported file format")
                st.stop()
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
        st.success(f"✅ Loaded {model_file.name} successfully!")
    else:
        st.warning("⚠️ Upload a cleaned dataset or complete Step 1 first.")
        st.stop()

    st.subheader("Preview of Dataset")
    st.dataframe(df.head(10))

    # Select target column
    numeric_cols = df.select_dtypes(include=['number']).columns
    target_options = ["Next-period Close"] + numeric_cols.tolist()
    target_choice = st.selectbox("Choose target column:", target_options)

    # Prepare features and target
    if target_choice == "Next-period Close":
        df['target_close'] = df['close'].shift(-1)
        df_model = df.dropna(subset=['target_close']).copy()
        target_col = 'target_close'
    else:
        df_model = df.copy()
        target_col = target_choice

    features = df_model.drop(columns=[target_col])
    target = df_model[target_col]

    st.subheader("Preview of Features and Target")
    st.dataframe(pd.concat([features.head(), target.head()], axis=1))

    # Train model
    if st.button("🚀 Train Model"):
        with st.spinner("Training regression models... ⏳"):
            # Reset index for PyCaret
            df_model_reset = df_model.reset_index(drop=True)

            # Setup PyCaret
            s = setup(data=df_model_reset, target=target_col, verbose=False, session_id=123)

            # Compare models
            best_model = compare_models()
            compare_df = pull()
            st.subheader("Model Comparison Results")
            st.dataframe(compare_df)

            # Predict
            preds = predict_model(best_model, data=df_model_reset)
            if "prediction_label" in preds.columns:
                df_model["Predicted"] = preds["prediction_label"].values
            else:
                st.error("❌ Could not find prediction column in PyCaret output!")
                st.stop()

            # Interactive Plot: Actual vs Predicted
            st.subheader(f"📊 Actual vs Predicted: {target_col}")
            if 'open_time' in df_model.columns:
                x_axis = pd.to_datetime(df_model['open_time'])
            else:
                x_axis = df_model.index

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=df_model[target_col],
                                     mode='lines', name='Actual', line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=x_axis, y=df_model['Predicted'],
                                     mode='lines', name='Predicted', line=dict(color='red', width=2, dash='dash')))
            fig.update_layout(title=f'Actual vs Predicted: {target_col}',
                              xaxis_title='Datetime', yaxis_title=target_col,
                              legend=dict(x=0, y=1.0), margin=dict(l=40,r=40,t=40,b=40),
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            # Download predicted data
            export_format_pred = st.selectbox("Choose export format", ["CSV", "Parquet", "JSON", "HDF5 (.h5)"], key="pred_export_format")
            if st.button("📥 Download Predicted Data"):
                if export_format_pred in ["CSV","Parquet","JSON"]:
                    buffer = BytesIO()
                    if export_format_pred=="CSV":
                        df_model.to_csv(buffer,index=True); mime="text/csv"; filename="binance_predictions.csv"
                    elif export_format_pred=="Parquet":
                        df_model.to_parquet(buffer,index=True); mime="application/octet-stream"; filename="binance_predictions.parquet"
                    elif export_format_pred=="JSON":
                        df_model.to_json(buffer, orient="records", date_format="iso"); mime="application/json"; filename="binance_predictions.json"
                    buffer.seek(0)
                    st.download_button(f"Download {export_format_pred}", buffer, file_name=filename, mime=mime)
                else:  # HDF5
                    temp_file="binance_predictions.h5"
                    df_model.to_hdf(temp_file,key='df',mode='w')
                    with open(temp_file,"rb") as f:
                        st.download_button("Download HDF5 (.h5)", f, file_name=temp_file, mime="application/octet-stream")

            # Save trained model
            save_model(best_model,"best_binance_model")
            with open("best_binance_model.pkl","rb") as f:
                st.download_button("💾 Download Trained Model", f,"best_binance_model.pkl")

        st.session_state["predicted_data"] = df_model
        st.success("✅ Model training complete! You can now proceed to Tableau Export or AI Insights.")


# ===================== TAB 3: TABLEAU EXPORT =====================
with tabs[2]:
    st.header("Step 3: Tableau Visualization Export")
    if "predicted_data" not in st.session_state and "cleaned_data" not in st.session_state:
        st.warning("⚠️ Complete Step 1 or Step 2 first."); st.stop()

    export_option = st.radio("Choose dataset to export:", ("Cleaned Data","Predicted Data"))
    df = st.session_state["cleaned_data"] if export_option=="Cleaned Data" else st.session_state["predicted_data"]
    filename = "binance_cleaned_for_tableau.csv" if export_option=="Cleaned Data" else "binance_predictions_for_tableau.csv"
    st.dataframe(df.head(15))
    buf=BytesIO(); df.to_csv(buf,index=False); buf.seek(0)
    st.download_button(f"📊 Download {filename} for Tableau", buf, file_name=filename, mime="text/csv")

# ===================== TAB 4: AI INSIGHTS =====================
with tabs[3]:
    st.header("Step 4: AI-Generated Insights 🧠")

    # Determine which dataset to use
    if "predicted_data" in st.session_state:
        df = st.session_state["predicted_data"]
        st.info("Using Predicted Data for insights")
    elif "cleaned_data" in st.session_state:
        df = st.session_state["cleaned_data"]
        st.info("Using Cleaned Data for insights")
    else:
        st.warning("⚠️ Please complete Step 1 or Step 2 first.")
        st.stop()

    # Preview data
    st.subheader("Preview of Data Used")
    st.dataframe(df.head(10))

    # --- Rename Predicted column for clarity ---
    if "Predicted" in df.columns:
        if "target_close" in df.columns:
            df.rename(columns={"Predicted": "Predicted Next Close"}, inplace=True)
        else:
            # Use target_col if available in session or default
            target_col_name = st.session_state.get("target_col", "Target")
            df.rename(columns={"Predicted": f"Predicted {target_col_name}"}, inplace=True)

    # --- Generate AI Insights ---
    st.markdown("### 🔍 AI Insights")
    insights = []

    # RSI analysis
    if "rsi" in df.columns:
        avg_rsi = df["rsi"].mean()
        if avg_rsi < 30:
            insights.append("RSI indicates oversold — potential rebound zones.")
        elif avg_rsi > 70:
            insights.append("RSI indicates overbought — possible corrections.")
        else:
            insights.append("RSI levels neutral — steady momentum.")

    # EMA crossovers
    if "ema_20" in df.columns and "ema_50" in df.columns:
        cross_count = ((df["ema_20"] > df["ema_50"]) & (df["ema_20"].shift(1) < df["ema_50"].shift(1))).sum()
        insights.append(f"Detected {cross_count} bullish EMA crossovers (EMA20>EMA50).")

    # Predicted vs Actual correlation
    predicted_cols = [col for col in df.columns if col.startswith("Predicted")]
    if predicted_cols and "close" in df.columns:
        for pcol in predicted_cols:
            corr = df[pcol].corr(df["close"])
            insights.append(f"{pcol} correlation with actual close: {corr:.2f}")

    # Volume insights
    if "volume" in df.columns:
        vol_avg = df["volume"].mean()
        insights.append(f"Average trading volume: {vol_avg:,.0f}")

    # Fallback
    if not insights:
        insights.append("Dataset lacks indicators or target variable for deeper analysis.")

    # Display insights
    for i, ins in enumerate(insights, 1):
        st.write(f"**Insight {i}:** {ins}")

    # --- Select indicators for interactive chart ---
    available_indicators = ["Close Price"]
    if "ema_20" in df.columns: available_indicators.append("EMA20")
    if "ema_50" in df.columns: available_indicators.append("EMA50")
    if "rsi" in df.columns: available_indicators.append("RSI")
    available_indicators.extend(predicted_cols)

    selected_indicators = st.multiselect(
        "Select indicators to display in interactive chart",
        options=available_indicators,
        default=["Close Price"]
    )

    # --- Interactive Plotly Chart ---
    st.subheader("📊 Interactive Chart")
    fig = go.Figure()

    # X-axis: use datetime if available
    if 'open_time' in df.columns:
        x_axis = pd.to_datetime(df['open_time'])
    else:
        x_axis = df.index

    # Add selected indicators
    if "Close Price" in selected_indicators:
        fig.add_trace(go.Scatter(x=x_axis, y=df['close'], mode='lines', name='Close', line=dict(color='blue')))
    if "EMA20" in selected_indicators:
        fig.add_trace(go.Scatter(x=x_axis, y=df['ema_20'], mode='lines', name='EMA20', line=dict(color='orange')))
    if "EMA50" in selected_indicators:
        fig.add_trace(go.Scatter(x=x_axis, y=df['ema_50'], mode='lines', name='EMA50', line=dict(color='green')))
    if "RSI" in selected_indicators:
        fig.add_trace(go.Scatter(
            x=x_axis, y=df['rsi'], mode='lines', name='RSI',
            line=dict(color='red', dash='dash'), yaxis='y2'
        ))
        fig.update_layout(
            yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0,100], showgrid=False, showline=True, showticklabels=True)
        )
    for pcol in predicted_cols:
        if pcol in selected_indicators:
            fig.add_trace(go.Scatter(x=x_axis, y=df[pcol], mode='lines', name=pcol, line=dict(color='purple', dash='dot')))

    # Layout
    fig.update_layout(
        title="Selected Indicators",
        xaxis_title="Datetime",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        hovermode='x unified',
        margin=dict(l=40,r=40,t=40,b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ AI insights generated successfully! Select indicators to explore interactively.")
