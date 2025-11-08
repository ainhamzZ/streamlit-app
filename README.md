<!-- Banner -->
<p align="center">
  <img src="https://raw.githubusercontent.com/streamlit/brand/main/logos/streamlit-logo-secondary-colormark-darktext.png" width="300" alt="Streamlit Logo">
</p>

<h1 align="center">🪙 Binance Data Analytics Dashboard</h1>

<p align="center">
  <b>Interactive Streamlit app for cleaning, modeling & visualizing Binance crypto data</b><br>
  <sub>Built with ❤️ by <a href="https://github.com/yourusername">Nurul Ain</a></sub>
</p>

---

### 🚀 Live App
👉 **[Launch on Streamlit Cloud](https://share.streamlit.io/)**  
*(Your live link will appear here after deployment)*

---

## ✨ Features

### 🔧 Data Cleaning
- Automatically converts `open_time` and `close_time` to proper datetime formats  
- Adds `candle_duration` for each trade interval  
- Detects numeric columns and fills missing values  
- Computes **RSI**, **EMA20**, and **EMA50** indicators using `ta`  
- Supports multiple import/export formats: CSV, JSON, Parquet, HDF5  

### 📈 Predictive Modeling (PyCaret)
- Train predictive models for **close price** or any numeric column  
- Auto-selects the best regression model using `compare_models()`  
- Displays **Actual vs Predicted** graph  
- Allows downloading trained models (`.pkl`) and predicted datasets  

### 📊 Tableau Export
- Export cleaned or modeled data for Tableau dashboarding  
- Ready for visualizations like:
  - Price trend analysis  
  - RSI & EMA crossover tracking  
  - Predicted vs actual model evaluation  

### 🧠 AI Insights
- Automatically generates data-driven insights:
  - RSI trends (overbought/oversold)
  - EMA crossover signals
  - Correlation between predicted and actual prices
  - Average trading volume  
- Interactive **Plotly chart** — toggle between RSI, EMA, Close, or Predicted series  

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|--------|
| **Framework** | Streamlit |
| **Machine Learning** | PyCaret |
| **Indicators & Analysis** | `ta`, pandas, numpy |
| **Visualization** | Plotly, Matplotlib |
| **Storage Formats** | CSV, JSON, Parquet, HDF5 |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/binance-data-analytics-dashboard.git
cd binance-data-analytics-dashboard
