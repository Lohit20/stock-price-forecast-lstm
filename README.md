
# 📈 Stock‑Price Forecasting with LSTM & Tableau

An end-to-end Python project for forecasting stock prices by scraping NASDAQ data, training LSTM models, and visualizing results with interactive Tableau dashboards.

---

## 📁 Repository Structure

```
.
├── data/
│   ├── raw/                         # Raw CSVs from Selenium scraper
│   └── processed/                   # Scaled and cleaned datasets
├── notebooks/
│   ├── 1‑data_collection.ipynb      # Scraping NASDAQ with Selenium
│   ├── 2‑preprocessing.ipynb        # Data transformation
│   └── 3‑lstm_model.ipynb           # LSTM model building and evaluation
├── models/                          # Saved Keras LSTM models
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Ignored files/folders
```

---

## 🧰 Tools & Technologies

- **Python**: Core programming logic and ML pipeline
- **Selenium**: Scraping real-time NASDAQ stock data
- **Pandas, NumPy**: Data manipulation
- **Keras (TensorFlow backend)**: LSTM neural network
- **Matplotlib**: Data visualization
- **RobustScaler**: Preprocessing to mitigate outlier impact
- **Tableau**: Interactive dashboards for financial analysis

---

## 📊 Tableau Dashboards

- 📌 [Overall Income Statement Comparison](https://public.tableau.com/views/incomestatement_16990451574550/Sheet10)
- 📌 [Company A – Balance Sheet Dashboard](https://public.tableau.com/views/AnalysisforcompanyA/Dashboard1)
- 📌 [Top 10 Companies – Balance Sheet](https://public.tableau.com/views/Balance-top10companies/Sheet5)
- 📌 [Income Statement Dashboard](https://public.tableau.com/views/incomestatement-dashboard/Dashboard1)

---

## 🧠 LSTM Modeling — Detailed Overview

### 🔍 Data Preprocessing

- Load CSV with `price` indexed by timestamp
- Normalize with `RobustScaler`
- Transform into supervised format using `split_sequence()`:
  - Lookback: 90 days
  - Forecast: 30 days
  - Shift: 30 days

### 🧱 LSTM Architecture

```python
model = Sequential()
model.add(LSTM(60, activation='softsign', return_sequences=True, input_shape=(90, 1)))
model.add(LSTM(30, activation='softsign'))
model.add(Dense(30))  # 30 future predictions
```

- Loss Function: MSE
- Optimizer: Adagrad
- Epochs: 10
- Batch Size: 32

### ✅ Training & Evaluation

- Validation split: 10%
- Loss convergence:
```
Epoch 1/10 - loss: 0.0440 - val_loss: 0.1697
...
Epoch 10/10 - loss: 0.0422 - val_loss: 0.1616
```
- Final MSE: **0.0680**
- Visual plots: Actual vs Predicted

### 🔮 Forecasting

- Predicts next 30 days from last 90
- Example Output:
```
2021-07-30 | 2.3155
2021-07-31 | 2.1947
...
2021-08-28 | 1.9687
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/Lohit20/stock-price-forecast-lstm.git
cd stock-price-forecast-lstm
pip install -r requirements.txt
```

- Run Jupyter notebooks in order: `1-data_collection` → `2-preprocessing` → `3-lstm_model`

---

## 📚 Future Enhancements

- Add technical indicators (MACD, RSI)
- Compare LSTM with GRU, TCN
- Real-time inference API
- Dashboard integration

---

## 🤝 Contributing

Feel free to fork and open pull requests with improvements or new ideas.

---

## 📬 Contact

For any questions or collaborations, open an issue or reach out via GitHub.

