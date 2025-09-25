import pandas as pd
import streamlit as st
import plotly.express as px
from prophet import Prophet
import io

# --- Page Config ---
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark/Light Theme ---
is_dark = st.sidebar.checkbox("Dark Mode", value=False)
bg_color = "#1e1e1e" if is_dark else "#f4f6f9"
text_color = "#ffffff" if is_dark else "#2c3e50"
secondary_bg = "#2c2c2c" if is_dark else "#ffffff"
trend_color = "#76c7c0" if is_dark else "#1abc9c"
forecast_color = "#ffa07a" if is_dark else "#ff7f0e"

# --- CSS Styling ---
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{background-color: {bg_color}; color: {text_color}; border-right: 2px solid #ddd;}}
h1, h2, h3 {{color: {text_color}; font-family: 'Segoe UI', sans-serif;}}
div[data-testid="metric-container"] {{background-color: {secondary_bg}; border-radius: 10px; padding: 15px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1); text-align:center;}}
.stDataFrame {{border-radius: 10px; overflow: hidden;}}
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
data = pd.read_csv("sales_data.csv")
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")
regions = data['Region'].unique()
products = data['Product'].unique()

selected_region = st.sidebar.selectbox("Select Region", regions)
selected_products = st.sidebar.multiselect("Select Products", products, default=[products[0]])

min_date, max_date = data['Date'].min(), data['Date'].max()
selected_dates = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Aggregation choice (for trend only, Prophet always monthly)
agg_choice = st.sidebar.radio("Aggregation Level (Trend Only)", ["Daily", "Weekly", "Monthly"], index=1)

# --- Filter Data ---
filtered_data = data[
    (data['Region'] == selected_region) &
    (data['Product'].isin(selected_products)) &
    (data['Date'] >= pd.to_datetime(selected_dates[0])) &
    (data['Date'] <= pd.to_datetime(selected_dates[1]))
]

if filtered_data.empty:
    st.warning("No data available for this selection.")
    st.stop()

# --- Aggregate Data for Trend ---
if agg_choice == "Weekly":
    filtered_data = filtered_data.groupby(
        [pd.Grouper(key="Date", freq="W"), "Product"]
    )["Sales"].sum().reset_index()
elif agg_choice == "Monthly":
    filtered_data = filtered_data.groupby(
        [pd.Grouper(key="Date", freq="M"), "Product"]
    )["Sales"].sum().reset_index()

# --- Dashboard Header ---
st.markdown(f"<h1 style='text-align:center; color:#1abc9c;'>📊 Sales Dashboard: {selected_region}</h1>", unsafe_allow_html=True)
st.write(f"Products: {', '.join(selected_products)} | Date Range: {selected_dates[0]} to {selected_dates[1]} | Aggregation (Trend): {agg_choice}")

# --- Product KPIs ---
st.subheader("📌 Product-wise KPIs")
for product in selected_products:
    prod_data = filtered_data[filtered_data['Product'] == product].sort_values('Date')
    total = prod_data['Sales'].sum()
    avg = prod_data['Sales'].mean()
    max_val = prod_data['Sales'].max()
    
    # Sparkline with rolling mean
    prod_data["Smooth"] = prod_data["Sales"].rolling(3).mean()
    spark_fig = px.line(prod_data, x='Date', y='Smooth', height=80, width=250, line_shape='linear',
                        color_discrete_sequence=[trend_color])
    spark_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
    
    col1, col2 = st.columns([1,3])
    with col1:
        st.metric(f"{product}", f"Total: {total:,}", f"Avg: {avg:.2f}")
    with col2:
        st.plotly_chart(spark_fig, use_container_width=True)

# --- Sales Trend Chart ---
st.subheader("📈 Sales Trend")
fig_trend = px.line(
    filtered_data, x="Date", y="Sales", color="Product",
    title=f"Sales Trend by Product ({agg_choice})",
    markers=True,
    hover_data={"Date": True, "Sales": True, "Product": True},
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_trend, use_container_width=True)

# --- Prophet Forecast (Always Monthly) ---
st.subheader("🔮 Sales Forecast (Next 6 Months)")

# Aggregate sales monthly for forecast
monthly_data = filtered_data.groupby(
    pd.Grouper(key="Date", freq="M")
)["Sales"].sum().reset_index().rename(columns={"Date": "ds", "Sales": "y"})

if len(monthly_data) < 6:
    st.info("Not enough monthly data to forecast. Showing trend line instead.")
    st.line_chart(monthly_data.set_index('ds')['y'])
else:
    model = Prophet()
    model.fit(monthly_data)

    # Forecast next 6 months
    future = model.make_future_dataframe(periods=6, freq="M")
    forecast = model.predict(future)

    # --- Custom Forecast Plot ---
    import plotly.graph_objects as go
    fig_forecast = go.Figure()

    # Confidence Interval (Shaded)
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        name='Confidence Interval'
    ))

    # Forecast Line
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines',
        line=dict(color=forecast_color, width=2),
        name='Forecast'
    ))

    # Actual Sales
    fig_forecast.add_trace(go.Scatter(
        x=monthly_data['ds'], y=monthly_data['y'],
        mode='markers+lines',
        marker=dict(color='blue', size=6, opacity=0.8),
        line=dict(color='blue', width=1, dash="dot"),
        name='Actual Sales'
    ))

    # Layout
    fig_forecast.update_layout(
        title="Monthly Sales Forecast with Confidence Interval",
        xaxis_title="Date (Monthly)",
        yaxis_title="Sales",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

# --- Download Filtered Data ---
st.subheader("📥 Download Data")
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Download CSV",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_region}_sales.csv",
        mime="text/csv"
    )
with col2:
    buffer = io.BytesIO()
    filtered_data.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)
    st.download_button(
        label="Download Excel",
        data=buffer,
        file_name=f"{selected_region}_sales.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )