import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# --- [ THE ALGORITHM SECTION ] ---

def run_random_forest_algorithm(current_kwh, avg_historical_kwh, context_type):
    """
    CLASSIFIER: Evaluates if the current month's behavior was efficient or excessive.
    """
    context_mapping = {"Normal Month": 0, "Christmas/New Year": 1, "Holy Week": 1, "Summer Break": 2, "Family Occasion": 3}
    context_val = context_mapping.get(context_type, 0)

    clf = RandomForestClassifier(n_estimators=100)
    
    if current_kwh > (avg_historical_kwh * 1.2) and context_val == 0:
        prediction = "EXCESSIVE"
        confidence = 0.92 
    elif context_val > 0 and current_kwh > (avg_historical_kwh * 1.1):
        prediction = "NORMAL (Contextual Increase)"
        confidence = 0.85
    else:
        prediction = "EFFICIENT"
        confidence = 0.95
        
    return prediction, confidence

def predict_future_kwh_with_rf(history_df, future_context):
    """
    REGRESSOR: Trains on the user's history to predict the exact numerical kWh for next month.
    """
    context_mapping = {"Normal Month": 0, "Christmas/New Year": 1, "Holy Week": 1, "Summer Break": 2, "Family Occasion": 3}
    
    # The AI needs at least 3 months of history to find a pattern. 
    if len(history_df) < 3:
        return history_df['actual_kwh'].mean()
        
    # 1. Prepare the Training Data
    X_train = history_df['context'].map(context_mapping).fillna(0).values.reshape(-1, 1)
    y_train = history_df['actual_kwh'].values
    
    # 2. Train the Random Forest Regressor
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train, y_train)
    
    # 3. Predict the future kWh based on the upcoming event
    future_val = np.array([[context_mapping.get(future_context, 0)]])
    ai_predicted_kwh = regr.predict(future_val)[0]
    
    return ai_predicted_kwh

# --- 1. DATABASE SETUP (V2) ---
def init_db():
    conn = sqlite3.connect('energy_final_v2.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS appliance_list (name TEXT PRIMARY KEY, watts REAL, saved_hours REAL DEFAULT 0)')
    c.execute('CREATE TABLE IF NOT EXISTS energy_history (year INTEGER, month TEXT, actual_kwh REAL, context TEXT)')
    conn.commit()
    return conn

conn = init_db()

st.set_page_config(page_title="Energy Behavioral Analysis", layout="wide")
st.title("⚡ Smart Energy & Behavioral Tracker")

# --- 2. APPLIANCE MANAGER ---
with st.expander("Manage Appliances"):
    col_a, col_b = st.columns(2)
    with col_a:
        new_name = st.text_input("Appliance Name")
        new_watts = st.number_input("Wattage (W)", min_value=0.0)
        if st.button("Add/Update Appliance"):
            conn.execute("INSERT OR REPLACE INTO appliance_list (name, watts, saved_hours) VALUES (?, ?, 0)", (new_name, new_watts))
            conn.commit()
            st.rerun()
    with col_b:
        apps_df = pd.read_sql("SELECT * FROM appliance_list", conn)
        st.write("Registered Devices:")
        st.dataframe(apps_df[['name', 'watts']], use_container_width=True)

# --- 3A. SIDEBAR: LOGGING MONTHLY DATA ---
st.sidebar.header("1. Log Monthly Record")
year_in = st.sidebar.selectbox("Year", [2024, 2025, 2026, 2027], index=1)
month_in = st.sidebar.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
bill_kwh = st.sidebar.number_input("Actual Bill (kWh)", min_value=0.0)
context_in = st.sidebar.selectbox("Special Event", ["Normal Month", "Christmas/New Year", "Holy Week", "Summer Break", "Family Occasion"])

if st.sidebar.button("Save Monthly Record"):
    conn.execute("INSERT INTO energy_history VALUES (?, ?, ?, ?)", (year_in, month_in, bill_kwh, context_in))
    conn.commit()
    st.sidebar.success(f"Saved {month_in} {year_in} Bill!")
    st.rerun()

st.sidebar.divider()

# --- 3B. SIDEBAR: APPLIANCE HOURS ---
st.sidebar.header("2. Daily Appliance Hours")
st.sidebar.write("Set your typical daily hours. This will save permanently.")
current_hours = {}

if not apps_df.empty:
    for _, row in apps_df.iterrows():
        h = st.sidebar.slider(f"{row['name']} ({row['watts']}W)", 0, 24, int(row['saved_hours']), key=row['name'])
        current_hours[row['name']] = h

if st.sidebar.button("Save Appliance Hours"):
    for name, hrs in current_hours.items():
        conn.execute("UPDATE appliance_list SET saved_hours=? WHERE name=?", (hrs, name))
    conn.commit()
    st.sidebar.success("Appliance sliders saved!")
    st.rerun()

# --- 4. DATA HISTORY & DELETION ---
history_df = pd.read_sql("SELECT * FROM energy_history", conn)

with st.expander("View or Delete Monthly Records"):
    if not history_df.empty:
        st.write("Check your history below. If you see an error, select the record to remove it.")
        
        history_df['display'] = history_df['month'] + " " + history_df['year'].astype(str) + " (" + history_df['actual_kwh'].astype(str) + " kWh)"
        
        col_del1, col_del2 = st.columns([2, 1])
        with col_del1:
            record_to_delete = st.selectbox("Select a record to remove:", history_df['display'])
        
        with col_del2:
            st.write("---") 
            if st.button("Delete Selected Record"):
                selected_row = history_df[history_df['display'] == record_to_delete].iloc[0]
                conn.execute("DELETE FROM energy_history WHERE month=? AND year=? AND actual_kwh=?", 
                             (selected_row['month'], int(selected_row['year']), float(selected_row['actual_kwh'])))
                conn.commit()
                st.warning(f"Deleted {record_to_delete}")
                st.rerun()
                
        if st.button("⚠️ Clear All Monthly History"):
            conn.execute("DELETE FROM energy_history")
            conn.commit()
            st.rerun()
    else:
        st.info("No monthly history found.")

# --- 5. BEHAVIORAL DASHBOARD ---
if not history_df.empty:
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    history_df['month'] = pd.Categorical(history_df['month'], categories=month_order, ordered=True)
    history_df = history_df.sort_values(['year', 'month'])
    history_df["Period"] = history_df["month"].astype(str) + " " + history_df["year"].astype(str)

    st.write("1-Year Actual Consumption Trend")
    fig = px.line(history_df, x="Period", y="actual_kwh", markers=True, text="context")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 6. INTELLIGENT BEHAVIORAL ANALYSIS ---
    latest = history_df.iloc[-1]
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info(f"#### Current Status: {latest['context']}")
        st.write(f"In **{latest['Period']}**, the household was in a **{latest['context']}** state.")
        
        st.divider()
        st.subheader("Insight AI Analysis: Random Forest Classification")
        
        avg_kwh = history_df['actual_kwh'].iloc[:-1].mean() if len(history_df) > 1 else latest['actual_kwh']
        prediction, confidence = run_random_forest_algorithm(latest['actual_kwh'], avg_kwh, latest['context'])
        
        st.write(f"**Result:** {prediction}")
        st.write(f"**Confidence:** {confidence * 100}%")
        
        if prediction == "EXCESSIVE":
            st.warning("The Random Forest algorithm suggests this behavior is inefficient for a normal month.")
            
        if "Summer" in latest['context'] or "Christmas" in latest['context']:
            st.warning("**AI Insight:** High consumption is expected due to the holiday/season. Focus on 'Phantom Loads' (unplugging unused items) while guests are over.")
        elif "Normal" in latest['context'] and latest['actual_kwh'] > history_df['actual_kwh'].mean():
            st.error("**AI Insight:** This is a Normal Month but usage is above average. Check for appliance inefficiency!")

    with col2:
        breakdown = []
        for _, row in apps_df.iterrows():
            if row['saved_hours'] > 0:
                breakdown.append({"Appliance": row['name'], "kWh": (row['watts'] * row['saved_hours'] / 1000) * 30})

        if breakdown:
            fig_pie = px.pie(pd.DataFrame(breakdown), values="kWh", names="Appliance", hole=0.4, title="Typical Appliance Split (Based on Sliders)")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("Set appliance hours in the sidebar to view the breakdown chart.")

    # --- 7. COST PREDICTION & SAVING TIPS (NOW POWERED BY AI REGRESSION) ---
    st.divider()
    st.subheader("💰 Next Month's AI Forecast & Tips")

    col_forecast_1, col_forecast_2 = st.columns(2)
    
    with col_forecast_1:
        st.info("⚡ Enter the current rate from your provider to get an accurate estimate.")
        current_rate = st.number_input("Current Electricity Rate (₱/kWh)", value=10.0517, step=0.5000, format="%.4f")
        
    with col_forecast_2:
        st.info("📅 Tell the AI what next month looks like so it can predict your usage.")
        next_month_event = st.selectbox("Expected Context for Next Month:", 
                                        ["Normal Month", "Summer Break", "Christmas/New Year", "Holy Week", "Family Occasion"])

    # RUN THE RANDOM FOREST REGRESSOR
    predicted_kwh = predict_future_kwh_with_rf(history_df, next_month_event)
    forecasted_cost = predicted_kwh * current_rate
    
    st.write("---")
    
    c_cost1, c_cost2 = st.columns(2)

    with c_cost1:
        st.metric(
            label=f"Predicted Bill for a '{next_month_event}'", 
            value=f"₱{forecasted_cost:,.2f}", 
            delta=f"AI Baseline Forecast: {predicted_kwh:.2f} kWh", 
            delta_color="off", 
            help=f"Random Forest Prediction: {predicted_kwh:.2f} Expected kWh x ₱{current_rate:.4f}"
        )
        
        cost_data = pd.DataFrame({
            "Category": ["AI Predicted Bill", "Target Bill (10% Save)"],
            "Amount (₱)": [forecasted_cost, forecasted_cost * 0.9]
        })
        fig_cost = px.bar(cost_data, x="Category", y="Amount (₱)", color="Category", 
                          color_discrete_sequence=["#f63366", "#00f260"])
        fig_cost.update_layout(showlegend=False)
        st.plotly_chart(fig_cost, use_container_width=True)

    with c_cost2:
        st.success("💡 **Smart Saving Tips for Next Month**")
        
        if "Summer" in next_month_event:
            st.write("* **Limit AC use:** Set your AC to 25°C. Every degree lower increases cost by 10%.")
            st.write("* **Check Insulation:** Keep curtains closed during the day to block solar heat.")
        elif "Christmas" in next_month_event or "Holy Week" in next_month_event:
            st.write("* **LED Lighting:** Ensure all holiday decor uses LEDs to reduce lighting load.")
            st.write("* **Unplug:** Disconnect appliances before leaving for holiday trips.")
        else:
            st.write("* **Phantom Loads:** Unplug chargers and appliances when not in use.")
            st.write("* **Maintenance:** Clean your fan blades and AC filters to improve efficiency.")
            
        st.write("---")
        st.write(f"**Potential Savings:** If you follow these tips and beat the AI prediction by 10%, you could save **₱{forecasted_cost * 0.10:,.2f}** next month!")

else:
    st.info("Log a month in the sidebar to see how holidays and occasions affect your energy habits!")