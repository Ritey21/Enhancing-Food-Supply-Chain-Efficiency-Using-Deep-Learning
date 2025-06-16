import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("supermarket_sales_data_updated.csv")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["Price_per_kg"] = df["Price_per_kg"].str.extract(r'(\d+)').astype(int)
    df["Discounted_Price"] = df["Discounted_Price"].str.extract(r'(\d+)').astype(int)
    df["Day_of_Week_Num"] = df["Day_of_Week"].astype('category').cat.codes
    df["Holiday"] = df["Holiday"].map({"Yes": 1, "No": 0})
    df["Promotion"] = df["Promotion"].map({"Yes": 1, "No": 0})
    df["Weather_Code"] = df["Weather"].astype('category').cat.codes
    df["Discount"] = df["Price_per_kg"] - df["Discounted_Price"]
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Modules")
module = st.sidebar.radio("Select Module", ["Supplier Upload", "Buyer Dashboard", "Demand Forecast", "Analytics & Reporting"])

if "supplier_data" not in st.session_state:
    st.session_state.supplier_data = pd.DataFrame(columns=df.columns)

# Supplier Upload Module
if module == "Supplier Upload":
    st.title("📤 Supplier Upload Portal")
    with st.form("upload_form"):
        supermarket = st.selectbox("Supermarket", df["Supermarket"].unique())
        product_name = st.text_input("Product Name")
        category = st.selectbox("Category", df["Category"].unique())
        quantity = st.number_input("Quantity Sold (kg)", min_value=1)
        price = st.number_input("Price per kg (INR)", min_value=1)
        discounted_price = st.number_input("Discounted Price (INR)", min_value=1, max_value=price)
        date = st.date_input("Date", value=datetime.today())
        day_of_week = date.strftime("%A")
        holiday = st.selectbox("Holiday", ["No", "Yes"])
        weather = st.selectbox("Weather", df["Weather"].unique())
        promotion = st.selectbox("Promotion", ["No", "Yes"])
        submitted = st.form_submit_button("Upload Product")

        if submitted:
            new_entry = {
                "Supermarket": supermarket,
                "Locality": "Indiranagar",
                "Product_Name": product_name,
                "Date": pd.to_datetime(date),
                "Quantity_Sold": quantity,
                "Price_per_kg": price,
                "Discounted_Price": discounted_price,
                "Category": category,
                "Day_of_Week": day_of_week,
                "Holiday": holiday,
                "Weather": weather,
                "Promotion": promotion
            }
            st.session_state.supplier_data = pd.concat([st.session_state.supplier_data, pd.DataFrame([new_entry])], ignore_index=True)
            st.success(f"Product '{product_name}' uploaded for supermarket {supermarket}!")
            st.toast(f"✅ '{product_name}' uploaded successfully!", icon="📦")

    if not st.session_state.supplier_data.empty:
        st.subheader("Uploaded Data (Unsaved)")
        st.dataframe(st.session_state.supplier_data)

# Buyer Dashboard
elif module == "Buyer Dashboard":
    st.title("🛒 Buyer Dashboard")
    supermarket_sel = st.selectbox("Select Supermarket", df["Supermarket"].unique())
    product_sel = st.selectbox("Select Product", df[(df["Supermarket"] == supermarket_sel)]["Product_Name"].unique())

    stock_df = df[(df["Supermarket"] == supermarket_sel) & (df["Product_Name"] == product_sel)]
    latest_date = stock_df["Date"].max()
    latest_stock = stock_df[stock_df["Date"] == latest_date]["Quantity_Sold"].sum()

    st.write(f"*Available stock for {product_sel} on {latest_date.date()}:* {latest_stock} kg")

    # Low stock alert with toast and warning message
    low_stock_threshold = 30
    if latest_stock < low_stock_threshold:
        st.toast(f"⚠ Low Stock Alert! Only {latest_stock} kg left for {product_sel}!", icon="⚠")
        st.warning(f"⚠ Warning: Low stock for {product_sel} ({latest_stock} kg left)! Consider restocking.")

    purchase_qty = st.number_input("Enter quantity to buy (kg)", min_value=1, max_value=int(latest_stock))
    if st.button("Simulate Purchase"):
        if purchase_qty <= latest_stock:
            st.success(f"Purchased {purchase_qty} kg of {product_sel} from {supermarket_sel}!")
            st.toast(f"🛒 Purchased {purchase_qty} kg of {product_sel}!", icon="🛍")
        else:
            st.error("Insufficient stock!")

# Demand Forecast
elif module == "Demand Forecast":
    st.title("📈 Demand Forecasting with XGBoost")
    supermarket_sel = st.selectbox("Select Supermarket", df["Supermarket"].unique())
    product_sel = st.selectbox("Select Product", df[df["Supermarket"] == supermarket_sel]["Product_Name"].unique())

    df_sel = df[(df["Supermarket"] == supermarket_sel) & (df["Product_Name"] == product_sel)]

    if len(df_sel) < 20:
        st.warning("Not enough data to build forecast model.")
    else:
        features = ["Day_of_Week_Num", "Holiday", "Promotion", "Weather_Code", "Discount"]
        target = "Quantity_Sold"

        X = df_sel[features]
        y = df_sel[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        st.metric("Test RMSE", f"{rmse:.2f}")

        # Forecast
        last_date = df_sel["Date"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Day_of_Week_Num": [d.dayofweek for d in future_dates],
            "Holiday": [0]*7,
            "Promotion": [1]*7,
            "Weather_Code": [0]*7,
            "Discount": [df_sel["Discount"].median()]*7
        })

        future_df["Predicted_Quantity"] = model.predict(future_df[features])

        max_forecast = future_df["Predicted_Quantity"].max()
        high_demand_threshold = 200
        if max_forecast > high_demand_threshold:
            st.toast(f"📈 High Demand Alert! Forecast up to {max_forecast:.0f} kg for {product_sel}!", icon="📈")
            st.warning(f"⚠ Warning: High demand forecast! Expect up to {max_forecast:.0f} kg for {product_sel}.")

        low_demand_threshold = 20
        min_forecast = future_df["Predicted_Quantity"].min()
        if min_forecast < low_demand_threshold:
            st.toast(f"⚠ Low Demand Warning! Forecast dips to {min_forecast:.0f} kg.", icon="⚠")
            st.warning(f"⚠ Warning: Low demand forecast! Dips to {min_forecast:.0f} kg for {product_sel}.")

        st.subheader("Forecasted Demand (Next 7 Days)")
        st.line_chart(future_df.set_index("Date")["Predicted_Quantity"])
        st.dataframe(future_df[["Date", "Predicted_Quantity"]])

# Analytics & Reporting
elif module == "Analytics & Reporting":
    st.title("📊 Analytics and Reporting")

    combined_df = pd.concat([df, st.session_state.supplier_data], ignore_index=True) if not st.session_state.supplier_data.empty else df

    sales_sum = combined_df.groupby("Supermarket")["Quantity_Sold"].sum().reset_index().rename(columns={"Quantity_Sold": "Total_Sold_kg"})
    st.subheader("Total Sales by Supermarket")
    fig1 = px.bar(sales_sum, x="Supermarket", y="Total_Sold_kg", title="Total Quantity Sold per Supermarket (kg)")
    st.plotly_chart(fig1)

    cat_sum = combined_df.groupby("Category")["Quantity_Sold"].sum().reset_index()
    st.subheader("Sales Distribution by Category")
    fig2 = px.pie(cat_sum, names="Category", values="Quantity_Sold", title="Sales by Category")
    st.plotly_chart(fig2)

    avg_sales = combined_df.groupby(["Supermarket", "Product_Name"])["Quantity_Sold"].mean().reset_index()
    low_stock = avg_sales[avg_sales["Quantity_Sold"] < 10]

    st.subheader("Inventory Optimization Suggestion")
    if low_stock.empty:
        st.write("All products have healthy sales volumes.")
    else:
        st.write("Products with low average sales (consider restocking or promotion):")
        st.dataframe(low_stock)

    # Total sales alert with toast and warning
    total_sales_all = combined_df["Quantity_Sold"].sum()
    high_sales_threshold = 1000
    if total_sales_all > high_sales_threshold:
        st.toast(f"🔥 High Sales Alert! Total sales reached {total_sales_all:.0f} kg!", icon="🔥")
        st.warning(f"⚠ Warning: High total sales detected! {total_sales_all:.0f} kg sold across supermarkets.")

    st.toast("📊 Analytics loaded successfully!", icon="✅")
