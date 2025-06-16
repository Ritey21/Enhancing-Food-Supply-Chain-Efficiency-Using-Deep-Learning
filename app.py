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

if module == "Supplier Upload":
    st.title("\U0001F4E4 Supplier Upload Portal")
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

    if not st.session_state.supplier_data.empty:
        st.subheader("Uploaded Data (Unsaved)")
        st.dataframe(st.session_state.supplier_data)

elif module == "Buyer Dashboard":
    st.title("\U0001F6D2 Buyer Dashboard")
    supermarkets = df["Supermarket"].unique()
    supermarket_sel = st.selectbox("Select Supermarket", supermarkets)
    available_products = df[(df["Supermarket"] == supermarket_sel) & (df["Quantity_Sold"] > 0)]["Product_Name"].unique()
    product_sel = st.selectbox("Select Product", available_products)

    stock_df = df[(df["Supermarket"] == supermarket_sel) & (df["Product_Name"] == product_sel)]
    latest_date = stock_df["Date"].max()
    latest_stock = stock_df[stock_df["Date"] == latest_date]["Soldout"].sum()

    st.write(f"*Available stock for {product_sel} on {latest_date.date()}:* {latest_stock} kg")

    if latest_stock < 10:
        st.warning(f"⚠ Low Stock Alert! Only {latest_stock} kg left for {product_sel} in supermarket {supermarket_sel}.")

    purchase_qty = st.number_input("Enter quantity to buy (kg)", min_value=1, max_value=int(latest_stock))
    if st.button("Simulate Purchase"):
        if purchase_qty <= latest_stock:
            st.success(f"Purchased {purchase_qty} kg of {product_sel} from supermarket {supermarket_sel}!")
        else:
            st.error("Insufficient stock!")

elif module == "Demand Forecast":
    st.title("\U0001F4C8 Demand Forecasting using XGBoost")

    supermarket_sel = st.selectbox("Select Supermarket", df["Supermarket"].unique())
    product_sel = st.selectbox("Select Product", df[df["Supermarket"] == supermarket_sel]["Product_Name"].unique())

    df_sel = df[(df["Supermarket"] == supermarket_sel) & (df["Product_Name"] == product_sel)].sort_values("Date")

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

        # Compare actual vs predicted
        compare_df = pd.DataFrame({"Actual Demand (kg)": y_test, "Predicted Demand (kg)": y_pred})
        compare_df["Date"] = df_sel.loc[y_test.index, "Date"].values  # <-- FIXED HERE using loc
        st.subheader("Actual vs Predicted Demand")
        fig = px.line(compare_df.sort_values("Date"), x="Date", y=["Actual Demand (kg)", "Predicted Demand (kg)"],
                     labels={"Date": "Date", "value": "Demand (kg)", "variable": "Type"},
                     title="Actual vs Predicted Demand Over Time")
        fig.update_layout(xaxis_title="Date", yaxis_title="Quantity (kg)")
        st.plotly_chart(fig)

        # Forecast next 7 days
        last_date = df_sel["Date"].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Day_of_Week_Num": [d.weekday() for d in future_dates],
            "Holiday": [0]*7,
            "Promotion": [1]*7,
            "Weather_Code": [0]*7,
            "Discount": [df_sel["Discount"].median()]*7
        })

        future_df["Predicted_Quantity"] = model.predict(future_df[features])

        st.subheader("Forecasted Demand for Next 7 Days")
        st.line_chart(future_df.set_index("Date")["Predicted_Quantity"])
        st.dataframe(future_df[["Date", "Predicted_Quantity"]])

        if future_df["Predicted_Quantity"].max() > 50:
            st.warning("\U0001F4C8 High Demand Alert! Demand may exceed 50 kg soon.")

elif module == "Analytics & Reporting":
    st.title("\U0001F4CA Analytics and Reporting")
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
        st.write("Products with low average sales (consider restocking or promotions):")
        st.dataframe(low_stock)
