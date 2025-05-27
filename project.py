import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import random
from streamlit_option_menu import option_menu
import time
import plotly.express as px
import smtplib
from email.message import EmailMessage
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "df" not in st.session_state:
    st.session_state.df = None
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "Data Overview"
if "UserEnterUserName" not in st.session_state:
    st.session_state.UserEnterUserName = ""
if "Company" not in st.session_state:
    st.session_state.Company = ""
if "OTP" not in st.session_state:
    st.session_state.OTP = ""
if hasattr(st, "experimental_rerun"):
    rerun = st.experimental_rerun
else:
    rerun = st.rerun

query_params = st.query_params
if query_params.get("auth") == "true":
    st.session_state.authenticated = True
    st.session_state.UserEnterUserName = query_params.get("user", "")
    st.session_state.Company = query_params.get("company" , "")

def side_bar():
    with st.sidebar:
        sidebar_select=option_menu("TrendVista",["Log-in","Sign-Up"],default_index=0,menu_icon="pie-chart",icons=["unlock","lock"],orientation="vertical",key="sidebar_menu_key")

    if sidebar_select=="Log-in":
        st.success("If You Do Not Have Account Then Sign-Up First")
        with st.form("Form1"):
            st.title("Log In")
            UserEnterUserName=st.text_input("Enter Your Username")
            UserEnterPassword=st.text_input("Enter Your Password",type="password")

            submit=st.form_submit_button("Submit")
            is_data_save=check_user_data(UserEnterUserName,UserEnterPassword)

            if submit:
               if is_data_save==True:
                  st.session_state.authenticated = True
                  st.session_state.UserEnterUserName = UserEnterUserName
                  st.query_params.update(auth = "true",user=UserEnterUserName)
                  st.success("Login Successfully")
                  rerun()
               else:
                  st.error("User name or Password is Wrong, please enter correct one")

    if sidebar_select=="Sign-Up":
        with st.form("Form2"):
            st.title("Sign Up")
            UserEnterNewUserName=st.text_input("Enter Your Username")
            UserEnterNewPassword=st.text_input("Enter Your Password",type="password")
            Company = st.text_input("Enter Your Company")
            user_email = st.text_input("Enter Your Email Id")
            send_submit = st.form_submit_button("Send OTP")
            is_data_save = check_user_company(Company)
            is_data_save = check_user_availabel(UserEnterNewUserName)

        if  send_submit:
            st.session_state.Company = Company
            st.query_params.update(company = Company)
            try:
                if UserEnterNewUserName == "":
                    st.error("Enter New Username")
                elif  UserEnterNewPassword == "":
                    st.error("Enter New Password")
                elif check_user_company(Company):
                    st.error("Company already taken ❌")
                elif user_email == "":
                    st.error("Enter Email")
                elif check_user_availabel(UserEnterNewUserName):
                    st.error("Username already taken ❌")
                else:
                    st.session_state.OTP = mail_varify(user_email,  UserEnterNewPassword)

            except smtplib.SMTPRecipientsRefused:
                st.error("Enter Valid Email ID")
        if len(st.session_state.OTP) == 6:
            with st.form("check_form"):
                user_otp = st.text_input("Enter OTP:")
                submit8 = st.form_submit_button("submit")

            if submit8:
                if check_user_availabel(UserEnterNewUserName):
                    st.success("User Successfully Registered 🎉")
                else:
                    if user_otp == st.session_state.OTP:
                        enter_new_data(UserEnterNewUserName, UserEnterNewPassword , Company)
                        st.success("User Successfully Registered 🎉")

                    elif user_otp == "":
                        st.error("Enter OTP First")
                    else:
                        print(st.session_state.OTP)
                        st.error("Invalid OTP")
def mail_varify(user_email, UserEnterNewPassword):
    otp = ""

    for i in range(6):
        otp += str(random.randint(0, 9))

    # print(otp)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    from_mail = "pateljimi53@gmail.com"
    server.login(from_mail, "zxyg xxom flgg fmmc")
    to_mail = user_email

    msg = EmailMessage()
    msg["Subject"] = "OTP Verification"
    msg["From"] = from_mail
    msg["To"] = to_mail
    msg.set_content(f" {UserEnterNewPassword} , Your OTP is :" + otp)
    server.send_message(msg)
    return otp


def check_user_company(Company):
    df = pd.read_excel("PROJECT_DATA.xlsx")
    rows , cols = df.shape
    is_data_in=False

    for i in range(0,rows):
        if df.iat[i,3] == Company:
            is_data_in = True
            break
    return is_data_in
def check_user_data(UserEnterUserName,UserEnterPassword):
    df = pd.read_excel("PROJECT_DATA.xlsx")
    rows , cols = df.shape
    is_data_in=False

    for i in range(0,rows):
        if df.iat[i,1]==UserEnterUserName:
            for j in range(0,cols):
                if df.iat[i,2]==int(UserEnterPassword):
                    is_data_in=True
                    break

    return is_data_in

def enter_new_data(UserEnterNewUserName,UserEnterNewPassword,Company):

    # Load existing data
    file_path = "PROJECT_DATA.xlsx"
    existing_data = pd.read_excel(file_path)
    rows , cols = existing_data.shape

    # New data
    new_data = pd.DataFrame([{'No': rows+1, 'User Name':UserEnterNewUserName, 'Password':int(UserEnterNewPassword),'Company':Company}])

    # Append and save
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.to_excel(file_path, index=False)


def check_user_availabel(UserEnterNewUserName):
    df = pd.read_excel("PROJECT_DATA.xlsx")
    rows, cols = df.shape
    is_data_in = False

    for i in range(0, rows):
        if df.iat[i, 1] == UserEnterNewUserName:
            is_data_in = True
            break

    return is_data_in

def main_app():
    with st.sidebar:
        selected = option_menu(
            "TrendVista",
            ["Home","Data Overview", "Revenue Forecasting", "Trend Analysis","Business Growth", "Sales Analysis","Best Sellers","Analytic Dashboard","About Us","Contact Us" ,"Log Out"],
            icons=["house","table", "graph-up-arrow", "activity","bar-chart-fill", "pie-chart-fill","trophy","clipboard", "people","telephone-fill","lock"],
            default_index=0,
            menu_icon="pie-chart",
            orientation="vertical",
            key="main_menu_key"
        )

        if selected != st.session_state.selected_menu:
            st.session_state.selected_menu = selected
            rerun()
    DEFAULT_FILE_PATH = "sales_data_large.csv"

    if st.session_state.selected_menu == "Home":
        st.toast("Welcome To TrendVista", icon="🎊")
        st.header(f"🎉 Welcome {st.session_state.UserEnterUserName}!")
        st.subheader("Welcome to the TrendVista! 🚀")
        st.image("picture ai.png" , width=1000)
        st.success("Explore the dashboard to make accurate business decisions based on reliable predictions.!")


    if st.session_state.selected_menu == "Data Overview":
        st.subheader(" 📂 Upload Your Historical Data Here!")
        uploaded_file = st.file_uploader("📤 Choose a file", type=["csv", "xlsx"])

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            file_extension = uploaded_file.name.split(".")[-1]

            if file_extension == "csv":
                st.session_state.df = pd.read_csv(uploaded_file)
            elif file_extension == "xlsx":
                st.session_state.df = pd.read_excel(uploaded_file)
        else:
            st.write("### No file uploaded. Loading default dataset...")
            file_extension = DEFAULT_FILE_PATH.split(".")[-1]

            if file_extension == "csv":
                st.session_state.df = pd.read_csv(DEFAULT_FILE_PATH)
            elif file_extension == "xlsx":
                st.session_state.df = pd.read_excel(DEFAULT_FILE_PATH)

        st.write("### 🔎 Preview of Data")
        st.dataframe(st.session_state.df)
    if st.session_state.selected_menu == "Revenue Forecasting":
        st.header("📊 Revenue Forecasting")
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df

            df.fillna(method='ffill', inplace=True)

            # Apply Label Encoding to categorical columns
            label_encoders = {}
            categorical_cols = ['Product', 'Region', 'City']
            object_columns = df.select_dtypes(include='object').columns

            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            # Select features (X) and target variable (y)
            X = df[['Product', 'Region', 'City', 'Quantity', 'Discount']]
            y = df['Revenue']  # <-- Target variable is Revenue

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Predict revenue for test set
            y_pred_test = rf_model.predict(X_test)

            # Compute model accuracy
            test_accuracy = r2_score(y_test, y_pred_test)

            # Streamlit UI

            st.title("📊 Revenue Forecasting")
            st.subheader("Predicted Revenue for Test Data")

            # Show sample of actual vs predicted revenue
            results_df = pd.DataFrame({"Actual Revenue": y_test.values, "Predicted Revenue": y_pred_test})
            st.write(results_df.head(20))  # Display first 20 rows

            # Show model accuracy
            st.write(f"**Model Accuracy (R² Score):** {test_accuracy:.2f}")
    if st.session_state.selected_menu == "Sales Analysis":
        st.header("📊 Sales Analysis")
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df

            if "selected_regions" not in st.session_state:
                st.session_state.selected_regions = []
            if "selected_cities" not in st.session_state:
                st.session_state.selected_cities = []
            if "selected_products" not in st.session_state:
                st.session_state.selected_products = []

            st.subheader("Choose Regions")
            all_regions = df["Region"].unique().tolist()
            selected_regions = st.multiselect("Select Regions", all_regions, default=st.session_state.selected_regions)

            filtered_df = df[df["Region"].isin(selected_regions)] if selected_regions else df
            st.subheader("Choose Cities")
            all_cities = filtered_df["City"].unique().tolist()
            selected_cities = st.multiselect("Select Cities", all_cities, default=st.session_state.selected_cities)

            st.subheader("Choose Products")
            all_products = filtered_df["Product"].unique().tolist()
            selected_products = st.multiselect("Select Products", all_products, default=st.session_state.selected_products)

            if st.button("Submit"):
                progress_bar=st.progress(0)

                for percent in range(1,101,10):
                    time.sleep(0.2)
                    progress_bar.progress(percent)

                st.session_state.selected_regions = selected_regions
                st.session_state.selected_cities = selected_cities
                st.session_state.selected_products = selected_products

            if st.session_state.selected_regions or st.session_state.selected_cities or st.session_state.selected_products:
                filtered_df = df[
                    df["Region"].isin(st.session_state.selected_regions)] if st.session_state.selected_regions else df
                city_filtered_df = filtered_df[filtered_df["City"].isin(
                    st.session_state.selected_cities)] if st.session_state.selected_cities else filtered_df
                product_filtered_df = city_filtered_df[city_filtered_df["Product"].isin(
                    st.session_state.selected_products)] if st.session_state.selected_products else city_filtered_df

                if st.session_state.selected_regions:
                    region_sales = filtered_df.groupby("Region")["Sales"].sum()
                    st.write("### Sales Data (Region-wise)")
                    st.dataframe(region_sales)

                    fig, ax = plt.subplots(figsize=(8, 5))
                    region_sales.plot(kind="bar", ax=ax, color="skyblue")
                    ax.set_ylabel("Total Sales")
                    ax.set_title("Region-wise Sales Distribution")
                    st.pyplot(fig)

                if st.session_state.selected_cities:
                    if not city_filtered_df.empty:
                        city_sales = city_filtered_df.groupby("City")["Sales"].sum()
                        st.write("### Sales Data (City-wise)")
                        st.dataframe(city_sales)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        city_sales.plot(kind="bar", ax=ax, color="orange")
                        ax.set_ylabel("Total Sales")
                        ax.set_title("City-wise Sales Distribution")
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for the selected cities.")

                if st.session_state.selected_products:
                    if not product_filtered_df.empty:
                        product_sales = product_filtered_df.groupby("Product")["Sales"].sum()
                        st.write("### Sales Data (Product-wise)")
                        st.dataframe(product_sales)

                        fig, ax = plt.subplots(figsize=(8, 5))
                        product_sales.plot(kind="bar", ax=ax, color="green")
                        ax.set_ylabel("Total Sales")
                        ax.set_title("Product-wise Sales Distribution")
                        st.pyplot(fig)
                    else:
                        st.warning("No data available for the selected products.")


    if st.session_state.selected_menu == "Best Sellers":
        st.header("📈🔥 Visualize the Best sellers")
        if "df" in st.session_state and st.session_state.df is not None:
            df = st.session_state.df

            with st.form("best_sellers_form"):
                region = st.selectbox("Select Region", df["Region"].unique())
                city = st.selectbox("Select City", df[df["Region"] == region]["City"].unique())
                top_n = st.slider("Select Top N Products", min_value=1, max_value=100, value=10)

                submit_button = st.form_submit_button("Analyze")

            if submit_button:
                # Filter data based on selected region and city
                filtered_df = df[(df["Region"] == region) & (df["City"] == city)]

                # Aggregate total sales and total price per product
                top_products = (
                    filtered_df.groupby("Product", as_index=False)
                    .agg(total_sales=("Price", "sum"), total_profit=("Profit", "sum"))
                    .nlargest(top_n, "total_sales")  # Sorting by total sales
                )

                # Display the table of top products
                st.subheader(f"📋 Top {top_n} Products in {city}, {region}")
                st.dataframe(top_products.rename(
                    columns={"product": "Product", "total_sales": "Sales", "total_profit": "Profit"}))
                # Filter data based on selected region and city
                filtered_df = df[(df["Region"] == region) & (df["City"] == city)]

                # Ensure correct column names for aggregation
                top_products = (
                    filtered_df.groupby("Product", as_index=False)["Price"].sum().nlargest(top_n, "Price")
                )
                top_profits = (
                    filtered_df.groupby("Product", as_index=False)["Profit"].sum().nlargest(top_n, "Profit")
                )

                # Pie chart for sales distribution
                st.subheader("🚀💰 Top Products By Sales")
                fig_sales = px.pie(
                    top_products, names="Product", values="Price"
                )
                st.plotly_chart(fig_sales, use_container_width=True)

                # Pie chart for profit distribution
                st.subheader("🚀💰 Top Products By Profit")
                fig_profit = px.pie(
                    top_profits, names="Product", values="Profit"
                )
                st.plotly_chart(fig_profit, use_container_width=True)

    if st.session_state.selected_menu == "Analytic Dashboard":
        st.header(f"🎉 Analytic Dashboard")
        st.subheader(st.session_state.Company)

    if st.session_state.selected_menu == "About Us":
        st.header("🚀AI-Driven Revenue Forecasting📊 and Trend Analysis  📈for Business Growth 💰")
        st.subheader(
            "T.R.E.N.D.V.I.S.T.A. – Technology-driven Revenue Estimation & Next-gen Data Visualization, Insights, Strategy, & Trend Analysis")
        st.markdown(
            """
            **Project Description/Abstract:** 
            AI-Driven Revenue Forecasting and Trend Analysis for Business 
            Growth" is a machine learning project that predicts future revenue trends by analyzing historical data. 
            It identifies patterns, detects anomalies, and forecasts revenue. The system accounts for seasonal 
            fluctuations, market conditions, and consumer behavior, providing businesses with actionable insights. 
            It aids in financial planning, risk mitigation, and strategic resource allocation. The solution is adaptable 
            to various industries such as retail, e-commerce, and SaaS. Ultimately, it enables data-driven decision
            making for sustained business growth.

            **Problems in the Existing System** 
            1. Inaccurate Revenue Predictions – Traditional forecasting methods lack precision. 
            2. Limited Consideration of Market Trends – Many forecasting models ignore external factors 
               like economic shifts. 
            3. Inability to Detect Anomalies – Unexpected market disruptions remain unaccounted for. 
            4. Manual Data Analysis – Businesses rely on time-consuming and error-prone manual 
               calculations. 
            5. Poor Financial Planning – Lack of accurate revenue forecasts leads to ineffective budgeting 
               and resource allocation.

            **Purpose of the Project** 
            • To predict future revenue trends using machine learning. 
            • To analyze historical financial data and identify revenue patterns. 
            • To detect anomalies and fluctuations for proactive decision-making. 
            • To provide accurate and actionable financial insights for business growth. 
            • To support strategic resource allocation and risk mitigation.   

            **Functional Requirements**
            1. Data Ingestion & Preprocessing – Collects, cleans, and normalizes historical revenue data. 
            2. Trend Analysis & Visualization – Generates charts and insights on revenue trends. 
            3. Revenue Forecasting Model – Uses ML algorithms to predict future revenue. 
            4. Anomaly Detection – Identifies outliers and revenue discrepancies. 
            5. Report Generation – Provides detailed financial summaries for stakeholders. 
            6. Industry-Specific Adaptability – Customizable for various business domains. 
            7. User Dashboard – Interactive UI displaying real-time analytics and insights.

            **System Modules**
            1. Data Collection Module – Gathers historical revenue and market data. 
            2. Preprocessing & Feature Engineering – Cleans and prepares data for analysis. 
            3. Machine Learning Model – Predicts revenue trends and detects anomalies. 
            4. Visualization & Reporting Module – Displays graphs, reports, and insights. 
            5. User Management Module – Allows different user roles (admin, analyst, manager).

            **System Requirements**

            *Hardware Requirements:* \n
               • Processor: Intel i5 or higher 
               • RAM: 8GB minimum 
               • Storage: 250GB SSD or more 
               • Internet Connectivity: Stable broadband connection 
            *Software Requirements:* \n
               • Operating System: Windows 
               • Pycharm , python 
               • Required AI Libraries 

            **Front End and Back End of System**\n
               • Front End (Client-Side): StreamLit \n
               • Back End (Server-Side): Python , Machine Learning Models , AI models   
            """
        )

    if st.session_state.selected_menu == "Contact Us":
        st.header("☎️Contact Us")
        st.markdown(
            """ We’d love to hear from you! Whether you have questions about our AI-driven revenue forecasting system, need support, or just want to share feedback, feel free to reach out.  
### 📧 Email Us  
For inquiries, collaborations, or support, email us at:  
**✉️ pateljimi53@gmail.com.com**  

### 📞 Call Us  
Prefer speaking with someone? Give us a call:  
**📱 +91 - 6351614963**  

### 💬 Stay Connected  
Follow us on social media for the latest updates and insights:  
- 🔹 **[LinkedIn](https://www.linkedin.com/in/jimi-patel-43a763228?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)**  
- 🔹 **[Twitter](#)**  
- 🔹 **[Instagram](#)**  

We look forward to connecting with you! 😊    
            """)

    if st.session_state.selected_menu == "Log Out":
        st.session_state.authenticated = False
        st.query_params["auth"] = "false"
        rerun()

# Display the appropriate section based on authentication status
if st.session_state.authenticated:
    main_app()
else:
    side_bar()






