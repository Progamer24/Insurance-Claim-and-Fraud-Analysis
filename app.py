import streamlit as st
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import os
import joblib
from textblob import TextBlob
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import logging
import io

# ---------- Setup ----------
DB_PATH = "users.db"
SUBMISSION_CSV = os.path.join("ml_model", "submission.csv")
MODEL_PATH = os.path.join("ml_model", "model.pkl")
LOG_PATH = "app.log"
FRAUD_THRESHOLD = 3  # Number of fraud claims before auto-ban
PDF_EXPORT_PATH = "claims_report.pdf"

# Configure logging
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Database Functions ----------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def create_users_table():
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone()
    
    expected_columns = {
        'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
        'username': 'TEXT UNIQUE NOT NULL',
        'password': 'TEXT NOT NULL',
        'is_admin': 'INTEGER NOT NULL CHECK(is_admin IN (0,1))',
        'status': 'TEXT DEFAULT "active"',
        'fraud_count': 'INTEGER DEFAULT 0'
    }
    
    if table_exists:
        cursor.execute("PRAGMA table_info(users)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        if columns != expected_columns:
            cursor.execute("DROP TABLE users")
            logging.warning("Dropped users table due to incorrect schema")
            table_exists = False
    
    if not table_exists:
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER NOT NULL CHECK(is_admin IN (0,1)),
                status TEXT DEFAULT 'active',
                fraud_count INTEGER DEFAULT 0
            )
        ''')
        logging.info("Created users table")
    
    cursor.execute("SELECT id FROM users WHERE username=?", ("admin",))
    if cursor.fetchone() is None:
        cursor.execute("INSERT INTO users (username, password, is_admin, status, fraud_count) VALUES (?, ?, ?, ?, ?)",
                      ("admin", hash_password("Password"), 1, "active", 0))
        logging.info("Created default admin user")
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, is_admin):
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, is_admin, fraud_count) VALUES (?, ?, ?, ?)",
                      (username, hash_password(password), int(is_admin), 0))
        conn.commit()
        logging.info(f"User registered: {username}, Admin: {is_admin}")
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT is_admin, status, id FROM users WHERE username=? AND password=?",
                   (username, hash_password(password)))
    result = cursor.fetchone()
    conn.close()
    if result:
        logging.info(f"User login: {username}, Status: {result[1]}")
        return result[0], result[1], result[2]
    return None, None, None

def update_user_status(user_id, status):
    with get_connection() as conn:
        conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        conn.commit()
        logging.info(f"User ID {user_id} status updated to {status}")

def increment_fraud_count(user_id):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET fraud_count = fraud_count + 1 WHERE id = ?", (user_id,))
        conn.commit()
        cursor.execute("SELECT fraud_count, username FROM users WHERE id = ?", (user_id,))
        fraud_count, username = cursor.fetchone()
        if fraud_count >= FRAUD_THRESHOLD:
            update_user_status(user_id, "banned")
            logging.warning(f"User {username} auto-banned due to {fraud_count} fraud claims")
            return True, username
        return False, username

def get_user_list():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT id, username, is_admin, status, fraud_count FROM users", conn)

def add_admin_user(username, password):
    if len(password) < 6:
        st.error("Password must be at least 6 characters long.")
        return
    with get_connection() as conn:
        conn.execute("INSERT INTO users (username, password, is_admin, status, fraud_count) VALUES (?, ?, 1, 'active', 0)",
                     (username, hash_password(password)))
        conn.commit()
        logging.info(f"Admin added: {username}")

# ---------- Prediction Functions ----------
def normalize_probability(prob):
    """Ensure a probability is between 0 and 1"""
    prob = float(prob)
    if pd.isna(prob):
        return 0.0
    return max(0.0, min(1.0, prob))

def get_probability(features):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found.")
    
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(features)[0][1]
    
    # Validation check
    proba = normalize_probability(proba)
    if not 0 <= proba <= 1:
        logging.warning(f"Model returned out-of-range probability: {proba}")
    
    return proba

def validate_scores(df):
    """Ensure all scores are valid percentages"""
    if 'Score' not in df.columns:
        return df
    
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df['Score'] = df['Score'].fillna(0)
    df['Score'] = df['Score'].clip(0, 100)
    return df

def get_prediction_score(provider_id):
    if not os.path.exists(SUBMISSION_CSV):
        return {
            "score": 0,
            "sentiment": "Unknown",
            "ai_confidence": 0,
            "prediction": "Unknown"
        }
    try:
        df = pd.read_csv(SUBMISSION_CSV)
        df = validate_scores(df)
        
        if provider_id in df["Provider"].values:
            row = df[df["Provider"] == provider_id].iloc[0]
            return {
                "score": row.get("Score", 0),
                "sentiment": row.get("Sentiment", "Neutral"),
                "ai_confidence": row.get("Score", 0),
                "prediction": row.get("Label", "Unknown")
            }
        return {
            "score": 0,
            "sentiment": "Unknown",
            "ai_confidence": 0,
            "prediction": "Unknown"
        }
    except Exception as e:
        logging.error(f"Error in get_prediction_score: {str(e)}")
        return {
            "score": 0,
            "sentiment": "Unknown",
            "ai_confidence": 0,
            "prediction": "Unknown"
        }

def retrain_model():
    if not os.path.exists(SUBMISSION_CSV):
        return False
    try:
        df = pd.read_csv(SUBMISSION_CSV)
        df = validate_scores(df)
        
        if df.empty:
            return False
            
        if "Score" in df.columns and "Label" in df.columns:
            X = df.drop(columns=["Username", "Provider", "Score", "Label", "Timestamp", "Comment", "Sentiment", "AdminReview"], errors='ignore')
            y = df["Label"].map({"Fraud": 1, "Not Fraud": 0})
            
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                use_label_encoder=False,
                eval_metric='logloss',
                objective='binary:logistic'
            )
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)
            logging.info("Model retrained successfully")
            return True
        return False
    except Exception as e:
        logging.error(f"Error in retrain_model: {str(e)}")
        return False

def retrain_on_labeled_data():
    try:
        df = pd.read_csv(SUBMISSION_CSV)
        df = validate_scores(df)
        df = df[df['AdminReview'].isin(["Approve", "Deny"])]
        
        if df.empty:
            st.warning("No labeled data available for retraining.")
            logging.info("No labeled data for retraining")
            return False
            
        X = df[[
            "InscClaimAmtReimbursed_sum", "InscClaimAmtReimbursed_mean", "InscClaimAmtReimbursed_count",
            "DeductibleAmtPaid_sum", "DeductibleAmtPaid_mean", "Gender_nunique", "Race_nunique",
            "RenalDiseaseIndicator", "ChronicCond_Alzheimer_sum", "ChronicCond_Heartfailure_sum",
            "ChronicCond_KidneyDisease_sum", "ChronicCond_Cancer_sum", "ChronicCond_Diabetes_sum",
            "ChronicCond_Depression_sum"
        ]]
        y = df['AdminReview'].map({"Deny": 1, "Approve": 0})
        
        from xgboost import XGBClassifier
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic'
        )
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        logging.info("Model retrained on labeled data")
        return True
    except Exception as e:
        st.error(f"Error retraining model: {str(e)}")
        logging.error(f"Error in retrain_on_labeled_data: {str(e)}")
        return False

# ---------- PDF Export ----------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Insurance Claims Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def export_to_pdf(df):
    if df.empty:
        st.warning("No data to export.")
        return None
    try:
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 10)

        headers = ["Username", "Provider", "Score", "Label", "Timestamp", "AdminReview"]
        col_widths = [40, 40, 30, 30, 50, 30]
        pdf.set_fill_color(200, 220, 255)
        for header, width in zip(headers, col_widths):
            pdf.cell(width, 10, header, 1, 0, 'C', 1)
        pdf.ln()

        for _, row in df.iterrows():
            for header, width in zip(headers, col_widths):
                pdf.cell(width, 10, str(row[header]), 1, 0, 'C')
            pdf.ln()

        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        return pdf_output
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        logging.error(f"Error in export_to_pdf: {str(e)}")
        return None

# ---------- Authentication Interface ----------
def auth_interface():
    st.sidebar.title("🔐 Login or Register")
    action = st.sidebar.radio("Action", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if action == "Register":
        is_admin = st.sidebar.checkbox("Register as Admin")
        if st.sidebar.button("Register"):
            success, message = register_user(username, password, is_admin)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
    else:
        if st.sidebar.button("Login"):
            is_admin, status, user_id = login_user(username, password)
            if is_admin is not None:
                if status != 'active':
                    st.sidebar.error(f"Your account is {status}. Contact admin.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.is_admin = bool(is_admin)
                    st.session_state.user_id = user_id
                    st.success(f"Welcome {'Admin' if is_admin else 'User'}: {username}")
            else:
                st.sidebar.error("Invalid credentials")

# ---------- User Dashboard ----------
def user_dashboard():
    st.title("🧾 Insurance Claim Portal")
    st.info(f"Logged in as **{st.session_state['username']}**")

    tab1, tab2 = st.tabs(["Submit New Claim", "View Past Claim"])

    with tab1:
        st.header("Submit New Claim")
        with st.form("claim_form"):
            provider = st.text_input("Provider ID")
            features = {}
            cols = st.columns(3)
            fields = [
                ("InscClaimAmtReimbursed_sum", 0.0, "Reimbursed (Sum)"),
                ("InscClaimAmtReimbursed_mean", 0.0, "Reimbursed (Mean)"),
                ("InscClaimAmtReimbursed_count", 0, "Claim Count"),
                ("DeductibleAmtPaid_sum", 0.0, "Deductible Paid (Sum)"),
                ("DeductibleAmtPaid_mean", 0.0, "Deductible Paid (Mean)"),
                ("Gender_nunique", 0, "Gender Diversity"),
                ("Race_nunique", 0, "Race Diversity"),
                ("RenalDiseaseIndicator", 0, "Renal Indicator"),
                ("ChronicCond_Alzheimer_sum", 0, "Alzheimer"),
                ("ChronicCond_Heartfailure_sum", 0, "Heart Failure"),
                ("ChronicCond_KidneyDisease_sum", 0, "Kidney Disease"),
                ("ChronicCond_Cancer_sum", 0, "Cancer"),
                ("ChronicCond_Diabetes_sum", 0, "Diabetes"),
                ("ChronicCond_Depression_sum", 0, "Depression"),
            ]
            for idx, (label, default, display_name) in enumerate(fields):
                col = cols[idx % 3]
                if label == "RenalDiseaseIndicator":
                    features[label] = 1 if col.selectbox(display_name, ["No", "Yes"], key=f"field_{idx}") == "Yes" else 0
                elif "sum" in label or "mean" in label:
                    features[label] = col.number_input(display_name, min_value=0.0, step=0.01, key=f"field_{idx}")
                else:
                    features[label] = col.number_input(display_name, min_value=0, step=1, key=f"field_{idx}")

            submitted = st.form_submit_button("Submit Claim")
            if submitted:
                if not provider:
                    st.error("Provider ID is required.")
                    logging.error("Claim submission failed: Provider ID missing")
                    return
                df_input = pd.DataFrame([features])
                try:
                    # Get and validate probability score
                    raw_prob = get_probability(df_input)
                    score = normalize_probability(raw_prob)
                    confidence = round(score * 100, 2)  # Convert to percentage
                    
                    label = "Fraud" if score > 0.5 else "Not Fraud"
                    comment = f"This claim looks {'suspicious' if score > 0.75 else 'fine'}"
                    sentiment = TextBlob(comment).sentiment.polarity

                    new_entry = pd.DataFrame([{
                        "Username": st.session_state.username,
                        "Provider": provider,
                        "Score": confidence,
                        "Label": label,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Comment": comment,
                        "Sentiment": sentiment,
                        "AdminReview": "Pending",
                        **features
                    }])

                    if os.path.exists(SUBMISSION_CSV):
                        df = pd.read_csv(SUBMISSION_CSV)
                        df = validate_scores(df)
                        df = pd.concat([df, new_entry], ignore_index=True)
                    else:
                        df = new_entry

                    df.to_csv(SUBMISSION_CSV, index=False, float_format='%.2f')
                    st.success(f"Prediction: **{label}** (Confidence: {confidence}%)")
                    logging.info(f"Claim submitted by {st.session_state.username}: {label}, Score: {confidence}%")

                    if label == "Fraud":
                        banned, username = increment_fraud_count(st.session_state.user_id)
                        if banned:
                            st.error(f"User {username} has been banned due to repeated fraud claims.")
                            st.session_state.logged_in = False
                            st.session_state.username = None
                            st.session_state.is_admin = False
                            st.session_state.user_id = None
                except FileNotFoundError:
                    st.error("Model file not found. Please contact admin.")
                    logging.error("Model file not found during claim submission")
                except Exception as e:
                    st.error(f"Error submitting claim: {str(e)}")
                    logging.error(f"Error in claim submission: {str(e)}")

    with tab2:
        st.header("View Past Claim")
        provider_id = st.text_input("Enter Provider ID to view past claim")
        if st.button("Check Claim"):
            if provider_id:
                result = get_prediction_score(provider_id)
                if result["score"] != 0:
                    st.write(f"**Provider ID**: {provider_id}")
                    st.write(f"**Prediction**: {result['prediction']}")
                    st.write(f"**Confidence Score**: {result['ai_confidence']}%")
                    st.write(f"**Sentiment**: {result['sentiment']}")
                    logging.info(f"User {st.session_state.username} checked past claim for Provider {provider_id}")
                else:
                    st.warning(f"No claim found for Provider ID: {provider_id}")
                    logging.info(f"No claim found for Provider ID: {provider_id}")
            else:
                st.error("Please enter a valid Provider ID.")
                logging.error("View past claim failed: Provider ID missing")

# ---------- Admin Dashboard ----------
def admin_dashboard():
    st.title("🛡️ Admin Dashboard")
    tab = st.sidebar.radio("Admin Tools", ["Review Claims", "Manage Users", "Add Admin", "Retrain Model"])
    
    if tab == "Review Claims":
        if not os.path.exists(SUBMISSION_CSV):
            st.warning("No claims submitted yet.")
            logging.info("No submission.csv found for review claims")
            return
        try:
            df = pd.read_csv(SUBMISSION_CSV)
            df = validate_scores(df)
            
            if df.empty:
                st.warning("No claims available in submission.csv.")
                logging.info("submission.csv is empty")
                return
            
            st.subheader("Filter Claims")
            col1, col2, col3 = st.columns(3)
            with col1:
                review_filter = st.selectbox("Review Status", ["All", "Pending", "Reviewed"], index=0)
            with col2:
                label_filter = st.selectbox("Prediction Label", ["All", "Fraud", "Not Fraud"], index=0)
            with col3:
                min_score, max_score = st.slider("Confidence Score Range", 0, 100, (0, 100))
            
            if review_filter != "All":
                if review_filter == "Pending":
                    df = df[df["AdminReview"] == "Pending"]
                else:
                    df = df[df["AdminReview"].isin(["Approve", "Deny"])]
            
            if label_filter != "All":
                df = df[df["Label"] == label_filter]
            
            df = df[(df["Score"] >= min_score) & (df["Score"] <= max_score)]
            
            if df.empty:
                st.info("No claims match the selected filters.")
                return
            
            st.write(f"Showing {len(df)} claims")
            
            edited_df = st.data_editor(
                df[["Username", "Provider", "Score", "Label", "AdminReview", "Timestamp"]],
                column_config={
                    "AdminReview": st.column_config.SelectboxColumn(
                        "Admin Review",
                        help="Update the review status",
                        width="medium",
                        options=["Pending", "Approve", "Deny"],
                        required=True
                    ),
                    "Score": st.column_config.NumberColumn(
                        "Score (%)",
                        help="Confidence score",
                        format="%.1f %%",
                        min_value=0,
                        max_value=100
                    )
                },
                key="claims_editor",
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("Save Changes"):
                df.update(edited_df)
                df.to_csv(SUBMISSION_CSV, index=False, float_format='%.2f')
                st.success("Changes saved successfully!")
                logging.info("Admin review changes saved to submissions.csv")
                
        except Exception as e:
            st.error(f"Error loading submission.csv: {str(e)}")
            logging.error(f"Error in Review Claims: {str(e)}")

    elif tab == "Manage Users":
        users = get_user_list()
        st.subheader("All Users")
        search = st.text_input("Search username")
        if search:
            users = users[users["username"].str.contains(search, case=False, na=False)]
        st.dataframe(users)
        user_id = st.selectbox("Select user ID", users["id"])
        new_status = st.selectbox("Set Status", ["active", "frozen", "banned"])
        if st.button("Update Status"):
            update_user_status(user_id, new_status)
            st.success("User status updated.")
            logging.info(f"User ID {user_id} status updated to {new_status}")

    elif tab == "Add Admin":
        uname = st.text_input("New Admin Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Create Admin"):
            add_admin_user(uname, pwd)
            st.success("Admin added.")

    elif tab == "Retrain Model":
        if st.button("Retrain on Labeled Data"):
            if retrain_on_labeled_data():
                st.success("Model retrained and saved!")
            else:
                st.error("Retraining failed. Check data.")

# ---------- Analytics Dashboard ----------
def analytics_dashboard():
    st.title("📊 Analytics Dashboard")
    if not os.path.exists(SUBMISSION_CSV):
        st.warning("No submissions found.")
        logging.info("No submission.csv found for analytics")
        return
    
    try:
        df = pd.read_csv(SUBMISSION_CSV)
        df = validate_scores(df)
        
        if df.empty:
            st.warning("No submissions available.")
            logging.info("submission.csv is empty for analytics")
            return
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = df['Timestamp'].dt.date
        
        st.sidebar.subheader("Filters")
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[df['Date'].min(), df['Date'].max()],
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
        
        review_status = st.sidebar.multiselect(
            "Review Status",
            options=df['AdminReview'].unique(),
            default=df['AdminReview'].unique()
        )
        
        prediction_label = st.sidebar.multiselect(
            "Prediction Label",
            options=df['Label'].unique(),
            default=df['Label'].unique()
        )
        
        df = df[
            (df['Date'] >= date_range[0]) & 
            (df['Date'] <= date_range[1]) &
            (df['AdminReview'].isin(review_status)) &
            (df['Label'].isin(prediction_label))
        ]
        
        if df.empty:
            st.warning("No data matches the selected filters.")
            return
        
        st.subheader("Claims Data")
        st.dataframe(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Export to CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="claims_data.csv",
                mime='text/csv'
            )
        with col2:
            pdf = export_to_pdf(df)
            if pdf:
                st.download_button(
                    "Export to PDF",
                    data=pdf.getvalue(),
                    file_name="claims_report.pdf",
                    mime='application/pdf'
                )
        
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims", len(df))
        col2.metric("Fraud Claims", len(df[df['Label'] == 'Fraud']))
        col3.metric("Avg Confidence", f"{df['Score'].mean():.1f}%")
        col4.metric("Pending Reviews", len(df[df['AdminReview'] == 'Pending']))
        
        st.subheader("Trend Analysis")
        tab1, tab2, tab3 = st.tabs(["Daily Trend", "User Activity", "Score Distribution"])
        
        with tab1:
            daily_data = df.groupby('Date').agg({
                'Provider': 'count',
                'Score': 'mean',
                'Label': lambda x: (x == 'Fraud').sum()
            }).rename(columns={
                'Provider': 'ClaimCount',
                'Score': 'AvgScore',
                'Label': 'FraudCount'
            }).reset_index()
            
            fig = px.line(
                daily_data,
                x='Date',
                y=['ClaimCount', 'FraudCount', 'AvgScore'],
                title="Daily Claims Activity",
                labels={'value': 'Count/Score', 'variable': 'Metric'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            user_data = df.groupby('Username').agg({
                'Provider': 'count',
                'Label': lambda x: (x == 'Fraud').sum()
            }).rename(columns={
                'Provider': 'TotalClaims',
                'Label': 'FraudClaims'
            }).reset_index()
            
            fig = px.bar(
                user_data.sort_values('TotalClaims', ascending=False).head(10),
                x='Username',
                y=['TotalClaims', 'FraudClaims'],
                barmode='group',
                title="Top 10 Users by Claim Activity",
                labels={'value': 'Number of Claims'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = px.histogram(
                df,
                x='Score',
                color='Label',
                nbins=20,
                title="Confidence Score Distribution by Label",
                labels={'Score': 'Confidence Score (%)'},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Analysis")
        if st.checkbox("Show Fraud Pattern Analysis"):
            fraud_df = df[df['Label'] == 'Fraud']
            if not fraud_df.empty:
                st.write("Common characteristics of fraud claims:")
                
                num_features = [
                    'InscClaimAmtReimbursed_sum', 'InscClaimAmtReimbursed_mean',
                    'DeductibleAmtPaid_sum', 'DeductibleAmtPaid_mean',
                    'Gender_nunique', 'Race_nunique'
                ]
                
                fig = px.box(
                    fraud_df[num_features],
                    title="Distribution of Numeric Features for Fraud Claims",
                    labels={'value': 'Value', 'variable': 'Feature'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                cat_features = [
                    'RenalDiseaseIndicator', 'ChronicCond_Alzheimer_sum',
                    'ChronicCond_Heartfailure_sum', 'ChronicCond_KidneyDisease_sum',
                    'ChronicCond_Cancer_sum', 'ChronicCond_Diabetes_sum',
                    'ChronicCond_Depression_sum'
                ]
                
                cat_data = fraud_df[cat_features].sum().reset_index()
                cat_data.columns = ['Feature', 'Count']
                
                fig = px.bar(
                    cat_data,
                    x='Feature',
                    y='Count',
                    title="Chronic Condition Indicators in Fraud Claims",
                    labels={'Count': 'Number of Claims'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No fraud claims found in the filtered data.")
    
    except Exception as e:
        st.error(f"Error loading analytics data: {str(e)}")
        logging.error(f"Error in Analytics Dashboard: {str(e)}")

# ---------- Main ----------
def main():
    create_users_table()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.is_admin = False
        st.session_state.user_id = None

    if not st.session_state.logged_in:
        auth_interface()
    else:
        menu = ["User Dashboard", "Analytics"]
        if st.session_state.is_admin:
            menu.insert(1, "Admin Dashboard")

        choice = st.sidebar.selectbox("Navigation", menu)

        if choice == "User Dashboard":
            user_dashboard()
        elif choice == "Admin Dashboard":
            admin_dashboard()
        elif choice == "Analytics":
            analytics_dashboard()

if __name__ == "__main__":
    main()