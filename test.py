import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Thiết lập cấu hình giao diện
st.set_page_config(page_title="Customer RFM Prediction", layout="wide", page_icon="📊")

# CSS để tùy chỉnh giao diện
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTextInput>div>input {
        border-radius: 8px;
        padding: 8px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .header {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load mô hình
@st.cache_resource
def load_model():
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.error("File model.pkl không tồn tại!")
        return None
    return joblib.load(model_path)

# Load dữ liệu CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("rfm.csv")  # Thay bằng tên file CSV của bạn
        return df
    except FileNotFoundError:
        st.error("File CSV không tồn tại!")
        return None
class_to_label = {
    0: 'At_risk',
    1: 'Hibernate',
    2: 'High_spender',
    3: 'Regular',
    4: 'Loyal'
}
model = load_model()
df = load_data()
df['Member_number'] = df['Member_number'].astype(str)  # Đảm bảo Member_number là kiểu chuỗi
df ['Class'] = model.predict(df[['Recency', 'Frequency', 'Monetary']]).astype(int)
df ['Class'] = df['Class'].map(class_to_label)

# Header
st.markdown('<div class="header"><h1>Customer RFM Prediction Dashboard</h1></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Chọn chức năng:", ["Random 5 Customers", "Predict RFM", "Add New Customers", "Search Customer"])

# Chức năng 1: Hiển thị ngẫu nhiên 5 khách hàng
if menu == "Random 5 Customers":
    st.subheader("Random 5 Customers")
    if st.button("Show random 5 members"):
        if df is not None:
            random_customers = df.sample(5, random_state=None)
            st.dataframe(random_customers, use_container_width=True)
        else:
            st.warning("Không thể tải dữ liệu khách hàng.")

# Chức năng 2: Dự đoán RFM từ input người dùng
elif menu == "Predict RFM":
    st.subheader("Predict Customer Class")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        r = st.number_input("Recency (R)", min_value=0, value=0, step=1)
    with col2:
        f = st.number_input("Frequency (F)", min_value=0, value=0, step=1)
    with col3:
        m = st.number_input("Monetary (M)", min_value=0, value=0, step=1)
    
    if st.button("Predict"):
        if model is not None:
            try:
                input_data = {'Recency': [r], 'Frequency': [f], 'Monetary': [m]}
                input_df = pd.DataFrame(input_data)
                input_df = input_df[['Recency', 'Frequency', 'Monetary']]
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted Class: {class_to_label[int(prediction)]}")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
        else:
            st.error("Mô hình chưa được tải.")

# Chức năng 3: Thêm và dự đoán cho 5 khách hàng mới
elif menu == "Add New Customers":
    st.subheader("Add and Predict for New Customers")
    new_customers = []
    tmp_member_ids = []
    for i in range(5):
        st.markdown(f"**Customer {i+1}**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            member_id = st.text_input(f"Member ID {i+1}", key=f"id_{i}")
            if member_id == "":
                st.warning("Vui lòng nhập Member ID.")
                continue
            
            if member_id in df['Member_number'].values or (member_id in tmp_member_ids):
                st.warning(f"Member ID {member_id} đã tồn tại trong dữ liệu.")
                continue
            tmp_member_ids.append(member_id)
            # elif member_id == "":
            #     st.warning("Vui lòng nhập Member ID.")
            #     continue
            # else:
            #     pass
        with col2:
            r = st.number_input(f"R {i+1}", min_value=0, value=0, step=1, key=f"r_{i}")
        with col3:
            f = st.number_input(f"F {i+1}", min_value=0, value=0, step=1, key=f"f_{i}")
        with col4:
            m = st.number_input(f"M {i+1}", min_value=0, value=0, step=1, key=f"m_{i}")
        new_customers.append({'Member_number':member_id,'Recency': r, 'Frequency': f, 'Monetary': m})
    new_df = pd.DataFrame(new_customers)
    df = pd.concat([df, new_df],axis=0, ignore_index=True)
    if st.button("Predict for New Customers"):
        if model is not None:
            try:
                predictions = model.predict(new_df[['Recency', 'Frequency', 'Monetary']])
                new_df['Predicted Class'] = [class_to_label[int(p)] for p in predictions]
                st.success("Dự đoán thành công!")
                st.write("**New Customers with Predictions:**")
                st.dataframe(new_df, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
        else:
            st.error("Mô hình chưa được tải.")

# Chức năng 4: Tìm kiếm khách hàng và dự đoán class
elif menu == "Search Customer":
    st.subheader("Search Customer by Member ID")
    search_id = st.text_input("Enter Member ID")
    
    if st.button("Search"):
        if df is not None and model is not None:
            customer = df[df['Member_number'] == search_id]
            if not customer.empty:
                rfm = customer[['Recency', 'Frequency', 'Monetary']]
                prediction = model.predict(rfm)[0]
                       
                st.write("**Customer Details:**")
                st.dataframe(customer[['Member_number', 'Recency', 'Frequency', 'Monetary']], use_container_width=True)
                st.success(f"Predicted Class: {class_to_label[int(prediction)]}")
            else:
                st.warning("Không tìm thấy khách hàng với ID này.")
        else:
            st.error("Dữ liệu hoặc mô hình chưa được tải.")

# Footer
st.markdown("---")
st.markdown("**CSC EDU**")