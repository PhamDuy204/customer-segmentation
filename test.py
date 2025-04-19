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
def load_data():
    try:
        df = pd.read_csv("rfm.csv")
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

# Kiểm tra nếu không tải được dữ liệu
if df is None:
    st.error("Không thể tải dữ liệu khách hàng. Vui lòng kiểm tra file rfm.csv và thử lại.")
    st.stop()  # Dừng toàn bộ ứng dụng nếu dữ liệu không tải được

# Xử lý dữ liệu
df['Member_number'] = df['Member_number'].astype(str)
df['Class'] = model.predict(df[['Recency', 'Frequency', 'Monetary']]).astype(int)
df['Class'] = df['Class'].map(class_to_label)

# Khởi tạo max_id trong session_state
if "value" not in st.session_state:
    st.session_state.value = df['Member_number'].astype(int).max()

min_id = df['Member_number'].astype(int).min()
max_id = df['Member_number'].astype(int).max()
# Header
st.markdown('<div class="header"><h1>Customer RFM Prediction Dashboard</h1></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Chọn chức năng:", ["Homepage", "Predict RFM", "Add New Customers", "Search Customer"])

# Chức năng 1: Homepage
if menu == "Homepage":
    st.subheader("Homepage")
    
    # Phần Overview
    st.markdown("""
    <div class="overview">
        <h2>Giới thiệu</h2>
        <p><strong>Thành viên dự án:</strong> Phạm Đình Anh Duy, Châu Nhật Minh</p>
        <p><strong>Mô tả trang web:</strong> Trang web này phục vụ các chủ cửa hàng có thể phân loại khách hàng thành các phân khúc dựa trên các chỉ số RFM (Recency, Frequency, Monetary). Trong dự án này, chúng tôi sử dụng thuật toán K-Means để phân khách hàng thành 5 phân khúc (At_risk, Hibernate, High_spender, Regular, Loyal) và áp dụng Decision Tree để đưa ra các rule nhằm phân loại khách hàng một cách chính xác.</p>
        <p><strong>Các chức năng chính:</strong></p>
        <ul>
            <li><strong>Predict RFM:</strong> Dự đoán phân khúc khách hàng dựa trên các giá trị RFM nhập vào.</li>
            <li><strong>Add New Customers:</strong> Thêm khách hàng mới và dự đoán phân khúc của họ.</li>
            <li><strong>Search Customer:</strong> Tìm kiếm và xem thông tin chi tiết của khách hàng theo Member ID.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Phần hình ảnh demo
    st.markdown("### Demo các chức năng")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("./public/predict_rfm_demo.png", use_container_width=True)
    with col2:
        st.image("./public/add_customers_demo.png", use_container_width=True)
    with col3:
        st.image("./public/search_customer_demo.png", use_container_width=True)
    
    # Phần hiển thị ngẫu nhiên 5 khách hàng
    st.markdown("### Ngẫu nhiên 5 khách hàng")
    if st.button("Hiển thị ngẫu nhiên 5 khách hàng"):
        random_customers = df.sample(5, random_state=None)
        st.dataframe(random_customers, use_container_width=True)

# Chức năng 2: Dự đoán RFM từ input người dùng
elif menu == "Predict RFM":
    st.subheader("Predict Customer Class")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        r = st.number_input("Recency (R)", min_value=1, max_value=1000, value=1, step=1)
    with col2:
        f = st.number_input("Frequency (F)", min_value=1, max_value=90, value=1, step=1)
    with col3:
        m = st.number_input("Monetary (M)", min_value=1.0, max_value=2000.0, value=1.0, step=0.01)
    
    if st.button("Predict"):
        if model is not None:
            try:
                input_data = {'Recency': [r], 'Frequency': [f], 'Monetary': [m]}
                input_df = pd.DataFrame(input_data)
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted Class: {class_to_label[int(prediction)]}")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
        else:
            st.error("Mô hình chưa được tải.")

# Chức năng 3: Thêm và dự đoán cho khách hàng mới
elif menu == "Add New Customers":
    st.subheader("Add and Predict for New Customers")
    
    # Hiển thị phạm vi Member_number hiện tại
    st.info(f"Current Member_number range: {min_id} to {st.session_state.value}")

    # Thanh trượt chọn số lượng khách hàng
    num_customers = st.slider("Chọn số lượng khách hàng muốn thêm", min_value=1, max_value=10, value=5)
    new_customers = []
    for i in range(num_customers):
        st.markdown(f"**Customer {i+1}**")
        # Tự động sinh MemberID mới
        new_member_id = st.session_state.value + 1 + i
        st.write(f"Member ID: {new_member_id}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.number_input(f"R {i+1}", min_value=1, max_value=1000, value=1, step=1, key=f"r_{i}")
        with col2:
            f = st.number_input(f"F {i+1}", min_value=1, max_value=90, value=1, step=1, key=f"f_{i}")
        with col3:
            m = st.number_input(f"M {i+1}", min_value=1.0, max_value=2000.0, value=1.0, step=0.01, key=f"m_{i}")
        
        new_customers.append({'Member_number': str(new_member_id), 'Recency': r, 'Frequency': f, 'Monetary': m})
    
    if st.button("Add and Predict"):
        try:
            new_df = pd.DataFrame(new_customers)
            predictions = model.predict(new_df[['Recency', 'Frequency', 'Monetary']])
            new_df['Class'] = [class_to_label[int(p)] for p in predictions]
            df = pd.concat([df, new_df], axis=0, ignore_index=True)
            df.to_csv("rfm.csv", index=False)
            st.success("Thêm và dự đoán thành công!")
            st.write("**New Customers with Predictions:**")
            st.dataframe(new_df, use_container_width=True)

        except Exception as e:
            st.error(f"Lỗi khi thêm và dự đoán: {e}")
    if st.button("Update Member ID"):
        st.session_state.value += num_customers
        st.success(f"Cập nhật Member ID thành công! Giá trị mới là: {st.session_state.value}")
        st.rerun()

# Chức năng 4: Tìm kiếm khách hàng và dự đoán class
elif menu == "Search Customer":
    df = load_data()
    st.subheader("Search Customer by Member ID")
    member_ids = list(range(min_id,st.session_state.value + 1))
    selected_id = st.selectbox("Select Member ID", member_ids)
    if st.button("Search"):
        df['Member_number'] = df['Member_number'].astype(str)
        customer = df[df['Member_number'] == str(selected_id)]
        if not customer.empty:
            rfm = customer[['Recency', 'Frequency', 'Monetary']]
            prediction = model.predict(rfm)[0]
            st.write("**Customer Details:**")
            st.dataframe(customer[['Member_number', 'Recency', 'Frequency', 'Monetary', 'Class']], use_container_width=True)
            st.success(f"Predicted Class: {class_to_label[int(prediction)]}")
        else:
            st.warning("Không tìm thấy khách hàng với ID này.")

# Footer
st.markdown("---")
st.markdown("**CSC EDU**")