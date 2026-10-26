# import streamlit as st
# st.title("Tiêu đề chính")
# st.header("Tiêu đề mục lớn")
# st.subheader("Tiêu đề phụ")
# st.caption("Ghi chú/Chú thích")
# st.write("Đa năng:", {"status": "OK"})

# # Cách 1: Ép kiểu Dictionary về String và nối chuỗi
# st.write("Đa năng: " + str({"status": "OK"}))

# # Cách 2: Sử dụng f-string (Lưu ý dùng ngoặc nhọn kép {{}} cho JSON bên trong f-string)
# st.write(f"Đa năng: {{'status': 'OK'}}")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# 1. Thiết lập cấu hình trang (BẮT BUỘC là lệnh Streamlit đầu tiên)
st.set_page_config(
    page_title="Data Mining & AI Web App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Hệ thống Khai phá Dữ liệu & Huấn luyện Mô hình AI")
# st.caption("Ứng dụng Web tương tác toàn diện xây dựng bằng Streamlit & Scikit-Learn")

# ==========================================
# HÀM TỐI ƯU HÓA BỘ NHỚ ĐỆM (CACHING)
# ==========================================
@st.cache_data
def load_default_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = [iris.target_names[i] for i in iris.target]
    return df

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

# ==========================================
# THANH BÊN (SIDEBAR - CONTROL PANEL)
# ==========================================
st.sidebar.header("⚙️ Bảng Điều Khiển")

# Chọn nguồn dữ liệu
st.sidebar.subheader("1. Nguồn dữ liệu")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV của bạn", type=['csv'])

if uploaded_file is not None:
    df_raw = load_csv(uploaded_file)
    st.sidebar.success("Đã tải file CSV thành công!")
else:
    df_raw = load_default_data()
    st.sidebar.info("Đang sử dụng tập dữ liệu mẫu: Iris Dataset (150 mẫu)")

# Tùy chọn tiền xử lý dữ liệu
st.sidebar.subheader("2. Tiền xử lý Dữ liệu")
missing_strategy = st.sidebar.radio(
    "Xử lý dữ liệu khuyết (Missing Data):",
    ["Dropna (Xóa dòng lỗi)", "Fill Mean (Điền trung bình)"]
)
use_scaler = st.sidebar.checkbox("Chuẩn hóa dữ liệu (StandardScaler)", value=True)

# Tùy chọn mô hình & siêu tham số
st.sidebar.subheader("3. Cấu hình Mô hình ML")
model_type = st.sidebar.selectbox(
    "Chọn thuật toán:",
    ["Random Forest", "Logistic Regression", "SVM"]
)

if model_type == "Random Forest":
    n_est = st.sidebar.slider("Số lượng cây (n_estimators)", 10, 500, 100, step=10)
    max_d = st.sidebar.slider("Độ sâu tối đa (max_depth)", 1, 20, 10)
elif model_type == "SVM":
    c_param = st.sidebar.slider("Hệ số điều chuẩn (C)", 0.1, 10.0, 1.0)
else:
    max_iter = st.sidebar.slider("Số vòng lặp tối đa (max_iter)", 100, 1000, 200)

test_size = st.sidebar.slider("Tỉ lệ tập Test (%)", 10, 50, 20) / 100.0

# ==========================================
# BƯỚC 1: KHÁM PHÁ DỮ LIỆU (EDA)
# ==========================================
st.header("Bước 1: Khám phá Dữ liệu (EDA)")

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Số dòng (Samples)", df_raw.shape[0])
col_info2.metric("Số cột (Columns)", df_raw.shape[1])
col_info3.metric("Giá trị bị khuyết (Missing)", int(df_raw.isnull().sum().sum()))

col_eda1, col_eda2 = st.columns([3, 2], gap="medium")

with col_eda1:
    st.subheader("Bản xem trước dữ liệu")
    st.dataframe(df_raw.head(10), use_container_width=True)

with col_eda2:
    st.subheader("Trực quan hóa nhanh")
    selected_col = st.selectbox("Chọn cột để vẽ biểu đồ:", df_raw.columns)
    st.bar_chart(df_raw[selected_col].value_counts() if df_raw[selected_col].dtype == 'object' else df_raw[selected_col])

with st.expander("🔍 Xem thống kê mô tả chi tiết"):
    st.write(df_raw.describe())

st.divider()

# ==========================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (PREPROCESSING)
# ==========================================
st.header("Bước 2: Tiền xử lý & Chọn Đặc trưng")

df_clean = df_raw.copy()
if missing_strategy.startswith("Dropna"):
    df_clean = df_clean.dropna()
else:
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())

col_prep1, col_prep2 = st.columns(2)
with col_prep1:
    target_col = st.selectbox("🎯 Chọn cột Nhãn dự đoán (Target):", df_clean.columns, index=len(df_clean.columns) - 1)

numeric_columns = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c != target_col]

with col_prep2:
    feature_cols = st.multiselect(
        "📊 Chọn các cột Đặc trưng số (Features):",
        options=numeric_columns,
        default=numeric_columns
    )

if not feature_cols:
    st.warning("⚠️ Vui lòng chọn ít nhất 1 cột đặc trưng số để tiếp tục!")
    st.stop()

X = df_clean[feature_cols].copy()
y = df_clean[target_col].copy()

scaler = StandardScaler()
if use_scaler:
    X_processed = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
else:
    X_processed = X.copy()

with st.expander("Xem dữ liệu Đặc trưng (X) sau khi Tiền xử lý"):
    st.dataframe(X_processed.head(), use_container_width=True)

st.divider()

# ==========================================
# BƯỚC 3: HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING)
# ==========================================
st.header("Bước 3: Huấn luyện Mô hình Machine Learning")

if st.button("▶ Bắt đầu Huấn luyện Mô hình", type="primary"):
    with st.spinner("Mô hình đang được huấn luyện..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=42
        )

        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
        elif model_type == "SVM":
            model = SVC(C=c_param, probability=True, random_state=42)
        else:
            model = LogisticRegression(max_iter=max_iter, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Lưu trạng thái vào Session State để duy trì qua các lần rerun
        st.session_state.model = model
        st.session_state.scaler = scaler if use_scaler else None
        st.session_state.feature_cols = feature_cols
        st.session_state.raw_X = X
        st.session_state.metrics = {
            "acc": accuracy_score(y_test, y_pred),
            "prec": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "rec": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "cm": confusion_matrix(y_test, y_pred),
            "labels": model.classes_
        }
        st.success(f"🎉 Đã huấn luyện thành công mô hình **{model_type}**!")

# ==========================================
# BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH (EVALUATION)
# ==========================================
if 'model' in st.session_state and st.session_state.feature_cols == feature_cols:
    st.divider()
    st.header("Bước 4: Trực quan hóa Đánh giá Mô hình")
    m = st.session_state.metrics

    # Hiển thị thẻ KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy (Độ chính xác)", f"{m['acc']*100:.2f}%")
    k2.metric("Precision", f"{m['prec']*100:.2f}%")
    k3.metric("Recall", f"{m['rec']*100:.2f}%")
    k4.metric("F1-Score", f"{m['f1']*100:.2f}%")

    # Hiển thị Confusion Matrix
    col_eval1, col_eval2 = st.columns([1, 1])
    with col_eval1:
        st.subheader("Ma trận Nhầm lẫn (Confusion Matrix)")
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=m['cm'], display_labels=m['labels'])
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        st.pyplot(fig)

    with col_eval2:
        if hasattr(st.session_state.model, "feature_importances_"):
            st.subheader("Mức độ Quan trọng của Đặc trưng (Feature Importance)")
            imp_df = pd.DataFrame({
                "Đặc trưng": feature_cols,
                "Độ quan trọng": st.session_state.model.feature_importances_
            }).set_index("Đặc trưng")
            st.bar_chart(imp_df)
        else:
            st.info("💡 Chọn thuật toán **Random Forest** để xem thêm biểu đồ mức độ quan trọng của từng đặc trưng (Feature Importance).")

    # ==========================================
    # BƯỚC 5: DỰ ĐOÁN THỜI GIAN THỰC (REAL-TIME PREDICTION)
    # ==========================================
    st.divider()
    st.header("Bước 5: Dự đoán Real-time trên Dữ liệu Mới")

    with st.form(key="predict_form"):
        st.write("Nhập thông số đặc trưng bên dưới và nhấn nút để AI dự đoán:")
        form_cols = st.columns(2)
        user_input = {}

        for idx, col_name in enumerate(feature_cols):
            with form_cols[idx % 2]:
                default_val = float(st.session_state.raw_X[col_name].mean())
                user_input[col_name] = st.number_input(
                    f"{col_name}",
                    value=round(default_val, 2)
                )

        submit_btn = st.form_submit_button("Dự đoán ngay")

    if submit_btn:
        input_df = pd.DataFrame([user_input])
        if st.session_state.scaler is not None:
            input_processed = st.session_state.scaler.transform(input_df)
        else:
            input_processed = input_df.values

        prediction = st.session_state.model.predict(input_processed)[0]
        st.success(f"**Kết quả dự đoán từ Mô hình AI:** `{prediction}`")