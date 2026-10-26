import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

st.title("🧪 ML Classifier Studio")
st.sidebar.header("Cấu hình mô hình")

# 1. Chọn Dataset & Thuật toán
dataset_name = st.sidebar.selectbox("1. Chọn Dataset", ("Iris", "Wine"))
classifier_name = st.sidebar.selectbox("2. Chọn thuật toán", ("KNN", "SVM", "Random Forest"))

# 2. Caching dữ liệu
@st.cache_data
def load_dataset(name):
  return datasets.load_iris() if name == "Iris" else datasets.load_wine()

data = load_dataset(dataset_name)
X, y = data.data, data.target

# 3. Dynamic Hyperparameters trên Sidebar
params = {}
if classifier_name == "KNN":
  params["K"] = st.sidebar.slider("K (Láng giềng)", 1, 15, 5)
  clf = KNeighborsClassifier(n_neighbors=params["K"])
elif classifier_name == "SVM":
  params["C"] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
  clf = SVC(C=params["C"])
else:
  params["trees"] = st.sidebar.slider("Số lượng cây", 10, 100, 50, step=10)
  clf = RandomForestClassifier(n_estimators=params["trees"], random_state=42)

# 4. Huấn luyện & Đánh giá
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))

col1, col2 = st.columns(2)
col1.metric("Thuật toán", classifier_name)
col2.metric("Độ chính xác Test", f"{acc * 100:.1f}%")

# 5. Trực quan hóa không gian dữ liệu bằng PCA 2D
pca = PCA(2)
X_projected = pca.fit_transform(X)
fig, ax = plt.subplots()
scatter = ax.scatter(X_projected[:, 0], X_projected[:, 1], c=y, alpha=0.8, cmap="viridis")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
st.pyplot(fig)