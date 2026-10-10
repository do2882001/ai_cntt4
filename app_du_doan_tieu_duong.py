from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVC


STUDENT_ID = "26DCN020"
STUDENT_NAME = "Trần Thủ Đô"
DATA_PATH = Path(__file__).with_name("CSDL_TieuDuong.csv")
TARGET = "NguyCoTieuDuong"
NUMERIC_COLUMNS = [
    "Tuoi",
    "Glucose",
    "HbA1C",
    "Cholesterol",
    "HDL",
    "LDL",
    "Triglycerid",
    "Creatinine",
    "Ure",
    "AcidUric",
    "AST",
    "ALT",
]
CATEGORICAL_COLUMNS = ["GioiTinh"]
PLAUSIBLE_RANGES = {
    "Tuoi": (0, 120),
    "Glucose": (0, 1000),
    "HbA1C": (2, 30),
    "Cholesterol": (0, 100),
    "HDL": (0, 20),
    "LDL": (0, 100),
    "Triglycerid": (0, 500),
    "Creatinine": (0, 5000),
    "Ure": (0, 500),
    "AcidUric": (0, 2000),
    "AST": (0, 5000),
    "ALT": (0, 5000),
}
INPUT_FIELDS = {
    "Tuoi": ("Tuổi", 0.0, 120.0, 49.0, 1.0),
    "Glucose": ("Glucose", 0.0, 1000.0, 5.6, 0.1),
    "HbA1C": ("HbA1C", 2.0, 30.0, 8.025, 0.1),
    "Cholesterol": ("Cholesterol", 0.0, 100.0, 5.5, 0.1),
    "HDL": ("HDL", 0.0, 20.0, 1.6, 0.1),
    "LDL": ("LDL", 0.0, 100.0, 3.45, 0.1),
    "Triglycerid": ("Triglycerid", 0.0, 500.0, 2.1, 0.1),
    "Creatinine": ("Creatinine", 0.0, 5000.0, 85.2, 0.1),
    "Ure": ("Ure", 0.0, 500.0, 5.3, 0.1),
    "AcidUric": ("Acid uric", 0.0, 2000.0, 426.2, 0.1),
    "AST": ("AST", 0.0, 5000.0, 25.0, 0.1),
    "ALT": ("ALT", 0.0, 5000.0, 21.6, 0.1),
}


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip outliers using IQR limits learned from the training partition."""

    def fit(self, values, y=None):
        first_quartile = np.nanpercentile(values, 25, axis=0)
        third_quartile = np.nanpercentile(values, 75, axis=0)
        spread = third_quartile - first_quartile
        self.lower_ = first_quartile - 1.5 * spread
        self.upper_ = third_quartile + 1.5 * spread
        return self

    def transform(self, values):
        return np.clip(values, self.lower_, self.upper_)


def clean_dataset(path):
    raw = pd.read_csv(path)
    required_columns = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET])
    missing_columns = required_columns.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Thiếu cột dữ liệu: {', '.join(sorted(missing_columns))}")

    data = raw[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET]].copy()
    data["_target"] = data[TARGET].astype("string").str.strip().str.casefold().map(
        {
            "true": 1,
            "1": 1,
            "1.0": 1,
            "có nguy cơ": 1,
            "co nguy co": 1,
            "false": 0,
            "0": 0,
            "0.0": 0,
            "không có nguy cơ": 0,
            "khong co nguy co": 0,
        }
    )
    unknown_labels = int(data["_target"].isna().sum())
    data = data.loc[data["_target"].notna()].copy()
    data[TARGET] = data.pop("_target").astype("int64")

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    scaled_hba1c = data["HbA1C"].between(100, 10000, inclusive="right")
    scaled_hba1c_count = int(scaled_hba1c.sum())
    data.loc[scaled_hba1c, "HbA1C"] /= 1000

    invalid_value_count = 0
    for column, (minimum, maximum) in PLAUSIBLE_RANGES.items():
        invalid = data[column].notna() & ~data[column].between(minimum, maximum)
        invalid_value_count += int(invalid.sum())
        data.loc[invalid, column] = np.nan

    gender_map = {
        "nam": "Nam",
        "male": "Nam",
        "nữ": "Nữ",
        "nu": "Nữ",
        "female": "Nữ",
    }
    data["GioiTinh"] = (
        data["GioiTinh"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .map(gender_map)
    )

    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().reset_index(drop=True)
    features = data[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    labels = data[TARGET]
    diagnostics = {
        "raw_rows": len(raw),
        "clean_rows": len(data),
        "missing_values": int(features.isna().sum().sum()),
        "duplicate_rows": duplicate_count,
        "unknown_labels": unknown_labels,
        "invalid_numeric_values": invalid_value_count,
        "scaled_hba1c_values": scaled_hba1c_count,
        "class_counts": labels.value_counts().sort_index().to_dict(),
    }
    return features, labels, diagnostics, data


def make_pipeline(classifier):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier_handler", IQRClipper()),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


@st.cache_resource
def train_models():
    features, labels, diagnostics, cleaned_data = clean_dataset(DATA_PATH)
    train_features, held_features, train_labels, held_labels = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    validation_features, test_features, validation_labels, test_labels = train_test_split(
        held_features,
        held_labels,
        test_size=0.50,
        random_state=42,
        stratify=held_labels,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced", random_state=42
        ),
    }
    candidates["SVM"] = CalibratedClassifierCV(candidates["SVM"], cv=3)

    validation_results = []
    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)
        pipeline.fit(train_features, train_labels)
        predictions = pipeline.predict(validation_features)
        validation_results.append(
            {
                "Mô hình": name,
                "Accuracy": accuracy_score(validation_labels, predictions),
                "Recall": recall_score(validation_labels, predictions, zero_division=0),
                "F1-score": f1_score(validation_labels, predictions, zero_division=0),
            }
        )

    validation_table = pd.DataFrame(validation_results).sort_values(
        "F1-score", ascending=False
    )
    best_name = validation_table.iloc[0]["Mô hình"]
    final_model = make_pipeline(candidates[best_name])
    final_model.fit(
        pd.concat([train_features, validation_features]),
        pd.concat([train_labels, validation_labels]),
    )
    test_predictions = final_model.predict(test_features)
    test_metrics = {
        "Accuracy": accuracy_score(test_labels, test_predictions),
        "Recall": recall_score(test_labels, test_predictions, zero_division=0),
        "F1-score": f1_score(test_labels, test_predictions, zero_division=0),
        "Confusion matrix": confusion_matrix(
            test_labels, test_predictions, labels=[0, 1]
        ),
    }
    return final_model, best_name, validation_table, test_metrics, diagnostics, cleaned_data


st.set_page_config(page_title="Dự đoán nguy cơ tiểu đường", page_icon="🩺", layout="wide")
st.title("Dự đoán nguy cơ bệnh tiểu đường")
st.caption(f"Bài kiểm tra Trí tuệ nhân tạo | {STUDENT_ID} | {STUDENT_NAME}")
st.warning(
    "Công cụ phục vụ học tập, không thay thế chẩn đoán hoặc tư vấn của nhân viên y tế."
)

try:
    model, best_model_name, validation_table, test_metrics, diagnostics, cleaned_data = (
        train_models()
    )
except FileNotFoundError:
    st.error("Không tìm thấy CSDL_TieuDuong.csv trong thư mục chương trình.")
    st.stop()
except Exception as error:
    st.error("Không thể huấn luyện mô hình từ dữ liệu hiện tại.")
    st.exception(error)
    st.stop()

overview, evaluation, data_tab = st.tabs(
    ["Dự đoán", "Đánh giá mô hình", "Dữ liệu"]
)

with overview:
    st.subheader("Thông tin người bệnh")
    with st.form("patient_form"):
        columns = st.columns(4)
        patient_values = {}
        for index, (column, field) in enumerate(INPUT_FIELDS.items()):
            label, minimum, maximum, default, step = field
            with columns[index % len(columns)]:
                patient_values[column] = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=default,
                    step=step,
                    key=f"patient_{column}",
                )
        patient_values["GioiTinh"] = st.selectbox(
            "Giới tính", ["Nữ", "Nam"], key="patient_gender"
        )
        submitted = st.form_submit_button(
            "Dự đoán nguy cơ", type="primary", use_container_width=True
        )

    if submitted:
        patient = pd.DataFrame([patient_values], columns=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
        risk_probability = float(model.predict_proba(patient)[0, 1])
        st.metric("Điểm nguy cơ theo mô hình", f"{risk_probability:.1%}")
        st.progress(risk_probability)
        if risk_probability >= 0.5:
            st.error("Mô hình xếp trường hợp này vào nhóm có nguy cơ cao hơn.")
        else:
            st.success("Mô hình xếp trường hợp này vào nhóm có nguy cơ thấp hơn.")

        reference_signals = []
        if patient_values["Glucose"] >= 7:
            reference_signals.append(
                "Glucose từ 7.0 trở lên; mốc này chỉ có ý nghĩa nếu đơn vị và điều kiện đo phù hợp (ví dụ glucose máu lúc đói)."
            )
        if patient_values["HbA1C"] >= 6.5:
            reference_signals.append(
                "HbA1C từ 6.5 trở lên; cần xác nhận đơn vị và diễn giải bằng xét nghiệm y tế."
            )
        st.markdown("**Diễn giải**")
        st.write(
            "Điểm trên là xác suất do mô hình học từ bộ dữ liệu cung cấp; ngưỡng 50% được dùng để phân nhóm trong bài tập, không phải ngưỡng chẩn đoán."
        )
        if reference_signals:
            st.write("Một số chỉ số cần lưu ý theo mốc tham khảo:")
            for signal in reference_signals:
                st.write(f"- {signal}")
        else:
            st.write(
                "Các chỉ số Glucose và HbA1C nhập vào chưa vượt các mốc tham khảo nêu trong hướng dẫn. Điều này không loại trừ nguy cơ bệnh."
            )
        st.info(
            "Nếu có triệu chứng hoặc kết quả xét nghiệm bất thường, hãy trao đổi với bác sĩ để được đánh giá chính thức."
        )

with evaluation:
    st.subheader("Chọn mô hình trên tập validation")
    st.dataframe(
        validation_table.style.format(
            {"Accuracy": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(f"Mô hình được chọn theo F1-score validation: {best_model_name}.")
    st.subheader("Kết quả cuối trên tập test")
    metric_columns = st.columns(3)
    for column, metric_name in zip(metric_columns, ["Accuracy", "Recall", "F1-score"]):
        column.metric(metric_name, f"{test_metrics[metric_name]:.3f}")
    st.write("Ma trận nhầm lẫn (hàng: nhãn thật, cột: nhãn dự đoán; 0 = không nguy cơ, 1 = có nguy cơ)")
    st.dataframe(
        pd.DataFrame(
            test_metrics["Confusion matrix"],
            index=["Thật: không nguy cơ", "Thật: có nguy cơ"],
            columns=["Dự đoán: không nguy cơ", "Dự đoán: có nguy cơ"],
        ),
        width="stretch",
    )
    st.caption("Tập test không được dùng để chọn mô hình; tỷ lệ chia là 70% train, 15% validation, 15% test.")

with data_tab:
    st.subheader("Tiền xử lý dữ liệu")
    st.write(
        f"{diagnostics['raw_rows']:,} dòng ban đầu; "
        f"{diagnostics['clean_rows']:,} dòng sau khi chuẩn hóa nhãn và loại bản ghi trùng."
    )
    st.write(
        f"Nhãn không nhận diện: {diagnostics['unknown_labels']}; "
        f"ô thiếu được điền trong pipeline: {diagnostics['missing_values']}; "
        f"giá trị ngoài phạm vi hợp lý chuyển thành thiếu: {diagnostics['invalid_numeric_values']}; "
        f"HbA1C đã sửa thang đo: {diagnostics['scaled_hba1c_values']}."
    )
    st.write(
        "Nhãn TRUE/1/Có nguy cơ được mã hóa thành 1; FALSE/0/Không có nguy cơ thành 0. "
        "Dữ liệu thiếu được điền bằng trung vị (số) hoặc giá trị phổ biến nhất (phân loại). "
        "Ngoại lai được giới hạn theo IQR, với các giới hạn chỉ được học trên tập train. "
        "Biến giới tính được One-Hot Encoding; biến số được Robust Scaling."
    )
    st.write("Số mẫu theo nhãn sau làm sạch:", diagnostics["class_counts"])
    st.download_button(
        "Tải dữ liệu đã làm sạch",
        data=cleaned_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="CSDL_TieuDuong_DaXuLy.csv",
        mime="text/csv",
    )