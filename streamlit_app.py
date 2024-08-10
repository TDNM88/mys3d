import streamlit as st
import torch
import random
from dotenv import load_dotenv
import os

# Load token từ file .env
load_dotenv()
APP_TOKEN = os.getenv("APP_TOKEN")

def authenticate():
    token = st.text_input("Nhập Token để truy cập:", type="password")
    if token == APP_TOKEN:
        st.success("Truy cập thành công!")
        return True
    else:
        st.error("Token không đúng.")
        return False

def load_model(path):
    try:
        model = torch.load(path)
        st.success(f"Đã tải mô hình từ {path}")
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình: {e}")
        return None

def save_model(model, path):
    try:
        torch.save(model, path)
        st.success(f"Đã lưu mô hình tại {path}")
    except Exception as e:
        st.error(f"Không thể lưu mô hình: {e}")

def layer_wise_merge(model_a, model_b, alpha=0.5, noise_factor=1e-5):
    merged_model = {}
    for key in model_a.keys():
        if "layer" in key:
            merged_weights = alpha * model_a[key] + (1 - alpha) * model_b[key]
            noise = noise_factor * torch.randn_like(merged_weights)
            merged_model[key] = merged_weights + noise
        else:
            merged_model[key] = model_a[key]
    return merged_model

def random_weight_merge(model_a, model_b, noise_factor=1e-5):
    merged_model = {}
    for key in model_a.keys():
        alpha = random.uniform(0.3, 0.7)
        merged_weights = alpha * model_a[key] + (1 - alpha) * model_b[key]
        noise = noise_factor * torch.randn_like(merged_weights)
        merged_model[key] = merged_weights + noise
    return merged_model

# Giao diện người dùng Streamlit
st.title("APP")

if authenticate():
    uploaded_file_a = st.file_uploader("Chọn mô hình A", type="pth")
    uploaded_file_b = st.file_uploader("Chọn mô hình B", type="pth")

    if uploaded_file_a and uploaded_file_b:
        model_a = load_model(uploaded_file_a)
        model_b = load_model(uploaded_file_b)

        if model_a and model_b:
            merged_model = layer_wise_merge(model_a, model_b)
            merged_model = random_weight_merge(merged_model, model_b)

            save_path = st.text_input("Nhập đường dẫn để lưu mô hình đã gộp:", "merged_model.pth")

            if st.button("Lưu mô hình"):
                save_model(merged_model, save_path)
