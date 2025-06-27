# 🚀 Deploy ML & DL Models using FastAPI + Streamlit

This project demonstrates how to **deploy machine learning and deep learning models** using **FastAPI** and create a **user-friendly frontend with Streamlit**.

## 📌 Main Objective

The goal is to show **how to expose ML/DL models as APIs** using FastAPI and interact with them via a clean, responsive Streamlit interface.

## 📂 Project Structure

```
├── app.py                # FastAPI backend (API server)
├── frontend.py             # Streamlit frontend app
├── models/
│   ├── ml\_model.pkl      # Trained ML model (e.g., RandomForest for Iris)
│   └── dl\_model.h5       # Trained DL model (e.g., CNN for digit recognition)
├── requirements.txt      # Required Python packages
└── README.md             # You're here!
````

## ⚙️ Backend (FastAPI)

### 🔥 Features
- `/predict_iris` → Accepts sepal/petal inputs and returns the predicted Iris species.
- `/predict_digit` → Accepts a 28x28 image and returns the predicted digit (0–9).

### ▶️ Run FastAPI server

```bash
uvicorn app:app --reload
````

## 🎨 Frontend (Streamlit) Demo 
![Image](https://github.com/user-attachments/assets/f001f95c-b879-4e3f-82b1-5d31fa89aa05)

### 🧠 Features

* Input form for Iris flower classification.
* Image uploader for handwritten digit recognition.
* Real-time API interaction and clean results display.

### ▶️ Run Streamlit app

```bash
streamlit run frontend.py
```

## 🔗 API Endpoints

| Endpoint         | Method | Description                  |
| ---------------- | ------ | ---------------------------- |
| `/predict_iris`  | POST   | Predicts Iris flower species |
| `/predict_digit` | POST   | Predicts handwritten digit   |

---

## 🛠️ Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/fastapi-ml-dl-deploy.git
   cd fastapi-ml-dl-deploy
   ```

2. Create virtual environment & install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server:

   ```bash
   uvicorn app:app --reload
   ```

4. Open a new terminal & run the frontend:

   ```bash
   streamlit run app_ui.py
   ```

## 🧠 Models Used

* **ML Model**: Scikit-learn (e.g., RandomForestClassifier for Iris dataset)
* **DL Model**: TensorFlow (CNN trained on MNIST-style digits)

## 📬 Contact

**Author:** Muhammad Hamza Khattak
📧 [[Gmail](mailto:mr.hamxa942@gmail.com)]
🔗 [LinkedIn](https://linkedin.com/in/muhammad-hamza-khattak/)

## 📝 License

This project is licensed under the [MIT License](LICENSE).
