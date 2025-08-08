import streamlit as st
import pandas as pd
import joblib

st.title('Iris Classifier') # judul aplikasi
st.write('This is a simple Iris Classifier app') #menampilkan teks

# cara cun streamlit di local
# 1. open terminal and run `streamlit run streamlit_app.py`

# == Inference Function
model = joblib.load('model.joblib')
def get_prediction(data:pd.DataFrame):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)
    return pred, pred_proba

# User Input
left, right = st.columns(2, gap = 'medium')

# sepal input
left.subheader('sepal')
sepal_length = left.slider('sepal length', min_value =1.0, max_value=10.0, value=5.4, step=0.1)
sepal_width = left.slider('sepal width', min_value =1.0, max_value=10.0, value=5.4, step=0.1)

# petal input
right.subheader('petal')
petal_length = right.slider('petal length', min_value =1.0, max_value=10.0, value=5.4, step=0.1)
petal_width = right.slider('petal width', min_value =1.0, max_value=10.0, value=5.4, step=0.1)

# dataframe
data = pd.DataFrame({"sepal length (cm)": [sepal_length],
                     "sepal width (cm)": [sepal_width], 
                     "petal length (cm)": [petal_length], 
                     "petal width (cm)": [petal_width]})
st.dataframe(data, use_container_width=True)

# Prediction Button
# Button memiliki nilai boolean (jika button ditekan maka bernilai True)
button = st.button("Predict", use_container_width=True)

if button:
    st.write("Prediksi Berhasil !")
    pred, pred_proba = get_prediction(data)

    label_map = {0: "Setosa", 1: "Versicolor",2: "Virginica"}
    
    label_pred = label_map[pred[0]]
    label_proba = pred_proba[0][pred[0]]
    output = f"Iris Anda diklasifikasikan sebagai {label_proba:.0%} {label_pred}"
    st.write(output)

# # Show input value
# data = pd.DataFrame({"sepal length (cm)": [sepal_length], 
#                        "sepal width (cm)": [sepal_width], 
#                        "petal length (cm)": [petal_length], 
#                        "petal width (cm)": [petal_width]})
# st.dataframe(data, use_container_width=True)
# st.write(f"Sepal Length: {sepal_length}")
# st.write(f"Sepal Width: {sepal_width}")
# st.write(f"Petal Length: {petal_length}")
# st.write(f"Petal Width: {petal_width}")

# # Prediction Button
# button = st.button("Predict", use_container_width=True)
# if button:

#     st.write("Prediksi Berhasil !")
