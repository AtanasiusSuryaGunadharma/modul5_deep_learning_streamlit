import streamlit as st
import numpy as np
import pickle
from PIL import Image
import os

# Definisikan jalur model
model_directory = r'D:\SURYA\UAJY\Semester 5\Asdos Machine Learning\Pemegang Modul\Modul Deep Learning\Notebook'
model_path = os.path.join(model_directory, r'best_model.pkl')

# Load the model
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Nama kelas untuk Fashion MNIST
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Fungsi untuk memproses gambar
        def preprocess_image(image):
            image = image.resize((28, 28))  # Ubah ukuran menjadi 28x28 piksel
            image = image.convert('L')      # Ubah menjadi grayscale
            image_array = np.array(image) / 255.0  # Normalisasi
            image_array = image_array.reshape(1, -1)  # Flatten ke bentuk 1D array
            return image_array

        # UI Streamlit
        st.title("Fashion MNIST Image Classifier")
        st.write("Unggah beberapa gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelas masing-masing.")

        # File uploader untuk input gambar (multiple files)
        uploaded_files = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Sidebar dengan tombol "Predict" dan hasil prediksi
        with st.sidebar:
            st.write("## Navigator")
            predict_button = st.button("Predict")  # Tombol di sidebar

            # Tampilkan hasil prediksi di bawah tombol "Predict"
            if uploaded_files and predict_button:
                st.write("### Hasil Prediksi")

                for uploaded_file in uploaded_files:
                    # Buka dan proses setiap gambar
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
                    predictions = model.predict_proba(processed_image)
                    predicted_class = np.argmax(predictions)
                    confidence = np.max(predictions) * 100

                    # Tampilkan nama file dan hasil prediksi
                    st.write(f"**Nama File:** {uploaded_file.name}")
                    st.write(f"Kelas Prediksi: **{class_names[predicted_class]}**")
                    st.write(f"Confidence: **{confidence:.2f}%**")
                    st.write("---")  # Garis pemisah antara hasil prediksi

        # Tampilkan gambar yang diunggah di halaman utama
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Gambar: {uploaded_file.name}", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.error("File model tidak ditemukan.")




# import streamlit as st
# import numpy as np
# import pickle
# import os
# from PIL import Image
# from sklearn.preprocessing import StandardScaler

# # Define paths
# model_directory = r'D:\SURYA\UAJY\Semester 5\Asdos Machine Learning\Pemegang Modul\Modul Deep Learning\Notebook'
# model_path = os.path.join(model_directory, r'best_model.pkl')

# # Load the model
# if os.path.exists(model_path):
#     try:
#         with open(model_path, 'rb') as model_file:
#             model = pickle.load(model_file)

#         # Define class names for Fashion MNIST
#         class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#         # Function to preprocess the image
#         def preprocess_image(image):
#             image = image.resize((28, 28))  # Resize to 28x28 pixels
#             image = image.convert('L')      # Convert to grayscale
#             image_array = np.array(image) / 255.0  # Normalize
#             image_array = image_array.reshape(1, -1)  # Flatten to 1D array
#             return image_array

#         # Streamlit UI
#         st.title("Fashion MNIST Image Classifier")
#         st.write("Upload an image of a fashion item (e.g., shoe, bag, shirt) and the model will predict its class.")

#         # File uploader for image input
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#         # Initialize session state for predict and reset actions
#         if "predicted" not in st.session_state:
#             st.session_state.predicted = False

#         # Display the uploaded image and add the "Predict" button
#         if uploaded_file is not None:
#             # Display uploaded image
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded Image", use_column_width=True)

#             # Show "Predict" button
#             if st.button("Predict"):
#                 # Preprocess and predict
#                 processed_image = preprocess_image(image)
#                 predictions = model.predict_proba(processed_image)
#                 predicted_class = np.argmax(predictions)
#                 confidence = np.max(predictions) * 100

#                 # Save prediction state
#                 st.session_state.predicted = True
#                 st.session_state.predicted_class = predicted_class
#                 st.session_state.confidence = confidence

#         # Display prediction results if prediction is done
#         if st.session_state.predicted:
#             st.write("### Prediction Results")
#             st.write(f"Predicted Class: **{class_names[st.session_state.predicted_class]}**")
#             st.write(f"Confidence: **{st.session_state.confidence:.2f}%**")

#         # Reset button
#         if st.button("Reset"):
#             # Clear session state
#             st.session_state.predicted = False
#             uploaded_file = None  # Reset the uploaded file

#     except Exception as e:
#         st.error(f"Error: {str(e)}")
# else:
#     st.error("Model file not found.")




# import streamlit as st
# import numpy as np
# import pickle
# import os
# from PIL import Image
# from sklearn.preprocessing import StandardScaler

# # Path to the directory containing the model and sample images
# model_directory = r'D:\SURYA\UAJY\Semester 5\Asdos Machine Learning\Pemegang Modul\Modul Deep Learning\Notebook'
# model_path = os.path.join(model_directory, 'fashion_mnist_model.pkl')

# # Load sample images for each class (assuming we have a folder with one sample per class)
# sample_images_dir = os.path.join(model_directory, 'sample_images')  # Update this path if needed
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# sample_images = {class_name: os.path.join(sample_images_dir, f"{class_name}.png") for class_name in class_names}

# if os.path.exists(model_path):
#     try:
#         # Load the model
#         with open(model_path, 'rb') as model_file:
#             model = pickle.load(model_file)

#         # Function to preprocess the uploaded image to match the model input format
#         def preprocess_image(image):
#             image = image.resize((28, 28))  # Resize to 28x28 pixels
#             image = image.convert('L')      # Convert to grayscale
#             image_array = np.array(image) / 255.0  # Normalize
#             image_array = image_array.reshape(1, -1)  # Flatten to 1D array
#             return image_array

#         # Streamlit UI
#         st.title("Fashion MNIST Image Classifier")
#         st.write("Upload an image of a fashion item (e.g., shoe, bag, shirt) and the model will predict its class.")

#         # File uploader for image input
#         uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#         if uploaded_file is not None:
#             # Display the uploaded image
#             image = Image.open(uploaded_file)
#             st.image(image, caption="Uploaded Image", use_column_width=True)
            
#             # Preprocess and predict
#             processed_image = preprocess_image(image)
#             predictions = model.predict_proba(processed_image)
#             predicted_class = np.argmax(predictions)
#             confidence = np.max(predictions) * 100

#             # Display prediction results
#             st.write("### Prediction Results")
#             st.write(f"Predicted Class: **{class_names[predicted_class]}**")
#             st.write(f"Confidence: **{confidence:.2f}%**")

#         # Separator line
#         st.write("---")

#         # Option to display a sample image for a specific class
#         st.write("Or, select a class to view a sample image:")
#         selected_class = st.selectbox("Choose a class", class_names)

#         if selected_class:
#             sample_image_path = sample_images.get(selected_class)
#             if sample_image_path and os.path.exists(sample_image_path):
#                 # Display the sample image
#                 sample_image = Image.open(sample_image_path)
#                 st.image(sample_image, caption=f"Sample Image of {selected_class}", use_column_width=True)
#             else:
#                 st.error(f"No sample image found for class '{selected_class}'")

#     except Exception as e:
#         st.error(f"Error: {str(e)}")
# else:
#     st.error("Model file not found.")

