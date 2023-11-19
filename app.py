import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from numpy import argmax
import pickle




# Load your tumor classification model
#cnn_model = tf.keras.models.load_model('cnn_tumor_model.h5')

# Function to perform image classification using CNN
def classify_image(img, cnn_model):
    img = Image.open(img)
    img = img.resize((128, 128))
    img = np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = cnn_model.predict(input_img)
    if res > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor"
    
    
    

# Load your SMS spam detection model
spam_model = tf.keras.models.load_model('spammodel.h5')
# Load the saved tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokeniser = pickle.load(handle)

max_length=20
def predict_message(input_text, tokeniser):
    # Process input text similarly to training data
    encoded_input = tokeniser.texts_to_sequences([input_text])
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(encoded_input, maxlen=max_length, padding='post')
    # Get the probabilities of being classified as "Spam" for each input
    predictions = spam_model.predict(padded_input)
    # Define a threshold (e.g., 0.5) for classification
    threshold = 0.5
    # Make the predictions based on the threshold for each input
    for prediction in predictions:
        if prediction > threshold:
            return "Spam"
        else:
            return "Not spam"


    

# Load the saved model
imdb_model = tf.keras.models.load_model('lstm_imdb_model.h5')
top_words = 5000
max_review_length = 500

# Function to predict sentiment for a given review
def predict_sentiment(review):
    # Process input text similarly to training data
    word_index = imdb.get_word_index()
    review = review.lower().split()
    review = [word_index[word] if word in word_index and word_index[word] < top_words else 0 for word in review]
    review = sequence.pad_sequences([review], maxlen=max_review_length)
    prediction = imdb_model.predict(review)
    if prediction > 0.5:
        return "Positive"
    else:
        return "Negative"




# Load the saved model
iris_dnn_model = tf.keras.models.load_model('iris_dnn_model.h5')

def predict_iris_class(input_data):
    # Make predictions using the loaded model
    prediction = iris_dnn_model.predict(input_data)
    predicted_class = argmax(prediction)
    
    class_names = ['setosa', 'versicolor', 'virginica']
    predicted_class_name = class_names[predicted_class]

    return prediction, predicted_class_name




# Load the saved model
mnist_model = tf.keras.models.load_model('mnist_cnn_model.h5')

def predict_digit(file_path):
    # Load the image using PIL
    image = Image.open(file_path)
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to 28x28 (same as MNIST dataset)
    image = image.resize((28, 28))
    # Convert image to array
    image_array = np.array(image)
    # Reshape and normalize the image (similar to training data)
    processed_image = image_array.reshape((1, 28, 28, 1))
    processed_image = processed_image.astype('float32') / 255.0   
    # Make predictions using the loaded model
    prediction = mnist_model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return predicted_class




# Load the model from the file using pickle
with open('iris_perceptron_model.pkl', 'rb') as file:
    iris_perceptron_model = pickle.load(file)
def predict_iris_species(input_data):
    # Make predictions using the loaded Perceptron model
    prediction = iris_perceptron_model.predict(input_data)
    predicted_class = prediction[0]  # Assuming the prediction is a single class
    classes = {0: 'Setosa', 1: 'Not Setosa'}  # Map prediction to class label
    predicted_class_name = classes[predicted_class]

    return predicted_class_name

# Load the model from the file using pickle
with open('iris_backprop_model.pkl', 'rb') as file:
    iris_backprop_model = pickle.load(file)
def predict_iris_species_backprop(input_data):
    # Make predictions using the loaded Perceptron model
    prediction = iris_backprop_model.predict(input_data)
    predicted_class = prediction[0]  # Assuming the prediction is a single class
    classes = {0: 'Setosa', 1: 'Not Setosa'}  # Map prediction to class label
    predicted_class_name = classes[predicted_class]

    return predicted_class_name









# Main function for Streamlit app
def main():
    st.title("Multitasking App for Image, Text and Data Analysis")
    st.subheader("Task Selecetion")

    # Dropdown for task selection
    task = st.selectbox("Select Task", ["Tumor Detection", "SMS Spam Detection", "IMDb Sentiment Analysis","Digit Recognition", "Iris Flower Classification-DNN","Iris Species Prediction-Perceptron","Iris Species Prediction-Backpropagation"])

    if task == "Tumor Detection":
        st.subheader("Tumor Detection")
        uploaded_file = st.file_uploader("Upload an image to check for tumor...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Display the image
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Uploaded Image", use_column_width=True)

            if st.button("Detect Tumor"):
                # Call the tumor detection function
                result = classify_image(uploaded_file, cnn_model)
                st.write("Tumor Detection Result:", result)
                

    elif task == "SMS Spam Detection":
        st.subheader("SMS Spam Detection")
        user_input = st.text_area("Enter a message to classify as 'Spam' or 'Not spam': ")
            
        if st.button("Predict"):
            if user_input:
                prediction_result = predict_message(user_input, tokeniser)
                st.write(f"The message is classified as: {prediction_result}")
            else:
                st.write("Please enter some text for prediction")

    
    elif task == "IMDb Sentiment Analysis":
        st.subheader("IMDb Sentiment Analysis")
        user_review = st.text_area("Enter a movie review: ")
        
        if st.button("Analyze Sentiment"):
            if user_review:
                sentiment_result = predict_sentiment(user_review)
                st.write(f"The sentiment of the review is: {sentiment_result}")
            else:
                st.write("Please enter a movie review for sentiment analysis")
                
                
    elif task == "Iris Flower Classification-DNN":
        st.subheader("Iris Flower Classification-DNN")
        
        # Input fields for user to enter data
        sepal_length = st.number_input("Sepal Length", min_value=0.1, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Sepal Width", min_value=0.1, max_value=10.0, value=3.5)
        petal_length = st.number_input("Petal Length", min_value=0.1, max_value=10.0, value=1.4)
        petal_width = st.number_input("Petal Width", min_value=0.1, max_value=10.0, value=0.2)

        if st.button("Predict Iris Class"):
            # Prepare input data for prediction
            input_row = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Get prediction results
            probabilities, predicted_class = predict_iris_class(input_row)

            # Display prediction results
            st.subheader("Prediction Results")
            st.write('Predicted probabilities:', probabilities)
            st.write('Predicted class:', predicted_class)
            

    elif task == "Digit Recognition":
        st.subheader("Digit Recognition")

        uploaded_digit = st.file_uploader("Upload an image of a digit (0-9) to predict...", accept_multiple_files=True)

        if uploaded_digit is not None:
            # Display the uploaded digit image(s)
            for digit_image in uploaded_digit:
                img = Image.open(digit_image)
                st.image(img, caption="Uploaded Image", use_column_width=True)

                if st.button("Predict Digit"):
                    # Call the digit prediction function
                    digit_prediction = predict_digit(digit_image)
                    st.write(f"The predicted digit is : {digit_prediction}")




    elif task == "Iris Species Prediction-Perceptron":
        st.subheader("Iris Species Prediction-Perceptron")
        
        # Input fields for user to enter data
        sepal_length = st.number_input("Sepal Length", min_value=0.1, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Sepal Width", min_value=0.1, max_value=10.0, value=3.5)

        if st.button("Predict Iris Species"):
            # Prepare input data for prediction
            input_row = np.array([[sepal_length, sepal_width]])
            
            # Get prediction results using Perceptron model
            predicted_class_perceptron = predict_iris_species(input_row)

            # Display prediction results
            st.subheader("Prediction Results")
            st.write('Predicted class:', predicted_class_perceptron)
            
            
    elif task == "Iris Species Prediction-Backpropagation":
        st.subheader("Iris Species Prediction-Backpropagation")
        
        # Input fields for user to enter data
        sepal_length = st.number_input("Sepal Length", min_value=0.1, max_value=10.0, value=5.0)
        sepal_width = st.number_input("Sepal Width", min_value=0.1, max_value=10.0, value=2.5)

        if st.button("Predict Iris Species"):
            # Prepare input data for prediction
            input_row = np.array([[sepal_length, sepal_width]])
            
            # Get prediction results using Perceptron model
            predicted_class = predict_iris_species_backprop(input_row)

            # Display prediction results
            st.subheader("Prediction Results")
            st.write('Predicted class:', predicted_class)
        
                
    


if __name__ == "__main__":
    main()
    
