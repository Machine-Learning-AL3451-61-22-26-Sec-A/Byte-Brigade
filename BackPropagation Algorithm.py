import streamlit as st
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def backpropagation(X, y, epochs, lr):
    input_layer_size = X.shape[1]
    hidden_layer_size = 4
    output_layer_size = 1

    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
    weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)
    bias_hidden = np.random.rand(1, hidden_layer_size)
    bias_output = np.random.rand(1, output_layer_size)

    for _ in range(epochs):

        hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
        predicted_output = sigmoid(output_layer_input)


        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * lr
        bias_output += np.sum(d_predicted_output) * lr
        weights_input_hidden += X.T.dot(d_hidden_layer) * lr
        bias_hidden += np.sum(d_hidden_layer) * lr

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

st.title('Backpropagation Algorithm Demonstration')
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])


epochs = st.sidebar.slider('Epochs', min_value=100, max_value=10000, value=1000)
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.1)

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = backpropagation(X, y, epochs, learning_rate)

st.subheader('Trained Model Parameters')
st.write('Weights (Input to Hidden Layer):')
st.write(weights_input_hidden)
st.write('Weights (Hidden to Output Layer):')
st.write(weights_hidden_output)
st.write('Bias (Hidden Layer):', bias_hidden)
st.write('Bias (Output Layer):', bias_output)
