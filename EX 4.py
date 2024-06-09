import streamlit as st
import numpy as np
# Neural Network class definition
class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
# Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (2x3) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer
    def forward(self, X):
        self.z = np.dot(X, self.W1)  # dot product of X (input) and first set of weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer and second set of weights
        o = self.sigmoid(self.z3)  # final activation function
        return o
   def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))  # activation function
  def sigmoidPrime(self, s):
        return s * (1 - s)  # derivative of sigmoid
  def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to  error
        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error
self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden -> output) weights
def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
# Data preparation
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)  # X = (hours sleeping, hours studying)
y = np.array(([92], [86], [89]), dtype=float)  # y = score on test
# Scale units
X = X / np.amax(X, axis=0)  # maximum of X array
y = y / 100  # max test score is 100
# Initialize neural network
NN = Neural_Network()
# Streamlit interface
st.title("Simple Neural Network Example")
if st.button("Train Neural Network"):
    # Train the neural network
    NN.train(X, y)
    st.write("Neural network trained successfully!")
# Display input, actual output, predicted output, and loss
predicted_output = NN.forward(X)
loss = np.mean(np.square(y - predicted_output))
st.subheader("Input Data (Scaled)")
st.write(X)
st.subheader("Actual Output (Scaled)")
st.write(y)
st.subheader("Predicted Output")
st.write(predicted_output)
st.subheader("Loss")
st.write(loss)
# Running the app
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        &copy; 2024 Your Company Name. All rights reserved.
    </div>
""", unsafe_allow_html=True)
if __name__ == "__main__":
    st.write("Press the button to train the neural network.")
