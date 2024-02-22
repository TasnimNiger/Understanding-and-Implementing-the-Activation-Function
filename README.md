# **Understanding and Implementing the Activation Function**

## **What are Activation Functions?**

Activation functions are a critical component of neural networks that introduce non-linearity into the model, allowing networks to learn complex patterns and relationships in the data. These functions play an important role in the hyperparameters of AI-based models. 

There are numerous different activation functions to choose from. Knowing which function or series of functions to train a neural network can be challenging for data scientists and machine learning engineers. 

In a neural network, the weighted sum of inputs is passed through the activation function.

Y = Activation function(∑ (weights*input + bias))

<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/ff9d94a2-3435-4431-823e-e7c913538b8c" width="700" height="500" align="center">

## **Why Neural Networks Need Activation Functions?**

*  Activation functions are used to remove the linearity from the neural network. If we do not apply an activation function, the output signal would be a simple linear function. In other words, it wouldn’t be able to handle large volumes of complex data. Activation functions are an additional step in each forward propagation layer but valuable. 

*  Activation functions used in ML models' output layers (think classification problems). The primary purpose of these activation functions is to squash the value between a bounded range like 0 to 1.
  <img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/80737c0e-3b75-4c1f-9f3d-bbb34c94b230" width="600" height="400" align="center">

* Activation functions that are used in hidden layers of neural networks. The primary purpose of these activation functions is to provide non-linearity, without which neural networks cannot model non-linear relationships.
  
*  During backpropagation, gradients calculated for each layer depend on the derivative of the activation function.
  
*  The choice of activation function affects the overall training speed, stability, and convergence of neural networks.


### **Activation functions are mainly of two types based on their use in an ML model.**
## 1. **Linear Activation Functions**
**Equation : f(Z) = Z**

**Function Input Range:** (- ∞,∞)

**Function Output Range:** (- ∞,∞)

**Graph:**

  
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/6c3db64d-d511-4e71-84e5-7ecea92203ee)

**Code Snippet:**


    def linear(z) :
  
          fn = z
  
          return(fn)

**Key features:**

*  A linear function is used at the output layer when the target variable is continuous, i.e., regression problem.

**Usage:**

The linear activation function is preferred at the output layer when it can not capture the non-linearity and complexity of data. Hence, we cannot apply it in the hidden or input layers.


## 2. **Non-Linear Activation Functions**

## **a) Sigmoid Activation Functions**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/bdfa870d-bbab-480e-b2ee-7d955f464c05)

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (0,1)

**Graph:**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/c2354b23-687e-42e1-ad8e-0ed8c75f6199)

**Code Snippet:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/0c5359cc-0b9f-4b10-a506-7eaac94cc6db)


**Key features:**

*  This is also called the logistic function used in logistic regression models.
*  The sigmoid function has an s-shaped graph.
*  Clearly, this is a non-linear function.
*  The sigmoid function converts its input into a probability value between 0 and 1.
*  It converts large negative values towards 0 and large positive values towards 1.
*  It returns 0.5 for the input 0. The value 0.5 is the threshold value, deciding which of two classes a given input belongs to.

**Usage:**

*  In the early days, the sigmoid function activated the hidden layers in MLPs, CNNs and RNNs.
*  However, the sigmoid function is still used in RNNs.
*  We do not usually use the sigmoid function for the hidden layers in MLPs and CNNs. Instead, we use ReLU or Leaky ReLU there.
*  The sigmoid function must be used in the output layer to build a binary classifier. The output is interpreted as a class label depending on the probability value of input returned by the function.
*  The sigmoid function is used when we build a multilabel classification model in which each mutually inclusive class has two outcomes. Do not confuse this with a multiclass classification model.

## **b) Tanh Activation Functions**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/df470087-bfff-4fba-b6c8-2e0a91fae527)

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-1,1)

**Graph:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/1aabd856-1d9a-42a4-a927-f05256d9f653)

**Code Snippet:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/2c6e6598-54fb-407f-8aab-0221ec98e61d)

**Key features:**

*  The output of the tanh (tangent hyperbolic) function always ranges between -1 and +1.
*  Like the sigmoid function, it has an s-shaped graph. This is also a non-linear function.
*  One advantage of using the tanh function over the sigmoid function is that the tanh function is zero centered. This makes the optimization process much easier.
*  The tanh function has a steeper gradient than the sigmoid function has.

**Usage:**

*  The tanh function activated the hidden layers in MLPs, CNNs and RNNs.However, the tanh function is still used in RNNs.
*  We do not usually use the tanh function for the hidden layers in MLPs and CNNs. Instead, we use ReLU or Leaky ReLU there.
*  We never use the tanh function in the output layer.

## **c) ReLU Activation Functions**

**Equation: f(Z) = max(0,Z)**

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (0,∞)

**Graph:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/de2ae401-3a97-4c88-8d37-172e511c9016)

**Code Snippet:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/dc9b687c-eccc-4686-82e8-8f64fa43faaf)

**Key features:**

*  The ReLU (Rectified Linear Unit) activation function is a great alternative to both sigmoid and tanh activation functions.
*  Inventing ReLU is one of the most important breakthroughs made in deep learning.
*  This function does not have the vanishing gradient problem.
*  This function is computationally inexpensive. It is considered that the convergence of ReLU is six times faster than sigmoid and tanh functions.
*  If the input value is 0 or greater than 0, the ReLU function outputs the input as it is. If the input is less than 0, the ReLU function outputs the value 0.
*  The ReLU function is made up of two linear components. Because of that, the ReLU function is piecewise linear. The ReLU function is non-linear.
*  The output of the ReLU function can range from 0 to positive infinity.
*  The convergence is faster than the sigmoid and tanh functions. This is because the ReLU function has a fixed derivate (slope) for one linear component and a zero derivative for the other. Therefore, the learning process is much faster with the ReLU function.
*  Calculations can be performed much faster with ReLU because no exponential terms are included in the function.

**Usage:**

*  The ReLU function is the default activation function for hidden layers in modern MLP and CNN neural network models.
*  We do not usually use the ReLU function in the hidden layers of RNN models. Instead, we use the sigmoid or tanh function there.
*  We never use the ReLU function in the output layer.


## **d) Leaky Relu**

**Equation: f(Z) = max(0.01Z,Z)**

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-∞,∞)

**Graph:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/d3510d18-f5f5-474e-8cff-e1597bd07803)

**Code Snippet:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/30cd20f1-d6ec-45b2-84bb-491364d309ad)

**Key features:**

*  The leaky ReLU activation function is a modified version of the default ReLU function.
*  Like the ReLU activation function, this function does not have the vanishing gradient problem.
*  If the input value is 0 greater than 0, the leaky ReLU function outputs the input as it is like the default ReLU function does. However, if the input is less than 0, the leaky ReLU function outputs a small negative value defined by αz (where α is a small constant value. Usually 0.01, and z is the input value).
*  It has no linear component with zero derivatives (slopes). Therefore, it can avoid the dying ReLU problem.
*  The learning process with leaky ReLU is faster than the default ReLU.

**Usage:**

*  The same usage of the ReLU function is also valid for the leaky ReLU function.


##  **e) Parametric ReLU(PReLU)**

**Equation: f(Z) = max(αZ,Z)**

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-∞,∞)

**Graph:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/9e955d70-90fc-45d0-b05e-b0bf09d89976)

**Code Snippet:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/22afaefc-d0cf-4de0-afd8-b1ed870dff12)

**Key features:**

*  This is another variant of the ReLU function.
*  This is almost similar to the leaky ReLU function. The only difference is that the value α becomes a learnable parameter (hence the name). We set α as a parameter for each neuron in the network. Therefore, the optimal value of α learns from the network.

**Usage:**

The parameterized ReLU function is used when the leaky ReLU function fails to solve the problem of dead neurons and the relevant information is not successfully passed to the next layer.
  
##  **f) Exponential Linear Unit(ELU)**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/489e243a-753d-416a-b574-208ee40e01ed)

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (-α,∞)

**Graph:**


<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/2a95d39a-ed13-475c-8822-37f7ef7e1f36" width=300, hight=300>

**Code Snippet:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/03d0b5da-c16d-481b-8ab9-f5cb1ba5528c)

**Key features:**

* ELU is a variant of Rectified Linear Units (ReLU) that modifies the slope of the negative part of the function.
* ELU offers zero-centered output and helps mitigate the vanishing gradient problem by providing positive and negative gradients.
* Unlike the leaky ReLU and parametric ReLU functions, ELU uses a log curve instead of a straight line to define the negative values.

**Usages:**

It is used for CNN, NLP, and Pattern Recognition models, which require deep neural networks.

##  **g) Softmax Activation Functions**

**Equation:** <img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/36f225d6-5309-497e-9e0e-5acb9b567269" hight=100, width=200>

**Function Input Range :** (- ∞,∞)

**Function Output Range :** (0,1)

**Graph:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/36d345fe-b51e-41e1-bcf2-3c3eebe0a15b)

**Code Snippet:**


![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/423e52f4-d8ad-4cd1-a668-5a5d1f1d238e)

**Key features:**

*  This is also a non-linear activation function.
*  The softmax function calculates the probability value of an event (class) over K different events (classes). It calculates the probability values for each class. The sum of all probabilities is 1 meaning that all events (classes) are mutually exclusive.

**Usage:**

We must use the softmax function in the output layer of a multiclass classification problem.

## **h) Binary Step Function**

**Equation:** f(x)  = 1, x>=0
                    = 0, x<0

**Function Input Range :** (- ∞,∞)

**Function Output Range :** True(1) or False(0)

**Graph:**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/c064e8be-bc52-4d57-961f-92b8ce3511a3)


**Code Snippet:**

![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/fe41adc2-e31a-40b9-aa3d-4f64c41f976b)

**Key features:**

It takes all real values as input. For positive values, the output is 1; for every negative value, the output is 0.

**Usage:**

*  The binary step function can be used as an activation function while creating a binary classifier.
*  This function will not be useful when multiple classes are in the target variable.          

## **i) Softplus:**

**Equation:**: f(x) = ln(1+exp x)

**Function Range:**  (0,∞)

**Graph:**


<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/c438866a-9791-4984-9c97-6034bc836ae0" width=400, hight=400>

**Key features:**

*  The softplus function is similar to the ReLU function but relatively smooth. It is unilateral suppression like ReLU. 
*  It has a wide acceptance range (0, + inf).
   

## **Choosing the right Activation Function**

<img src="https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/8c91052c-3e28-4990-8346-e98a1a31b59a" width="600" height="400" align="center">

Now that we have seen so many activation  functions, we need some logic/heuristics to know which activation function should be used in which situation. Good or bad – there is no rule of thumb.

However, depending on the properties of the problem, we can make a better choice for easy and quicker network convergence.

*  Sigmoid functions and their combinations generally work better in the case of classifiers
*  Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
*  ReLU function is a general activation function and is used in most cases these days
*  If we encounter a case of dead neurons in our networks, the leaky ReLU function is the best choice
*  Always keep in mind that the ReLU function should only be used in the hidden layers
*  As a rule of thumb, you can begin with using the ReLU function and then move over to other activation functions in case ReLU doesn’t provide optimum results. 


## **Advantages and Disadvantages**
![image](https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/9609a39e-ca9b-41bf-bc82-8eb3b967255b)

##  **Vanishing Gradients:**

*  Vanishing gradients occur when the derivatives of activation functions become extremely small, causing slow convergence or stagnation in training.
*  Sigmoid and tanh activation functions are known for causing vanishing gradients, especially in deep networks.

##  **Mitigating the Vanishing Gradient Problem:**

*  Rectified Linear Unit (ReLU) and its variants, such as Leaky ReLU, address the vanishing gradient problem by providing a non-zero gradient for positive inputs.
*  ReLU functions result in faster convergence due to the lack of vanishing gradients when inputs are positive.

##  **Role of Zero-Centered Activation Functions:**

Zero-centering is the most common data pre-processing technique that involves subtracting the mean from each data point to center it around zero. It is also known as Mean subtraction. A zero-centered activation function ensures that the gradients are not all positive or all negative. It contributes to stable weight updates and optimization during training. The zero-centered activation function is preferred in neural networks for several reasons:

*  **Symmetry around zero:** Zero-centered activation functions have the property of being symmetric around zero, which can help the network converge faster during training.
*  **Gradient descent:** Zero-centered activation functions make it easier for the gradients to propagate through the network during backpropagation, leading to more stable and efficient training.
*  **Reduced vanishing gradients:** By being centered around zero, these activation functions can help mitigate the vanishing gradient problem, which can occur with other activation functions.
*  **Better weight initialization:** Zero-centered activation functions can work well with certain weight initialization methods, such as the Xavier or He initialization, leading to faster convergence and better performance.

## **Quick Summary of different activation functions**
<img src= "https://github.com/TasnimNiger/Deep-Learning-Resources/assets/85071596/978a0327-28c9-4afa-817f-365e918c6a59" width="1500" height="600" align="center">

## **References:**
1. [Activation functions in Neural Networks](https://medium.com/@datta.nagraj/activation-functions-in-neural-networks-6ffe7b723420)
2. [Linear Activation Function](https://iq.opengenus.org/linear-activation-function/)
3. [Unlocking The Power of Activation Functions in Neural Networks](https://www.analyticsvidhya.com/blog/2023/10/activation-functions-in-neural-networks/)
4. [How to Choose the Right Activation Function for Neural Networks](https://towardsdatascience.com/how-to-choose-the-right-activation-function-for-neural-networks-3941ff0e6f9c)
5. [Neural Networks and Activation Function](https://www.analyticsvidhya.com/blog/2021/04/neural-networks-and-activation-function/)
6. [Introductory Guide on the Activation Functions](https://www.analyticsvidhya.com/blog/2022/03/introductory-guide-on-the-activation-functions/)
