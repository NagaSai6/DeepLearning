import numpy as np
from scipy.special import softmax


# after_activation will contain a matrix with cols as the activation layer hi, and each col for each data example
# Send inputs.transpose() for first layer alone

class Layer:
    def __init__(self,n_neurons,prev_layer_dim):
        self.weight=np.random.randn(n_neurons,prev_layer_dim)
        self.bias=np.random.randn(n_neurons,1)

    def calculate_hidden_layer(self,inputs):                                                       # Contains ai and hi values
        self.pre_activation=np.matmul(self.weight,inputs) + self.bias                                #Computing ai
        self.after_activation= 1.0/1.0+np.exp(-self.pre_activation)

    def calculate_output_layer(self,inputs):
        self.pre_activation_output=np.matmul(self.weight,inputs) + self.bias
        self.output_final=softmax(self.pre_activation_output,axis=0)                                # I'll be having 10 * m matrix 
    
    def calculate_gradient(layer_above_this_layer,layer_below_this_layer):
        self.gradient_after_activation=np.matmul(layer_above_this_layer.weight.transpose(),layer_above_this_layer.gradient_pre_activation)
        self.gradient_pre_activation=self.gradient_after_activation * _________________
        self.gradient_weight=np.matmul(self.gradient_pre_activation,layer_below_this_layer.after_activation.transpose())
        self.gradient_bias=self.gradient_pre_activation

    def calculate_gradient_output_layer(layer_below_this_layer):
        self.gradient_pre_activation_output=        # Have to write
        self.gradient_weight_output=np.matmul(self.gradient_pre_activation_output,layer_below_this_layer.after_activation.transpose())
        self.gradient_bias_output=self.gradient_pre_activation_output       # What to do here? Bias is 1d, whereas pre_activation is 2d

    def calculate_gradient_first_layer(layer_above_this_layer,input_data):
        self.gradient_after_activation=np.matmul(layer_above_this_layer.weight.transpose(),layer_above_this_layer.gradient_pre_activation)
        self.gradient_pre_activation=self.gradient_after_activation * _________________
        self.gradient_weight=np.matmul(self.gradient_pre_activation,input_data.transpose())
        self.gradient_bias=self.gradient_pre_activation
        


def forward_prop(no_hidden_layers, layer_objects,output_layer,input_data):

    layer_objects[0].calculate_hidden_layer(input_data)                                          # Finding A1, H1 of first layer
    for i in range(1,no_hidden_layers-1):                                                        # Iterating from 1,2... till L-2 (Since our list indexes from 0). This means I'm going from Layer 2 to L-1
        layer_objects[i].calculate_hidden_layer(layer_objects[i-1].after_activation)
    output_layer.calculate_output_layer(layer_objects[-1].after_activation)                     # I'll be having 10 * m matrix      


def backward_prop(no_hidden_layers,layer_objects,output_layer,input_data):

    output_layer.calculate_gradient_output_layer(layer_objects[-1])                             # Calculate gradients associated with output layer
    layer_objects[-1].calculate_gradient(output_layer,layer_objects[-2])
    for i in range(no_hidden_layers-3, 0,-1):
        layer_objects[i].calculate_gradient(layer_objects[i+1],layer_objects[i-1])
    layer_objects[0].calculate_gradient_first_layer(layer_objects[1],input_data)


def gradient_descent(Max_Iters,no_of_neurons_hidden_layer,no_hidden_layers,input_data,etha):
    
    # Intialize each hidden layer's weights and bias
    layer_objects=[Layer(no_of_neurons_hidden_layer[i],no_of_neurons_hidden_layer[i-1]) for i in range(1,len(no_of_neurons_hidden_layer))]  # Created 3 layer objects, i can access each one using layer_objects[i-1]
    # Same for output layer
    output_layer=Layer(10,no_of_neurons_hidden_layer[-1])

    while(Max_Iters>0):
        forward_prop(no_hidden_layers, layer_objects, output_layer,input_data.transpose())
        backward_prop(no_hidden_layers,layer_objects,output_layer,input_data.transpose())
        
        # Updating parameters
        for i in range(no_hidden_layers-1):               # Goes from 0 to L-2 (included)
            layer_objects[i].weight = layer_objects[i].weight - layer_objects[i].gradient_weight*etha
            layer_objects[i].bias = layer_objects[i].bias - layer_objects[i].gradient_bias*etha
        output_layer.weight = output_layer.weight - output_layer.gradient_weight_output*etha
        output_layer.bias = output_layer.bias - output_layer.gradient_bias_output*etha            

        Max_Iters=Max_Iters-1



# Give no of neurons in each hidden layer in the list, output layer is fixed with 10 and try to pythonize this later
no_of_neurons_hidden_layer=[input_dim,2,5,3]     #means 3 hidden layers and 2,5,3 neurons in each
no_hidden_layers=len(no_of_neurons_hidden_layer)        # it gives 4 (3+1 for output)
gradient_descent(Max_Iters=1000,no_of_neurons_hidden_layer,no_hidden_layers)
