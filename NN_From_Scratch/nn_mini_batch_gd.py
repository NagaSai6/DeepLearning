import numpy as np

no_neurons_each_layer=[784,3,5,4,10]                     # 784 for input layer, followed by hidden layers and then 10 for output 
L= len(no_neurons_each_layer)                            # L=5 



class Layer:
    def __init__(self,layer_dim,prev_layer_dim):
        self.weight=np.random.randn(layer_dim,prev_layer_dim)
        self.weight_grad=np.zeros((layer_dim,prev_layer_dim))
        self.bias=np.random.randn(layer_dim,1)
        self.bias_grad=np.zeros((layer_dim,1))
        self.a_=np.zeros((layer_dim,1))
        self.a_grad=np.zeros((layer_dim,1))
        self.h_=np.zeros((layer_dim,1))
        self.h_grad=np.zeros((layer_dim,1))

class Total_Grad_Layer:
    def __init__(self,layer_dim,prev_layer_dim):
        self.grad_weight=np.zeros((layer_dim,prev_layer_dim))
        self.grad_bias=np.zeros((layer_dim,1))

def do_sigmoid(layer_a):                                                                                   
    return 1.0/1.0+np.exp(-layer_a)

def do_softmax(layer_a):                                                                                    # HAVE TO WRITE THIS
    pass


def forward_prop(layer_objects,x,L):

    layer_objects[1].a_= np.matmul(layer_objects[1].weight,x) + layer_objects[1].bias                                       # Input layer
    layer_objects[1].h_=do_sigmoid(layer_objects[1].a_)

    for i in range(2,L-1):
        layer_objects[i].a_= np.matmul(layer_objects[i].weight,layer_objects[i-1].h_) + layer_objects[i].bias               # Error with matmul may come
        layer_objects[i].h_=do_sigmoid(layer_objects[i].a_)

    layer_objects[L-1].a_= np.matmul(layer_objects[L-1].weight,layer_objects[L-2].h_) + layer_objects[L-1].bias             # Output layer
    layer_objects[L-1].h_=do_softmax(layer_objects[L-1].a_)


def backward_prop(layer_objects,x,y,L):

    layer_objects[L-1].a_grad= layer_objects[L-1].h_ - y                                                                # Grad wrt A_L

    for i in range(L-1,1,-1):
        layer_objects[i].weight_grad= np.dot(layer_objects[i].a_grad,layer_objects[i-1].h_.transpose())                     # 
        layer_objects[i].bias_grad=layer_objects[i].a_grad
        layer_objects[i-1].h_grad= np.matmul(layer_objects[i].weight.transpose(),layer_objects[i].a_grad)
        layer_objects[i-1].a_grad= layer_objects[i-1].h_grad * do_sigmoid(layer_objects[i-1].a_) * 1-do_sigmoid(layer_objects[i-1].a_)          

    layer_objects[1].weight_grad= np.dot(layer_objects[1].a_grad,x.transpose())                                             # For L1 layer, we need Input
    layer_objects[1].bias_grad=layer_objects[1].a_grad



def mini_batch_gradient_descent(no_neurons_each_layer,L,eta=0.1,batch_size=16,max_epochs=10):

    layer_objects=["layer0"]                                          # List which stores all layer objects, layer0 is simply put for easier indexing 
    for i in range(1,L):
        layer_objects.append(Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    for i in range(max_epochs):
        
        # Initialize
        num_points_seen=0
        total_grad_layer_objects=["layer0"]                                   # List which stores all grad layer objects, which are used to add and store gradients of all examples upto batch_size
        for i in range(1,L):
            total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

        for x,y in zip(X,Y):

            forward_prop(layer_objects,x,L)                     # Forward pass
            backward_prop(layer_objects,x,y,L)                  # Backward Pass

            for i in range(1,L):                                                                                            # Adding grad of each layer to each layer of Total Grad
                total_grad_layer_objects[i].grad_weight = total_grad_layer_objects[i].grad_weight + layer_objects[i].weight_grad
                total_grad_layer_objects[i].grad_bias = total_grad_layer_objects[i].grad_bias + layer_objects[i].bias_grad

            num_points_seen+=1

            if(num_points_seen%batch_size==0):

                for i in range(1,L):                                                                                    # Updating weight and bias of each layer with grad from Total Grad
                    layer_objects[i].weight=layer_objects[i].weight - eta*total_grad_layer_objects[i].grad_weight
                    layer_objects[i].bias=layer_objects[i].bias - eta*total_grad_layer_objects[i].grad_bias

                # Make the grad_objects 0 for next batch
                total_grad_layer_objects=["layer0"]
                for i in range(1,L):
                    total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))




#layers=mini_batch_gradient_descent(no_neurons_each_layer,L)
#print(layers[4].weight.shape)



