import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import wandb



from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()            # X_train is (60000,28,28) and X_test is (10000,28,28)     

labels={0:'T-shirt/Top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle Boot'}
label_check=[]
image_label_combination={}    # Contains Label and the first image it comes across of that label

def Get_One_Each():
    for i in range(X_train.shape[0]):
        if y_train[i] not in label_check:
            label_check.append(y_train[i])
            image_label_combination[labels[y_train[i]]]=X_train[i]
        if len(label_check)==10:
            break

def Log_Images_Labels():
    Get_One_Each()
    wandb.log({"Images With Labels": [wandb.Image(img, caption=label) for label,img in image_label_combination.items()]})

# QUESTION 1:  Log the images  1st Question - Stored in fancy-fireband-3 in Assignment_1_try
#Log_Images_Labels()


Y_train=np.zeros((y_train.shape[0],10))
Y_train[np.arange(y_train.shape[0]),y_train]=1                                          # One-Hot encoding y_train to form Y_train which is now (m,10) matrix
Y_test=np.zeros((y_test.shape[0],10))
Y_test[np.arange(y_test.shape[0]),y_test]=1                                             # One-Hot encoding y_test to form Y_test
X_train_vectorized=np.reshape(X_train,(X_train.shape[0],-1))/255                            # X_train_vectorized is (m,784) matrix
X_test_vectorized=np.reshape(X_test,(X_test.shape[0],-1))/255                               # X_test_vectorized is (m,784) matrix

# WHEN SENDING DATA, SEND LIKE (m,784) THIS ONLY, .transpose() IN ZIP(X,Y) TAKES CARE OF THAT


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

class Momentum_Grad_Layer:                                                                          
    def __init__(self,layer_dim,prev_layer_dim):                                                       
        self.weight_update=np.zeros((layer_dim,prev_layer_dim))
        self.bias_update=np.zeros((layer_dim,1))

class LookAhead_Grad_Layer:                                                                         
    def __init__(self,layer_dim,prev_layer_dim):                                                       
        self.weight_before_lh=np.zeros((layer_dim,prev_layer_dim))                                      
        self.bias_before_lh=np.zeros((layer_dim,1))

def do_sigmoid(layer_a):                                                                                   
    return 1.0/(1.0+np.exp(-layer_a))

def do_softmax(layer_a):     
    exp_a = np.exp(layer_a - np.max(layer_a))                                                                               
    return exp_a/np.sum(exp_a,axis=0)

def do_tanh(layer_a):
    return np.tanh(layer_a)


def forward_prop(layer_objects,x,L):
    #print(x.shape)
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
        layer_objects[i].weight_grad= np.dot(layer_objects[i].a_grad,layer_objects[i-1].h_.transpose())                     
        layer_objects[i].bias_grad=layer_objects[i].a_grad
        layer_objects[i-1].h_grad= np.matmul(layer_objects[i].weight.transpose(),layer_objects[i].a_grad)
        layer_objects[i-1].a_grad= layer_objects[i-1].h_grad * do_sigmoid(layer_objects[i-1].a_) * (1-do_sigmoid(layer_objects[i-1].a_))          
    layer_objects[1].weight_grad= np.dot(layer_objects[1].a_grad,x.transpose())                                             # For L1 layer, we need Input
    layer_objects[1].bias_grad=layer_objects[1].a_grad



def cross_entropy_loss(Y_hat,Y):
    return np.sum(-1* Y * np.log(Y_hat))

def measure_accuracy(Y_hat,Y):
    Y_hat_indices=np.argmax(Y_hat,axis=0)
    Y_indices=np.argmax(Y,axis=0)
    return 100 * (np.sum(Y_hat_indices==Y_indices)/Y_hat.shape[1])

def sq_error_loss(Y_hat,Y):
    return np.sum((Y_hat-Y) * (Y_hat-Y))


def calculate_error_accuracy(X_train,Y_train,X_Val,Y_Val,layer_objects,L,error_type):                                           # Function returns Error and Accuracy as a tuple after taking Y_hat and Y  

    print("Entered Calculation Phase")
    forward_prop(layer_objects,X_train.transpose(),L)
    Y_hat=layer_objects[L-1].h_                                                                                                 # Later check if both Y_hat and Y have same shape
    Y=Y_train.transpose()
    
    forward_prop(layer_objects,X_Val.transpose(),L)
    Y_hat_val=layer_objects[L-1].h_
    Y_val=Y_Val.transpose()

    if(error_type=='cross'):
        train_loss=cross_entropy_loss(Y_hat,Y)
        val_loss=cross_entropy_loss(Y_hat_val,Y_val)
    elif (error_type=='sq_error'):
        train_loss=sq_error_loss(Y_hat,Y)
        val_loss=sq_error_loss(Y_hat_val,Y_val)

    train_accuracy=measure_accuracy(Y_hat,Y)
    val_accuracy=measure_accuracy(Y_hat_val,Y_val)

    return (train_loss,train_accuracy,val_loss,val_accuracy)       


# VANILLA GRADIENT DESCENT
def mini_batch_gradient_descent(X,Y,no_neurons_each_layer,L,eta=0.1,batch_size=16,max_epochs=10):

    num_steps_taken=0
    layer_objects=["layer0"]                                          # List which stores all layer objects, layer0 is simply put for easier indexing 
    for i in range(1,L):
        layer_objects.append(Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    for j in range(max_epochs):
        
        # Initialize
        num_points_seen=0
        total_grad_layer_objects=["layer0"]                                   # List which stores all grad layer objects, which are used to add and store gradients of all examples upto batch_size
        for i in range(1,L):
            total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

        #print(X.shape,Y.shape)

        for x,y in zip(X,Y):

            x = x[:,np.newaxis]
            #print(x.shape)
            y = y[:,np.newaxis]
            #print(y.shape)
            forward_prop(layer_objects,x,L)                     # Forward pass
            backward_prop(layer_objects,x,y,L)                  # Backward Pass

            for i in range(1,L):                                                                                            # Adding grad of each layer to each layer of Total Grad
                total_grad_layer_objects[i].grad_weight = total_grad_layer_objects[i].grad_weight + layer_objects[i].weight_grad
                total_grad_layer_objects[i].grad_bias = total_grad_layer_objects[i].grad_bias + layer_objects[i].bias_grad

            num_points_seen+=1

            if(num_points_seen%batch_size==0):
                num_steps_taken+=1
                for i in range(1,L):                                                                                    # Updating weight and bias of each layer with grad from Total Grad
                    layer_objects[i].weight=layer_objects[i].weight - eta*total_grad_layer_objects[i].grad_weight
                    layer_objects[i].bias=layer_objects[i].bias - eta*total_grad_layer_objects[i].grad_bias

                # Make the grad_objects 0 for next batch
                total_grad_layer_objects=["layer0"]
                for i in range(1,L):
                    total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
                
                if(num_steps_taken%500==0):
                    print("Num of steps completed:",num_steps_taken)
        print("Epoch completed:",j+1)

    return layer_objects

# MOMENTUM BASED GRADIENT DESCENT
def momentum_gradient_descent(X,Y,no_neurons_each_layer,L,eta=0.1,batch_size=16,max_epochs=10,gamma=0.9):

    num_steps_taken=0
    layer_objects=["layer0"]                                          # List which stores all layer objects, layer0 is simply put for easier indexing 
    for i in range(1,L):
        layer_objects.append(Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    # Initialize list which contains Momentum based update_t
    mom_update_objects=["layer0"]                                                                              
    for i in range(1,L):
        mom_update_objects.append(Momentum_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    for j in range(max_epochs):
        
        # Initialize
        num_points_seen=0
        total_grad_layer_objects=["layer0"]                                   # List which stores all grad layer objects, which are used to add and store gradients of all examples upto batch_size
        for i in range(1,L):
            total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

        for x,y in zip(X,Y):

            x = x[:,np.newaxis]
            y = y[:,np.newaxis]
            forward_prop(layer_objects,x,L)                     # Forward pass
            backward_prop(layer_objects,x,y,L)                  # Backward Pass

            for i in range(1,L):                                                                                            # Adding grad of each layer to each layer of Total Grad
                total_grad_layer_objects[i].grad_weight = total_grad_layer_objects[i].grad_weight + layer_objects[i].weight_grad
                total_grad_layer_objects[i].grad_bias = total_grad_layer_objects[i].grad_bias + layer_objects[i].bias_grad

            num_points_seen+=1

            if(num_points_seen%batch_size==0):
                num_steps_taken+=1

                for i in range(1,L):                                                                                    #Updating weight and bias of each layer with grad from Total Grad
                                                                                                                       
                    mom_update_objects[i].weight_update= gamma * mom_update_objects[i].weight_update + eta * total_grad_layer_objects[i].grad_weight
                    mom_update_objects[i].bias_update= gamma * mom_update_objects[i].bias_update + eta * total_grad_layer_objects[i].grad_bias

                    layer_objects[i].weight=layer_objects[i].weight - mom_update_objects[i].weight_update
                    layer_objects[i].bias=layer_objects[i].bias - mom_update_objects[i].bias_update

                # Make the grad_objects 0 for next batch
                total_grad_layer_objects=["layer0"]
                for i in range(1,L):
                    total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

                if(num_steps_taken%500==0):
                    print("Num of steps completed:",num_steps_taken)

        print("Epoch completed:", j+1)

    return layer_objects


# NAG GRADIENT DESCENT
def nag_gradient_descent(X,Y,no_neurons_each_layer,L,eta=0.1,batch_size=16,max_epochs=10,gamma=0.9):

    num_steps_taken=0
    layer_objects=["layer0"]                                          # List which stores all layer objects, layer0 is simply put for easier indexing 
    for i in range(1,L):
        layer_objects.append(Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    lookahead_objects=["layer0"]                                          # List which stores all LookAhead objects, layer0 is simply put for easier indexing 
    for i in range(1,L):
        lookahead_objects.append(LookAhead_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    # Initialize list which contains Momentum based update_t
    mom_update_objects=["layer0"]                                                                               
    for i in range(1,L):
        mom_update_objects.append(Momentum_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    for j in range(max_epochs):

        # Initialize num_points_seen and Total_Grad
        num_points_seen=0
        total_grad_layer_objects=["layer0"]                                   # List which stores all grad layer objects, which are used to add and store gradients of all examples upto batch_size
        for i in range(1,L):
            total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

        for i in range(1,L):                                                        # Storing the weight and bias before doing the look ahead and also updating the weights to look ahead
            lookahead_objects[i].weight_before_lh=layer_objects[i].weight
            lookahead_objects[i].bias_before_lh=layer_objects[i].bias
            layer_objects[i].weight= layer_objects[i].weight - gamma * mom_update_objects[i].weight_update
            layer_objects[i].bias= layer_objects[i].bias - gamma * mom_update_objects[i].bias_update 

        for x,y in zip(X,Y):
            
            x = x[:,np.newaxis]
            y= y[:,np.newaxis]
            forward_prop(layer_objects,x,L)                     # Forward pass
            backward_prop(layer_objects,x,y,L)                  # Backward Pass

            for i in range(1,L):                                                                                            # Adding grad of each layer to each layer of Total Grad
                total_grad_layer_objects[i].grad_weight = total_grad_layer_objects[i].grad_weight + layer_objects[i].weight_grad
                total_grad_layer_objects[i].grad_bias = total_grad_layer_objects[i].grad_bias + layer_objects[i].bias_grad

            num_points_seen+=1
            if(num_points_seen%batch_size==0):
                num_steps_taken+=1

                for i in range(1,L):                                                                                    # Updating weight and bias of each layer with grad from Total Grad
                                                                                                                        #
                    mom_update_objects[i].weight_update= gamma * mom_update_objects[i].weight_update + eta * total_grad_layer_objects[i].grad_weight
                    mom_update_objects[i].bias_update= gamma * mom_update_objects[i].bias_update + eta * total_grad_layer_objects[i].grad_bias

                    layer_objects[i].weight=lookahead_objects[i].weight_before_lh - mom_update_objects[i].weight_update         # Updates W and b with old weights and b
                    layer_objects[i].bias=lookahead_objects[i].bias_before_lh - mom_update_objects[i].bias_update

                for i in range(1,L):                                                        # Storing the weight and bias before doing the look ahead and also updating the weights to look ahead
                    lookahead_objects[i].weight_before_lh=layer_objects[i].weight
                    lookahead_objects[i].bias_before_lh=layer_objects[i].bias
                    layer_objects[i].weight= layer_objects[i].weight - gamma * mom_update_objects[i].weight_update
                    layer_objects[i].bias= layer_objects[i].bias - gamma * mom_update_objects[i].bias_update
                
                # Make the grad_objects 0 for next batch
                total_grad_layer_objects=["layer0"]
                for i in range(1,L):
                    total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

                if(num_steps_taken%500==0):
                    print("Num of steps completed:",num_steps_taken)
        print("Epoch Completed:", j+1)

    return layer_objects



#RMSprop learning Algorithm
class v_Layer:
    def __init__(self,layer_dim,prev_layer_dim):
        self.v_weight=np.zeros((layer_dim,prev_layer_dim))
        self.v_bias=np.zeros((layer_dim,1))

def do_rmsprop(X,Y,no_neurons_each_layer,L,eta=0.1,batch_size=16,max_epochs=10, eps = 1e-8, beta1 = 0.9): #beta1 value as given in lecture slides

    num_steps_taken=0
    layer_objects=["layer0"]                                          # List which stores all layer objects, layer0 is simply put for easier indexing 
    for i in range(1,L):
        layer_objects.append(Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    for j in range(max_epochs):
        
        # Initialize
        num_points_seen=0
        total_grad_layer_objects=["layer0"]                                   # List which stores all grad layer objects, which are used to add and store gradients of all examples upto batch_size
        for i in range(1,L):
            total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
            
        #initialising the velocity/momemtum variables
        v_layer_objects = ["layer0"]
        for i in range(1,L):
            v_layer_objects.append(v_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

        for x,y in zip(X,Y):
            x = x[:,np.newaxis]
            y = y[:,np.newaxis]
            forward_prop(layer_objects,x,L)                     # Forward pass
            backward_prop(layer_objects,x,y,L)                  # Backward Pass

            for i in range(1,L):                                                                                            # Adding grad of each layer to each layer of Total Grad
                total_grad_layer_objects[i].grad_weight = total_grad_layer_objects[i].grad_weight + layer_objects[i].weight_grad
                total_grad_layer_objects[i].grad_bias = total_grad_layer_objects[i].grad_bias + layer_objects[i].bias_grad

            num_points_seen+=1

            if(num_points_seen%batch_size==0):
                num_steps_taken+=1
                for i in range(1,L):
                    v_layer_objects[i].v_weight = beta1 * v_layer_objects[i].v_weight + (1-beta1) * total_grad_layer_objects[i].grad_weight**2
                    v_layer_objects[i].v_bias = beta1 * v_layer_objects[i].v_bias + (1-beta1) * total_grad_layer_objects[i].grad_bias**2
                for i in range(1,L):                                                                                    # Updating weight and bias of each layer with grad from Total Grad
                    layer_objects[i].weight=layer_objects[i].weight - (eta/(np.sqrt(eps + v_layer_objects[i].v_weight)))*total_grad_layer_objects[i].grad_weight
                    layer_objects[i].bias=layer_objects[i].bias - (eta/(np.sqrt(eps + v_layer_objects[i].v_bias)))*total_grad_layer_objects[i].grad_bias

                # Make the grad_objects 0 for next batch
                total_grad_layer_objects=["layer0"]
                for i in range(1,L):
                    total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

                if(num_steps_taken%500==0):
                    print("Num of steps completed:",num_steps_taken)
        print("Epoch Completed:", j+1)
    
    return layer_objects


#ADAM learning Algorithm

class v_Hat_Layer:
    def __init__(self,layer_dim,prev_layer_dim):
        self.v_hat_weight=np.zeros((layer_dim,prev_layer_dim))
        self.v_hat_bias=np.zeros((layer_dim,1))

class m_Layer:
    def __init__(self,layer_dim,prev_layer_dim):
        self.m_weight=np.zeros((layer_dim,prev_layer_dim))
        self.m_bias=np.zeros((layer_dim,1))

class m_Hat_Layer:
    def __init__(self,layer_dim,prev_layer_dim):
        self.m_hat_weight=np.zeros((layer_dim,prev_layer_dim))
        self.m_hat_bias=np.zeros((layer_dim,1))

def do_adam( X,Y,no_neurons_each_layer,L,eta=0.1,batch_size=16,max_epochs=10, eps = 1e-8, beta1 = 0.999, beta2 = 0.9): #beta1 and beta2 values as given in lecture slides

    num_steps_taken=0
                                                                                                                    #beta1 and beta2 are used for v and m variables respectively
    layer_objects=["layer0"]                                          # List which stores all layer objects, layer0 is simply put for easier indexing 
    for i in range(1,L):
        layer_objects.append(Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))

    for j in range(max_epochs):
        
        # Initialize
        num_points_seen=0
        total_grad_layer_objects=["layer0"]                                   # List which stores all grad layer objects, which are used to add and store gradients of all examples upto batch_size
        for i in range(1,L):
            total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
            
        #initialising the velocity/momemtum variables and velocity_hat variables
        v_layer_objects, v_hat_layer_objects = ["layer0"] , ["layer0"]
        for i in range(1,L):
            v_layer_objects.append(v_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
        for i in range(1,L):
            v_hat_layer_objects.append(v_Hat_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
            
        #initialising the moment variables , moment_hat variables
        m_layer_objects , m_hat_layer_objects = ["layer0"] , ["layer0"]
        for i in range(1,L):
            m_layer_objects.append(m_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
        for i in range(1,L):
            m_hat_layer_objects.append(m_Hat_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
        
        
        for x,y in zip(X,Y):
            x = x[:,np.newaxis]
            y = y[:,np.newaxis]
            forward_prop(layer_objects,x,L)                     # Forward pass
            backward_prop(layer_objects,x,y,L)                  # Backward Pass

            for i in range(1,L):                                                                                            # Adding grad of each layer to each layer of Total Grad
                total_grad_layer_objects[i].grad_weight = total_grad_layer_objects[i].grad_weight + layer_objects[i].weight_grad
                total_grad_layer_objects[i].grad_bias = total_grad_layer_objects[i].grad_bias + layer_objects[i].bias_grad

            num_points_seen+=1

            if(num_points_seen%batch_size==0):
                num_steps_taken+=1
                for i in range(1,L):
                    v_layer_objects[i].v_weight = beta1 * v_layer_objects[i].v_weight + (1-beta1) * total_grad_layer_objects[i].grad_weight**2
                    v_layer_objects[i].v_bias = beta1 * v_layer_objects[i].v_bias + (1-beta1) * total_grad_layer_objects[i].grad_bias**2
                for i in range(1,L):
                    m_layer_objects[i].m_weight = beta2 * m_layer_objects[i].m_weight + (1-beta2) * total_grad_layer_objects[i].grad_weight
                    m_layer_objects[i].m_bias = beta2 * m_layer_objects[i].m_bias + (1-beta2) * total_grad_layer_objects[i].grad_bias
                    
                #calculate m_hat and v_hat variables
                update_number = num_points_seen/batch_size
                for i in range(1,L):
                    v_hat_layer_objects[i].v_hat_weight , m_hat_layer_objects[i].m_hat_weight = v_layer_objects[i].v_weight/(1-beta1**update_number), m_layer_objects[i].m_weight/(1-beta2**update_number)
                    v_hat_layer_objects[i].v_hat_bias , m_hat_layer_objects[i].m_hat_bias = v_layer_objects[i].v_bias/(1-beta1**update_number), m_layer_objects[i].m_bias/(1-beta2**update_number)
                    
                for i in range(1,L):                                                                                    # Updating weight and bias of each layer with grad from Total Grad
                    layer_objects[i].weight=layer_objects[i].weight - (eta/(np.sqrt(eps + v_hat_layer_objects[i].v_hat_weight)))*m_hat_layer_objects[i].m_hat_weight
                    layer_objects[i].bias=layer_objects[i].bias - (eta/(np.sqrt(eps + v_hat_layer_objects[i].v_hat_bias)))*m_hat_layer_objects[i].m_hat_bias

                # Make the grad_objects 0 for next batch
                total_grad_layer_objects=["layer0"]
                for i in range(1,L):
                    total_grad_layer_objects.append(Total_Grad_Layer(no_neurons_each_layer[i],no_neurons_each_layer[i-1]))
                
                if(num_steps_taken%500==0):
                    print("Num of steps completed:",num_steps_taken)
        print("Epoch Completed:", j+1)

    return layer_objects



# Give input in (m*784) and (m*10) form.  no_neurons_each_layer is a list. 

def train(X_train,Y_train,X_Val,Y_Val,no_neurons_each_layer,L,batch_size,epochs,optimizer_type,eta=0,gamma=0,eps=0,beta1=0,beta2=0,error_type='cross'):                                         

    if(optimizer_type=='vanilla_gd'):
        print("Using Vanilla_gd")
        layer_objects= mini_batch_gradient_descent(X_train,Y_train,no_neurons_each_layer,L,eta,batch_size,epochs)               # Function trains and returns Layer_Objects which contain the weights and bias
    if(optimizer_type=='mom_gd'):
        print("Using Mom_gd")
        layer_objects=momentum_gradient_descent(X_train,Y_train,no_neurons_each_layer,L,eta,batch_size,epochs,gamma)
    if(optimizer_type=='nag_gd'):
        print("Using NAG_gd")
        layer_objects=nag_gradient_descent(X_train,Y_train,no_neurons_each_layer,L,eta,batch_size,epochs,gamma)
    if(optimizer_type=='rms_prop'):
        print("Using RMS Prop")
        layer_objects=do_rmsprop(X_train,Y_train,no_neurons_each_layer,L,eta,batch_size,epochs,eps,beta1)
    if(optimizer_type=='adam'):
        print("Using Adam")
        layer_objects=do_adam(X_train,Y_train,no_neurons_each_layer,L,eta,batch_size,epochs,eps,beta1,beta2)
    if(optimizer_type=='nadam'):
        pass

    train_loss,train_accuracy,val_loss,val_accuracy=calculate_error_accuracy(X_train,Y_train,X_Val,Y_Val,layer_objects,L,error_type)
    print("Train and Val loss:", train_loss,val_loss)
    print("Train and Val acc:", train_accuracy,val_accuracy)


no_neurons_each_layer=[784,128,128,10]                     # 784 for input layer, followed by hidden layers and then 10 for output 
L= len(no_neurons_each_layer)                            # L=5 


#train(X_train=X_train_vectorized[:54000],Y_train=Y_train[:54000],X_Val=X_train_vectorized[54000:],Y_Val=Y_train[54000:],no_neurons_each_layer=no_neurons_each_layer,L=L,batch_size=16,epochs=2,optimizer_type="vanilla_gd",eta=0.001,gamma=0,eps=0,beta1=0,beta2=0,error_type='cross')
#train(X_train=X_train_vectorized[:54000],Y_train=Y_train[:54000],X_Val=X_train_vectorized[54000:],Y_Val=Y_train[54000:],no_neurons_each_layer=no_neurons_each_layer,L=L,batch_size=16,epochs=2,optimizer_type="mom_gd",eta=0.001,gamma=0.9,eps=0,beta1=0,beta2=0,error_type='cross')
train(X_train=X_train_vectorized[:54000],Y_train=Y_train[:54000],X_Val=X_train_vectorized[54000:],Y_Val=Y_train[54000:],no_neurons_each_layer=no_neurons_each_layer,L=L,batch_size=32,epochs=5,optimizer_type="nag_gd",eta=0.001,gamma=0.9,eps=0,beta1=0,beta2=0,error_type='cross')
#train(X_train=X_train_vectorized[:54000],Y_train=Y_train[:54000],X_Val=X_train_vectorized[54000:],Y_Val=Y_train[54000:],no_neurons_each_layer=no_neurons_each_layer,L=L,batch_size=16,epochs=2,optimizer_type="rms_prop",eta=0.001,gamma=0,eps=1e-8,beta1=0.9,beta2=0,error_type='cross')
#train(X_train=X_train_vectorized[:54000],Y_train=Y_train[:54000],X_Val=X_train_vectorized[54000:],Y_Val=Y_train[54000:],no_neurons_each_layer=no_neurons_each_layer,L=L,batch_size=16,epochs=2,optimizer_type="adam",eta=0.001,gamma=0,eps=1e-8,beta1=0.999,beta2=0.9,error_type='cross')
