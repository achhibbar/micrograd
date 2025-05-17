import random
from micrograd.engine import Value

class Module:

    # zero the gradients of all the parameters. This can be called on a Neuron or Layer object, 
    # but is more useful for the MLP object (which contains layers of neurons)
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    #nin = no of inputs to the neuron, nonlin = whether a non linearity is applied (set to True by default)
    #self allows the class to store its attributes for use later. Without self, the method would be static
    def __init__(self, nin, nonlin=True):
        #create a weight for every input using Value class from the engine, init with a random val between -1 and 1
        #inputs can be the inputs to a MLP or the outputs of neurons from previous layer in the MLP
        # since weights are created without any mathematical operations, they don't have any children
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        #create a single bias value for the neuron using Value class
        self.b = Value(0)
        #store whether the neuron has a non-linearity applied to it
        self.nonlin = nonlin

    #call the neuron using () and send as args the inputs
    #this is where the relationships between Values is created, as Values are mulitplied created parents and children
    def __call__(self, x):
        #zip together the weights (w) and inputs (x) to create a list of tuples
        #multiply each tuple wi, xi with each other, then sum them and add the bias value
        # in practice the bias is added to the 1st element of the sum, and then the sum is added to the 2nd element of the sum and so on
        # that's why when visualizing the graph using graphviz, the bias is added to the 1st (input * weight) and then the sum is added to the 2nd (input * weight) and so on
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        #apply a nonlinearity relu to the summed output + bias if the neuron has one
        return act.relu() if self.nonlin else act

    #return the list of weights and the bias for the neuron
    def parameters(self):
        return self.w + [self.b]

    #if the name of the neuron is typed and entered, this returns whether it uses ReLU or is linear (repr = representation overloaded)
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    #create a layer of Neurons, each with no. of inputs in nin and the no. of neurons in nout
    #**kwargs is used to indicate that the function can accept any number of keyword arguments
    def __init__(self, nin, nout, **kwargs):
        #create the neurons and store the list of them in self.neurons
        #neurons each have no. of weights in nin, and no. of neurons created are nout
        #**kwargs can pass whether the neurons should have a nonlin or not
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    # x are a set of inputs, either the MLP's inputs or the outputs from the preceding layer
    def __call__(self, x):
        #out is a list of outputs of all the neurons in the layer when input x is passed to them as an input
        out = [n(x) for n in self.neurons]
        #return a scaler value if there is only 1 neuron in the layer
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    #nin stores the no. of inputs to the MLP and nouts is a list that defines the no. of layers and no. of neurons in each layer
    #MLP(3, [4, 4, 1]) = 3 inputs, 2 hidden layers and 1 neuron in the ouput layer. Each hidden layer contains 4 neurons
    def __init__(self, nin, nouts):
        sz = [nin] + nouts #prepend the inputs to the MLP layers so we'll know how many weights the 1st hidden layer should have per neuron
        #create the no. of layers as there are elements in the nouts list (this doesn't contain the input layer)
        #each layers no. of neuron's inputs should be == the previous layer's no/ of neurons
        #1st hidden layer of MLP will have no. of weights per neuron == the no. of MLP inputs and no of neurons specified in nouts
        #since the MLP inputs aren't created here, the 1st layer created is the 1st hidden layer which needs to have weights == to no. of inputs
        #e.g. 1st hiddent layer created has inputs of sz[i] which is the no. of inputs of the MLP (3) and no of neurons = sz[i+1] which is 4 in this case
        #since sz prepended the no of inputs to the MLP architecture
        #nonlin=i!=len(nouts)-1 ensure each layer's neurons has a nonlinearity apart from the last layer (ouput layer)
        #if i is not == to the last layer, set nonlin to true, otherwise set to false
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    #do forward pass using the inputs passed in x
    #starting with the 1st hidden layer and finishing with the final output layer
    def __call__(self, x):
        for layer in self.layers:
            #at the beginning of the 1st iteration, x is the MLP's inputs
            #at the end of the iteration x is overwritten with the current layers output
            #the current layer's output is then fed into the next layer
            #this extablishes the relationships of the neurons between layers which are used for back prop
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
