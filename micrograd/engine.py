class Value:
    """ stores a single scalar value and its gradient """

    # Explanation of how backward() and _backward works:
    # the _backward() function is defined inside __add__() and assigned to out._backward, the variables self.grad, 
    # other.grad, and out.grad refer to the specific Value objects involved in that particular addition operation. 
    # This behavior is due to Python's closure mechanism, which captures the surrounding scope's variables by reference.



    #__init__ is used when a val is initialized either by making a new Value or by using an op like __pow__
    #if a Value is created by an op, the values used in the op are store in children and the op stored in op
    #data = value of the object
    #childred = the values that were used in an op to create this value (e.g. __mul__)
    #_op = if an op (e.g. __add__) was to use to create this val, it's string to identify the op is store in _op
    def __init__(self, data, _children=(), _op=''):
        self.data = data  # the actual scalar value
        self.grad = 0  # the gradient of this value
        # internal variables used for autograd graph construction
        self._backward = lambda: None  # This will recalculate the gradients of the children's Values using the gradient of this Value
        self._prev = set(_children)  # tuple of children nodes. A set is used to avoid duplicates
        self._op = _op  # the operation that produced this node stored as a string, for graphviz / debugging / etc

    #using double underscores overloads the addition (+) operation in python, so + operations used by Value objects will be redirected here
    def __add__(self, other):
        #check if other is a Value object, if not then make it a Value. 
        #This allows us to add/subtract etc. without needed to make a Value object each time e.g. Value(4) + 5
        other = other if isinstance(other, Value) else Value(other)  # ensure other is a Value
        out = Value(self.data + other.data, (self, other), '+')  # create new Value for the sum

        #the underscore before backward indicates this is a private function not meant for API usage unlike the backward function defined later
        #define what happens in backprop for addition using the chain rule
        #this will be called when _backward is called on the output value, typically the loss function
        #the children's gradient is calculated by the their parent's (out) gradient using the chain rule
        def _backward():
            self.grad += out.grad  # using closer, self.grad is the gradient of the Value object that was used in the op to create out
            other.grad += out.grad  # propagate gradient to other

        #assigns the above _backward function to out._backward and is called during back prop to calculate out's children's grads.
        out._backward = _backward  # when _backward is called on the Value out, it will recalculate the gradients in self and other using out's gradient

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # ensure other is a Value
        out = Value(self.data * other.data, (self, other), '*')  # create new Value for the product

        def _backward():
            self.grad += other.data * out.grad  # propagate gradient to self
            other.grad += self.data * out.grad  # propagate gradient to other
        out._backward = _backward  # set the backward function

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"  # ensure other is int/float
        out = Value(self.data**other, (self,), f'**{other}')  # create new Value for the power

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad  # propagate gradient to self
        out._backward = _backward  # set the backward function

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')  # if self.data < 0, set to 0

        def _backward():
            self.grad += (out.data > 0) * out.grad  #if out.data is greater than 0, * by out.grad and add to self.grad
        out._backward = _backward  # set the backward function

        return out

    def backward(self):
        # backward is called on a single scalar output that is typically the loss loss
        # topological order all of the children in the graph
        topo = [] #empty list that will store the nodes
        visited = set() #will be populated by nodes as they are visited during back prop to avoid redundant vists. Set is used to avoid duplicates
        #the function within backward that builds the topological graph. It's used recursively to build the graph
        def build_topo(v): #depth 1st traversal
            if v not in visited: #check if the node has been visited
                visited.add(v) #if it hasn't been visitied, add it to the visited list
                for child in v._prev: #establish a for loop that goes through the children of the node (the values used in an op to create it)
                    build_topo(child) #recursively build the topo of v's child (which will then build the topo of v's child's, child recursively)
                topo.append(v) #append the node/value to topo, since this is done after the recursive traversal, it start with the deepest depth (closest to the input)
        build_topo(self) #the 1st call to building the topographical graph within back prop

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1 #as is standard, set the gradient of the final output value to 1
        # since the topo list is depth 1st (start with values closts to input and end with loss), 
        # the order is reversed to start with the node/Value that is typically the loss
        for v in reversed(topo): 
            v._backward() #call the back prop functions store in the loss Value 1st and go backward from there using a reversed topo list. Back prop is done using the chain rule to calculate the gradient of the children nodes

    # the following methods cleverly reuse the __add__ and __mul__ methods to perform new arithmetic operations

    #get the negative of a Value by making use of the existing __mul__ method
    def __neg__(self): # -self
        return self * -1

    #since this is called on the right side value (e.g, 5 + Value), self is the Value and other is the number value
    def __radd__(self, other): # other + self
        return self + other

    #a simple way of defining subtraction by using the existing __add__ method
    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    #remember that this is only called when a Value is on the right, e.g. 5 * x, and therefore rmul is called on the x which is store in self
    def __rmul__(self, other): # other * self
        return self * other

    #a simple way to divide using the existing __mul__ and __pow__ methods
    def __truediv__(self, other): # self / other
        return self * other**-1

    # a simple way to divide using the existing __mul__ and __pow__ methods when the Value is on the right
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    # repr is used to return a string representation of the Value object. This can be accessed by using print(Value)
    def __repr__(self):
        #return f"Value(data={self.data}, grad={self.grad})"  # string representation of Value
        # return the Value object, the gradient of the Value object, the operation used to create the Value object, and the children of the Value object as a string
        return f"Value(data={self.data}, grad={self.grad}, op={self._op}, prev={len(self._prev)})"
