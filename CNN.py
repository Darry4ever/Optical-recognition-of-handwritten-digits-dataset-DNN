from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from math import ceil
from numpy import mat
from sklearn.neighbors import KNeighborsClassifier

#load digits dataset from sklearn
digits = load_digits()
X = digits.data
Y = digits.target
X,Y=shuffle(X,Y)
#make the targets ranging in 0-1
X -= X.min()
X /= X.max()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
labels_train = LabelBinarizer().fit_transform(Y_train)
X = digits.data
Y = digits.target
X,Y = shuffle(X,Y)
images_and_labels = list(zip(digits.images, digits.target))
#the Convolutional layer of CNN
class ConvLayer():
    def __init__(self, n_filters, kernel_size, eta = 0.0001, activation='linear', input_shape=(28, 28, 1), padding='same'):
        self.a_in = None
        np.random.seed(100)
        # Initialize filters
        self.filters = self.initialize_filter(f_size=[n_filters, kernel_size[0], kernel_size[1]])
        self.n_filters = n_filters
        self.kernel_size = kernel_size[0]
        self.biases = np.zeros(n_filters)
        self.out = None
        self.z = []
        self.activation = activation
        # Learning rate
        self.eta = eta
    def feed_forward(self, images):
        """
        Feed forward for convolution layer
        :param images: input matrices/images
        :return: input convolved with the filters in convolution layer
        """
        self.a_in = images
        n_images, img_size, _ = np.shape(images)
        stride = 1
        out_img_size = int((img_size-self.kernel_size)/stride + 1)
        self.out = np.zeros(shape=[n_images*self.n_filters, out_img_size, out_img_size])
        self.z = []
        out_nr = 0
        for img_nr in range(n_images):
            image = self.a_in[img_nr, :, :]
            for filter_nr, filter in enumerate(self.filters):
                z = self.convolution2D(image, filter, self.biases[filter_nr])
                self.z.append(self.convolution2D(image, filter, self.biases[filter_nr]))
                self.out[out_nr, :, :] = self.activation_func(z)
                out_nr += 1
        return self.out
    def activation_func(self, a):#ReLU activation function
        a[a <= 0] = 0
        return a
    def convolution2D(self, image, filter, bias, stride=1, padding=0):
        '''
        Convolution of 'filter' over 'image' using stride length 'stride'
        Param1: image, given as 2D np.array
        Param2: filter, given as 2D np.array
        Param3: bias, given as float
        Param4: stride length, given as integer
        Return: 2D array of convolved image
        '''
        h_filter, _ = filter.shape  # get the filter dimensions
        if padding > 0:
            image = np.pad(image, pad_width=padding, mode='constant')
        in_dim, _ = image.shape  # image dimensions (NB image must be [NxN])
        out_dim = int(((in_dim - h_filter) / stride) + 1)  # output dimensions
        out = np.zeros((out_dim, out_dim))  # create the matrix to hold the values of the convolution operation
        # convolve each filter over the image
        # Start at y=0
        curr_y = out_y = 0
        # move filter vertically across the image
        while curr_y + h_filter <= in_dim:
            curr_x = out_x = 0
            # move filter horizontally across the image
            while curr_x + h_filter <= in_dim:
                # perform the convolution operation and add the bias
                out[out_y, out_x] = np.sum(filter * image[curr_y:curr_y + h_filter, curr_x:curr_x + h_filter]) + bias
                curr_x += stride
                out_x += 1
            curr_y += stride
            out_y += 1
        return out
    def softmax(self, raw_preds):
        '''
        pass raw predictions through softmax activation function
        Param1: raw_preds - np.array
        Return: np.array
        '''
        out = np.exp(raw_preds)  # exponentiate vector of raw predictions
        return out / np.sum(out)  # divide the exponentiated vector by its sum. All values in the output sum to 1.
    def initialize_filter(self, f_size, scale=1.0):
        """
        Initialize filters with random numbers
        """
        stddev = scale / np.sqrt(np.prod(f_size))
        return np.random.normal(loc=0, scale=stddev, size=f_size)
    def back_propagation(self, error):
        """
        Backpropagation in convolution layer
        :param error: error from layer L+1
        F = filter
        d_F = derivative with regards to filter
        d_X = error in input, input for backpropagation in layer L-1
        """
        error_out = np.zeros(shape=np.shape(self.a_in))
        n_images, _, _ = np.shape(self.a_in)
        # Backpropagate through activation function part
        for img_nr in range(n_images):
            image = self.a_in[img_nr, :, :]
            for filter_nr in range(len(self.filters)):
                prop_error = error[filter_nr, :, :]*self.d_activation(self.z[filter_nr])
                d_F = self.convolution2D(image, np.rot90(prop_error, 2), bias=0)

                # Update weights
                self.filters[filter_nr] += self.eta*d_F/np.shape(self.a_in)[0]
                self.biases[filter_nr] += self.eta*np.sum(prop_error)/np.shape(self.a_in)[0]
                error_out[img_nr, :, :] += self.convolution2D(prop_error, np.rot90(self.filters[filter_nr], 2), bias=0, padding=1)
        return error_out
    def d_activation(self, a): # Return ReLU derivative    
        a[a <= 0] = 0
        a[a > 0] = 1
        return a
class MaxPoolLayer():#ReLU activation function
    def __init__(self, stride):
        self.a_in = None
        self.stride = stride
        self.a_out = None
    def feed_forward(self, a_in):#Max pooling of input a_in
        self.a_in = a_in
        n_images, img_size, _ = np.shape(self.a_in)
        self.a_out = np.zeros(shape=[n_images, int(ceil(img_size/self.stride)), int(ceil(img_size/self.stride))])
        for img_nr in range(n_images):
            self.a_out[img_nr, :, :] = block_reduce(self.a_in[img_nr, :, :], (self.stride, self.stride), np.max, cval=-np.inf)
        return self.a_out
    def back_propagation(self, d_error):
        """
        Backpropagation on max pool layer
        :param d_error: error from subsequent layer
        :return: error of input to max pool layer
        """
        # Pass error to pixel with largest value
        e_i = 0
        e_j = 0
        i = 0
        j = 0
        d_p = np.zeros(self.a_in.shape)
        for img_nr in range(np.shape(self.a_in)[0]):
            while i + self.stride <= np.shape(d_p)[1]:
                e_j = 0
                j = 0
                while j + self.stride <= np.shape(d_p)[2]:
                    block = self.a_in[img_nr, i:i+self.stride, j:j+self.stride]
                    x, y = np.unravel_index(block.argmax(), [self.stride, self.stride])
                    d_p[img_nr, x+i, y+j] = d_error[img_nr, e_i, e_j]
                    e_j += 1
                    j += self.stride

                e_i += 1
                i += self.stride
        return d_p
    
class FullyConnectedLayer:#Fully connected layer class
    def __init__(self, n_categories, n_images, eta=0.001, activation='softmax'):
        # Input
        self.a_in = None
        # Vectorized
        self.S = None
        # Weights
        self.weights = None
        self.bias = np.random.randn(n_categories)
        self.n_categories = n_categories
        self.n_images = n_images
        self.activation = activation
        self.out = None
        # Learning rate
        self.eta = eta
    def update_images(self, n_images): #Change nr of outputs
        self.n_images = n_images
    def new_shape(self, a):#Vectorize input to FC layer
        # Vectorize step
        lengde, size, size = np.shape(a)
        per_image = int(lengde / self.n_images)
        reshaped = np.zeros(shape=[self.n_images, per_image*size*size])
        count = 0
        img_count = per_image
        for img in range(self.n_images):
            vec = np.zeros(per_image*size*size)
            pos = 0
            while count < img_count:
                vec[pos:pos + size*size] = (np.ravel(a[count].T))
                count += 1
                pos += size*size
            reshaped[img, :] = vec
            img_count += per_image
        return reshaped
    def reshape_back(self, a):#Undo vectorization
        out = np.zeros(np.shape(self.a_in))
        length, size, size = np.shape(self.a_in)
        l = 0
        for i in range(np.shape(a)[0]):
            pos = 0
            while pos < np.shape(a)[1]:
                out[l, :, :] = a[i, pos:pos+size*size].reshape((size, size)).T
                pos += size*size
                l += 1
        return out
    def feed_forward(self, a_in): #Feed forward step in FC layer
        # Vectorize step
        self.a_in = a_in
        self.S = self.new_shape(a_in)
        if self.weights is None:
            # initialize weights
            self.weights = np.random.randn(self.n_categories, np.shape(self.S)[1])
        self.out = self.activation_function(np.matmul(self.S, self.weights.T) + self.bias)
        return self.out
    def activation_function(self, a):
        if self.activation == 'softmax':
            return np.exp(a) / (np.exp(a).sum(axis=1))[:, None]
    def back_propagation(self, error):
        """
        Backpropagation of Fully Connected layer
        :param error: Error in final output w.r.t softmax
        :return: error in input to FC layer
        """
        # Error w.r.t weights
        w_d = np.matmul(error.T, self.S)
        b_d = np.sum(error)/len(error)
        # Calculate error in input
        d_S = np.matmul(error, self.weights)
        # Reshape
        d_a_in = self.reshape_back(d_S)
        # Update weights & bias
        self.weights -= self.eta*w_d/np.shape(error)[0]
        #print(self.weights)
        self.bias -= self.eta*b_d/np.shape(error)[0]
        return d_a_in
    
class CNN():
    def __init__(self, X_data, Y_data):
        self.layers = []
        self.X = X_data
        self.Y = Y_data
        self.n_categories = np.shape(self.Y)[1]
        self.n_images = np.shape(self.Y)[0]
    # Functions to add various layers to the model
    def add_conv_layer(self, n_filters, kernel_size, eta=0.001):
        self.layers.append(ConvLayer(n_filters, kernel_size, eta))
    def add_maxpool_layer(self, stride=2):
        self.layers.append(MaxPoolLayer(stride))
    def add_fullyconnected_layer(self, eta=0.1):
        self.layers.append(FullyConnectedLayer(self.n_categories, self.n_images, eta))
    # Define data for training or testing
    def new_input(self, X_data, Y_data):
        self.X = X_data
        self.Y = Y_data
        self.n_images = np.shape(self.Y)[0]
        self.layers[-1].update_images(self.n_images)
    # Forward propagation through all layers
    def forward_propagation(self):
        input = self.X
        for layer in self.layers:
            new_input = layer.feed_forward(input)
            input = new_input
        return new_input
    # Back propagation through all layers in revers order
    def back_propagation(self, y_hat):
        # Derivative of loss function wrt softmax
        error = y_hat - self.Y
        # propagate error through layers
        for layer in reversed(self.layers):
            new_error = layer.back_propagation(error)
            error = new_error
    # Loss function for multiclass classifier
    def loss(self, y, target):
        return -np.sum(target*np.log(y), axis=1)
    # Predict class from model output
    def predict(self, y_hat):
        out = np.zeros(shape=np.shape(y_hat))
        # Predict by setting largest propbability to 1
        out[np.arange(np.shape(y_hat)[0]), np.argmax(y_hat, axis=1).T] = 1
        return out
    # Accuracy
    def accuracy(self, y_hat):
        y_ = self.predict(y_hat)
        a = np.sum(np.all(y_ == self.Y, axis=1))
        print( a / np.shape(y_)[0])
        
def transform_targets(targets):
    #transform targets from a n array with values 0-9 to a nx10 array where each row is zero, except at the indice corresponding to the value in the original array
    n = len(targets)
    new_targets = np.zeros([n, 10])
    for i in range(n):
        value = int(targets[i])
        new_targets[i, value] = 1.0
    return new_targets

def transform_targets_back(targets):
    return np.where(targets == 1)[1]
#load cnn from saved file CnnModel and output the training and test accuracy
def loadCNN():
    X_train, X_test, Y_train, Y_test = train_test_split(digits.images, transform_targets(digits.target), test_size=0.2)
    model = joblib.load('CnnModel.m') 
    model.new_input(X_train, Y_train)
    print('Training data accuracy')
    model.accuracy(Y_train)# Training data accuracy
    model.new_input(X_test, Y_test)
    predict_test = model.forward_propagation()
    print('Test data accuracy')
    model.accuracy(predict_test)
#train cnn   
def trainCNN(X_train,Y_train,X_test,Y_test,rangeNum):
    nn_start_time = time.time()
    model = CNN(X_train[0:100, :], Y_train[0:100, :])
    model.add_conv_layer(n_filters=5, kernel_size=[2, 2])#add layers to cnn model
    model.add_maxpool_layer(stride=2)
    model.add_conv_layer(n_filters=2, kernel_size=[2, 2])
    model.add_maxpool_layer(stride=2)
    model.add_fullyconnected_layer()
    # Train model with batch gradient descent
    n_images = np.shape(X_train)[0]
    indices = np.arange(0, n_images)
    batch = 10
    for i in range(rangeNum):#iterate number of rangeNum times
        for j in range(int(n_images/batch)):
            r_indices = np.random.choice(indices, size=batch)
            model.new_input(X_train[r_indices, :], Y_train[r_indices, :])
            pred = model.forward_propagation()
            model.back_propagation(pred)
    nn_end_time = time.time()
    model.new_input(X_train, Y_train)
    print('Training data accuracy:')
    model.accuracy(Y_train)#calculate test accuracy
    model.new_input(X_test, Y_test)
    predict_test = model.forward_propagation()#calculate test accuracy
    print('Test data accuracy:')
    model.accuracy(predict_test)
    print('train my cnn uses time:',nn_end_time-nn_start_time)#output the consumed time
    return model
#save cnn model to CnnModel file
def saveCNN(model):
    joblib.dump(model, 'CnnModel.m')
#cross validation of CNN    
def CrossValidationCNN():
    x = digits.images[:1795]
    y = digits.target[:1795]
    x,y = shuffle(x,y)
    folds = 5   #5_fold cross validation
    X_folds = []
    y_folds = []
    X_folds = np.vsplit(x,folds)
    y_folds = np.hsplit(y,folds) 
    print('\nPlease wait for around 5 mins for the cross validation of my cnn')
    nn_start_time = time.time()
    for i in range(folds):#split X_train and X_test datasets
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:]) 
        X_val = X_folds[i]
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        y_train=transform_targets(y_train)
        y_val = y_folds[i]
        y_val=transform_targets(y_val)
        print ('\n',i+1,':', 'Total digits dataset shape: ',y_train.shape[0], '. Shape for this test dataset: ',y_val.shape[0])
        trainCNN(X_train,y_train,X_val,y_val,3)  
    nn_end_time = time.time()
    print('cross validation my CNN uses time:',nn_end_time-nn_start_time)
#the confusion matrix method
def confusionmatrix(actual, predicted, normalize = False):
    """
    Generate a confusion matrix for multiple classification
    @params:
        actual      - a list of integers or strings for known classes
        predicted   - a list of integers or strings for predicted classes
        normalize   - optional boolean for matrix normalization
    @return:
        matrix      - a 2-dimensional list of pairwise counts
    """
    unique = sorted(set(actual))
    matrix = [[0 for _ in unique] for _ in unique]
    imap   = {key: i for i, key in enumerate(unique)}
    # Generate Confusion Matrix
    for p, a in zip(predicted, actual):
        matrix[imap[p]][imap[a]] += 1
    # Matrix Normalization
    if normalize:
        sigma = sum([sum(matrix[imap[i]]) for i in unique])
        matrix = [row for row in map(lambda i: list(map(lambda j: j / sigma, i)), matrix)]
    return matrix
#to plot the confusion matrix of cnn
def CMOfCNN():
    X_train, X_test, Y_train, Y_test = train_test_split(digits.images, transform_targets(digits.target), test_size=0.2)
    model = joblib.load('CnnModel.m')
    model.new_input(X_test, Y_test)
    predict_test = model.forward_propagation()
    pred = transform_targets_back(model.predict(predict_test))
    cm=confusionmatrix(transform_targets_back(Y_test), pred) #call confusionmatrix()method to plot the confusion matrix
    print('\nThe confusion matrix of cnn:\n', mat(cm))
#to plot the ROC curve of cnn
def plotROCCNN():
    # Test data accuracy
    X_train, X_test, Y_train, Y_test = train_test_split(digits.images, transform_targets(digits.target), test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(digits.images, digits.target, test_size=0.2)    
    predictions_confidence=[]
    model = joblib.load('CnnModel.m')#load model from file
    model.new_input(X_test, Y_test)
    predict_test = model.forward_propagation()
    for i in range (0,360):
        predictions_confidence.append(predict_test[i][1])
    predictions=[]
    for i in range(0,360):
        out=predict_test[i][1]
        predictions.append(out)
    y_test_pre=[]
    num_neg=0
    num_pos=0#calculate the number of positive and negative digits
    for i in y_test:
        if i==9:
            y_test_pre.append(0)#change the value of class-9 to 0
            num_neg=num_neg+1
        else:
            y_test_pre.append(1)#change the value of all the other classes to 0
            num_pos=num_pos+1
    sorted_pre=[]#sorted confidence
    sorted_y_test_pre=[]#sorted y test value of 1 and 0
    for i in range(0,360):
        pre_max=np.max(predictions)
        sorted_pre.append(pre_max)
        pre_max_index=np.argmax(predictions)
        sorted_y_test_pre.append(y_test_pre[pre_max_index])
        predictions = np.delete(predictions, pre_max_index)
    ROC(sorted_pre,sorted_y_test_pre, num_neg, num_pos)
#the method the plot the ROC curve
def ROC(confidence,y_pre,num_neg, num_pos):    
    fprs = [0]
    tprs = [0]
    TP=0
    FP=0
    last_TP=0
    for i in range(0, 360):
        if i>0 and confidence[i]!=confidence[i-1] and y_pre[i]==0 and TP>last_TP:
            FPR=FP/num_neg
            TPR=TP/num_pos
            fprs.append(FPR)
            tprs.append(TPR)
            last_TP=TP
        if y_pre[i]==1:
            TP=TP+1
        else:
            FP=FP+1
    FPR=FP/num_neg
    TPR=TP/num_pos
    fprs.append(1)
    tprs.append(1)
    for i in range(0,len(fprs)):
        if fprs[i]>tprs[i]:
            middle=tprs[i]
            tprs[i]=fprs[i]
            fprs[i]=middle
        if fprs[i]>1:
            fprs[i]=1
        if tprs[i]>1:
            tprs[i]=1
    plt.figure()
    lw = 2
    plt.plot(fprs, tprs, color='darkorange',lw=lw, 
             label='ROC curve ' )
    plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()  
    
###########construct Neural Network model################
def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)
def logistic(x):
    return 1 / (1 + np.exp(-x))
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))
def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

class NeuralNetwork:
    #constructor
    def __init__(self, layers, activation='logistic'):
        '''
        :param layers: list
        :param activation: activation function
        '''
        self.layers = layers
        #choose the activation function to be used later
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        #define the number of layers
        self.num_layers = len(layers)
        #produce the number of biases
        self.biases = [np.random.randn(x) for x in layers[1:]]
        #randomly produce the weights of each layer, in the range（-1,1）
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]
    #train the model
    def fit(self, X, y, learning_rate=0.2, epochs=1):
        '''
        :param self: the object itself
        :param X: training dataset
        :param y: training dataset target
        :param learning_rate: learning rate
        :param epochs: training times
        :return: void
        '''
        for k in range(epochs):
            #iterate the dataset each time
            for i in range(len(X)):
                i = np.random.randint(X.shape[0])
                #save the input of this layer and the output of next layers
                activations = [X[i]]
                for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activations[-1])+b
                    #calculate the output
                    output = self.activation(z)
                    #save the output into list
                    activations.append(output)
                error = y[i]-activations[-1]
                #calculate the error rate 
                deltas = [error * self.activation_deriv(activations[-1])]
                # calculation of hidden layer's error rate
                for l in range(self.num_layers-2, 0, -1):
                    deltas.append(self.activation_deriv(activations[l]) * np.dot( deltas[-1],self.weights[l]))
                #reverse the deltas
                deltas.reverse()
                #update the weights and biases
                for j in range(self.num_layers-1):
                    layers = np.array(activations[j])
                    delta = learning_rate * ((np.atleast_2d(deltas[j]).T).dot(np.atleast_2d(layers)))
                    #update the weights
                    self.weights[j] += delta
                    delta = learning_rate * deltas[j]
                    #update the weights
                    self.biases[j] += delta       
    def predict(self, x):
        '''
        :param x: test dataset
        :return: the predictions 
        '''
        for b, w in zip(self.biases, self.weights):
            # 计算权重相加再加上偏向的结果
            z = np.dot(w, x) + b
            # 计算输出值
            x = self.activation(z)
        return x
    def score(self,x,y): #calculate the score(accuracy)
        pred = self.predict(x)
        err = 0.0
        for i in range(x.shape[0]):
            if pred[i]!=y[i]:
                err = err+1
        return 1-float(err/x.shape[0])
    def scoreNN(self,X_val, y_val): #calculate the accuracy of NN
        predictions = []
        err=0
        for i in range(y_val.shape[0]):
            out = self.predict(X_val[i])
            predictions.append(np.argmax(out))
        for i in range(y_val.shape[0]):
            #print(predictions[i], y_val[i])
            if predictions[i]!=y_val[i]:
                err = err+1
        print('error number:', err)
        accu=err/y_val.shape[0]
        nnscore=1-accu
        return nnscore
#train the nn model
def trainNN():
    print('Please wait for no more than 1 mins')
    nn = NeuralNetwork([64, 100, 10], 'logistic')
#train the model
    nn.fit(X_train, labels_train, learning_rate=0.2, epochs=100)
#save the model
    print('training accuracy:', nn.scoreNN(X_train,Y_train))
    print('testing accuracy:',nn.scoreNN(X_test,Y_test))
    joblib.dump(nn, 'nnModel.m')
    return nn
#load the nn model
def loadNN():
    nn = joblib.load('nnModel.m')
    print('Training accuracy',nn.scoreNN(X_train,Y_train))
    print('Testing accuracy',nn.scoreNN(X_test,Y_test))
#calculate the score of nn
def ScoreNN(X_val, y_val):   
    nn = joblib.load('nnModel.m')
    predictions = []
    err=0
    #predict the target of test dataset
    for i in range(y_val.shape[0]):
        out = nn.predict(X_val[i])
        predictions.append(np.argmax(out))
    for i in range(y_val.shape[0]):
        if predictions[i]!=y_val[i]:
            err = err+1
    print('error number:', err)
    accu=err/y_val.shape[0]
    nnscore=1-accu
    return nnscore
# the cross validation of NN
def CrossValidationNN():
    x = digits.data[:1795]
    y = digits.target[:1795]
    x,y = shuffle(x,y)
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.vsplit(x,folds)
    y_folds = np.hsplit(y,folds) 
    nn_start_time = time.time()
    nn = NeuralNetwork([64, 100, 10], 'tanh')
#split the dataset with 5-folds
    for i in range(folds):
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:]) 
        X_val = X_folds[i]
        Y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        Y_val = y_folds[i]
        print ('\n',i+1,':', 'Total digits dataset shape: ',Y_train.shape, 'Shape for this test dataset: ',Y_val.shape)
        nn.fit(X_train, labels_train, learning_rate=0.2, epochs=100)
        print('training accuracy:', ScoreNN(X_train,Y_train))
        print('testing accuracy:', ScoreNN(X_val, Y_val))
    nn_end_time = time.time()
    print('cross validation my nn uses time:',nn_end_time-nn_start_time)
#the confusion matrix of nn
def CMOfNN():
    nn = joblib.load('nnModel.m')
    a = []
    a.append(Y_test)
    a=a[0]
    b=[]
    for i in range(Y_test.shape[0]):
        out = nn.predict(X_test[i])
        b.append(np.argmax(out))
    cm=confusionmatrix(a, b)
    print('\nThe confusion matrix of nn:\n', mat(cm))
#plot the roc curve of nn
def plotROC_NN():
    NN = NeuralNetwork([64,100,10])
    print('Please wait for around 10 seconds')
    NN.fit(X_train,labels_train,epochs=10)
    predictions=[]
    for i in range(X_test.shape[0]):
        out = NN.predict(X_test[i])      
        out = 1-NN.predict(X_test[i])[9]
        predictions.append(out)
    pre=predictions
    num_neg=0
    num_pos=0
    y_test_pre=[]
    for i in Y_test:
        if i==9:
            y_test_pre.append(0)
            num_neg=num_neg+1#calculate the number of negative digits
        else:
            y_test_pre.append(1)
            num_pos=num_pos+1
    sorted_pre=[]#sorted confidence
    sorted_y_test_pre=[]#sorted y test value of 1 and 0
    for i in range(0,360):#sort the two lists
        pre_max=np.max(pre)
        sorted_pre.append(pre_max)
        pre_max_index=np.argmax(pre)
        #print(pre_max_index)
        sorted_y_test_pre.append(y_test_pre[pre_max_index])
        pre = np.delete(pre, pre_max_index)
    ROC(sorted_pre,sorted_y_test_pre, num_neg, num_pos)
#the class for my knn model
class KNNClassfier(object):
    def fit(self,X, Y):     
        self.x = X
        self.y = Y
    def predict(self,X_test):
        output = np.zeros((X_test.shape[0],1))
        for i in range(X_test.shape[0]):
            dis = [] 
            for j in range(self.x.shape[0]):
                if self.distance == 'euc': #Euclidean Distance
                    dis.append(np.linalg.norm(X_test[i]-self.x[j,:]))
            #print('dis',dis)
            labels = []
            index=sorted(range(len(dis)), key=dis.__getitem__)
            for j in range(self.k):
                labels.append(self.y[index[j]])
            #print('labels',labels)
            counts = []
            for label in labels:
                counts.append(labels.count(label))
            #print(counts)
            output[i] = labels[np.argmax(counts)]
        return output
    def confidence(self,X_test):#calculate the confidence 
        confidence=[]
        for i in range(X_test.shape[0]):
            dis = [] 
            for j in range(self.x.shape[0]):
                if self.distance == 'euc': #Euclidean Distance
                    dis.append(np.linalg.norm(X_test[i]-self.x[j,:]))
            labels = []
            index=sorted(range(len(dis)), key=dis.__getitem__)
            for j in range(self.k):
                labels.append(self.y[index[j]])
            temp = 0
            for i in labels:
                if labels.count(i) > temp:
                    temp = labels.count(i)
            confidence.append(temp/9)
        return confidence
    def score(self,x,y): #calculate the score(accuracy)
        pred = self.predict(x)
        err = 0.0
        for i in range(x.shape[0]):
            if pred[i]!=y[i]:
                err = err+1
        return 1-float(err/x.shape[0])
    def __init__(self, k=5, distance='euc'):
        self.k = k
        self.distance = distance
        self.x = None
        self.y = None
#do cross validation on my knn    
def CrossValidationMyKNN():
    x = digits.data[:1795]
    y = digits.target[:1795]
    x,y = shuffle(x,y)
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.vsplit(x,folds)
    y_folds = np.hsplit(y,folds) 
#split the train sets and validation sets
    print('\nPlease wait for no more than 2 mins for the cross validation of my knn')
    myknn_start_time = time.time()
    for i in range(folds):
        clf = KNNClassfier(k=3)
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:]) 
        X_val = X_folds[i]
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        y_val = y_folds[i]
        print ('\n', i+1, ":", 'train size:',X_train.shape[0],'test size:', X_val.shape[0])
        clf.fit(X_train,y_train)
        print('training accuracy:', clf.score(X_train,y_train))
        print('testing accuracy:', clf.score(X_val, y_val))
    myknn_end_time = time.time()
    print('cross validation myknn uses time:',myknn_end_time-myknn_start_time)
#do cross validation on sklearn knn  
def CrossValidationSkLearnKNN():
    print('\nPlease wait for no more than 10s for the cross validation of sklearn knn')
    x = digits.data[:1795]
    y = digits.target[:1795]
    x,y = shuffle(x,y)
    knn = KNeighborsClassifier(n_neighbors=3)
    folds = 5
    X_folds = []
    y_folds = []
    X_folds = np.vsplit(x,folds)
    y_folds = np.hsplit(y,folds) 
    #split the train sets and validation sets
    for i in range(folds):
        knn = KNeighborsClassifier(n_neighbors=3)
        X_train =np.vstack(X_folds[:i] + X_folds[i+1:]) 
        X_val = X_folds[i]
        y_train = np.hstack(y_folds[:i] + y_folds[i+1:])
        y_val = y_folds[i]
        print ('\n', i+1, ":", 'train size:',X_train.shape[0],'test size:', X_val.shape[0])
        knn.fit(X_train,y_train)
        print('training accuracy:', knn.score(X_train,y_train))
        print('testing accuracy:', knn.score(X_val, y_val))
#the confusion matrix of sklearn knn
def CMOfsklearnKnn():
    knn = KNeighborsClassifier(n_neighbors=3)
    # Fit the classifier to the training data
    knn.fit(X_train,Y_train)
    a = []
    a.append(Y_test)
    a=a[0]
    b=[]
    for i in range(0,359):
        b.append(knn.predict(X_test[i:i+1])[0])
    cm=confusionmatrix(a, b)
    print('\nThe confusion matrix of sklearn knn:\n', mat(cm))
#the confusion matrix of my knn
def CMOfMyKnn():
    print("Please wait for no more than 10s to print the confusion matrix of both models.")
    clf = KNNClassfier(k=3)
    clf.fit(X_train,Y_train)
    a = []
    a.append(Y_test)
    a=a[0]
    b=[]
    for i in range(0,359):
        predictnum=clf.predict(X_test[i:i+1])
        predictnum=int(predictnum)
        b.append(predictnum)
    cm=confusionmatrix(a, b)
    print('\nThe confusion matrix of my knn:\n', mat(cm))
#to plot the roc curve of sklearn knn
def ROCcurvesklearnKnn():
    print('ROC curve of sklearn Knn:')
    algo = KNeighborsClassifier(n_neighbors=9)
    algo.fit(X_train,Y_train)
    num_neg=0
    num_pos=0
    y_test_pre=[]
    for i in Y_test:
        if i==1:
            y_test_pre.append(0)
            num_pos=num_pos+1#calculate the number of positive digits
        else:
            y_test_pre.append(1)
            num_neg=num_neg+1
    test_predict_prob = algo.predict_proba(X_test)  
    predictions=[]
    for i in range(0,360):
        out=test_predict_prob[i][1]
        predictions.append(out)
    #calculate the confidence
    test_predict_prob = test_predict_prob[:, 1]
    ROC(test_predict_prob,y_test_pre,num_neg, num_pos)
#to plot the roc curve of sklearn knn
def ROCcurveMyKnn():
    print('ROC curve of my Knn:')
    X = digits.data
    Y = digits.target
    X,Y = shuffle(X,Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    clf = KNNClassfier(k=9)
    clf.fit(X_train,Y_train)
    num_pos=0
    num_neg=0
    y_test_pre=[]
    for i in Y_test:
        if i==5:
            y_test_pre.append(0)
            num_pos=num_pos+1#calculate the number of positive digits
        else:
            y_test_pre.append(1)
            num_neg=num_neg+1
    confidence=clf.confidence(X_test)
    sorted_pre=[]#sorted confidence
    sorted_y_test_pre=[]#sorted y test value of 1 and 0
    for i in range(0,360):#sort the two lists
        pre_max=np.max(confidence)
        sorted_pre.append(pre_max)
        pre_max_index=np.argmax(confidence)
        sorted_y_test_pre.append(y_test_pre[pre_max_index])
        confidence = np.delete(confidence, pre_max_index)
    ROC(sorted_pre,sorted_y_test_pre,num_neg, num_pos)
#the user interface    
def UI():            
    print('\n\nPlease choose one option from the following (Enter number from 1 to 11):\n\
      1. F1: Load NN model and show results\n\
      2. F2: Train and save NN model\n\
      3. F3: Cross validation of NN\n\
      4. F4: Confusion matrix of NN\n\
      5. F5: ROC curve of NN\n\
      6. F6: Load CNN model and show results\n\
      7. F7: Train and save CNN model\n\
      8. F8: Cross validation of CNN\n\
      9. F9: Confusion matrix of CNN\n\
      10.F10:ROC curve of CNN\n\
      11.F11:Cross Validation, Confusion matrix and ROC curve of both traditional models.\n\
      12.The end')
    while  True :   #input validation
        UserInput = input()
        try:
            if isinstance(eval(UserInput) ,(int))==True and 13>eval(UserInput) >0:
                InputNum=int(UserInput)
                break
            else:
                print("Invalid input with a wrong number. Please input again.")
        except:
            print("Invalid input with digits or other input. Please input again.")
    if InputNum==1:
        loadNN()
        UI()
    elif InputNum==2: 
        trainNN()
        UI()         
    elif InputNum==3:
        CrossValidationNN()  
        UI()
    elif InputNum==4:
        CMOfNN()
        UI()
    elif InputNum==5:
        plotROC_NN()       
        UI()
    elif InputNum==6: 
        loadCNN()
        UI()
    elif InputNum==7: 
        print('Please wait for around 2 mins.')
        X_train, X_test, Y_train, Y_test = train_test_split(digits.images, transform_targets(digits.target), test_size=0.2)
        model=trainCNN(X_train,Y_train,X_test,Y_test,5)
        saveCNN(model)
        UI()  
    elif InputNum==8:
        CrossValidationCNN()
        UI()
    elif InputNum==9:
        CMOfCNN()
        UI()
    elif InputNum==10:
        plotROCCNN()
        UI()
    elif InputNum==11:
        ROCcurveMyKnn()
        ROCcurvesklearnKnn()    
        CMOfMyKnn()
        CMOfsklearnKnn()
        CrossValidationSkLearnKNN()
        CrossValidationMyKNN()
        UI()           
    else:
        print('The end.')
UI()