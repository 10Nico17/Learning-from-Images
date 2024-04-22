import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import torch


# set a fixed seed for reproducability
np.random.seed(0)

nn_img_size = 32
num_classes = 3
learning_rate = 0.0001
num_epochs = 500
#num_epochs = 5

batch_size = 4

loss_mode = 'crossentropy'   #mse
loss_train_hist = []

##################################################
## Please implement a two layer neural network  ##
##################################################


def relu(x):
    """ReLU activation function"""
    return np.maximum(x, 0)


def relu_derivative(output):
    """derivative of the ReLU activation function"""
    output[output <= 0] = 0
    output[output > 0] = 1
    return output


def softmax(z):
    """softmax function to transform values to probabilities"""
    z -= z.max()
    z = np.exp(z)
    sum_z = z.sum(1, keepdims=True)
    return z / sum_z


def loss_mse(activation, y_batch):
    """mean squared loss function"""
    # use MSE error as loss function
    # Hint: the computed error needs to get normalized over the number of samples
    loss = ((activation - y_batch)**2).sum()
    mse = 1.0 / activation.shape[0] * loss
    return mse


def loss_crossentropy(activation, y_batch):
    """cross entropy loss function"""
    batch_size = y_batch.shape[0]
    loss = (-y_batch * np.log(activation)).sum() / batch_size
    return loss


def loss_deriv_mse(activation, y_batch):
    """derivative of the mean squared loss function"""
    dCda2 = (1 / activation.shape[0]) * (activation - y_batch)
    return dCda2


def loss_deriv_crossentropy(activation, y_batch):
    """derivative of the mean cross entropy loss function, that includes the derivate of the softmax
       for further explanations see here: https://deepnotes.io/softmax-crossentropy
    """
    batch_size = y_batch.shape[0]
    dCda2 = activation
    dCda2[range(batch_size), np.argmax(y_batch, axis=1)] -= 1
    dCda2 /= batch_size
    return dCda2


def setup_train():
    """train function"""
    # load and resize train images in three categories
    # cars = 0, flowers = 1, faces = 2 ( true_ids )
    train_images_cars = glob.glob('./images/db/train/cars/*.jpg')
    train_images_flowers = glob.glob('./images/db/train/flowers/*.jpg')
    train_images_faces = glob.glob('./images/db/train/faces/*.jpg')
    if not train_images_cars or not train_images_flowers or not train_images_faces:
        raise ValueError(
            'No image found! Please make sure the images are in the correct location.'
        )

    train_images = [train_images_cars, train_images_flowers, train_images_faces]
    num_rows = len(train_images_cars) + len(train_images_flowers) + len(
        train_images_faces)
    X_train = np.zeros((num_rows, nn_img_size * nn_img_size))
    y_train = np.zeros((num_rows, num_classes))

    counter = 0
    for (label, fnames) in enumerate(train_images):
        for fname in fnames:
            print(label, fname)
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (nn_img_size, nn_img_size),
                             interpolation=cv2.INTER_AREA)
            # print(label, fname, img.shape)
            # fill matrices X_train - each row is an image vector
            # y_train - one-hot encoded, put only a 1 where the label is correct for the row in X_train
            y_train[counter, label] = 1
            X_train[counter] = img.flatten().astype(np.float32)
            counter += 1
    # print(y_train)
    return X_train, y_train


def forward(X_batch, y_batch, W1, W2, b1, b2):
    """forward pass in the neural network """
    ### YOUR CODE ####setup_train
    o1 = np.dot(X_batch, W1) + b1
    #print('o1: ', o1.shape)
    a1 = relu(o1)
    #print('a1: ', a1.shape)
    o2 = np.dot(a1, W2) + b2    
    
    if loss_mode == 'mse':
        a2=o2
        loss=loss_mse(a2, y_batch)  
    elif loss_mode == 'crossentropy':   
        a2 = softmax(o2)     
        loss=loss_crossentropy(a2, y_batch) 
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")   
    return loss, a2, a1




def backward(a2, a1, X_batch, y_batch, W2):
    """backward pass in the neural network """
    # Implement the backward pass by computing
    # the derivative of the complete function
    # using the chain rule as discussed in the lecture    
    if loss_mode == 'mse':
        d_a2= loss_deriv_mse(a2, y_batch)
    elif loss_mode == 'crossentropy':
        d_a2= loss_deriv_crossentropy(a2, y_batch)   
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")   
    d_o2=d_a2
    dCdW2 = np.dot(a1.T, d_o2)
    dCdb2 = np.sum(d_o2, axis=0)
    d_a1 = np.dot(d_o2, W2.T)
    d_ReLU = relu_derivative(a1)
    d_o1 = d_a1 * d_ReLU
    dCdW1 = np.dot(X_batch.T, d_o1)
    dCdb1 = np.sum(d_o1, axis=0)
    # please use the appropriate loss functions
    # YOUR CODE HERE
    # function should return 4 derivatives with respect to
    # W1, W2, b1, b2
    return dCdW1, dCdW2, dCdb1, dCdb2



def train(X_train, y_train):
    """ train procedure """
    # for simplicity of this execise you don't need to find useful hyperparameter
    # I've done this for you already and every test image should work for the
    # given very small trainings database and the following parameters.
    h = 1500
    std = 0.001
    # YOUR CODE HERE
    # initialize W1, W2, b1, b2 randomly
    W1 = std*np.random.randn(X_train.shape[1],h)
    b1 = np.zeros((1, h))
    W2 = std* np.random.randn(h,y_train.shape[1])
    b2 = np.zeros((1, y_train.shape[1]))
    # Note: W1, W2 should be scaled by variable std
    #print("W1 shape:", W1.shape)
    #print("b1 shape:", b1.shape)
    #print("W2 shape:", W2.shape)
    #print("b2 shape:", b2.shape)
    
    # run for num_epochs
    for i in range(num_epochs):
        #print('interation: ', i)
        #X_batch = Nonesetup_train
        #y_batch = None
        # use only a batch of batch_size of the training images in each run
        # sample the batch images randomly from the training set
        # YOUR CODE HERE
        random_indices = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
        #print('random_indices: ', random_indices)
        X_batch = X_train[random_indices]
        #print('X_batch: ', X_batch.shape)
        y_batch = y_train[random_indices]

        # forward pass for two-layer neural network using ReLU as activation function
        loss, a2, a1=forward(X_batch, y_batch, W1, W2, b1, b2)

        # add loss to loss_train_hist for plotting
        loss_train_hist.append(loss)

        if i % 10 == 0:
            print("iteration %d: loss %f" % (i, loss))
        # backward pass
        dCdW1, dCdW2, dCdb1, dCdb2= backward(a2, a1, X_batch, y_batch, W2)
        # print("dCdb2.shape:", dCdb2.shape, dCdb1.shape)
        # depending on the derivatives of W1, and W2 regaring the cost/lossfür denm 
        # we need to adapt the values in the negative direction of the
        # gradient decreasing towards the minimum
        # we weight the gradient by a learning rate
        W1 -= learning_rate * dCdW1
        b1 -= learning_rate * dCdb1
        W2 -= learning_rate * dCdW2
        b2 -= learning_rate * dCdb2
    
    return W1, W2, b1, b2

X_train, y_train = setup_train()
#print('X_train: ', type(X_train.shape[1]))


W1, W2, b1, b2 = train(X_train, y_train)

# predict the test images, load all test images and
# run prediction by computing the forward pass
test_images = []
test_images.append((cv2.imread('./images/db/test/flower2.jpg',
                               cv2.IMREAD_GRAYSCALE), 1))
test_images.append((cv2.imread('./images/db/test/car.jpg',
                               cv2.IMREAD_GRAYSCALE), 0))
test_images.append((cv2.imread('./images/db/test/face.jpg',
                               cv2.IMREAD_GRAYSCALE), 2))
for ti in test_images:
    resized_ti = cv2.resize(ti[0], (nn_img_size, nn_img_size),
                            interpolation=cv2.INTER_AREA)
    x_test = resized_ti.reshape(1, -1)
    # convert test images to pytorch
    x_test_tensor = torch.from_numpy(x_test).float()
    # do forward pass depending mse or softmax
    o1_test = np.dot(x_test, W1) + b1
    a1_test = relu(o1_test)
    o2_test = np.dot(a1_test, W2) + b2
    a2_test = softmax(o2_test)
    print(f"Test output - values: {a2_test} \t pred_id: {np.argmax(a2_test)} \t true_id: {ti[1]}")

print("------------------------------------")
print("Test model output Weights:", W1, W2)
print("Test model output bias:", b1, b2)


plt.title("Training Loss vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.plot(range(1, num_epochs + 1), loss_train_hist, label="Train")
plt.ylim((0, 3.))
plt.xticks(np.arange(1, num_epochs + 1, 50.0))
plt.legend()
plt.savefig(f"My_Solution_two_layer_nn_train_{loss_mode}.png")
plt.show()
