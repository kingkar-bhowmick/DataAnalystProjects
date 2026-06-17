#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[6]:


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
X_train.shape


# In[7]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[17]:


print("Y_train shape:", y_train.shape)

y_train[:5]


# In[18]:


y_train = y_train.reshape(-1, )
y_train[:5]


# In[24]:


classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

classes[9]


# In[25]:


#plt.figure(figsize = (15,2))
#plt.imshow(X_train[0])

def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X_train[index])
    plt.xlabel(classes[y[index]])


# In[33]:


plot_sample(X_train, y_train, 2)


# In[35]:


# Normalisierung

X_train = X_train / 255
X_test = X_test / 255


# In[38]:


# 1. Define the Artificial Neural Network (ANN) architecture

ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')  # Added output layer for CIFAR-10 (10 classes)
])

 # 2. Compile the model with the correct arguments
ann.compile(
    optimizer='SGD',
    loss='sparse_categorical_crossentropy',  # Ideal for integer labels like CIFAR-10
    metrics=['accuracy']
)

# 3. Train the model
ann.fit(X_train, y_train, epochs=5)


# In[39]:


ann.evaluate(X_test, y_test)


# In[40]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Get predictions from the trained ANN model
y_pred = ann.predict(X_test)

# 2. Convert raw probability outputs to class indices using argmax
y_pred_classes = [np.argmax(element) for element in y_pred]

# 3. Print out the precision, recall, and f1-score report
print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[41]:


cnn = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second Convolutional Block (Notice: input_shape is removed here)
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Dense Classifier Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Perfect for CIFAR-10's 10 classes
])


# In[42]:


cnn.compile(
    optimizer='SGD',
    loss='sparse_categorical_crossentropy',  # Ideal for integer labels like CIFAR-10
    metrics=['accuracy']
)


# In[43]:


cnn.fit(X_train, y_train, epochs = 10)


# In[44]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 1. Get predictions from the trained ANN model
y_pred = cnn.predict(X_test)

# 2. Convert raw probability outputs to class indices using argmax
y_pred_classes = [np.argmax(element) for element in y_pred]

# 3. Print out the precision, recall, and f1-score report
print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[45]:


y_test[:5]


# In[46]:


y_test = y_test.reshape(-1,)
y_test[:5]


# In[47]:


plot_sample(X_test, y_test, 1)


# In[48]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[53]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[54]:


y_test[:5]


# In[75]:


plot_sample(X_train, y_train, 1)


# In[72]:


classes[y_classes[0]]


# In[ ]:




