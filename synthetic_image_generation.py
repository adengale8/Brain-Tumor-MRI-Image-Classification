import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import cv2
import os
import seaborn as sns
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, Reshape, Input, Conv2DTranspose
from keras.layers import Activation, LeakyReLU, BatchNormalization, Dropout, Resizing
from keras.losses import BinaryCrossentropy
from tensorflow.keras.applications import VGG16
from PIL import Image 

import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.optimizers import Adam
except:
    from keras.optimizers import Adam


# In[3]:


NOISE_DIM = 100  
BATCH_SIZE = 8
STEPS_PER_EPOCH = 3750
EPOCHS =25
SEED = 40
WIDTH, HEIGHT, CHANNELS = 128, 128, 1

OPTIMIZER = Adam(0.0002, 0.5)


# In[4]:


no_tumor_dir="/kaggle/input/brain-tumor-classification-mri/Training/no_tumor"
glioma_tumor_dir="/kaggle/input/glioma-similar"
meningioma_tumor_dir="/kaggle/input/me-similar"
pituitary_tumor_dir="/kaggle/input/pituit-similar"


# # Loading and Preprocessing the Images

# In[5]:


def load_images(folder):
    
    imgs = []
    target = 1
    labels = []
    for i in os.listdir(folder):
        img_dir = os.path.join(folder,i)
        try:
            img = cv2.imread(img_dir)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128,128))
            imgs.append(img)
            labels.append(target)
        except:
            continue
        
    imgs = np.array(imgs)
    labels = np.array(labels)
    
    return imgs, labels


# In[6]:


load_images(no_tumor_dir)


# In[7]:


data1, labels1 = load_images(no_tumor_dir)
print(data1.shape, labels1.shape)

data2, labels2 = load_images(glioma_tumor_dir)
print(data2.shape, labels2.shape)

data3, labels3 = load_images(meningioma_tumor_dir)
print(data3.shape, labels3.shape)

data4, labels4 = load_images(pituitary_tumor_dir)
print(data4.shape, labels4.shape)


# ## Generate 20 random numbers to index images from data

# In[8]:


np.random.seed(SEED)
idxs1 = np.random.randint(0, 395, 30)
idxs2 = np.random.randint(0, 107, 30)
idxs3 = np.random.randint(0, 106, 30)
idxs4 = np.random.randint(0, 117, 30)


# In[9]:


X_train1 = data1[idxs1]
print(X_train1.shape)

X_train2 = data2[idxs2]
print(X_train2.shape)

X_train3 = data3[idxs3]
print(X_train3.shape)

X_train4 = data4[idxs4]
print(X_train4.shape)


# ## Normalize and Reshape the Data

# In[10]:


# Normalize the Images
X_train1 = (X_train1.astype(np.float32) - 127.5) / 127.5

# Reshape images 
X_train1 = X_train1.reshape(-1, WIDTH,HEIGHT,CHANNELS)

# Check shape
print(X_train1.shape)

# Normalize the Images
X_train2 = (X_train2.astype(np.float32) - 127.5) / 127.5

# Reshape images 
X_train2 = X_train2.reshape(-1, WIDTH,HEIGHT,CHANNELS)

# Check shape
print(X_train2.shape)

# Normalize the Images
X_train3 = (X_train3.astype(np.float32) - 127.5) / 127.5

# Reshape images 
X_train3 = X_train3.reshape(-1, WIDTH,HEIGHT,CHANNELS)

# Check shape
print(X_train3.shape)

# Normalize the Images
X_train4 = (X_train4.astype(np.float32) - 127.5) / 127.5

# Reshape images 
X_train4 = X_train4.reshape(-1, WIDTH,HEIGHT,CHANNELS)

# Check shape
print(X_train4.shape)


# ## Plotting The Real Images

# In[11]:


plt.figure(figsize=(20,8))
for i in range(10):
    axs = plt.subplot(2,5,i+1)
    plt.imshow(X_train1[i], cmap="gray")
    plt.axis('off')
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    plt.subplots_adjust(wspace=None, hspace=None)
plt.tight_layout()

plt.figure(figsize=(20,8))
for i in range(10):
    axs = plt.subplot(2,5,i+1)
    plt.imshow(X_train2[i], cmap="gray")
    plt.axis('off')
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    plt.subplots_adjust(wspace=None, hspace=None)
plt.tight_layout()

plt.figure(figsize=(20,8))
for i in range(10):
    axs = plt.subplot(2,5,i+1)
    plt.imshow(X_train3[i], cmap="gray")
    plt.axis('off')
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    plt.subplots_adjust(wspace=None, hspace=None)
plt.tight_layout()

plt.figure(figsize=(20,8))
for i in range(10):
    axs = plt.subplot(2,5,i+1)
    plt.imshow(X_train4[i], cmap="gray")
    plt.axis('off')
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    plt.subplots_adjust(wspace=None, hspace=None)
plt.tight_layout()


# # The Architecture

# In[12]:


def build_generator():

    """
        Generator model "generates" images using random noise. The random noise AKA Latent Vector
        is sampled from a Normal Distribution which is given as the input to the Generator. Using
        Transposed Convolution, the latent vector is transformed to produce an image
        We use 3 Conv2DTranspose layers (which help in producing an image using features; opposite
        of Convolutional Learning)

        Input: Random Noise / Latent Vector
        Output: Image
    """

    model = Sequential([

        Dense(32*32*256, input_dim=NOISE_DIM),
        LeakyReLU(alpha=0.4),
        Reshape((32,32,256)),
        
        Conv2DTranspose(256, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.4),

        Conv2DTranspose(256, (3, 3), strides=2, padding='same'),
        LeakyReLU(alpha=0.4),

        Conv2D(CHANNELS, (3, 3), padding='same', activation='tanh')
    ], 
    name="generator")
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)

    return model


# In[13]:


def build_discriminator():
    
    """
        Discriminator is the model which is responsible for classifying the generated images
        as fake or real. Our end goal is to create a Generator so powerful that the Discriminator
        is unable to classify real and fake images
        A simple Convolutional Neural Network with 2 Conv2D layers connected to a Dense output layer
        Output layer activation is Sigmoid since this is a Binary Classifier

        Input: Generated / Real Image
        
        Output: Validity of Image (Fake or Real)

    """

    model = Sequential([

        Conv2D(64, (2, 2), padding='same', input_shape=(WIDTH, HEIGHT, CHANNELS)),
        LeakyReLU(alpha=0.2),

        Conv2D(128, (2, 2), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),

        Conv2D(128, (2, 2), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        
        Conv2D(256, (2, 2), strides=2, padding='same'),
        LeakyReLU(alpha=0.2),
        
        Flatten(),
        Dropout(0.4),
        Dense(1, activation="sigmoid", input_shape=(WIDTH, HEIGHT, CHANNELS))
    ], name="discriminator")
    model.summary()
    model.compile(loss="binary_crossentropy",
                        optimizer=OPTIMIZER)

    return model


# # Putting it together

# In[14]:


print('\n')
discriminator = build_discriminator()
print('\n')
generator = build_generator()

# Adjust the GAN model
discriminator.trainable = False
gan_input = Input(shape=(NOISE_DIM,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = Model(gan_input, gan_output, name="gan_model")
gan.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.0002, beta_1=0.5))

print("The Combined Network:\n")
gan.summary()


# In[15]:


def sample_images(noise, subplots, figsize=(22,8), save=False):
   generated_images = generator.predict(noise)
   plt.figure(figsize=figsize)
   
   for i, image in enumerate(generated_images):
       plt.subplot(subplots[0], subplots[1], i+1)
       if CHANNELS == 1:
           plt.imshow(image.reshape((WIDTH, HEIGHT)), cmap='gray')    
                                                                           
       else:
           plt.imshow(image.reshape((WIDTH, HEIGHT, CHANNELS)))
       if save == True:
           img_name = "gen" + str(i)
           plt.savefig(img_name)
       plt.subplots_adjust(wspace=None, hspace=None)
       plt.axis('off')
   
   plt.tight_layout()
   plt.show()


# In[16]:


import os

output_path = '/kaggle/working/glioma_similar/'  # Define your desired directory name
os.makedirs(output_path, exist_ok=True)  # Create the directory


# In[17]:


# Function to save generated images to a directory after the entire training process
def save_images_after_training(noise, num_images):
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Scale images back to [0, 1] from [-1, 1]

    output_path = '/kaggle/working/glioma_similar/'  # Define your output directory in Kaggle

    for i in range(num_images):
        image = generated_images[i]
        image = ((image + 1) * 127.5).astype(np.uint8)  # Convert back to uint8 in the range [0, 255]
        image = Image.fromarray(image.squeeze(), mode='L')  # 'L' mode for grayscale images
        image.save(f"{output_path}generated_image_{i + 1}.png")


# ## The Training

# In[ ]:


from keras.callbacks import EarlyStopping

np.random.seed(SEED)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

for epoch in range(20):
    for batch in tqdm(range(STEPS_PER_EPOCH)):

        noise = np.random.normal(0,1, size=(BATCH_SIZE, NOISE_DIM))
        fake_X = generator.predict(noise)
        
        idx = np.random.randint(0, X_train2.shape[0], size=BATCH_SIZE)
        real_X = X_train2[idx]

        X = np.concatenate((real_X, fake_X))

        disc_y = np.zeros(2*BATCH_SIZE)
        disc_y[:BATCH_SIZE] = 1

        d_loss = discriminator.train_on_batch(X, disc_y)
        y_gen = np.ones(BATCH_SIZE)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss} Discriminator Loss: {d_loss}")
    noise = np.random.normal(0, 1, size=(10, NOISE_DIM))
    sample_images(noise, (2, 5))
    
    # Check for early stopping
    if early_stopping.on_epoch_end(epoch + 1, {'gen_loss': g_loss}):
        print("Early stopping triggered.")
        break  # Stop training if early stopping criteria are met


# # Let's generate some images !

# In[ ]:


# Save generated images to the Kaggle output directory after training
num_images_to_save = 100  # Change this to the number of images you want to save
noise_for_saving = np.random.normal(0, 1, size=(num_images_to_save, NOISE_DIM))
save_images_after_training(noise_for_saving, num_images_to_save)


# # Testing the Generated sample: Plotting the Distributions
# 
# <p style="font-size:20px">In this test, we compare the generated images with the real samples by plotting their distributions. If the distributions overlap, that indicates the generated samples are very close to the real ones
# </p>

# In[ ]:


generated_images = generator.predict(noise)
generated_images.shape


# In[ ]:


fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(18,10))

sns.distplot(X_train2, label='Real Images', hist=True, color='#fc0328', ax=axs)
sns.distplot(generated_images, label='Generated Images', hist=True, color='#0c06c7', ax=axs)

axs.legend(loc='upper right', prop={'size': 12})

plt.show()


# In[ ]:


import zipfile
import os

directory_to_zip = '/kaggle/working/glioma_similar/'  # Replace with the directory path you want to download
output_zip_path = '/kaggle/working/glioma_similar.zip'  # Replace with the desired output zip file path

# Create a zip archive of the directory
with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(directory_to_zip):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory_to_zip))

# Download the created zip file
from IPython.display import FileLink
FileLink(output_zip_path)


# # **Meningioma tumor**

# In[ ]:


import os

output_path = '/kaggle/working/meningioma_images/'  # Define your desired directory name
os.makedirs(output_path, exist_ok=True)  # Create the directory


# In[ ]:


# Function to save generated images to a directory after the entire training process
def save_images_after_training(noise, num_images):
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Scale images back to [0, 1] from [-1, 1]

    output_path = '/kaggle/working/meningioma_images/'  # Define your output directory in Kaggle

    for i in range(num_images):
        image = generated_images[i]
        image = ((image + 1) * 127.5).astype(np.uint8)  # Convert back to uint8 in the range [0, 255]
        image = Image.fromarray(image.squeeze(), mode='L')  # 'L' mode for grayscale images
        image.save(f"{output_path}generated_image_{i + 1}.png")


# In[ ]:


from keras.callbacks import EarlyStopping

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
np.random.seed(SEED)

for epoch in range(12):
    for batch in tqdm(range(STEPS_PER_EPOCH)):

        noise = np.random.normal(0,1, size=(BATCH_SIZE, NOISE_DIM))
        fake_X = generator.predict(noise)
        
        idx = np.random.randint(0, X_train3.shape[0], size=BATCH_SIZE)
        real_X = X_train3[idx]

        X = np.concatenate((real_X, fake_X))

        disc_y = np.zeros(2*BATCH_SIZE)
        disc_y[:BATCH_SIZE] = 1

        d_loss = discriminator.train_on_batch(X, disc_y)
        
        y_gen = np.ones(BATCH_SIZE)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")
    noise = np.random.normal(0, 1, size=(10, NOISE_DIM))
    sample_images(noise, (2, 5))
    
    # Check for early stopping
    if early_stopping.on_epoch_end(epoch + 1, {'gen_loss': g_loss}):
        print("Early stopping triggered.")
        break  # Stop training if early stopping criteria are met


# In[ ]:


# Save generated images to the Kaggle output directory after training
num_images_to_save = 100  # Change this to the number of images you want to save
noise_for_saving = np.random.normal(0, 1, size=(num_images_to_save, NOISE_DIM))
save_images_after_training(noise_for_saving, num_images_to_save)


# In[ ]:


generated_images = generator.predict(noise)
generated_images.shape


# In[ ]:


fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(18,10))

sns.distplot(X_train3, label='Real Images', hist=True, color='#fc0328', ax=axs)
sns.distplot(generated_images, label='Generated Images', hist=True, color='#0c06c7', ax=axs)

axs.legend(loc='upper right', prop={'size': 12})

plt.show()


# In[ ]:


import zipfile
import os

directory_to_zip = '/kaggle/working/meningioma_images/'  # Replace with the directory path you want to download
output_zip_path = '/kaggle/working/meningioma_images.zip'  # Replace with the desired output zip file path

# Create a zip archive of the directory
with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(directory_to_zip):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory_to_zip))

# Download the created zip file
from IPython.display import FileLink
FileLink(output_zip_path)


# # **Pituitary tumor**

# In[ ]:


import os

output_path = '/kaggle/working/pituitary_images/'  # Define your desired directory name
os.makedirs(output_path, exist_ok=True)  # Create the directory


# In[ ]:


# Function to save generated images to a directory after the entire training process
def save_images_after_training(noise, num_images):
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Scale images back to [0, 1] from [-1, 1]

    output_path = '/kaggle/working/pituitary_images/'  # Define your output directory in Kaggle

    for i in range(num_images):
        image = generated_images[i]
        image = ((image + 1) * 127.5).astype(np.uint8)  # Convert back to uint8 in the range [0, 255]
        image = Image.fromarray(image.squeeze(), mode='L')  # 'L' mode for grayscale images
        image.save(f"{output_path}generated_image_{i + 1}.png")


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

np.random.seed(SEED)

for epoch in range(12):
    for batch in tqdm(range(STEPS_PER_EPOCH)):

        noise = np.random.normal(0, 1, size=(BATCH_SIZE, NOISE_DIM))
        fake_X = generator.predict(noise)
        
        idx = np.random.randint(0, X_train4.shape[0], size=BATCH_SIZE)
        real_X = X_train4[idx]

        X = np.concatenate((real_X, fake_X))

        disc_y = np.zeros(2 * BATCH_SIZE)
        disc_y[:BATCH_SIZE] = 1

        d_loss = discriminator.train_on_batch(X, disc_y)
        
        y_gen = np.ones(BATCH_SIZE)
        g_loss = gan.train_on_batch(noise, y_gen)
        
    # Add early stopping condition
    if g_loss < 0.2 and d_loss < 0.2:
        print("Early stopping criteria met. Stopping training.")
        break

    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")
    noise = np.random.normal(0, 1, size=(10, NOISE_DIM))
    sample_images(noise, (2, 5))


# In[ ]:


# Save generated images to the Kaggle output directory after training
num_images_to_save = 100  # Change this to the number of images you want to save
noise_for_saving = np.random.normal(0, 1, size=(num_images_to_save, NOISE_DIM))
save_images_after_training(noise_for_saving, num_images_to_save)


# In[ ]:


generated_images = generator.predict(noise)
generated_images.shape


# In[ ]:


fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(18,10))

sns.distplot(X_train4, label='Real Images', hist=True, color='#fc0328', ax=axs)
sns.distplot(generated_images, label='Generated Images', hist=True, color='#0c06c7', ax=axs)

axs.legend(loc='upper right', prop={'size': 12})

plt.show()


# In[ ]:


import zipfile
import os

directory_to_zip = '/kaggle/working/pituitary_images/'  # Replace with the directory path you want to download
output_zip_path = '/kaggle/working/pituitary_images.zip'  # Replace with the desired output zip file path

# Create a zip archive of the directory
with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(directory_to_zip):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory_to_zip))

# Download the created zip file
from IPython.display import FileLink
FileLink(output_zip_path)


# # **No tumour images**

# In[ ]:


import os

output_path = '/kaggle/working/no_tumor_images/'  # Define your desired directory name
os.makedirs(output_path, exist_ok=True)  # Create the directory


# In[ ]:


# Function to save generated images to a directory after the entire training process
def save_images_after_training(noise, num_images):
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Scale images back to [0, 1] from [-1, 1]

    output_path = '/kaggle/working/no_tumor_images/'  # Define your output directory in Kaggle

    for i in range(num_images):
        image = generated_images[i]
        image = ((image + 1) * 127.5).astype(np.uint8)  # Convert back to uint8 in the range [0, 255]
        image = Image.fromarray(image.squeeze(), mode='L')  # 'L' mode for grayscale images
        image.save(f"{output_path}generated_image_{i + 1}.png")


# In[ ]:


# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

np.random.seed(SEED)

for epoch in range(10):
    for batch in tqdm(range(STEPS_PER_EPOCH)):

        noise = np.random.normal(0,1, size=(BATCH_SIZE, NOISE_DIM))
        fake_X = generator.predict(noise)
        
        idx = np.random.randint(0, X_train1.shape[0], size=BATCH_SIZE)
        real_X = X_train1[idx]

        X = np.concatenate((real_X, fake_X))

        disc_y = np.zeros(2*BATCH_SIZE)
        disc_y[:BATCH_SIZE] = 1

        d_loss = discriminator.train_on_batch(X, disc_y)
        
        y_gen = np.ones(BATCH_SIZE)
        g_loss = gan.train_on_batch(noise, y_gen)

    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss:.4f} Discriminator Loss: {d_loss:.4f}")
    noise = np.random.normal(0, 1, size=(10, NOISE_DIM))
    sample_images(noise, (2, 5))
    
    # Add early stopping condition
    if g_loss < 0.2 and d_loss < 0.2:
        print("Early stopping criteria met. Stopping training.")
        break


# In[ ]:


# Save generated images to the Kaggle output directory after training
num_images_to_save = 100  # Change this to the number of images you want to save
noise_for_saving = np.random.normal(0, 1, size=(num_images_to_save, NOISE_DIM))
save_images_after_training(noise_for_saving, num_images_to_save)


# In[ ]:


generated_images = generator.predict(noise)
generated_images.shape


# In[ ]:


fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(18,10))

sns.distplot(X_train1, label='Real Images', hist=True, color='#fc0328', ax=axs)
sns.distplot(generated_images, label='Generated Images', hist=True, color='#0c06c7', ax=axs)

axs.legend(loc='upper right', prop={'size': 12})

plt.show()


# In[ ]:


import zipfile
import os

directory_to_zip = '/kaggle/working/no_tumor_images/'  # Replace with the directory path you want to download
output_zip_path = '/kaggle/working/no_tumor_images.zip'  # Replace with the desired output zip file path

# Create a zip archive of the directory
with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(directory_to_zip):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory_to_zip))

# Download the created zip file
from IPython.display import FileLink
FileLink(output_zip_path)


# In[ ]:


import zipfile
import os

directory_to_zip = '/kaggle/input/brain-tumor-classification-mri/Training/glioma_tumour'  # Replace with the directory path you want to download
output_zip_path = '/kaggle/input/brain-tumor-classification-mri/Training/glioma_tumour.zip'  # Replace with the desired output zip file path

# Create a zip archive of the directory
with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(directory_to_zip):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), directory_to_zip))

# Download the created zip file
from IPython.display import FileLink
FileLink(output_zip_path)


# In[ ]:





# In[ ]:


no_tumour_dir = "../input/brain-mri-images-for-brain-tumor-detection/no"


# In[ ]:


data1, labels1 = load_images(no_tumour_dir)
data1.shape, labels1.shape


# In[ ]:


np.random.seed(SEED)
idxss = np.random.randint(0, 98, 20)


# In[ ]:


X_train1 = data1[idxss]
X_train1.shape


# In[ ]:


# Normalize the Images
X_train1 = (X_train1.astype(np.float32) - 127.5) / 127.5

# Reshape images 
X_train1 = X_train1.reshape(-1, WIDTH,HEIGHT,CHANNELS)

# Check shape
X_train1.shape


# In[ ]:


plt.figure(figsize=(20,8))
for i in range(10):
    axs = plt.subplot(2,5,i+1)
    plt.imshow(X_train1[i], cmap="gray")
    plt.axis('off')
    axs.set_xticklabels([])
    axs.set_yticklabels([])
    plt.subplots_adjust(wspace=None, hspace=None)
plt.tight_layout()


# In[ ]:


import os

output_path = '/kaggle/working/no_tumour_images/'  # Define your desired directory name
os.makedirs(output_path, exist_ok=True)  # Create the directory


# In[ ]:


np.random.seed(SEED)

for epoch in range(10):
    for batch in tqdm(range(STEPS_PER_EPOCH)):

        noise1 = np.random.normal(0,1, size=(BATCH_SIZE, NOISE_DIM))
        fake_X1 = generator.predict(noise1)
        
        idx1 = np.random.randint(0, X_train1.shape[0], size=BATCH_SIZE)
        real_X1 = X_train1[idx1]

        X1 = np.concatenate((real_X1, fake_X1))

        disc_y1 = np.zeros(2*BATCH_SIZE)
        disc_y1[:BATCH_SIZE] = 1

        d_loss1 = discriminator.train_on_batch(X1, disc_y1)
        
        y_gen1 = np.ones(BATCH_SIZE)
        g_loss1 = gan.train_on_batch(noise1, y_gen1)

    print(f"EPOCH: {epoch + 1} Generator Loss: {g_loss1:.4f} Discriminator Loss: {d_loss1:.4f}")
    noise1 = np.random.normal(0, 1, size=(10, NOISE_DIM))
    sample_images(noise1, (2, 5))


# In[ ]:


# Function to save generated images to a directory after the entire training process
def save_images_after_training1(noise, num_images):
    generated_images1 = generator.predict(noise1)
    generated_images1 = 0.5 * generated_images1 + 0.5  # Scale images back to [0, 1] from [-1, 1]

    output_path1 = '/kaggle/working/no_tumour_images/'  # Define your output directory in Kaggle

    for i in range(num_images):
        image1 = generated_images1[i]
        image1 = ((image1 + 1) * 127.5).astype(np.uint8)  # Convert back to uint8 in the range [0, 255]
        image1 = Image.fromarray(image1.squeeze(), mode='L')  # 'L' mode for grayscale images
        image1.save(f"{output_path1}generated_image_{i + 1}.png")


# In[ ]:


# Save generated images to the Kaggle output directory after training
num_images_to_save1 = 100  # Change this to the number of images you want to save
noise_for_saving1 = np.random.normal(0, 1, size=(num_images_to_save1, NOISE_DIM))
save_images_after_training1(noise_for_saving, num_images_to_save)
