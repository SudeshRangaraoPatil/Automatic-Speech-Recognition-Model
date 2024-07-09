#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer
from pydub import AudioSegment
from multiprocessing import Pool
import os


# In[2]:


data_url="https://asr.iitm.ac.in/Gramvaani/NEW/GV_Eval_3h.tar.gz"
data_path = keras.utils.get_file("GV_Eval_3h", data_url, untar=True)


# In[3]:


wavs_path = data_path + "/Audio/"
matadata_path = data_path + "/text"


# In[4]:


metadata_df = pd.read_csv(matadata_path, sep="|", header=None, quoting=3)


# In[5]:


metadata_df.head(10)


# In[ ]:





# In[6]:


split_data_df = metadata_df[0].str.split(' ', expand=True)


# In[7]:


split_data_df.iloc[:, 1:101].fillna('')


# In[8]:


split_data_df['Merged_Column'] = split_data_df.iloc[:, 1:101].astype(str).apply(' '.join, axis=1)


# In[9]:


metaadata_df = split_data_df.drop(split_data_df.iloc[:, 1:101], axis=1)


# In[10]:


metaadata_df.head(10)


# In[11]:


metaadata_df.columns = ["file_name", "translation"]
metaadata_df = metaadata_df[["file_name", "translation"]]
metaadata_df = metaadata_df.sample(frac=1).reset_index(drop=True)
metaadata_df.head(3)


# In[12]:


split = int(len(metaadata_df)* 0.90)
df_train = metaadata_df[:split]
df_val = metaadata_df[split:]

print(f"size of train set: {len(df_train)}")
print(f"size of train set: {len(df_val)}")


# In[13]:


# Define the characters with unique entries
characters = list("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञ'?! ")

# Remove any duplicate entries
characters = list(set(characters))

# Create a StringLookup layer to map characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")

# Create a StringLookup layer to map integers back to characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# Print the vocabulary
print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size={char_to_num.vocabulary_size()})"
)


# In[ ]:





# In[14]:


frame_length = 256
frame_step = 160
fft_length = 384

def encode_single_sample(wav_file, label):  # Corrected typo in parameter name
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)
    spectrogram = tf.signal.stft(
        audio, frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    
    label = tf.strings.lower(label)
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    
    label = char_to_num(label)
    
    return spectrogram, label


# In[15]:


batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["file_name"]), list(df_train["translation"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]), list(df_val["translation"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# In[16]:


fig = plt.figure(figsize=(8, 5))
for batch in train_dataset.take(1):
    spectrogram = batcch[0][0].numpy()
    spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
    label = batch[1][0]
    
    label = tf.strings.reduce_join(num_to_char(label)).numpy.decode("utf-8")
    ax = plt.subplot(2, 1, 1)
    ax.imshow(spectrogram, vmax=1)
    ax.set_title(label)
    ax.axis("off")
    
    file = tf.io.read_file(wavs_path + list(df_train["file_name"])[0] + ".mp3")
    audio, _ =tf.audio.decode_mp3(file)
    audio = audio.numpy()
    ax = plt.subplot(2, 1, 2)
    plt.plot(audio)
    ax.set_title("Signal Wave")
    ax.set_xlim(0, len(audio))
    display.display(display.Audio(np.transpose(audio), rate=16000))
plt.show()


# In[17]:


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    input_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# In[18]:


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")

    # Expand the dimension to use 2D CNN
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)

    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv1"
    )(x)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)

    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="conv2_bn")(x)
    x = layers.ReLU(name="conv2_relu")(x)

    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN Layers
    for i in range(1, rnn_layers + 1):
        x = layers.Bidirectional(
            layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}"
            ),
            name=f"bidirectional_{i}",
            merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)

    # Classification Layer
    output = layers.Dense(units=output_dim, activation="softmax", name="output")(x)

    # Model
    model = keras.Model(inputs=input_spectrogram, outputs=output, name="DeepSpeech2")

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)

    # Compile the model and return
    model.compile(optimizer=opt, loss="categorical_crossentropy")

    return model


model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=512,
)
model.summary(line_length=110)


# In[ ]:





# In[21]:


batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["file_name"]), list(df_train["translation"]))
)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]), list(df_val["translation"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# In[ ]:


get_ipython().system('pip install pydub')

