{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "RealTimeFireDetection.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyPloN2ScLRBxJj+j1e/AE5T",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Aman-Kumar-Srivastava/Realtimefiredetection/blob/main/RealTimeFireDetection.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "juqbywK0j1Jj",
        "outputId": "86451ace-39c1-43ba-e6e2-433508c01a2f"
      },
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vjlC55tckGN_"
      },
      "source": [
        "!pip install -q keras\n",
        "import keras"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vu31XMv2kKlE",
        "outputId": "15d39b6e-d907-4ebd-9a87-fa49405e3e8f"
      },
      "source": [
        "import tensorflow as tf\n",
        "import keras_preprocessing\n",
        "from keras_preprocessing import image\n",
        "from keras_preprocessing.image import ImageDataGenerator\n",
        "\n",
        "TRAINING_DIR = \"/content/drive/MyDrive/fire/train\"\n",
        "training_datagen = ImageDataGenerator(rescale=1./255,\n",
        "zoom_range=0.15,\n",
        "horizontal_flip=True,\n",
        "fill_mode='nearest')\n",
        "\n",
        "VALIDATION_DIR = \"/content/drive/MyDrive/fire/validation\"\n",
        "validation_datagen = ImageDataGenerator(rescale = 1./255)\n",
        "\n",
        "train_generator = training_datagen.flow_from_directory(\n",
        "TRAINING_DIR,\n",
        "target_size=(224,224),\n",
        "shuffle = True,\n",
        "class_mode='categorical',\n",
        "batch_size = 128)\n",
        "\n",
        "validation_generator = validation_datagen.flow_from_directory(\n",
        "VALIDATION_DIR,\n",
        "target_size=(224,224),\n",
        "class_mode='categorical',\n",
        "shuffle = True,\n",
        "batch_size= 14)"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Found 412 images belonging to 2 classes.\n",
            "Found 90 images belonging to 2 classes.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aIRV2lq_kYkF",
        "outputId": "83a30c22-6569-4e5d-c1c6-720a80dadb1e"
      },
      "source": [
        "#traning\n",
        "from tensorflow.keras.applications.inception_v3 import InceptionV3\n",
        "from tensorflow.keras.preprocessing import image\n",
        "from tensorflow.keras.models import Model\n",
        "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout\n",
        "input_tensor = Input(shape=(224, 224, 3))\n",
        "base_model = InceptionV3(input_tensor=input_tensor,\n",
        " weights='imagenet', include_top=False)\n",
        "\n",
        "x = base_model.output\n",
        "x = GlobalAveragePooling2D()(x)\n",
        "x = Dense(2048, activation='relu')(x)\n",
        "x = Dropout(0.25)(x)\n",
        "x = Dense(1024, activation='relu')(x)\n",
        "x = Dropout(0.2)(x)\n",
        "predictions = Dense(2, activation='softmax')(x)\n",
        "\n",
        "model = Model(inputs=base_model.input, outputs=predictions)\n",
        "for layer in base_model.layers:\n",
        "  layer.trainable = False\n",
        "\n",
        "model.compile(optimizer='rmsprop', loss='categorical_crossentropy',\n",
        "              metrics=['acc'])\n",
        "history = model.fit(\n",
        "train_generator,\n",
        "steps_per_epoch = 6,\n",
        "epochs = 5,\n",
        "validation_data = validation_generator,\n",
        "validation_steps = 6)"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
            "87916544/87910968 [==============================] - 1s 0us/step\n",
            "87924736/87910968 [==============================] - 1s 0us/step\n",
            "Epoch 1/5\n",
            "4/6 [===================>..........] - ETA: 59s - loss: 0.5791 - acc: 0.6917 WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 30 batches). You may need to use the repeat() function when building your dataset.\n",
            "6/6 [==============================] - 137s 23s/step - loss: 0.5791 - acc: 0.6917 - val_loss: 0.0000e+00 - val_acc: 1.0000\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OzML_uD-ki3r",
        "outputId": "930d4eff-ec84-4c19-e5ba-da9f042a78a5"
      },
      "source": [
        "#To train the top 2 inception blocks, freeze the first 100 layers and unfreeze the rest.\n",
        "#from tensorflow.keras.model import load_model\n",
        "for layer in model.layers[:100]:\n",
        " layer.trainable = False\n",
        "\n",
        "for layer in model.layers[100:]:\n",
        "  layer.trainable = True\n",
        "#Recompile the model for these modifications to take effectfrom tensorflow.keras.optimizers import SGD\n",
        "model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),\n",
        "              loss='categorical_crossentropy',\n",
        "              metrics=['acc'])\n",
        "\n",
        "history = model.fit(\n",
        "train_generator,\n",
        "steps_per_epoch = 6,\n",
        "epochs = 5,\n",
        "validation_data = validation_generator,\n",
        "validation_steps = 6)\n",
        "model.save('/content/drive/MyDrive/fire/InceptionV3.h5')\n",
        "del model"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/optimizer_v2.py:356: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.\n",
            "  \"The `lr` argument is deprecated, use `learning_rate` instead.\")\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/5\n",
            "4/6 [===================>..........] - ETA: 1:10 - loss: 0.0000e+00 - acc: 1.0000WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 30 batches). You may need to use the repeat() function when building your dataset.\n",
            "6/6 [==============================] - 132s 23s/step - loss: 0.0000e+00 - acc: 1.0000 - val_loss: 0.0000e+00 - val_acc: 1.0000\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 582
        },
        "id": "mO6LlCpFkohi",
        "outputId": "a5e5d904-3eef-455e-ea84-1ac7a0a54b66"
      },
      "source": [
        "%matplotlib inline\n",
        "import matplotlib.pyplot as plt\n",
        "acc = history.history['acc']\n",
        "val_acc = history.history['val_acc']\n",
        "loss = history.history['loss']\n",
        "val_loss = history.history['val_loss']\n",
        "epochs = range(len(acc))\n",
        "\n",
        "plt.plot(epochs, acc, 'g', label='Training accuracy')\n",
        "plt.plot(epochs, val_acc, 'b', label='Validation accuracy')\n",
        "plt.title('Training and validation accuracy')\n",
        "\n",
        "plt.legend(loc=0)\n",
        "plt.figure()\n",
        "plt.show()\n",
        "\n",
        "plt.plot(epochs, loss, 'r', label='Training loss')\n",
        "plt.plot(epochs, val_loss, 'orange', label='Validation loss')\n",
        "plt.title('Training and validation loss')\n",
        "\n",
        "plt.legend(loc=0)\n",
        "plt.figure()\n",
        "plt.show()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 0 Axes>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 0 Axes>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ceXfYVqmkv-v",
        "outputId": "a3073f71-8c8d-4429-9236-881660106520"
      },
      "source": [
        "import cv2\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "import tensorflow as tf\n",
        "from keras.preprocessing import image\n",
        "!pip install playsound\n",
        "import playsound\n",
        "import smtplib\n",
        "import threading\n",
        "\n",
        "Alarm_Status = False\n",
        "Email_Status = False\n",
        "Fire_Reported = 0\n",
        "\n",
        "def play_alarm_sound():\n",
        "    while True:\n",
        "        playsound.playsound('Alarm Sound.mp3', True)\n",
        "\n",
        "def send_mail():\n",
        "    Email = \"firefix05@gmail.com\"\n",
        "    Email = Email.lower()\n",
        "\n",
        "    try:\n",
        "        server = smtplib.SMTP('smtp.gmail.com', 587)\n",
        "        server.ehlo()\n",
        "        server.starttls()\n",
        "        server.login(\"firefix05@gmail.com\", 'MaChodDenge')\n",
        "        server.sendmail('firefix05@gmail.com',Email,\n",
        "                        \"Warning A Fire Accident has been reported at Saket Bhawan\")\n",
        "        print(\"sent to {}\".format(Email))\n",
        "        server.close()\n",
        "    except Exception as e:\n",
        "        print(e)\n",
        "\n",
        "#Load the saved model\n",
        "model = tf.keras.models.load_model('/content/drive/MyDrive/fire/InceptionV3.h5')\n",
        "video = cv2.VideoCapture(0)\n",
        "while True:\n",
        "        ret, frame = video.read()\n",
        "        try:\n",
        "          #Resizing into 224x224 because we trained the model with this image size.\n",
        "         frame = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)\n",
        "        except:\n",
        "         break\n",
        "         frame = cv2.flip(frame,1)\n",
        "         blur = cv2.GaussianBlur(frame,(15,15),0)\n",
        "         hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)\n",
        "\n",
        "\n",
        "         lower = [18,50,50]\n",
        "         upper = [35,255,255]\n",
        "\n",
        "         lower = np.array(lower,dtype='uint8')\n",
        "         upper = np.array(upper,dtype='uint8')\n",
        "\n",
        "         mask = cv2.inRange(hsv,lower,upper)\n",
        "\n",
        "         output = cv2.bitwise_and(frame,hsv,mask=mask)\n",
        "         res = cv2.bitwise_and(frame, frame, mask=mask)\n",
        "\n",
        "         gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)\n",
        "         canny = cv2.Canny(gray, 100, 200)\n",
        "\n",
        "         _, thr = cv2.threshold(mask, 100, 255, 0)\n",
        "         contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)\n",
        "         cv2.drawContours(res, contours, -1, (255, 0, 0), 3)\n",
        "\n",
        "        #cv2.imshow('Original Frame', frame)\n",
        "        #cv2.imshow('Mask', mask)\n",
        "        #cv2.imshow('Res', res)\n",
        "        #cv2.imshow('canny', canny)\n",
        "        #Convert the captured frame into RGB\n",
        "        im = Image.fromarray(frame)\n",
        "        img_array = image.img_to_array(im)\n",
        "        img_array = np.expand_dims(img_array, axis=0) / 255\n",
        "        probabilities = model.predict(img_array)[0]\n",
        "        #Calling the predict method on model to predict 'fire' on the image\n",
        "        prediction = np.argmax(probabilities)\n",
        "        #if prediction is 0, which means there is fire in the frame.\n",
        "        if prediction == 0:\n",
        "         print(\"fire detected\")\n",
        "         Fire_Reported = Fire_Reported + 1\n",
        "\n",
        "         if Fire_Reported >= 1:\n",
        "\n",
        "            if Alarm_Status == False:\n",
        "                threading.Thread(target=play_alarm_sound).start()\n",
        "                Alarm_Status = True\n",
        "\n",
        "            if Email_Status == False:\n",
        "                threading.Thread(target=send_mail).start()\n",
        "                Email_Status = True\n",
        "\n",
        "        if ret == False:\n",
        "            break\n",
        "        print(probabilities[prediction])\n",
        "        cv2.imshow('output', output)\n",
        "        key=cv2.waitKey(1)\n",
        "        if key == ord('q'):\n",
        "                break\n",
        "video.release()\n",
        "cv2.destroyAllWindows()"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: playsound in /usr/local/lib/python3.7/dist-packages (1.3.0)\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}