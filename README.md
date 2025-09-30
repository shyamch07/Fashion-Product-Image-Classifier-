
# Fashion Product Image Classifier

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify fashion product images from the **Fashion-MNIST dataset**. It is a beginner-friendly project that demonstrates **image classification using deep learning** with TensorFlow and Keras.

---

## Features

* Classifies 10 fashion categories: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
* Achieves **~89% test accuracy** on unseen images.
* Visualizes training vs. validation accuracy using matplotlib.
* Simple and beginner-friendly for learning CNNs and deep learning basics.

---

## Dataset

* **Fashion-MNIST**
* 70,000 grayscale images of size 28x28 pixels
* 60,000 training images, 10,000 test images
* 10 fashion categories

---


## Libraries Used

* Python 3.8+
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## Future Improvements

* Add **web-based interface** using Flask for real-time predictions.
* Enhance preprocessing to improve accuracy.
* Extend to handle **custom image uploads**.

---
<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/1036500f-1d2d-43a9-888c-bf39c962905f" />
## Model Training Output

The following plot shows the **training vs validation accuracy** of the CNN model over 5 epochs:

![Training vs Validation Accuracy](output.png)

As seen in the plot:  
- The training accuracy increases steadily and reaches around 89%.  
- The validation accuracy closely follows the training accuracy, indicating that the model is **not overfitting**.  
- This demonstrates that the model learns to classify fashion images effectively across all 10 categories.


