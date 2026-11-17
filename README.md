# Potato Disease Classifier

Potato Disease Classifier is a small project that uses a CNN model to identify
potato leaf conditions from an image.

The app is made with Streamlit. Upload a potato leaf photo, click
**Detect Disease**, and the model will show its prediction and
confidence score.

## What it can detect

-   Potato Early Blight
-   Potato Late Blight
-   Potato Healthy

## Tools used

-   Python
-   TensorFlow / Keras
-   CNN (Convolutional Neural Network)
-   Streamlit
-   NumPy
-   Pillow

## Project files

Keep these files in the same folder:

``` text
Pottato_disease_cnn/
│
├── app.py
├── potato_cnn_model.keras
├── classification_report.txt
└── requirements.txt
```

-   `app.py` contains the Streamlit application.
-   `potato_cnn_model.keras` is the trained CNN model.
-   `classification_report.txt` contains the class names in the same
    order as the model output.
-   `requirements.txt` contains the Python libraries needed to run the
    app.

## Setup

### 1. Open the project folder

Open PowerShell in your project folder:

``` powershell
cd E:\Pottato_disease_cnn
```

### 2. Activate the virtual environment

If you already created a virtual environment named `venv`, activate it:

``` powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

``` powershell
python -m pip install -r requirements.txt
```

### 4. Run the app

``` powershell
python -m streamlit run app.py
```

Streamlit will show a local URL in the terminal. Open that URL in your
browser.

## How to use

1.  Open PotatoCare AI in your browser.
2.  Upload a potato leaf image in JPG, JPEG, or PNG format.
3.  Click **Detect Disease**.
4.  Check the predicted class and confidence score.
5.  Open **View all class probabilities** if you want to see the scores
    for all classes.

## Model details

The CNN model uses RGB images resized to `128 x 128` pixels. It has
convolution, batch normalization, max-pooling, and dense layers. The
final layer uses softmax to predict one of the three classes.

The notebook uses an 80/20 training and validation split. The validation
results are not the same as results on a separate test dataset.

## Note

This project is made for learning and demonstration. The prediction is
based on the image provided and should not be treated as a confirmed
agricultural diagnosis.
