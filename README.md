Skin Cancer Classification and Lesion Detection
This project is a web application for skin cancer classification and lesion detection. It uses a UNet model for lesion detection and a ResNet model for skin cancer classification.

Features
Upload an image of a skin lesion.
Detect the lesion in the uploaded image.
Classify the type of skin cancer based on the lesion.
How to Run the Application
Prerequisites
Make sure you have the following installed:

Python 3.x
Required Python libraries (see requirements.txt for dependencies).
Installation
Clone the repository:


git clone https://github.com/your-repository/skin-cancer-classification.git
cd skin-cancer-classification
Install the dependencies:

pip install -r requirements.txt

Start the application:


python app.py
Open your browser and navigate to:


http://127.0.0.1:5000
Upload an image of a skin lesion using the UPLOAD button.

View the results:

Lesion Detection: An image highlighting the detected lesion.
Classification Result: The type of skin cancer detected.
Models Used
Lesion Detection: UNet
Cancer Classification: ResNet
Both models are pre-trained and fine-tuned for this specific application.

Example Output
Input Image:

Lesion Detection Result:
An image with Lesion will be displayed here

Classification Result:
Type: Melanoma
