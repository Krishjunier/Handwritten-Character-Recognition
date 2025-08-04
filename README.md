# Handwritten Digit Recognition

A real-time handwritten digit recognition system built with deep learning, featuring an interactive web interface for drawing digits and instant prediction.

## 🎯 Project Overview

This project implements a complete end-to-end machine learning solution for recognizing handwritten digits (0-9). Users can draw digits on an HTML5 canvas, and the system provides real-time predictions using a trained Convolutional Neural Network (CNN) model.

**CodeAlpha Machine Learning Internship - Task 1**

## ✨ Features

- **Interactive Drawing Canvas**: HTML5 canvas with smooth drawing capabilities
- **Real-time Prediction**: Instant digit recognition as you draw
- **Deep Learning Model**: CNN trained on MNIST dataset
- **RESTful API**: FastAPI backend for seamless communication
- **Responsive Design**: Clean, modern interface with dark theme
- **High Accuracy**: Leverages proven CNN architecture for reliable predictions

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **TensorFlow/Keras**: Deep learning framework for model training and inference
- **PIL (Pillow)**: Image processing and manipulation
- **NumPy**: Numerical computing for array operations
- **Python 3.7+**: Core programming language

### Frontend
- **HTML5 Canvas**: Interactive drawing interface
- **Vanilla JavaScript**: Client-side logic and API communication
- **CSS3**: Styling and responsive design

### Machine Learning
- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Model Architecture**: Convolutional Neural Network (CNN)
- **Image Processing**: Grayscale conversion, resizing, normalization

## 📁 Project Structure

```
digit-recognition/
│
├── app.py                 # FastAPI backend server
├── index.html            # Frontend interface
├── model.h5              # Trained CNN model (not included)
├── static/               # Static files directory
│   └── index.html        # HTML file served by FastAPI
└── README.md             # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/CodeAlpha_HandwrittenDigitRecognition.git
cd CodeAlpha_HandwrittenDigitRecognition
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install fastapi uvicorn tensorflow pillow numpy
```

### Step 4: Prepare the Model
- Train your CNN model on MNIST dataset and save as `model.h5`
- Ensure the model expects input shape `(1, 28, 28, 1)`
- Place the model file in the project root directory

### Step 5: Set up Static Files
```bash
mkdir static
cp index.html static/
```

## 🏃‍♂️ Running the Application

### Start the FastAPI Server
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Access the Application
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

## 🎮 How to Use

1. **Draw a Digit**: Use your mouse to draw a digit (0-9) on the black canvas
2. **Get Prediction**: Click the "Predict" button to see the model's prediction
3. **Clear Canvas**: Use the "Clear" button to reset and draw a new digit
4. **Try Different Digits**: Experiment with various writing styles and digits

## 🧠 Model Architecture

The CNN model processes 28x28 grayscale images and typically includes:

- **Input Layer**: 28x28x1 grayscale images
- **Convolutional Layers**: Feature extraction with ReLU activation
- **Pooling Layers**: Spatial dimension reduction
- **Fully Connected Layers**: Classification with softmax output
- **Output Layer**: 10 neurons (digits 0-9) with probability distribution

## 📊 Model Performance

- **Dataset**: MNIST (60,000 training + 10,000 test images)
- **Expected Accuracy**: ~98-99% on test set
- **Image Format**: 28x28 grayscale
- **Classes**: 10 (digits 0-9)

## 🔧 API Endpoints

### GET `/`
- **Description**: Serves the main HTML interface
- **Response**: HTML page with drawing canvas

### POST `/predict`
- **Description**: Predicts digit from uploaded image
- **Input**: Multipart form data with image file
- **Output**: JSON with predicted digit
- **Example Response**:
```json
{
  "digit": 7
}
```

## 🎨 Frontend Features

- **Smooth Drawing**: Responsive canvas with brush-like drawing
- **Dark Theme**: Modern black background with white drawing
- **Intuitive Controls**: Clear and predict buttons for easy interaction
- **Visual Feedback**: Real-time display of prediction results

## 🔍 Technical Implementation

### Image Processing Pipeline
1. **Canvas Capture**: Extract image data from HTML5 canvas
2. **Format Conversion**: Convert to PNG blob
3. **Server Processing**: Resize to 28x28, convert to grayscale
4. **Normalization**: Scale pixel values to 0-1 range
5. **Model Input**: Reshape to (1, 28, 28, 1) for CNN inference

### Error Handling
- Graceful handling of network errors
- Console logging for debugging
- User-friendly error messages

## 🚀 Future Enhancements

- [ ] Support for multiple digits and equations
- [ ] Confidence score display
- [ ] Model training interface
- [ ] Mobile touch support optimization
- [ ] Batch prediction capabilities
- [ ] Model comparison features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of the CodeAlpha Machine Learning Internship program. Please refer to CodeAlpha's terms and conditions.

## 👨‍💻 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn Profile]
- Email: your.email@example.com

## 🙏 Acknowledgments

- **CodeAlpha**: For providing this internship opportunity
- **MNIST Dataset**: Yann LeCun and collaborators
- **TensorFlow Team**: For the excellent deep learning framework
- **FastAPI**: For the modern web framework

## 📞 Support

For questions or support regarding this project:
- **CodeAlpha Website**: www.codealpha.tech
- **WhatsApp**: +91 8052293611
- **Email**: services@codealpha.tech

---

**⭐ If you found this project helpful, please give it a star!**
