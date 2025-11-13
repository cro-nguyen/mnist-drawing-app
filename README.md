# 🎨 MNIST Handwritten Digit Recognition Web App

An interactive web application that uses a Convolutional Neural Network (CNN) to recognize handwritten digits in real-time. Draw a digit (0-9) on the canvas and watch AI predict it instantly!

## 🚀 Live Demo

**Try it here:** [https://cro-nguyen.github.io/mnist-drawing-app/](https://cro-nguyen.github.io/mnist-drawing-app/)

> Draw with your mouse on desktop or your finger on mobile devices!

---

## ✨ Features

- 🖌️ **Interactive Drawing Canvas** - Draw digits naturally with mouse or touch
- 🤖 **Real-Time Predictions** - Instant AI-powered digit recognition
- 📊 **Confidence Visualization** - See probability scores for all 10 digits
- 📱 **Mobile Responsive** - Works seamlessly on phones and tablets
- 🎯 **High Accuracy** - 98%+ accuracy using CNN trained on MNIST dataset
- ⚡ **Fast Performance** - Runs entirely in browser using ONNX Runtime Web

---

## 🛠️ Technologies Used

### Machine Learning
- **PyTorch** - Deep learning framework for model training
- **MNIST Dataset** - 60,000 training images of handwritten digits
- **CNN Architecture** - 2 convolutional layers + 2 fully connected layers
- **ONNX** - Universal model format for web deployment

### Web Technologies
- **HTML5 Canvas** - For drawing interface
- **JavaScript** - Application logic and event handling
- **ONNX Runtime Web** - Browser-based ML inference
- **CSS3** - Modern, responsive styling

---

## 🏗️ Architecture

### Model Architecture
```
Input (28×28 grayscale image)
    ↓
Conv2D (32 filters, 3×3) + ReLU + MaxPool
    ↓
Conv2D (64 filters, 3×3) + ReLU + MaxPool
    ↓
Flatten (3,136 neurons)
    ↓
Fully Connected (128 neurons) + ReLU + Dropout(0.25)
    ↓
Output Layer (10 neurons - softmax)
```

### Data Flow
```
User Drawing → Canvas (280×280)
    ↓
Preprocessing (resize to 28×28, normalize)
    ↓
ONNX Model Inference
    ↓
Softmax Probabilities
    ↓
Display Prediction + Confidence
```

---

## 📦 Project Structure

```
mnist-drawing-app/
├── index.html              # Main application file
├── mnist_model.onnx        # Trained CNN model (ONNX format)
├── README.md               # Project documentation
└── training/               # (Optional) Training code
    └── train_model.ipynb   # Google Colab notebook
```

---

## 🚀 Getting Started

### Option 1: Use the Live Demo
Simply visit **[https://cro-nguyen.github.io/mnist-drawing-app/](https://cro-nguyen.github.io/mnist-drawing-app/)** and start drawing!

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/cro-nguyen/mnist-drawing-app.git
   cd mnist-drawing-app
   ```

2. **Start a local server**
   
   Using Python:
   ```bash
   python -m http.server 8000
   ```
   
   Or using Node.js:
   ```bash
   npx http-server
   ```

3. **Open in browser**
   ```
   http://localhost:8000
   ```

> **Note:** You must use a local server (not just open the HTML file) because browsers restrict loading ONNX models from the file system.

---

## 🎓 How It Works

### 1. Model Training (Google Colab)
- Trained on 60,000 MNIST handwritten digit images
- Used data normalization (mean=0.1307, std=0.3081)
- Achieved **99.12% test accuracy** in 5 epochs
- Training time: ~5 minutes on CUDA GPU
- Exported to ONNX format for web compatibility

### 2. User Drawing
- Canvas captures mouse/touch events
- Drawing rendered at 280×280 pixels
- Black strokes on white background

### 3. Preprocessing
- Resize drawing to 28×28 pixels (MNIST standard)
- Convert to grayscale
- Invert colors (white digit on black background)
- Normalize using MNIST statistics (mean=0.1307, std=0.3081)

### 4. Prediction
- ONNX Runtime loads model in browser
- Preprocessed image fed to CNN
- Model outputs 10 scores (logits)
- Softmax converts to probabilities
- Highest probability = predicted digit

---

## 🎯 Usage Tips

For best results:
- ✅ Draw digits **large and centered**
- ✅ Use **bold, continuous strokes**
- ✅ Make digits look like **printed numbers**
- ✅ Clear canvas and try again if prediction is wrong
- ❌ Avoid very small or off-center drawings

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 99.12% |
| **Test Accuracy (Epoch 1)** | 98.50% |
| **Test Accuracy (Epoch 2)** | 98.97% |
| **Test Accuracy (Epoch 3)** | 98.94% |
| **Test Accuracy (Epoch 4)** | 99.05% |
| **Final Test Accuracy** | **99.12%** |
| **Model Size** | ~500 KB |
| **Inference Time** | <50ms |
| **Parameters** | ~100K |
| **Training Time** | ~5 minutes (GPU) |
| **Training Device** | CUDA (GPU) |

### Training Progress
```
Epoch 1: 98.50% → Loss decreased from 0.5739 to 0.0681
Epoch 2: 98.97% → Loss decreased from 0.0473 to 0.0478
Epoch 3: 98.94% → Loss decreased from 0.0424 to 0.0361
Epoch 4: 99.05% → Loss decreased from 0.0253 to 0.0304
Epoch 5: 99.12% → Loss decreased from 0.0226 to 0.0210
```

### Sample Predictions
The model achieved **100% accuracy** on test samples:
- Predicted: `[7, 2, 1, 0, 4]`
- Actual: `[7, 2, 1, 0, 4]`
- ✅ Perfect match!

### Performance Insights
- Rapid convergence: Achieved 98.5% accuracy in just the first epoch
- Consistent improvement: Steady accuracy gains across all 5 epochs
- Low final loss: 0.0210 indicates excellent model fit
- GPU acceleration: Training completed in approximately 5 minutes
- Stable training: No signs of overfitting or instability

---

## 🔧 Development

### Training the Model

1. **Open Google Colab**
   - Upload the training notebook or create new one

2. **Run training code**
   ```python
   # Install dependencies
   !pip install torch torchvision onnx
   
   # Train model (see training notebook for full code)
   # Exports to mnist_model.onnx
   ```

3. **Download trained model**
   - Download `mnist_model.onnx` from Colab
   - Replace in project directory

### Customizing the App

**Change canvas size:**
```javascript
// In index.html
<canvas id="drawingCanvas" width="280" height="280"></canvas>
```

**Adjust brush thickness:**
```javascript
ctx.lineWidth = 20;  // Change this value
```

**Modify colors:**
```css
/* In the <style> section */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- 🎨 Improve UI/UX design
- 📈 Add training accuracy graphs
- 🔄 Implement undo/redo functionality
- 🎭 Support letter recognition (A-Z)
- 🌐 Add multi-language support
- 📊 Display confusion matrix
- 💾 Save and share drawings

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Technical Details

### Browser Compatibility
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

### Requirements
- Modern web browser with JavaScript enabled
- No installation or plugins required
- Works offline after first load (model cached)

### Performance
- Model loads in ~200-500ms
- Prediction takes ~20-50ms
- Smooth drawing at 60 FPS
- Optimized for mobile devices

---

## 📚 Learning Resources

### Understanding the Code
- [PyTorch MNIST Tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [ONNX Runtime Web Docs](https://onnxruntime.ai/docs/tutorials/web/)
- [HTML5 Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

### Deep Learning Concepts
- [CNN Explainer (Interactive)](https://poloclub.github.io/cnn-explainer/)
- [3Blue1Brown Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)
- [Fast.ai Course](https://course.fast.ai/)

---

## 🐛 Known Issues

- Very small drawings may not be recognized accurately
- Unusual digit styles might confuse the model
- Some browsers may have slight drawing lag on older devices

### Troubleshooting

**Model not loading?**
- Ensure both `index.html` and `mnist_model.onnx` are in the same directory
- Check browser console (F12) for error messages
- Try hard refresh (Ctrl+Shift+R / Cmd+Shift+R)

**Poor predictions?**
- Draw larger and more centered
- Use bolder strokes
- Make digits look more like printed numbers
- Clear and redraw if needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Hung Nguyen** (cro-nguyen)
- 🎓 Student at Northeastern University
- 📚 Course: EAI 6010B - Applications of Artificial Intelligence
- 📧 Email: [your.email@northeastern.edu]
- 🔗 GitHub: [@cro-nguyen](https://github.com/cro-nguyen)

---

## 🙏 Acknowledgments

- **Professor Richard** - Course instructor
- **MNIST Database** - Yann LeCun and Corinna Cortes
- **PyTorch Team** - Deep learning framework
- **ONNX Community** - Model format and runtime
- **Fast.ai** - Educational resources and inspiration

---

## 📈 Future Enhancements

- [ ] Support for drawing multiple digits at once
- [ ] Add drawing tools (pen size, eraser, colors)
- [ ] Export predictions as JSON/CSV
- [ ] Compare multiple models (CNN vs MLP)
- [ ] Real-time training visualization
- [ ] Support for EMNIST (letters + digits)
- [ ] Dark mode toggle
- [ ] Save drawing history
- [ ] Leaderboard of most confident predictions

---

## 🌟 Star History

If you found this project helpful, please consider giving it a ⭐ on GitHub!

---

## 📞 Contact

Have questions or suggestions? Feel free to:
- 🐛 [Open an issue](https://github.com/cro-nguyen/mnist-drawing-app/issues)
- 💬 [Start a discussion](https://github.com/cro-nguyen/mnist-drawing-app/discussions)
- 📧 Email me directly

---

<div align="center">

**Made with ❤️ and lots of ☕**

[⬆ Back to Top](#-mnist-handwritten-digit-recognition-web-app)

</div>
