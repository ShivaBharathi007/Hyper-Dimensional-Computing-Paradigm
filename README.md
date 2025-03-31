# **HDC-Based Spam Detector**

## **Project Overview**
This project implements a **Hyperdimensional Computing (HDC)-based Spam Detector** using **Python, NumPy, Pandas, and Scikit-learn**. The model classifies SMS messages as either **spam or ham (not spam)** using hyperdimensional vectors and Hamming distance for classification.

## **How It Works**
### 1. **Data Preprocessing**
- Loads the SMS spam dataset (`spam.csv`).
- Retains only the `text` (message) and `label` (ham/spam) columns.
- Converts labels into **binary format** (0 = ham, 1 = spam).
- Splits the dataset into **training and testing sets** (80% train, 20% test).

### 2. **Hyperdimensional Computing (HDC) Representation**
- Uses **character n-grams (1 to 3)** for encoding text into hyperdimensional vectors.
- Each n-gram is assigned a **random binary hypervector**.
- Text is converted into a **single hypervector** using bundling (majority vote).

### 3. **Step-by-Step HDC Process**
1. **Item Memory Creation:**
   - Each unique n-gram (sequence of characters) in the dataset is assigned a random binary hypervector.
   
2. **Text Encoding:**
   - A given message is split into n-grams.
   - The hypervectors corresponding to the n-grams are retrieved and combined using **XOR binding**.
   - The final representation of the message is obtained using a **majority vote bundling** operation.
   
3. **Class Prototype Formation:**
   - Hypervectors of spam and ham messages are bundled separately to form **class prototype hypervectors**.
   
4. **Classification:**
   - For a new message, its hypervector representation is compared against the class prototypes.
   - **Hamming distance** (bitwise difference) is computed.
   - The class with the **smallest Hamming distance** is assigned to the message.

### 4. **Why HDC is 1000x More Efficient on CPU than GPU**
- **Binary Operations:** HDC relies on simple bitwise operations (XOR, majority vote) that are **highly parallelizable on CPUs** without requiring floating-point computations like deep learning models.
- **No Backpropagation:** Unlike deep learning, HDC does not require gradient descent, making it much faster.
- **Memory Efficiency:** HDC uses compact binary vectors instead of large floating-point tensors.
- **Lightweight Computation:** Hamming distance calculations require only XOR operations and bit counting, which are significantly faster than matrix multiplications in deep learning models.
- **Scalability on CPUs:** Modern CPUs can perform **SIMD (Single Instruction Multiple Data) operations**, accelerating HDC operations efficiently without requiring GPUs.

---

## **Training the HDC Model**
- Constructs an **item memory** for mapping n-grams to hypervectors.
- Generates **class prototype hypervectors** for spam and ham messages.
- Uses **Hamming distance** to classify new messages.

### 5. **Model Evaluation**
- Evaluates the model using:
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report**
- Tests on sample SMS messages.

---

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/HDC-Spam-Detector.git
cd HDC-Spam-Detector
```

### **2. Install Dependencies**
```bash
pip install numpy pandas scikit-learn
```

### **3. Run in Google Colab**
- Open the **Colab Notebook**: [HDC_Spam_Detector.ipynb](https://colab.research.google.com/)
- Upload `spam.csv` or modify the dataset path accordingly.

---

## **Usage**
### **1. Run the Model on Sample Messages**
After training, test the model on custom messages:
```python
test_messages = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts",
    "Hey, how are you doing today?",
    "URGENT! You have won a 1 week FREE membership",
    "Let's meet for lunch tomorrow"
]

for msg in test_messages:
    prediction = hdc_model.predict(msg)
    print(f"Message: {msg}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}\n")
```

---

## **Potential Improvements**
✅ Improve text preprocessing using **NLTK/SpaCy**.  
✅ Store and reload trained models using **joblib/pickle**.  
✅ Use **cosine similarity** instead of Hamming distance.  
✅ Implement **hyperparameter tuning** for `D` and `ngram_range`.  

---

**Contributions are welcome!** Feel free to fork and improve the project 🚀

