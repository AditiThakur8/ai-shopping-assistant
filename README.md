# AI Shopping Assistant

An intelligent e-commerce assistant that helps users find products through text queries and image uploads, powered by advanced AI models.

![AI Shopping Assistant](https://via.placeholder.com/800x400?text=AI+Shopping+Assistant)

## 🚀 Features

- **Text-Based Product Search**: Find products by typing natural language queries
- **Visual Search**: Upload images to find matching or complementary products
- **Price Filtering**: Search for products under a specific price point
- **Product Recommendations**: Get personalized product suggestions based on your queries
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **AI/ML Models**:
  - **CLIP** (Contrastive Language-Image Pre-Training) for image analysis
  - **Sentence Transformers** for semantic text search
  - **PyTorch** for deep learning operations

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers library
- Sentence Transformers
- Flask
- Pandas
- Pillow (PIL)
- OpenCV

## 🔧 Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-shopping-assistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the product dataset:
   - The application uses `products_cleaned.csv` which should be in the root directory
   - This dataset contains product information including titles, descriptions, prices, and image URLs

## 🚀 Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the application:
   - Enter text queries in the search box (e.g., "blue sneakers", "kitchen gadgets under 50")
   - Or upload an image to find similar products
   - View product recommendations with prices, ratings, and links

## 💡 How It Works

### Text Search
1. User enters a text query
2. The query is encoded using a Sentence Transformer model
3. Semantic similarity is calculated between the query and product descriptions
4. Most relevant products are returned and displayed

### Image Search
1. User uploads an image
2. The CLIP model analyzes the image content
3. The system finds products that match the visual elements in the image
4. Relevant products are displayed to the user

### Price Filtering
- When a query includes price constraints (e.g., "under 50"), the system filters results accordingly

## 📁 Project Structure

```
ai_shopping_assistant/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── products_cleaned.csv   # Product dataset
├── uploads/               # Directory for uploaded images
├── static/                # Static files (CSS, JS, images)
└── templates/
    └── index.html         # Main HTML template
```

## 🔍 Technical Details

- The application uses the `all-MiniLM-L6-v2` model for text embeddings
- CLIP (`clip-vit-base-patch32`) is used for image-text matching
- Product data is pre-processed and embedded at application startup
- The system combines product titles, descriptions, and reviews for better matching

## 🛠️ Future Improvements

- User accounts and personalized recommendations
- Shopping cart functionality
- Integration with actual e-commerce platforms
- Enhanced filtering options (category, brand, etc.)
- Mobile app version
- Product comparison feature

## 📄 License

[MIT License](LICENSE)

## 👥 Contributors

- [Your Name/Organization]

---

*This project was created as a demonstration of AI-powered e-commerce assistance capabilities.*
