from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import pytesseract
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your reference file
reference_file_path = r"C:\Users\vivek\Downloads\OCR_APP\flask_ocr_app\templates\reference.txt"
with open(reference_file_path, "r", encoding="utf-8") as reference_file:
    reference_text = reference_file.read()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        extracted_text = extract_text_from_image(filename)
        similarity_score = compare_text_to_reference(extracted_text)

        return render_template('result.html', extracted_text=extracted_text, score=similarity_score)

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def compare_text_to_reference(text):
    # Create TF-IDF vectors for the reference and extracted text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference_text, text])

    # Calculate the cosine similarity between the two vectors
    similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    # The similarity score is the value in the similarity matrix
    similarity_score = similarity_matrix[0][0]

    # Map the similarity score to a 0-100 range
    score_out_of_100 = similarity_score * 100

    return score_out_of_100

if __name__ == '__main__':
    app.run(debug=True)
