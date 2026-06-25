from flask import Flask, request, render_template
import os
from ranker import extract_all_resumes, rank_resumes

app = Flask(__name__)

# folder to temporarily store uploaded PDFs
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # create folder if it doesn't exist

# ---------- ROUTE 1: Show upload page ----------
@app.route('/')
def home():
    return render_template('index.html')

# ---------- ROUTE 2: Accept uploads + run ranker ----------
@app.route('/rank', methods=['POST'])
def rank():
    # get job description text from the form
    jd_text = request.form.get('jd_text', '').strip()

    if not jd_text:
        return "Please enter a Job Description.", 400

    # get uploaded PDF files from the form
    uploaded_files = request.files.getlist('resumes')  # list of uploaded files

    if not uploaded_files or uploaded_files[0].filename == '':
        return "Please upload at least one resume.", 400

    # save each uploaded PDF to the uploads folder temporarily
    saved_paths = []
    for file in uploaded_files:
        if file.filename.endswith('.pdf'):
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            saved_paths.append(save_path)

    if not saved_paths:
        return "No valid PDF files found.", 400

    # extract text from all saved PDFs
    resumes = extract_all_resumes(saved_paths)

    # rank resumes against JD
    rankings = rank_resumes(jd_text, resumes)

    # clean up - delete uploaded files after processing
    for path in saved_paths:
        os.remove(path)

    # send rankings to results page
    return render_template('result.html', rankings=rankings)

# ---------- RUN APP ----------
if __name__ == '__main__':
    app.run(debug=True)