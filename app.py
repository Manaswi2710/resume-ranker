from flask import Flask, request, render_template
import os
from ranker import extract_all_resumes, rank_resumes

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/rank', methods=['POST'])
def rank():

    jd_text = request.form.get('jd_text', '').strip()

    if not jd_text:
        return "Please enter a Job Description.", 400

  
    uploaded_files = request.files.getlist('resumes') 

    if not uploaded_files or uploaded_files[0].filename == '':
        return "Please upload at least one resume.", 400

    saved_paths = []
    for file in uploaded_files:
        if file.filename.endswith('.pdf'):
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            saved_paths.append(save_path)

    if not saved_paths:
        return "No valid PDF files found.", 400

    
    resumes = extract_all_resumes(saved_paths)

    # rank resumes against JD
    rankings = rank_resumes(jd_text, resumes)


    for path in saved_paths:
        os.remove(path)

    # send rankings to results page
    return render_template('result.html', rankings=rankings)


if __name__ == '__main__':
    app.run(debug=True)