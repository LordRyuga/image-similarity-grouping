from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from werkzeug.utils import secure_filename
import numpy as np
from models.model_loader import ModelLoader
from models.clustering import cluster_embeddings

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model loader at startup
model_loader = ModelLoader()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return list of available models"""
    return jsonify({
        'models': model_loader.get_available_models()
    })

@app.route('/api/upload', methods=['POST'])
def upload_images():
    """Process uploaded images and return grouped albums"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        model_name = request.form.get('model', 'simclr')
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({'error': 'No images selected'}), 400
        
        # Save uploaded files temporarily
        image_paths = []
        image_data = []
        
        for file in files:
            filename = file.filename
            if not filename or not allowed_file(filename):
                continue  # skip files without valid names

            safe_name = secure_filename(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            file.save(filepath)
            image_paths.append(filepath)

            # Read image as base64 for frontend display
            with open(filepath, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
                ext = safe_name.rsplit('.', 1)[1].lower()
                image_data.append({
                    'name': safe_name,
                    'data': f'data:image/{ext};base64,{img_base64}'
                })

        
        if not image_paths:
            return jsonify({'error': 'No valid images uploaded'}), 400
        
        # Generate embeddings
        embeddings = model_loader.generate_embeddings(model_name, image_paths)
        
        # Cluster images
        clusters = cluster_embeddings(embeddings, n_clusters=None)
        
        # Group images by cluster
        albums = {}
        for idx, cluster_id in enumerate(clusters):
            cluster_id = int(cluster_id)
            if cluster_id not in albums:
                albums[cluster_id] = []
            albums[cluster_id].append(image_data[idx])
        
        # Clean up uploaded files
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        
        # Convert to list format
        album_list = [{'id': k, 'images': v} for k, v in albums.items()]
        
        return jsonify({
            'success': True,
            'albums': album_list,
            'total_images': len(image_paths),
            'num_albums': len(albums)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)