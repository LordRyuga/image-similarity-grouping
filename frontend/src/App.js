import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ImageUpload from './components/ImageUpload';
import ModelSelector from './components/ModelSelector';
import AlbumDisplay from './components/AlbumDisplay';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [albums, setAlbums] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/models`);
      setModels(response.data.models);
      if (response.data.models.length > 0) {
        setSelectedModel(response.data.models[0]);
      }
    } catch (err) {
      setError('Failed to fetch models. Make sure the backend is running.');
      console.error(err);
    }
  };

  const handleFilesSelected = (files) => {
    setSelectedFiles(files);
    setError('');
    setAlbums([]);
    setResults(null);
  };

  const handleModelChange = (model) => {
    setSelectedModel(model);
  };

  const handleProcessImages = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select some images first');
      return;
    }

    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    setLoading(true);
    setError('');
    setAlbums([]);

    const formData = new FormData();
    selectedFiles.forEach((file) => {
      formData.append('images', file);
    });
    formData.append('model', selectedModel);

    try {
      const response = await axios.post(`${API_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setAlbums(response.data.albums);
        setResults({
          totalImages: response.data.total_images,
          numAlbums: response.data.num_albums,
        });
      }
    } catch (err) {
      setError(
        err.response?.data?.error || 'Failed to process images. Please try again.'
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFiles([]);
    setAlbums([]);
    setError('');
    setResults(null);
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🖼️ Image Similarity Grouping</h1>
          <p>Upload images and group them by visual similarity using deep learning</p>
        </header>

        {error && (
          <div className="error-message">
            <span>⚠️</span> {error}
          </div>
        )}

        {!albums.length && (
          <div className="upload-section">
            <ImageUpload onFilesSelected={handleFilesSelected} />
            
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              onModelChange={handleModelChange}
            />

            {selectedFiles.length > 0 && (
              <div className="selected-files-info">
                <p>✓ {selectedFiles.length} images selected</p>
              </div>
            )}

            <div className="button-group">
              <button
                className="btn btn-primary"
                onClick={handleProcessImages}
                disabled={loading || selectedFiles.length === 0}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Processing...
                  </>
                ) : (
                  'Process Images'
                )}
              </button>
              
              {selectedFiles.length > 0 && (
                <button className="btn btn-secondary" onClick={handleReset}>
                  Clear
                </button>
              )}
            </div>
          </div>
        )}

        {loading && (
          <div className="loading-container">
            <div className="loader"></div>
            <p>Generating embeddings and clustering images...</p>
          </div>
        )}

        {results && albums.length > 0 && (
          <div className="results-section">
            <div className="results-header">
              <div className="results-stats">
                <div className="stat">
                  <span className="stat-value">{results.totalImages}</span>
                  <span className="stat-label">Images Processed</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{results.numAlbums}</span>
                  <span className="stat-label">Albums Created</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{selectedModel}</span>
                  <span className="stat-label">Model Used</span>
                </div>
              </div>
              <button className="btn btn-secondary" onClick={handleReset}>
                Process New Images
              </button>
            </div>
            <AlbumDisplay albums={albums} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;