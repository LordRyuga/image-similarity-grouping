import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import './ImageUpload.css';

function ImageUpload({ onFilesSelected }) {
  const onDrop = useCallback((acceptedFiles) => {
    onFilesSelected(acceptedFiles);
  }, [onFilesSelected]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    },
    multiple: true
  });

  return (
    <div className="image-upload">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          <div className="upload-icon">📁</div>
          {isDragActive ? (
            <p className="dropzone-text">Drop the images here...</p>
          ) : (
            <>
              <p className="dropzone-text">
                Drag & drop images here, or click to select files
              </p>
              <p className="dropzone-hint">
                Supports PNG, JPG, JPEG, GIF, BMP
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default ImageUpload;