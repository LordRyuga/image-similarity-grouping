import React from 'react';
import './AlbumDisplay.css';

function AlbumDisplay({ albums }) {
  return (
    <div className="album-display">
      <h2 className="albums-title">Grouped Albums</h2>
      <div className="albums-grid">
        {albums.map((album) => (
          <div key={album.id} className="album-card">
            <div className="album-header">
              <h3>Album {album.id + 1}</h3>
              <span className="image-count">{album.images.length} images</span>
            </div>
            <div className="images-grid">
              {album.images.map((image, idx) => (
                <div key={idx} className="image-item">
                  <img
                    src={image.data}
                    alt={image.name}
                    className="album-image"
                  />
                  <div className="image-name">{image.name}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default AlbumDisplay;