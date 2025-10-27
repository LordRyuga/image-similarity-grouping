import React from 'react';
import './ModelSelector.css';

function ModelSelector({ models, selectedModel, onModelChange }) {
  return (
    <div className="model-selector">
      <label htmlFor="model-select" className="selector-label">
        Select Model:
      </label>
      <select
        id="model-select"
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        className="model-select"
      >
        {models.map((model) => (
          <option key={model} value={model}>
            {model.toUpperCase()}
          </option>
        ))}
      </select>
    </div>
  );
}

export default ModelSelector;