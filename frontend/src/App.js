import React, { useState } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [processedImage, setProcessedImage] = useState("");
  const [detections, setDetections] = useState([]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("", selectedFile); // Ensure this key matches the Flask backend

    try {
      const response = await axios.post("http://127.0.0.1:5000/detect", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log(response.data); // Check the response data from the backend

      if (response.data.error) {
        alert("Error: " + response.data.error);
      } else {
        // Assuming the response data structure is as expected (output_image and detections)
        const imagePath = response.data.output_image.replace("\\", "/"); // Fix backslash issue for URL
        setProcessedImage(`http://127.0.0.1:5000/${imagePath}`);
        setDetections(response.data.detections); // Update detections
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Failed to process the image.");
    }
  };

  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h2>Face Detection Upload</h2>

      {/* File input for image selection */}
      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />
      <button onClick={handleUpload} style={{ marginLeft: "10px" }}>
        Upload
      </button>

      {/* Display uploaded image preview */}
      {preview && (
        <div>
          <h3>Uploaded Image:</h3>
          <img
            src={preview}
            alt="Uploaded Preview"
            style={{ maxWidth: "100%", border: "2px solid #ddd" }}
          />
        </div>
      )}

      {/* Display processed image with detections */}
      {processedImage && (
        <div>
          {/* <h3>Processed Image with Detections:</h3> */}
          {/* <img
            src={processedImage}
            alt="Processed"
            style={{ maxWidth: "100%", border: "2px solid green" }}
          /> */}
        </div>
      )}

      {/* Display detections if available */}
      {detections.length > 0 && (
        <div>
          {/* <h3>Face Detection Results:</h3> */}
          <h3>Face Detection Results (Total Detections: {detections.length}):</h3>
          <ul>
            {detections.map((det, index) => (
              <li key={index}>
                Confidence: {det.confidence.toFixed(2)}
                <br />
                Bounding Box: [{det.x1}, {det.y1}] to [{det.x2}, {det.y2}]
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
