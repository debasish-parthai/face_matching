# Face Matching Frontend

A modern web interface for the Face Matching API built with HTML, CSS, and JavaScript.

## Features

- **Dual Upload Areas**: Separate sections for candidate and reference images
- **Image Previews**: Visual preview of uploaded images with remove options
- **Real-time Processing**: Shows progress during face matching
- **Face Display**: Shows cropped face images from both candidate and reference photos
- **Detailed Results**: JSON output with comprehensive matching data
- **Responsive Design**: Works on desktop and mobile devices
- **Modern UI**: Clean, gradient-based design with smooth animations

## Files

- `index.html` - Main HTML structure
- `styles.css` - Modern CSS styling with responsive design
- `script.js` - JavaScript functionality for uploads and API calls

## Usage

1. **Start the Backend**: Make sure the FastAPI server is running on `http://localhost:8000`

2. **Open the Frontend**: Open `index.html` in your web browser

3. **Upload Images**:
   - Click "Choose File" in the Candidate Image section to select the person you want to match
   - Click "Choose Files" in the Reference Images section to select one or more reference images

4. **Compare Faces**: Click the "Compare Faces" button to start the matching process

5. **View Results**:
   - Cropped face images are displayed in the results section
   - Detailed JSON results show matching scores, face attributes, and statistics
   - Use "Copy Results" to copy the JSON data to clipboard

## API Integration

The frontend communicates with the `/match-faces/` endpoint with the following parameters:
- `candidate_file`: The candidate image file
- `reference_files`: Array of reference image files
- `save_cropped_faces`: Set to `true` to enable face cropping

## Browser Support

- Chrome 70+
- Firefox 65+
- Safari 12+
- Edge 79+

## Development

To modify the frontend:

1. Edit `index.html` for structure changes
2. Modify `styles.css` for styling updates
3. Update `script.js` for functionality changes

Make sure to test changes across different browsers and screen sizes.
