# DeepFakeDetector

A simple DeepFake detection web app with a Flask backend and React/Vite frontend.

## Project structure

- `backend/` - Flask API, PyTorch model loading, Grad-CAM visualization
- `frontend/` - React + Vite UI for image upload and prediction

## Backend

1. Create and activate a Python virtual environment:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend:
   ```bash
   python app.py
   ```

4. Default backend URL:
   - `http://localhost:3000`

> Note: The backend in `backend/app.py` currently runs on port `3000`.

## Frontend

1. Install dependencies from the `frontend` folder:
   ```bash
   cd frontend
   npm install --no-bin-links
   ```

2. Start the frontend development server:
   ```bash
   npm run dev -- --host 127.0.0.1 --port 5173
   ```

3. Open the app in your browser at:
   - `http://127.0.0.1:5173`

## API configuration

- The frontend uses `frontend/src/App.jsx` and sends requests to `http://localhost:3000/predict`.
- If you change the backend port, update `API_URL` in `frontend/src/App.jsx` to match.

## Common issues

- On NTFS mounts, npm may fail when creating symlinks. Use:
  ```bash
  npm install --no-bin-links
  ```

- If the app always predicts `Fake`, this is usually caused by the model checkpoint or label order, not by the React UI.

## Troubleshooting

- Verify backend startup output and confirm `Model loaded successfully!` appears.
- Verify frontend startup output and confirm Vite is ready.
- If port `5173` is busy, Vite will choose another available port.
- If you change backend port, update `API_URL` in the frontend accordingly.
