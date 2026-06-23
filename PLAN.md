# Implementation Plan: React + FastAPI Architecture

## Overview

Convert the Streamlit-based UAP Data Analysis Tool to a modern React frontend with FastAPI backend, maintaining all existing functionality.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │ Parsing │ │Analysis │ │ Search  │ │Magnetic │ │  Map  │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └───┬───┘ │
└───────┼──────────┼──────────┼──────────┼──────────┼───────┘
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                          │
│  /api/parse  /api/analyze  /api/search  /api/magnetic /api/map │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              Existing Python Services                       │
│  UAPParser  UAPAnalyzer  UAPVisualizer  Cohere  InterMagnet │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
UAP-Data-Analysis-Tool/
├── api/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Settings and API keys
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── upload.py             # File upload endpoints
│   │   ├── parse.py              # OpenAI parsing endpoints
│   │   ├── analyze.py            # UMAP/HDBSCAN/XGBoost endpoints
│   │   ├── search.py             # Cohere rerank endpoints
│   │   ├── magnetic.py           # InterMagnet correlation endpoints
│   │   └── map.py                # Geospatial data endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── parser_service.py     # Wraps UAPParser
│   │   ├── analyzer_service.py   # Wraps UAPAnalyzer
│   │   ├── visualizer_service.py # Wraps UAPVisualizer
│   │   ├── search_service.py     # Cohere rerank logic
│   │   ├── magnetic_service.py   # InterMagnet API + DTW
│   │   └── map_service.py        # Kepler.gl data prep
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py            # Pydantic request/response models
│   │   └── jobs.py               # Background job tracking
│   └── utils/
│       ├── __init__.py
│       ├── file_handler.py       # CSV/Excel/HDF5 handling
│       └── serialization.py      # NumPy/DataFrame serialization
│
├── frontend/                     # React frontend
│   ├── package.json
│   ├── tailwind.config.js
│   ├── vite.config.js
│   ├── index.html
│   ├── src/
│   │   ├── main.jsx
│   │   ├── App.jsx
│   │   ├── api/
│   │   │   └── client.js         # Axios/fetch wrapper
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Navbar.jsx
│   │   │   │   ├── Sidebar.jsx
│   │   │   │   └── Layout.jsx
│   │   │   ├── common/
│   │   │   │   ├── FileUpload.jsx
│   │   │   │   ├── DataTable.jsx
│   │   │   │   ├── LoadingSpinner.jsx
│   │   │   │   └── ErrorBoundary.jsx
│   │   │   ├── charts/
│   │   │   │   ├── Treemap.jsx
│   │   │   │   ├── Histogram.jsx
│   │   │   │   ├── ScatterPlot.jsx
│   │   │   │   ├── Heatmap.jsx
│   │   │   │   └── ConfusionMatrix.jsx
│   │   │   └── map/
│   │   │       └── KeplerMap.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Parsing.jsx
│   │   │   ├── Analysis.jsx
│   │   │   ├── Search.jsx
│   │   │   ├── Magnetic.jsx
│   │   │   └── Map.jsx
│   │   ├── hooks/
│   │   │   ├── useFileUpload.js
│   │   │   ├── useAnalysis.js
│   │   │   └── useWebSocket.js
│   │   ├── store/
│   │   │   └── index.js          # Zustand or React Context
│   │   └── styles/
│   │       └── globals.css
│   └── public/
│       └── assets/
```

## Implementation Steps

### Phase 1: FastAPI Backend Setup

#### Step 1.1: Create API structure and main entry point
- Create `api/` directory structure
- Set up FastAPI app with CORS middleware
- Configure settings from environment/secrets
- Add health check endpoint

#### Step 1.2: Create Pydantic schemas
- Define request models for each endpoint
- Define response models with proper typing
- Create job status models for async operations

#### Step 1.3: Implement file upload endpoint
- POST `/api/upload` - Accept CSV/Excel files
- Store uploaded files temporarily
- Return file ID and column list
- Support chunked uploads for large files

#### Step 1.4: Implement parsing endpoints
- POST `/api/parse/start` - Start async parsing job
- GET `/api/parse/status/{job_id}` - Check job status
- GET `/api/parse/result/{job_id}` - Get parsed results
- Wrap existing UAPParser with proper error handling

#### Step 1.5: Implement analysis endpoints
- POST `/api/analyze/start` - Start clustering analysis
- GET `/api/analyze/status/{job_id}` - Check status
- GET `/api/analyze/clusters/{job_id}` - Get cluster data
- GET `/api/analyze/embeddings/{job_id}` - Get 2D embeddings for visualization
- GET `/api/analyze/predictions/{job_id}` - Get XGBoost results

#### Step 1.6: Implement search endpoints
- POST `/api/search/rerank` - Cohere rerank search
- Return ranked results with relevance scores

#### Step 1.7: Implement magnetic endpoints
- GET `/api/magnetic/stations` - List InterMagnet stations
- POST `/api/magnetic/correlate` - Run DTW correlation
- Return correlation results and time series data

#### Step 1.8: Implement map endpoints
- GET `/api/map/sightings` - Get sighting GeoJSON
- GET `/api/map/bases` - Get military bases data
- GET `/api/map/plants` - Get nuclear facilities data
- GET `/api/map/config` - Get Kepler.gl config

### Phase 2: React Frontend Setup

#### Step 2.1: Initialize React project
- Create Vite + React project in `frontend/`
- Install dependencies: react-router, axios, recharts/plotly, kepler.gl
- Configure Tailwind CSS
- Set up project structure

#### Step 2.2: Create layout components
- Navbar with navigation links
- Sidebar for feature options
- Main layout wrapper
- Responsive design

#### Step 2.3: Create common components
- FileUpload with drag-and-drop
- DataTable with sorting/filtering
- LoadingSpinner and progress indicators
- Error boundary and toast notifications

#### Step 2.4: Create chart components
- Treemap using Plotly.js
- Histogram using Recharts
- ScatterPlot for embeddings visualization
- Heatmap for Cramer's V and confusion matrix
- Feature importance bar chart

#### Step 2.5: Create Kepler.gl map component
- Integrate kepler.gl React component
- Handle data layers dynamically
- Support filtering by attributes

### Phase 3: Feature Pages

#### Step 3.1: Parsing page
- File upload interface
- Column selector
- Custom JSON schema editor (optional)
- Progress indicator for parsing
- Results table with download option

#### Step 3.2: Analysis page
- Dataset loader (upload or use parsed data)
- Column multi-selector for analysis
- Visualization tabs: Embeddings, Clusters, Predictions, Correlations
- Interactive charts with tooltips

#### Step 3.3: Search page
- Dataset display
- Query input
- Column selector for search scope
- Ranked results with relevance scores
- Click-to-expand details

#### Step 3.4: Magnetic page
- Date range selector
- Location input (lat/lon or from dataset)
- Station selector
- Correlation results with time series chart

#### Step 3.5: Map page
- Full-screen Kepler.gl map
- Layer toggles (sightings, bases, plants)
- Filter controls
- Export functionality

### Phase 4: Integration and Polish

#### Step 4.1: State management
- Set up Zustand store for global state
- Persist uploaded data across pages
- Handle authentication state (API keys)

#### Step 4.2: WebSocket for long-running tasks
- Add WebSocket endpoint for job progress
- Real-time updates during parsing/analysis

#### Step 4.3: Error handling
- Consistent error responses from API
- User-friendly error messages in frontend
- Retry logic for failed requests

#### Step 4.4: Testing
- API endpoint tests with pytest
- Component tests with React Testing Library

## Key API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload CSV/Excel file |
| POST | `/api/parse/start` | Start parsing job |
| GET | `/api/parse/status/{job_id}` | Get parsing status |
| GET | `/api/parse/result/{job_id}` | Get parsed data |
| POST | `/api/analyze/start` | Start analysis job |
| GET | `/api/analyze/clusters/{job_id}` | Get cluster results |
| GET | `/api/analyze/embeddings/{job_id}` | Get 2D embeddings |
| POST | `/api/search/rerank` | Semantic search |
| GET | `/api/magnetic/stations` | List stations |
| POST | `/api/magnetic/correlate` | Run correlation |
| GET | `/api/map/sightings` | Get sighting GeoJSON |
| GET | `/api/map/bases` | Get bases GeoJSON |

## Dependencies to Add

### Backend (add to pyproject.toml)
```
fastapi
uvicorn[standard]
python-multipart
aiofiles
websockets
```

### Frontend (package.json)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.x",
    "axios": "^1.x",
    "plotly.js": "^2.x",
    "react-plotly.js": "^2.x",
    "kepler.gl": "^3.x",
    "react-dropzone": "^14.x",
    "@tanstack/react-table": "^8.x",
    "zustand": "^4.x",
    "react-hot-toast": "^2.x"
  },
  "devDependencies": {
    "vite": "^5.x",
    "tailwindcss": "^3.x",
    "autoprefixer": "^10.x",
    "postcss": "^8.x"
  }
}
```

## Running the Application

```bash
# Terminal 1: Start FastAPI backend
cd api && uvicorn main:app --reload --port 8000

# Terminal 2: Start React frontend
cd frontend && npm run dev
```

## Notes

- Long-running tasks (parsing, analysis) use background jobs with polling or WebSocket updates
- Embeddings are stored server-side and referenced by job_id to avoid large payloads
- Visualizations are generated as Plotly JSON for interactive frontend rendering
- The existing `uap_analyzer.py` and `utils/` modules are reused as services
