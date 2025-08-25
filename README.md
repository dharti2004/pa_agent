
This document provides an overview of the available API endpoints in `api.py` and `main.py`.

## Prerequisites

Before running the application, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)
- MongoDB (for data storage)
- MongoDB Compass (for database management)
- Google API key (for Gemini AI model)

## MongoDB Setup

### 1. Install MongoDB Compass

**On Ubuntu/Debian:**
```bash
# Download MongoDB Compass
wget https://downloads.mongodb.com/compass/mongodb-compass_1.40.4_amd64.deb

# Install the package
sudo dpkg -i mongodb-compass_1.40.4_amd64.deb

# Fix any dependency issues
sudo apt-get install -f
```

**On macOS:**
```bash
# Using Homebrew
brew install --cask mongodb-compass

# Or download from official website
# https://www.mongodb.com/try/download/compass
```

**On Windows:**
```bash
# Using Chocolatey
choco install mongodb-compass

# Or download from official website
# https://www.mongodb.com/try/download/compass
```

### 2. Set up MongoDB Connection

1. **Open MongoDB Compass**
2. **Click "New Connection"**
3. **Enter your connection string:**
4. **Click "Connect"**

### 3. Create Database and Collections

1. **Create Database:**
   - Click "Create Database"
   - Database Name: `finpal_db`
   - Collection Name: `finance_profiles`
   - Click "Create Database"

2. **Create Additional Collection:**
   - In the `finpal_db` database, click "Create Collection"
   - Collection Name: `finance_memory`
   - Click "Create Collection"

### 4. Import Data to finance_profiles Collection

1. **Select the `finance_profiles` collection**
2. **Click "Add Data" → "Import File"**
3. **Choose your data file (JSON, CSV, etc.)**
4. **Configure import settings and click "Import"**


## Installation

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd /home/desk0046/Documents/pa_agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root with the following variables:
   ```env
   MONGO_URI= your MongoDB connection string with authentication
   DATABASE_NAME=your Database name
   COLLECTION_NAME= finance_profiles
   TARGET_COLLECTION_NAME= finance_memory
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

### Running the API Server (api.py)
To start the suggestions API server:
```bash
python api.py
```
This will start the server on `http://localhost:8000` and provides endpoints for:
- Building user financial memory
- Generating financial suggestions
- Health checks

### Running the Main Application (main.py)
To start the financial assistant chat application:
```bash
python main.py
```
This will start the server on `http://localhost:8000` and provides endpoints for:
- File uploads and processing
- Chat with financial assistant
- User memory management
- Finance profile processing

**Note:** Both applications run on port 8000, so you can only run one at a time.

## `api.py` Endpoints

### GET /health
- **Description**: Checks the health of the API and verifies if dependencies are initialized.
- **Responses**: Returns `{"status": "ok", "initialized": true/false}`.

### POST /memory/{user_id}
- **Description**: Builds a user's financial memory based on their `user_id`. It fetches user documents, processes them, and stores the financial memory.
- **Parameters**: `user_id` (path parameter, string).
- **Responses**: Returns a dictionary containing `finance`, `calculations`, and `profile_summary`.

### POST /suggestions/{user_id}
- **Description**: Generates financial suggestions for a user based on their `user_id` and existing financial memory.
- **Parameters**: `user_id` (path parameter, string).
- **Responses**: Returns a dictionary containing `short_msg` and `suggestion`.

## `main.py` Endpoints

### POST /upload
- **Description**: Uploads a file, parses it, and stores its embeddings in a vector store. It also keeps track of the last uploaded `file_id` for a given `user_id`.
- **Parameters**:
    - `user_id` (Form parameter, string, required).
    - `file` (Form parameter, UploadFile, required): The file to upload.
- **Example Request (Form Data)**:
```
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=test_user" \
  -F "file=@/path/to/your/file.pdf;type=application/pdf"
```
- **Responses**: Returns `{"status": "ok", "message": "File ingested to vector store", "file_id": file_id}`.

### POST /chat
- **Description**: Processes a chat message from a user, potentially using a previously uploaded file's context.
- **Parameters**:
    - `user_id` (Form parameter, string, required).
    - `messages` (Form parameter, string, required): A JSON string representing a list of messages.
    - `file_id` (Form parameter, string, optional): The ID of the file to use for context. If not provided, the last uploaded file for the user will be used.
- **Example Request (Form Data)**:
```
curl -X POST "http://localhost:8000/chat" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "user_id=test_user" \
  -F "messages=[{\"role\": \"user\", \"content\": \"What is my current balance?\"}]" \
  -F "file_id=optional_file_id"
```
- **Responses**: Returns `{"response": "..."}` containing the chat response.

### GET /memory/{user_id}
- **Description**: Retrieves the stored user memory (profiles) for a given `user_id`.
- **Parameters**: `user_id` (path parameter, string).
- **Responses**: Returns a dictionary containing the `user_id` and the user's profiles.

### POST /process-finance-profiles/{user_id}
- **Description**: Processes and updates a user's financial profile data.
- **Parameters**: `user_id` (path parameter, string).
- **Responses**: Returns `{"status": "ok", "message": "Finance profile for {user_id} processed and updated."}`.

### GET /suggestions/{user_id}
- **Description**: Fetches financial suggestions for a given `user_id`.
- **Parameters**: `user_id` (path parameter, string).
- **Responses**: Returns a dictionary containing financial suggestions.
