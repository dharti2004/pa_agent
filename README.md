# API Endpoints Documentation

This document provides an overview of the available API endpoints in `api.py` and `main.py`.

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
