# Video Gallery Server

A simple Python HTTP server for hosting the video gallery website with database persistence support.

## Features

- 📁 Static file serving for HTML, CSS, JavaScript, and video files
- 💾 RESTful API for database persistence
- 🔄 Automatic database synchronization
- 💿 Automatic database backups every 5 minutes (configurable)
- 🌐 CORS support for local development
- ⚡ No external dependencies (uses Python standard library)

## Requirements

- Python 3.6 or higher (no additional packages needed)

## Quick Start

1. **Navigate to the videogallery directory:**
   ```bash
   cd /home/stefano/Codebase/stereo4d-code/videogallery
   ```

2. **Start the server:**
   ```bash
   python server.py
   ```

3. **Open your browser and visit:**
   ```
   http://localhost:8000
   ```

## Usage

### Basic Usage

Start the server on default port (8000):
```bash
python server.py
```

### Custom Port

Run on a different port:
```bash
python server.py --port 8080
```

### Allow External Access

Bind to all network interfaces (accessible from other machines):
```bash
python server.py --host 0.0.0.0 --port 8000
```

Then access from other machines using:
```
http://<your-ip-address>:8000
```

### Command Line Options

```
--port PORT              Port to run the server on (default: 8000)
--host HOST              Host to bind to (default: localhost)
--backup-interval MIN    Backup interval in minutes (default: 5, 0 to disable)
```

### Custom Backup Interval

Change backup frequency to every 10 minutes:
```bash
python server.py --backup-interval 10
```

Disable automatic backups:
```bash
python server.py --backup-interval 0
```

## Automatic Backups

The server automatically creates timestamped backups of your database to protect against data loss.

### Backup Features

- **Automatic**: Backups are created every 5 minutes by default (configurable)
- **Timestamped**: Each backup has a unique timestamp in the filename
- **Organized**: All backups are stored in the `database_backups/` directory
- **Auto-cleanup**: Only the 50 most recent backups are kept
- **Initial backup**: A backup is created immediately when the server starts

### Backup File Format

Backups are named with the pattern:
```
database_backup_YYYYMMDD_HHMMSS.json
```

Example: `database_backup_20251211_143022.json`

### Restoring from Backup

To restore from a backup:

1. **Stop the server** (Ctrl+C)

2. **Find the backup** in the `database_backups/` directory:
   ```bash
   ls -lt database_backups/
   ```

3. **Copy the backup** to restore it:
   ```bash
   cp database_backups/database_backup_20251211_143022.json database.json
   ```

4. **Restart the server**:
   ```bash
   python server.py
   ```

### Manual Backup

You can also create manual backups at any time:
```bash
cp database.json database_manual_backup_$(date +%Y%m%d_%H%M%S).json
```

## API Endpoints

### Get Entire Database
```http
GET /api/database
```

Returns the complete database JSON object.

**Example:**
```bash
curl http://localhost:8000/api/database
```

### Update Entire Database
```http
POST /api/database
Content-Type: application/json

{
  "video_name.mp4": {
    "tags": ["all-good"],
    "timestamp": "2025-12-11T12:00:00.000Z",
    "lastModified": "2025-12-11T12:00:00.000Z"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/database \
  -H "Content-Type: application/json" \
  -d '{"video1.mp4": {"tags": ["all-good"], "timestamp": "2025-12-11T12:00:00.000Z"}}'
```

### Get Specific Video Data
```http
GET /api/database/<video_name>
```

**Example:**
```bash
curl http://localhost:8000/api/database/example_video.mp4
```

### Update Specific Video Data
```http
POST /api/database/<video_name>
Content-Type: application/json

{
  "tags": ["all-good", "warning"],
  "timestamp": "2025-12-11T12:00:00.000Z",
  "lastModified": "2025-12-11T12:30:00.000Z"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/database/example_video.mp4 \
  -H "Content-Type: application/json" \
  -d '{"tags": ["all-good"], "timestamp": "2025-12-11T12:00:00.000Z"}'
```

## How It Works

### Server-Side
1. The Python server serves static files (HTML, CSS, JS, videos) from the `videogallery/` directory
2. API endpoints at `/api/database` handle GET and POST requests
3. Database is stored in `database.json` in the same directory
4. CORS headers are added to allow local development

### Client-Side
1. When the page loads, JavaScript fetches the database from the server API
2. When you click tag buttons (✅, ❌, ⚠️), the database is updated locally
3. Changes are automatically saved to the server via POST request
4. If the server is not available, changes are downloaded as a JSON file (fallback)

## Database Format

The database is stored as a JSON file with the following structure:

```json
{
  "video_name.mp4": {
    "tags": ["all-good", "warning"],
    "classes": ["person", "car"],
    "timestamp": "2025-12-11T12:00:00.000Z",
    "lastModified": "2025-12-11T12:30:00.000Z"
  },
  "another_video.mp4": {
    "tags": ["all-wrong"],
    "classes": [],
    "timestamp": "2025-12-11T13:00:00.000Z",
    "lastModified": "2025-12-11T13:00:00.000Z"
  }
}
```

### Fields:
- **tags**: Array of tag strings (`"all-good"`, `"all-wrong"`, `"warning"`)
- **timestamp**: ISO 8601 timestamp when the entry was first created
- **lastModified**: ISO 8601 timestamp of the last modification
- **classes**: (Optional) Array of semantic class strings present in the video (e.g., `["person", "car", "bicycle"]`)

## Video List Format

Videos are listed in `videos.csv` with the following CSV format:

```csv
filename,classes
video1.mp4,person
video2.mp4,person,car,bicycle
video3.mp4,
```

- **First column**: Video filename (including .mp4 extension)
- **Second column**: Comma-separated list of semantic classes detected in the video (can be empty)

The classes from the CSV file are automatically loaded into the database when the page loads. If a video already has classes in the database, the CSV data won't override it.

## Video Filtering

The gallery provides two independent filtering options:

### Tag-Based Filtering
Filter videos based on their assigned tags:
- **All Videos**: Shows all videos (default)
- **All Good**: Shows only videos tagged as ✅ all-good
- **All Wrong**: Shows only videos tagged as ❌ all-wrong
- **Warning**: Shows only videos tagged as ⚠️ warning

### Semantic Class Filtering
Filter videos based on semantic classes they contain:
- **All Classes**: Shows all videos (default)
- **Multiple Classes**: Select multiple classes to show videos containing ANY of the selected classes
- **No Class**: Shows videos with no detected classes (empty or missing `classes` field) - only appears if there are videos without classes

The class filter dropdown is **dynamically populated** based on the unique classes found in `videos.csv`. Only classes that appear at least once in the video dataset will be shown as filter options.

Click the "Filter by class" dropdown to open a checkbox menu where you can:
- Select multiple classes simultaneously
- Combine "No Class" with specific classes (if videos without classes exist)
- Click "All Classes" to reset the filter

**Note**: The available filter options automatically update based on the content of `videos.csv`.

```json
{
  "video_name.mp4": {
    "tags": ["all-good"],
    "classes": ["person", "car", "bicycle"],
    "timestamp": "2025-12-11T12:00:00.000Z",
    "lastModified": "2025-12-11T12:30:00.000Z"
  },
  "video_with_no_detections.mp4": {
    "tags": ["warning"],
    "classes": [],
    "timestamp": "2025-12-11T12:00:00.000Z",
    "lastModified": "2025-12-11T12:30:00.000Z"
  }
}
```

Both filters can be used simultaneously - videos must match both the tag filter AND at least one of the selected class filters to be displayed.

## Troubleshooting

### Server won't start
- **Port already in use**: Try a different port with `--port 8080`
- **Permission denied**: Don't use ports below 1024 without sudo

### Database not saving
- **Check file permissions**: Make sure the script can write to `database.json`
- **Server not running**: Start the server with `python server.py`
- **Check console**: Open browser DevTools to see error messages

### Videos not loading
- **Check video path**: Make sure videos are in the `videos/` subdirectory
- **Update videos.csv**: Ensure all video filenames are listed in `videos.csv`

### Can't access from another machine
- **Use `--host 0.0.0.0`**: This binds to all network interfaces
- **Check firewall**: Make sure the port is not blocked
- **Use correct IP**: Access using your machine's IP address, not localhost

## Development

### Testing the API

You can test the API endpoints using curl or any HTTP client:

```bash
# Get database
curl http://localhost:8000/api/database

# Update database
curl -X POST http://localhost:8000/api/database \
  -H "Content-Type: application/json" \
  -d @database.json
```

### Modifying the Server

The server code is in `server.py`. Key components:

- `VideoGalleryHandler`: Custom HTTP request handler
- `do_GET()`: Handles GET requests (static files + API)
- `do_POST()`: Handles POST requests (API only)
- `load_database()`: Loads database from JSON file
- `save_database()`: Saves database to JSON file

## Tips

1. **Keep the server running**: The database only persists when the server is running
2. **Backup your database**: Periodically copy `database.json` to a safe location
3. **Use version control**: Add `database.json` to git to track changes over time
4. **Monitor the console**: Server logs all requests and errors

## License

This server is part of the stereo4d-code project.
