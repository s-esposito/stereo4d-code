#!/usr/bin/env python3
"""
Simple HTTP Server for Video Gallery with Database Support

This server provides:
1. Static file serving for the video gallery website
2. RESTful API endpoints for reading and writing the database.json file
3. CORS support for local development
4. Automatic database backups every 5 minutes

Usage:
    python server.py [--port PORT] [--host HOST] [--backup-interval MINUTES]

Arguments:
    --port PORT              Port to run the server on (default: 8000)
    --host HOST              Host to bind to (default: localhost)
    --backup-interval MIN    Backup interval in minutes (default: 5, 0 to disable)

Examples:
    python server.py
    python server.py --port 8080
    python server.py --host 0.0.0.0 --port 8080  # To make it visible to other machines on your local network
    python server.py --backup-interval 10
    
ngrok Example:
# Start server (Terminal 1)
python server.py --host localhost --port 8080

# Start ngrok with password (Terminal 2)
ngrok http 8080 --basic-auth "gallery:secure2025"

API Endpoints:
    GET  /api/database          - Get the entire database
    POST /api/database          - Update the entire database
    GET  /api/database/<video>  - Get data for a specific video
    POST /api/database/<video>  - Update data for a specific video
"""

import json
import os
import argparse
import shutil
import threading
import time
import hashlib
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote

# Database file path
DATABASE_FILE = 'database.json'
BACKUP_DIR = 'database_backups'
LAST_BACKUP_HASH_FILE = os.path.join(BACKUP_DIR, '.last_backup_hash')

class VideoGalleryHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler with database API support"""
    
    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)
    
    def do_OPTIONS(self):
        """Handle preflight CORS requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def send_cors_headers(self):
        """Add CORS headers to response"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path.startswith('/api/database'):
            self.handle_database_get()
        else:
            # Serve static files
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path.startswith('/api/database'):
            self.handle_database_post()
        else:
            self.send_error(404, "Not Found")
    
    def handle_database_get(self):
        """Handle GET requests to /api/database"""
        try:
            # Load database
            database = self.load_database()
            
            # Check if requesting specific video
            if self.path != '/api/database':
                # Extract video name from path
                video_name = unquote(self.path.split('/api/database/')[1])
                if video_name in database:
                    data = {video_name: database[video_name]}
                else:
                    data = {}
            else:
                data = database
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2).encode())
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")
    
    def handle_database_post(self):
        """Handle POST requests to /api/database"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            new_data = json.loads(post_data.decode())
            
            # Load existing database
            database = self.load_database()
            
            # Check if updating specific video
            if self.path != '/api/database':
                # Extract video name from path
                video_name = unquote(self.path.split('/api/database/')[1])
                database[video_name] = new_data
            else:
                # Update entire database
                database = new_data
            
            # Save database
            self.save_database(database)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode())
            
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")
    
    def load_database(self):
        """Load database from file"""
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def save_database(self, database):
        """Save database to file"""
        with open(DATABASE_FILE, 'w') as f:
            json.dump(database, f, indent=2)
    
    def end_headers(self):
        """Override to add CORS headers to all responses"""
        self.send_cors_headers()
        super().end_headers()


def create_backup():
    """Create a timestamped backup of the database only if it has changed since last backup"""
    if not os.path.exists(DATABASE_FILE):
        return
    
    # Create backup directory if it doesn't exist
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"Created backup directory: {BACKUP_DIR}")
    
    # Calculate hash of current database
    current_hash = calculate_file_hash(DATABASE_FILE)
    
    # Check if database has changed since last backup
    if os.path.exists(LAST_BACKUP_HASH_FILE):
        try:
            with open(LAST_BACKUP_HASH_FILE, 'r') as f:
                last_hash = f.read().strip()
            
            if current_hash == last_hash:
                print("⊘ No changes detected, skipping backup")
                return
        except Exception as e:
            print(f"⚠ Error reading last backup hash: {e}")
    
    # Generate timestamp for backup filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"database_backup_{timestamp}.json"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    try:
        # Copy database to backup
        shutil.copy2(DATABASE_FILE, backup_path)
        print(f"✓ Backup created: {backup_filename}")
        
        # Save current hash
        with open(LAST_BACKUP_HASH_FILE, 'w') as f:
            f.write(current_hash)
        
        # Clean old backups (keep last 50)
        cleanup_old_backups(50)
        
    except Exception as e:
        print(f"✗ Error creating backup: {e}")


def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return ""


def cleanup_old_backups(keep_count=50):
    """Remove old backups, keeping only the most recent ones"""
    if not os.path.exists(BACKUP_DIR):
        return
    
    # Get all backup files
    backups = [f for f in os.listdir(BACKUP_DIR) if f.startswith('database_backup_') and f.endswith('.json')]
    backups.sort(reverse=True)  # Sort by name (timestamp) in descending order
    
    # Remove old backups beyond keep_count
    for old_backup in backups[keep_count:]:
        try:
            os.remove(os.path.join(BACKUP_DIR, old_backup))
            print(f"  Cleaned old backup: {old_backup}")
        except Exception as e:
            print(f"  Error removing old backup {old_backup}: {e}")


def backup_loop(interval_minutes):
    """Run backup in a loop at specified interval"""
    if interval_minutes <= 0:
        return
    
    interval_seconds = interval_minutes * 60
    print(f"Backup thread started (interval: {interval_minutes} minutes)")
    
    while True:
        time.sleep(interval_seconds)
        create_backup()


def run_server(port=8000, host='localhost', backup_interval=5):
    """
    Start the HTTP server with automatic backups
    
    Args:
        port (int): Port number to listen on
        host (str): Host address to bind to
        backup_interval (int): Backup interval in minutes (0 to disable)
    """
    server_address = (host, port)
    httpd = HTTPServer(server_address, VideoGalleryHandler)
    
    # Start backup thread if interval > 0
    backup_thread = None
    if backup_interval > 0:
        # Create initial backup
        create_backup()
        
        # Start backup thread
        backup_thread = threading.Thread(target=backup_loop, args=(backup_interval,), daemon=True)
        backup_thread.start()
    
    backup_status = f"Enabled (every {backup_interval} minutes)" if backup_interval > 0 else "Disabled"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Video Gallery Server Started                    ║
╚══════════════════════════════════════════════════════════════╝

Server running at: http://{host}:{port}
Database file: {DATABASE_FILE}
Backup status: {backup_status}
Backup directory: {BACKUP_DIR}

API Endpoints:
  • GET  http://{host}:{port}/api/database
  • POST http://{host}:{port}/api/database
  • GET  http://{host}:{port}/api/database/<video_name>
  • POST http://{host}:{port}/api/database/<video_name>

Press Ctrl+C to stop the server
""")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()
        print("Server stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simple HTTP server for video gallery with database support and automatic backups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py
  python server.py --port 8080
  python server.py --host 0.0.0.0 --port 8080
  python server.py --backup-interval 10
  python server.py --backup-interval 0  # Disable backups
        """
    )
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on (default: 8000)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host to bind to (default: localhost)')
    parser.add_argument('--backup-interval', type=int, default=5,
                        help='Backup interval in minutes (default: 5, 0 to disable)')
    
    args = parser.parse_args()
    
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    run_server(port=args.port, host=args.host, backup_interval=args.backup_interval)
