const nr_videos_to_load = 100;

// API base URL - change this if running server on different host/port
const API_BASE = '/api/database';

// Database object to store video tags
let database = {};

// Server connection status
let serverConnected = false;

// Show server warning banner
function showServerWarning() {
    const warningBanner = document.getElementById('server-warning');
    if (warningBanner) {
        warningBanner.classList.remove('hidden');
    }
}

// Hide server warning banner
function hideServerWarning() {
    const warningBanner = document.getElementById('server-warning');
    if (warningBanner) {
        warningBanner.classList.add('hidden');
    }
}

// Load database from server API
async function loadDatabase() {
    try {
        const response = await fetch(API_BASE);
        if (response.ok) {
            database = await response.json();
            console.log('Database loaded from server:', database);
            serverConnected = true;
            hideServerWarning();
        } else {
            console.log('No existing database found on server, starting fresh');
            database = {};
            serverConnected = false;
            showServerWarning();
        }
    } catch (error) {
        console.log('Error loading database from server, trying local file:', error);
        serverConnected = false;
        showServerWarning();
        
        // Fallback to local file if server is not running
        try {
            const response = await fetch('database.json');
            if (response.ok) {
                database = await response.json();
                console.log('Database loaded from local file:', database);
            } else {
                database = {};
            }
        } catch (err) {
            console.log('Error loading local database:', err);
            database = {};
        }
    }
}

// Save database to server API
async function saveDatabase() {
    try {
        const response = await fetch(API_BASE, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(database)
        });
        
        if (response.ok) {
            console.log('Database saved to server:', database);
            serverConnected = true;
            hideServerWarning();
        } else {
            console.error('Failed to save database to server');
            serverConnected = false;
            showServerWarning();
            // Fallback: download as file
            downloadDatabase();
        }
    } catch (error) {
        console.error('Error saving to server, downloading instead:', error);
        serverConnected = false;
        showServerWarning();
        // Fallback: download as file
        downloadDatabase();
    }
}

// Download database as JSON file (fallback method)
function downloadDatabase() {
    const dataStr = JSON.stringify(database, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'database.json';
    link.click();
    URL.revokeObjectURL(url);
    console.log('Database downloaded as file');
}

// Update video tag in database
function updateVideoTag(fileName, tag) {
    if (!database[fileName]) {
        database[fileName] = {
            tags: [],
            timestamp: new Date().toISOString()
        };
    }
    
    // Toggle tag - if already exists, remove it; otherwise add it
    const tagIndex = database[fileName].tags.indexOf(tag);
    if (tagIndex > -1) {
        database[fileName].tags.splice(tagIndex, 1);
    } else {
        database[fileName].tags.push(tag);
    }
    
    database[fileName].lastModified = new Date().toISOString();
    
    console.log(`Updated ${fileName}:`, database[fileName]);
    
    // Auto-save after each update
    saveDatabase();
}

// Check if video has a specific tag
function hasTag(fileName, tag) {
    return database[fileName] && database[fileName].tags.includes(tag);
}

// videos are listed in videos.csv file in the same folder as this HTML file
// read the file and create an array of video file names
let videoFiles = [];
let videoClassesMap = {}; // Map of filename -> classes array
let currentTagFilter = 'all';
let currentClassFilters = ['all']; // Array to support multiple class filters

// Filter videos based on selected tag and classes
function filterVideos(tagFilter = null, classFilters = null) {
    if (tagFilter !== null) currentTagFilter = tagFilter;
    if (classFilters !== null) currentClassFilters = classFilters;
    
    let filteredFiles = videoFiles;
    
    // Apply tag filter
    if (currentTagFilter !== 'all') {
        filteredFiles = filteredFiles.filter(fileName => {
            return hasTag(fileName, currentTagFilter);
        });
    }
    
    // Apply class filters (multi-select)
    if (!currentClassFilters.includes('all') && currentClassFilters.length > 0) {
        filteredFiles = filteredFiles.filter(fileName => {
            const videoData = database[fileName];
            
            // Handle "no-class" filter
            if (currentClassFilters.includes('no-class')) {
                // Video should have no classes or empty classes array
                const hasNoClasses = !videoData || !videoData.classes || videoData.classes.length === 0;
                if (hasNoClasses) return true;
            }
            
            // Handle specific class filters
            if (videoData && videoData.classes && Array.isArray(videoData.classes)) {
                // Check if video has ANY of the selected classes
                return currentClassFilters.some(selectedClass => 
                    selectedClass !== 'no-class' && videoData.classes.includes(selectedClass)
                );
            }
            
            return false;
        });
    }
    
    // Update video count
    updateVideoCount(filteredFiles.length, videoFiles.length);
    
    // Reload videos with filtered list
    loadVideos(filteredFiles);
}

// Update the class filter button text based on selected checkboxes
function updateClassFilterText() {
    const selectedCheckboxes = document.querySelectorAll('#class-filter-dropdown input[type="checkbox"]:checked');
    const selectedValues = Array.from(selectedCheckboxes).map(cb => cb.value);
    const filterText = document.getElementById('class-filter-text');
    
    if (selectedValues.includes('all') || selectedValues.length === 0) {
        filterText.textContent = 'All Classes';
    } else if (selectedValues.length === 1) {
        const checkbox = selectedCheckboxes[0];
        filterText.textContent = checkbox.nextElementSibling.textContent;
    } else {
        filterText.textContent = `${selectedValues.length} classes selected`;
    }
}

// Populate class filter dropdown with unique classes from videos
function populateClassFilter() {
    const classFilterDropdown = document.getElementById('class-filter-dropdown');
    if (!classFilterDropdown) return;
    
    // Collect all unique classes from videoClassesMap
    const uniqueClasses = new Set();
    let hasVideosWithNoClasses = false;
    
    Object.values(videoClassesMap).forEach(classes => {
        if (classes.length === 0) {
            hasVideosWithNoClasses = true;
        } else {
            classes.forEach(cls => uniqueClasses.add(cls));
        }
    });
    
    // Sort classes alphabetically
    const sortedClasses = Array.from(uniqueClasses).sort();
    
    // Build the dropdown HTML
    let html = `
        <label class="checkbox-item">
            <input type="checkbox" value="all" checked>
            <span>All Classes</span>
        </label>`;
    
    // Add "No Class" option only if there are videos with no classes
    if (hasVideosWithNoClasses) {
        html += `
        <label class="checkbox-item">
            <input type="checkbox" value="no-class">
            <span>No Class (Nothing Detected)</span>
        </label>`;
    }
    
    // Add separator if there are any classes to show
    if (sortedClasses.length > 0) {
        html += '<div class="dropdown-separator"></div>';
        
        // Add each unique class
        sortedClasses.forEach(cls => {
            // Capitalize first letter for display
            const displayName = cls.charAt(0).toUpperCase() + cls.slice(1);
            html += `
        <label class="checkbox-item">
            <input type="checkbox" value="${cls}">
            <span>${displayName}</span>
        </label>`;
        });
    }
    
    classFilterDropdown.innerHTML = html;
    
    // Re-attach event listeners to the new checkboxes
    setupClassFilterListeners();
}

// Setup event listeners for class filter checkboxes
function setupClassFilterListeners() {
    const classFilterDropdown = document.getElementById('class-filter-dropdown');
    if (!classFilterDropdown) return;
    
    const checkboxes = classFilterDropdown.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const value = e.target.value;
            
            // Handle "all" checkbox
            if (value === 'all') {
                if (e.target.checked) {
                    // Uncheck all other checkboxes
                    checkboxes.forEach(cb => {
                        if (cb.value !== 'all') cb.checked = false;
                    });
                }
            } else {
                // If any specific class is selected, uncheck "all"
                const allCheckbox = classFilterDropdown.querySelector('input[value="all"]');
                if (allCheckbox) allCheckbox.checked = false;
            }
            
            // Get selected values
            const selectedCheckboxes = classFilterDropdown.querySelectorAll('input[type="checkbox"]:checked');
            const selectedValues = Array.from(selectedCheckboxes).map(cb => cb.value);
            
            // If nothing is selected, select "all" by default
            if (selectedValues.length === 0) {
                const allCheckbox = classFilterDropdown.querySelector('input[value="all"]');
                if (allCheckbox) {
                    allCheckbox.checked = true;
                    selectedValues.push('all');
                }
            }
            
            // Update button text
            updateClassFilterText();
            
            // Apply filter
            filterVideos(null, selectedValues);
        });
    });
}

// Update video count display
function updateVideoCount(filtered, total) {
    const countElement = document.getElementById('video-count');
    if (countElement) {
        if (filtered === total) {
            countElement.textContent = `(${total} videos)`;
        } else {
            countElement.textContent = `(${filtered} of ${total} videos)`;
        }
    }
}

// Initialize: load database first, then load videos
loadDatabase().then(() => {
    fetch('videos.csv')
        .then(response => response.text())
        .then(data => {
            // Parse CSV file
            const lines = data.split('\n').filter(line => line.trim() !== '');
            // Skip header row
            const dataLines = lines.slice(1);
            
            videoFiles = [];
            videoClassesMap = {};
            
            dataLines.forEach(line => {
                // Split by comma, handle potential commas in class lists
                const firstComma = line.indexOf(',');
                if (firstComma === -1) return; // Skip invalid lines
                
                const filename = line.substring(0, firstComma).trim();
                const classesStr = line.substring(firstComma + 1).trim();
                
                if (filename) {
                    videoFiles.push(filename);
                    
                    // Parse classes - they could be comma-separated or empty
                    if (classesStr) {
                        // Split by comma and clean up
                        const classes = classesStr.split(',').map(c => c.trim()).filter(c => c);
                        videoClassesMap[filename] = classes;
                        
                        // Update database with classes if not already present
                        if (!database[filename]) {
                            database[filename] = {
                                tags: [],
                                classes: classes,
                                timestamp: new Date().toISOString(),
                                lastModified: new Date().toISOString()
                            };
                        } else if (!database[filename].classes) {
                            database[filename].classes = classes;
                            database[filename].lastModified = new Date().toISOString();
                        }
                    } else {
                        videoClassesMap[filename] = [];
                        
                        // Ensure database entry has empty classes array
                        if (!database[filename]) {
                            database[filename] = {
                                tags: [],
                                classes: [],
                                timestamp: new Date().toISOString(),
                                lastModified: new Date().toISOString()
                            };
                        } else if (!database[filename].classes) {
                            database[filename].classes = [];
                            database[filename].lastModified = new Date().toISOString();
                        }
                    }
                }
            });
            
            // limit list to first N videos for performance
            videoFiles = videoFiles.slice(0, nr_videos_to_load);
            
            // Populate class filter dropdown with unique classes from loaded videos
            populateClassFilter();
            
            loadVideos(videoFiles);
            updateVideoCount(videoFiles.length, videoFiles.length);
        })
        .catch(error => {
            console.error('Error loading video list:', error);
        });
});

// Setup close button for warning banner and filter dropdown
document.addEventListener('DOMContentLoaded', () => {
    const closeButton = document.getElementById('close-warning');
    if (closeButton) {
        closeButton.addEventListener('click', hideServerWarning);
    }
    
    const filterSelect = document.getElementById('tag-filter');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            filterVideos(e.target.value, null);
        });
    }
    
    // Class filter dropdown functionality
    const classFilterButton = document.getElementById('class-filter-button');
    const classFilterDropdown = document.getElementById('class-filter-dropdown');
    
    if (classFilterButton && classFilterDropdown) {
        // Toggle dropdown on button click
        classFilterButton.addEventListener('click', (e) => {
            e.stopPropagation();
            classFilterDropdown.classList.toggle('hidden');
            classFilterButton.classList.toggle('open');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!classFilterButton.contains(e.target) && !classFilterDropdown.contains(e.target)) {
                classFilterDropdown.classList.add('hidden');
                classFilterButton.classList.remove('open');
            }
        });
        
        // Note: Checkbox event listeners are set up dynamically in setupClassFilterListeners()
        // after the dropdown is populated with actual classes from the CSV file
    }
});

const grid = document.getElementById('video-grid');
const videoFolder = 'videos/'; // The subfolder where your videos are
// const videoFolder = "/home/stefano/Codebase/stereo4d-code/videogallery/videos/"; // The subfolder where your videos

function loadVideos(videoFiles) {
    
    // Clear existing videos
    grid.innerHTML = '';

    // Sort videoFiles alphabetically
    videoFiles.sort();

    videoFiles.forEach(fileName => {

        const videoContainer = document.createElement('div');
        videoContainer.classList.add('video-item');
        
        // Check database for existing tags and apply appropriate border color
        if (database[fileName] && database[fileName].tags) {
            if (database[fileName].tags.includes('all-good')) {
                videoContainer.classList.add('tagged-all-good');
            }
            if (database[fileName].tags.includes('all-wrong')) {
                videoContainer.classList.add('tagged-all-wrong');
            }
            if (database[fileName].tags.includes('warning')) {
                videoContainer.classList.add('tagged-warning');
            }
        }

        const videoElement = document.createElement('video');
        videoElement.controls = true; // Show video controls (play, pause, etc.)
        videoElement.muted = true;    // Mute by default for a smoother grid experience
        videoElement.autoplay = true; // Play automatically (can be resource intensive)
        videoElement.loop = true;     // Loop the video

        const sourceElement = document.createElement('source');
        // Add .mp4 extension since CSV stores filenames without extension
        let videoPath = videoFolder + fileName + '.mp4';
        console.log("Loading video from path:", videoPath);
        sourceElement.src = videoPath;
        sourceElement.type = 'video/mp4'; // Explicitly state the type

        videoElement.appendChild(sourceElement);

        const title = document.createElement('p');
        title.textContent = fileName;
        title.classList.add('video-title');

        // add a button to copy visualization script 
        // e.g. python view_sample.py --view --scene=4uCq66L-tFs --timestamp=100650651
        
        const copyButton = document.createElement('button');
        // remove .mp4 extension from fileName
        const scene_timestamp = fileName.endsWith('.mp4') ? fileName.slice(0, -4) : fileName;
        const [scene, timestamp] = scene_timestamp.split('_');
        copyButton.innerHTML = "📋";
        copyButton.title = "Copy visualization script";
        copyButton.onclick = () => {
            const script = `python view_sample.py --view --scene=${scene} --timestamp=${timestamp}`;
            navigator.clipboard.writeText(script);
            // .then(() => {
            //     alert('Visualization script copied to clipboard!');
            // });
        };
        // add class to button
        copyButton.classList.add('video-button');

        // add an "all good" button
        const allGoodButton = document.createElement('button');
        allGoodButton.innerHTML = "✅";
        allGoodButton.title = "Mark as all good";
        allGoodButton.classList.add('video-button');
        if (hasTag(fileName, 'all-good')) {
            allGoodButton.classList.add('active');
        }
        allGoodButton.onclick = () => {
            updateVideoTag(fileName, 'all-good');
            allGoodButton.classList.toggle('active');
            // Toggle green border on video container
            videoContainer.classList.toggle('tagged-all-good');
            // Refresh filter if needed
            if (currentTagFilter !== 'all' || !currentClassFilters.includes('all')) {
                setTimeout(() => filterVideos(), 100);
            }
        };

        // add an "all wrong" button
        const allWrongButton = document.createElement('button');
        allWrongButton.innerHTML = "❌";
        allWrongButton.title = "Mark as all wrong";
        allWrongButton.classList.add('video-button');
        if (hasTag(fileName, 'all-wrong')) {
            allWrongButton.classList.add('active');
        }
        allWrongButton.onclick = () => {
            updateVideoTag(fileName, 'all-wrong');
            allWrongButton.classList.toggle('active');
            // Toggle red border on video container
            videoContainer.classList.toggle('tagged-all-wrong');
            // Refresh filter if needed
            if (currentTagFilter !== 'all' || !currentClassFilters.includes('all')) {
                setTimeout(() => filterVideos(), 100);
            }
        };

        // add a "warning" button
        const warningButton = document.createElement('button');
        warningButton.innerHTML = "⚠️";
        warningButton.title = "Mark as warning";
        warningButton.classList.add('video-button');
        if (hasTag(fileName, 'warning')) {
            warningButton.classList.add('active');
        }
        warningButton.onclick = () => {
            updateVideoTag(fileName, 'warning');
            warningButton.classList.toggle('active');
            // Toggle yellow border on video container
            videoContainer.classList.toggle('tagged-warning');
            // Refresh filter if needed
            if (currentTagFilter !== 'all' || !currentClassFilters.includes('all')) {
                setTimeout(() => filterVideos(), 100);
            }
        };

        const titleContainer = document.createElement('div');
        titleContainer.classList.add('title-container');
        titleContainer.appendChild(title);
        titleContainer.appendChild(copyButton);
        titleContainer.appendChild(allGoodButton);
        titleContainer.appendChild(allWrongButton);
        titleContainer.appendChild(warningButton);

        videoContainer.appendChild(videoElement);
        videoContainer.appendChild(titleContainer);
        grid.appendChild(videoContainer);

    });
}
