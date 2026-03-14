# Photo Location Tagger

A web app that scans a folder of travel photos, uses the Gemini Vision API to identify locations, and annotates each image with a stylish location name overlay.

## Features

- Automatically identifies locations in photos using Gemini Vision AI
- Annotates photos with elegant Great Vibes cursive font
- Smart text contrast — white text on dark backgrounds, dark text on light
- Web UI with live processing progress
- Manual tagging for photos the AI couldn't identify
- Supports JPEG and HEIC formats
- Preserves original EXIF metadata

## Prerequisites

- Python 3.9+
- A [Gemini API key](https://aistudio.google.com/apikey) (free tier available)

## Installation

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd photo-location-tagger
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   ```

## Usage

### Web UI (recommended)

```bash
python app.py
```

Opens automatically at **http://localhost:5000**. Enter your photos folder path, click Start, and follow the on-screen steps.

### CLI

```bash
python main.py
```

## Supported Formats

- JPEG (`.jpg`, `.jpeg`)
- HEIC (`.heic`) — automatically converted to JPEG in output

## Output

All output is saved to `<your-folder>/enriched/`:

| File | Description |
|------|-------------|
| Annotated images | Original photos with location name overlaid |
| `photo_data.json` | Full analysis results per photo |
| `photo_data.csv` | Flat CSV for spreadsheet analysis |
| `unidentified.txt` | Photos where location could not be determined |
| `progress.json` | Session progress (allows resuming interrupted runs) |

## Cost

Uses `gemini-2.5-flash` — approximately **$0.0001 per photo**.
Processing 250 photos costs roughly **$0.025** (under 3 cents).
