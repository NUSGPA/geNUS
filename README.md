# geNUS

geNUS is an academic planner and visualization tool designed for National University of Singapore (NUS) students. It facilitates the tracking of academic progression, evaluating S/U (Satisfactory/Unsatisfactory) grading impacts, and visualizing module data through generated reports.

## Key Features

1. **Dashboard Manager**: Tracks course modules, visualizes grade distributions using interactive charts, and monitors cumulative GPA metrics.
2. **S/U Scenario Planner**: Provides a calculator to model the impact of applying S/U grading options. Users can toggle S/U status on specific modules to observe the resulting changes to their cumulative GPA.
3. **geNUS Wrapped**: Generates an "Academic Archetype" report and aggregates academic statistics (e.g., grade streaks, A-grade rates, and top-performing semesters) into a formatted, 480px-wide exportable image.

---

## Technical Architecture & Implementation

geNUS is built utilizing the **Streamlit** framework in Python, customized with injected CSS and client-side JavaScript to extend the framework's default UI capabilities.

### Frontend & UI/UX
- **CSS Customization**: Incorporates custom CSS implementations to override standard Streamlit layouts, applying specific color palettes, responsive grid structures, and custom typography (Google's "Outfit" font).
- **Client-Side Export**: The report card generation bypasses server-side image rendering. It injects an HTML block that loads the `html2canvas.js` library, enabling the user's browser to render the DOM elements onto an HTML5 Canvas and trigger a direct `image/png` download locally.

### Language Model & RAG Implementation 
The Academic Archetype feature utilizes a **Retrieval-Augmented Generation (RAG)** pipeline backed by the **Gemini 2.5 Flash** large language model (LLM).

#### Pipeline Workflow (`wrapped_engine.py`):
1. **Prefix Extraction**: The engine processes the user's input transcript to extract the 2-to-4 letter alphabetic prefix from each module code (e.g., `CS1010S` yields `CS`). It calculates the frequency of each prefix to establish the user's academic focus.
2. **Knowledge Retrieval (`filtered_raw.csv`)**: A pre-processed static database containing NUS module codes and their titles. The application queries this database against the user's extracted prefixes.
3. **Context Assembly**: The pipeline constructs a context block mapping prefixes to their descriptive titles. For example: *Prefix CS (taken 3 times): Programming Methodology, Data Structures and Algorithms, Software Engineering*.
4. **LLM Prompting**: The retrieved context and frequency data are passed to the Gemini model with a specific system prompt. The model analyzes the weighted module data to generate a short archetype title and a descriptive summary of the student's academic focus.
5. **Fallback Mechanisms**: If the Gemini API request fails due to rate limits (HTTP 429) or quota restrictions, the process falls back to sending only the raw prefix frequencies without the full title context, ensuring the feature remains functional under API constraints.

---

## Setup & Installation

To run geNUS locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/NUSGPA/geNUS.git
   cd geNUS
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure API Keys:
   Create a folder named `.streamlit` in the root directory and add a `secrets.toml` file with your Gemini API key:
   ```toml
   GEMINI_API_KEY = "your-google-gemini-api-key"
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```
