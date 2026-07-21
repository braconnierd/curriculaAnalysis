# curriculaAnalysis
Analytical tool developed for topic based curricular analysis to enable systematic examination of curricular content and implementation of data-driven decisions regarding curricula. It uses a sentence embedding model to identify whether topics are covered within a curriculum. doi: (tba) ####-###-#####

# Capabilities of the Repository
- Process large collections of PDF documents and create corresponding txt files
- Create topic lists from given PDFs
- Compare topic lists and course content

# Requirements
- Python 3.11+
- Ollama
- Additional python libraries as specified within the requirements file
This script was originally designed for MacOS, and there has been limited testing on Windows.

Install Ollama at https://ollama.com/download. The default models used within the project are qwen3:4b-instruct-2507-q4_k_M and qwen2.5vl:3b.

# Project Workflow
1. Load topic lists and/or generate your topic lists
      - Use topic_list_generation.py to generate your own topic lists. Feel free to incorporate your own weighting structure within the topic lists by including a percentage next to the item
         - ex: Sandcasting (20%)
      - We have also included the txt files that we included within our research in the repository. 
2. Convert course content into txt files
      - Use course_content_handling.py to convert your course materials into text files. 
      - This script only converts PDFs into text files - please convert any non-PDF files into PDFs before using this script.
      - It is recommended that you structure your course material like so:
          ```text
          - SchoolName/
            ├── CourseName1/
            │   ├── course_material_1.pdf
            │   └── course_material_2.pdf
            ├── CourseName1/
            │   ├── course_material_1.pdf
            │   └── course_material_2.pdf
3. Organize your folders to fit with the analysis
4. Complete the analysis
      - Use full_lta.py to compare between the course content and topic lists.
   
# Script-Specific Notes
Generally, all of the scripts have their parameters at the top of the script, after the imports.

## topic_list_generation.py
