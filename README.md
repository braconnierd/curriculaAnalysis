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

Install [Ollama](https://ollama.com/download) at https://ollama.com/download. The default models used within the project are [qwen3:4b-instruct-2507-q4_k_M](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M) and [qwen2.5vl:3b](https://ollama.com/library/qwen2.5vl:3b).

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
            SchoolName/
            ├── CourseName1/
            │   ├── course_material_1.pdf
            │   └── course_material_2.pdf
            ├── CourseName2/
            │   ├── course_material_1.pdf
            │   └── course_material_2.pdf
          ```
3. Organize your folders to fit with the analysis
4. Complete the analysis
      - Use full_lta.py to compare between the course content and topic lists.
   
# Script-Specific Notes
Generally, all of the scripts have their parameters at the top of the script, after the imports.

## topic_list_generation.py
- Within the paper the [qwen3:4b-instruct-2507-q4_k_M model](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M) was used for generating lists and consolidating lists, but this can be changed in the parameters section.

## course_content_handling.py
- Note that this script turns *exclusively* PDFs into txt files. It will skip over any files that are not PDFs.
- Within the paper the qwen2.5vl:3b model was used to do turn the handwriting into text - this can be changed in the parameters section.
- It is recommended that you structure your course material like so:
  ```text
      InputFolder/
      ├── CourseName1/
      │   ├── course_material_1.pdf
      │   └── course_material_2.pdf
      ├── CourseName2/
      │   ├── course_material_1.pdf
      │   └── course_material_2.pdf
  ```
- The output folder will be made to mirror the input folder, so given the previous example, the output will look like this:
  ```text
      OutputFolder/
      ├── CourseName1/
      │   ├── course_material_1.txt
      │   └── course_material_2.txt
      ├── CourseName2/
      │   ├── course_material_1.txt
      │   └── course_material_2.txt
  ```
## full_lta.py
### Parameters
- Within the paper Google's [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma) model was used for generating the sentence embeddings. This can be changed in the parameters section.
- The cosine similarity thresholds are also set in the parameters section, but they can be tuned to make the system stricter and/or more lenient.
- Batch size and docs_per_chunk are the batching parameters used to ensure that the system doesn't run out of memory when it is run locally. When being run on a M4 Pro MacBook Pro with 24GB RAM, a batch size of 16 and 5 documents per chunk were used. 
- The input_folder should be the path to your course content as txt files. It should be structured like so:
  ```text
      Input_Folder/
      ├── CourseName1/
      │   ├── course_material_1.txt
      │   └── course_material_2.txt
      ├── CourseName2/
      │   ├── course_material_1.txt
      │   └── course_material_2.txt
  ```
- The topics_folder should be the path to the folder of topic lists. It should be structured like so:
    ```text
      Topics_Folder/
      ├── Building_Block_1/
      │   ├── topic_list_1.txt
      │   └── topic_list_2.txt
      ├── Building_Block_2/
      │   ├── topic_list_1.txt
      │   └── topic_list_2.txt
    ```
- The evidence_folder should be the path to the folder where you want to store all the evidence/documentation. After running the script, it will result in an output corresponding to the structure of the folder of topic lists. Based on the sample topic list folder structure, this would be the structure of the evidence folder.
    ```text
      Evidence_Folder/
      ├── Building_Block_1/
      │   ├── topic_list_1_Evidence.txt
      │   └── topic_list_2_Evidence.txt
      ├── Building_Block_2/
      │   ├── topic_list_1_Evidence.txt
      │   └── topic_list_2_Evidence.txt
      └── SchoolName.json
   ```
   - Each evidence file includes the final score for the corresponding topic list, as well as the hits found in for each sub-topic, and the closest miss for sub-topics that are completely missed.
   - The json file includes the percentage of coverage for each topic list. It is structured like so:
     ```text
     "school_name": "Program/School Name",
     "results": {
        "Building Block 1": {
           "Topic List 1": xx.xx,
           "Topic List 2": xx.xx,
         },
         "Building Block 2": {
           "Topic List 1": xx.xx,
           "Topic List 2": xx.xx,
         },
     ```

### Matching Schema
The cosine similarity of sentence embeddings of the topic lists and course content are used throughout this process to determine whether sub-topics are addressed within the course content. There are 3 kinds of comparisons that are made. 
