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
3. Complete the analysis
      - Use full_lta.py to compare between the course content and topic lists.
   
# Script-Specific Notes
Generally, all of the scripts have their parameters at the top of the script, after the imports.

## topic_list_generation.py
- Within the paper the [qwen3:4b-instruct-2507-q4_k_M model](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M) was used for generating lists and consolidating lists, but this can be changed in the parameters section.

## course_content_handling.py
- Note that this script *only* turns PDFs into txt files. It will skip over any files that are not PDFs.
- Within the paper the qwen2.5vl:3b model was used to turn the handwriting into text - this can be changed in the parameters section.
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
- The cosine similarity thresholds (`atom_threshold`, `query_threshold` and `context_threshold`) are also set in the parameters section, but they can be tuned to make the system stricter and/or more lenient. More details about these thresholds are included in the [Matching Schema](https://github.com/braconnierd/curriculaAnalysis/tree/main#matching-schema) section. 
- `batch_size` and `docs_per_chunk` are the batching parameters used to ensure that the system doesn't run out of memory when it is run locally. When being run on a M4 Pro MacBook Pro with 24GB RAM, a batch size of 16 and 5 documents per chunk were used.
    - `batch_size` refers to the number of strings that are sent to the embedding model at once. 
    - `docs_per_chunk` refers to the number of documents within the course content that are compared to the topic lists before the cache is emptied.  
- The `input_folder` should be the path to your course content as txt files. It should be structured like so:
  ```text
      Input_Folder/
      ├── CourseName1/
      │   ├── course_material_1.txt
      │   └── course_material_2.txt
      ├── CourseName2/
      │   ├── course_material_1.txt
      │   └── course_material_2.txt
  ```
- The `topics_folder` should be the path to the folder of topic lists. It should be structured like so:
    ```text
      Topics_Folder/
      ├── Building_Block_1/
      │   ├── topic_list_1.txt
      │   └── topic_list_2.txt
      ├── Building_Block_2/
      │   ├── topic_list_1.txt
      │   └── topic_list_2.txt
    ```
- The `evidence_folder` should be the path to the folder where you want to store all the evidence/documentation. After running the script, the folder will be populated corresponding to the structure of the folder of topic lists. Based on the sample topic list folder structure, this would be the structure of the evidence folder.
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
The cosine similarity of sentence embeddings of the topic lists and course content are used throughout this process to determine whether sub-topics are addressed within the course content. 

When the topic lists are parsed, they are parsed to find an atomic query and a full context query. As an example, within a list about metals, *yield strength* would be an atom, while the full context query would be *Content about yield strength as it relates to Mechanical Behavior of Metals*.

There are two kinds of matches included within the schema: lexical and semantic matching. **Semantic matching** means that the cosine similarity between the atom and query exceed the threshold - in other words, based on similarity, the topic has been determined to be mentioned within the text. These thresholds are controlled by `atom_threshold`, which sets the needed cosine similarity to the atom, and `query_threshold`, which sets the needed cosine similarity to the full context query. **Lexical matching** means that the atom and/or keywords from the phrase have been mentioned exactly within the text. This includes common variations on the word (i.e. plurals and verb conjugations count for the word as well). In order to ensure that the atom is appearing in the appropriate context, it still compares the full query to the sentence embeddings, and for it to be considered covered within the content, it must exhibit a cosine similarity above `context_threshold`. Note that some atomic topics are excluded from semantic matching. If atoms are identified as acronyms or names, they are only considered covered if they are mentioned exactly within the text (i.e. only lexical matches are considered). Lastly, the efficacy of the thresholds can be evaluated by looking through the evidence files. They will show what passed the threshold and if nothing passed the threshold, what was the closest to passing, as well as its score. This allows you to evaluate whether the thresholds need to be adjusted. 
