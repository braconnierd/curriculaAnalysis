# ------------------------------------------------------------------------------
# Import and Library Management
# ------------------------------------------------------------------------------
from ollama_utils import ensure_ollama_running, stop_ollama
import ollama
import pdfplumber 
from sys import exit, stdout
from time import sleep
import time
from itertools import cycle
from threading import Thread
from cursor import hide, show
from tkinter import filedialog
from pathlib import Path
from datetime import datetime

# -------------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------------

def preprocess_pdf(pdf_path):
    """
    Extracts and cleans text from a PDF using pdfplumber.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # extract_text() may return None
    cleaned_text = " ".join(text.split())  # remove extra whitespace
    return cleaned_text

class LoadingAnimation:
    def __init__(self, message):
        self.message = message
        self.done = False
        self.elapsed = 0

    def _animate(self):
        for s in cycle(["|", "/", "-", "\\"]):
            if self.done:
                print(f"\r{self.message}...DONE | {self.elapsed:.1f} sec")
                break

            print(f"\r{self.message}...{s}", end="", flush=True)
            sleep(0.1)
            self.elapsed += 0.1

    def start(self):
        self.thread = Thread(target=self._animate)
        self.thread.start()

    def stop(self):
        self.done = True
        self.thread.join()


# -------------------------------------------------------------------------------
# Ollama Functions
# -------------------------------------------------------------------------------

def ask_ollama(model,prompt):
    try:
        result = ollama.generate(model=model, 
                                prompt=prompt)
        return result['response']
    except Exception:
        pass    

def train_ollama(base_model,new_model_name,reset_model=False):
    """ Send all the PDFs of the reference materials to an ollama model to query the documents.
    """
    print('Opening folder to select reference materials...')

    filePaths = filedialog.askopenfilenames(title='Select PDF(s)...', filetypes= [('PDF','.pdf')])
    
    if filePaths == '':
        user_input = input('No reference material(s) selected. Do you want to continue? (Y/n)\n>>> \x1b[33m')
        if user_input == "Y":
            stdout.write('\x1b[0m')
            pass
        elif user_input == '/exit' or user_input == 'n':
                stdout.write('\x1b[0m')
                print('\x1b[95mAskPDFsLLM has been closed.\x1b[0m')
                hide()
                stop_ollama()
                exit()

    if reset_model and new_model_name in ollama.list():
        ollama.delete(new_model_name)

    if new_model_name not in ollama.list():
        ollama.copy(base_model,new_model_name)    

    
    #Collect all the texts and put them into a table to send to the Ollama model
    if filePaths != '': 
        reference_txts = []
        pdf_titles = []
        pdf_names = []
        ref_table_headers = ['Document Title\tDocument Text\n']
        total = len(filePaths)

        for i, file in enumerate(filePaths, start=1):

            pdf_name = file.split("/")[-1].replace(".pdf", "")

            pdf_titles.append(pdf_name)
            pdf_names.append(f'"{pdf_name}"')

            loader = LoadingAnimation(
                f"Collecting text from document {i} of {total}"
            )

            loader.start()

            text = preprocess_pdf(file)

            loader.stop()

            reference_txts.append(
                f'"{pdf_name}"\t{text}\n'
            )

        ref_table = ref_table_headers + reference_txts

    #System message for the new Ollama model
    input_info = f'You only know the information from the following document table. If the answer is not contained within the provided information, just say: "There is no information regarding your request". You also ignore any citations within the document text.\n\n{''.join(ref_table)})'

    ollama.create(model=new_model_name,
                        from_=new_model_name,
                        system=input_info,
                        parameters={'mirostat':2,
                                    'mirostat_eta':1,
                                    'mirostat_tau':0,
                                    'num_ctx':4096,
                                    'temperature':0,
                                    'repeat_penalty':1.5,
                                    'top_k':10,
                                    'top_p':0.5,
                                    'min_p':0.05})    
 
    return new_model_name, pdf_names

# -------------------------------------------------------------------------------
# File Ingestion Ollama Functions
# -------------------------------------------------------------------------------

def make_topic_list(topic,iterations,new_model_name,pdf_names, output_folder, genericmodel="qwen3:4b-instruct-2507-q4_k_M"):
    """ Send prompt n times, save all of the responses in the output folder as a session, then send the list to the LLM to delete duplicates. Return the topic list. """

    output_folder.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    
    
    # Ollama Session Log
    file_path = output_folder / f"Ollama Session - {timestamp}.txt"
    
    with open(file_path, "w", encoding="utf-8") as m:
        timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        m.write(f"Ollama Session - {timestamp}\n")
        m.write("==========================================================\n\n")
        m.write("Model Settings:\n")
    
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write("Reference Documents: \n")
        for title in pdf_names:
            f.write(f"- {title}\n")
        f.write("==========================================================\n\n")
        f.write("Session Log:\n")
    
    prompt = (
        f"Please give me a list of required topics in an college course on {topic}. Be as precise, specific and granular as possible — include subtopics for each topic. I should be able to take this list and make a course syllabus. List everything in order of importance."
    )

    print("Starting to make the topic list")
    
    masterlist = []
    totalstart = time.time()

    for i in range(iterations): 
        iterstart = time.time()
        summary = ask_ollama(
            new_model_name,
            f"{prompt}\n\nYou may only use information from the following documents: f{', '.join(pdf_names)}. Provide your response in under 500 words. In your response, label each line with the document title(s) of which the information came from. Follow your response with a list of these referenced documents. Labels and reference lists are not included in word count.\n\nBe as direct as possible."
        )
        masterlist.append(summary)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"Iteration {(i+1)} \n")
            f.write(f"{summary} \n\n")
        iterend = time.time()
        print(f"Completed iteration number {i}. Check the output file to see the output. Time taken: {iterend - iterstart:.2f} seconds")


    # Join all summaries into one string
    combined_list = "\n\n".join(masterlist)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"Combined List\n")
        f.write(f"{combined_list} \n\n")
    print(f"All the iterations have been completed, now sending to Ollama to consolidate the list.")

    # Send to Ollama to consolidate
    topic_list = ask_ollama(
        genericmodel,
        f"{prompt}\n\nConsolidate the following list of relevant topics and subtopics to delete duplicates and group similar ideas together, making one masterlist. Ensure that every subtopic and topic originally included in the list is represented. The topics and subtopics should be listed in order of importance. Keep the formatting the same as the individual lists (Do not make a syllabus). *Only output the list*, do not give any conversational response. \n\n Here is the combined list:\n\n{combined_list}"
    )
    list_path = output_folder / f"{topic}.txt"
    with open(list_path, 'w', encoding='utf-8') as f:
        f.write(f"{topic_list}")
    print(topic_list)
    print("The topic list has been generated.")
    totalend = time.time()
    print(f"time taken: {totalend - totalstart:.2f} seconds")
    return topic_list


# -------------------------------------------------------------------------------
# Main Loop
# -------------------------------------------------------------------------------

ensure_ollama_running()

output_folder = Path("JMSE Paper/JMSE data/Ollama Sessions/") ## Set the output destination for the session log - This stores the information from each individual iteration)
new_model_name, pdf_names = train_ollama(base_model='qwen3:4b-instruct-2507-q4_k_M', new_model_name='PDF_LLM', reset_model=True)

while True:
    topic = None
    show()
    user_input = input('Please type your topic below: (type \033[31m"/exit"\033[0m to close)\n>>> \x1b[33m')
    stdout.write('\x1b[0m')

    # Handle exit command
    if user_input.strip().lower() == '/exit':
        print("Exiting Ollama session.")
        stop_ollama()
        break  # Exit the loop completely

    if user_input.strip() != '':
        topic = user_input.strip()
        make_topic_list(topic, 3, new_model_name, pdf_names, output_folder)
    else:
        print("Please enter a valid topic or type /exit to quit.\n")
