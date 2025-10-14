from qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
    
from multiprocessing import Process, Queue
def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):
    def worker(q, prediction, reference):
        result = qwen_math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()
    
    # Add timeout handling
    p.join(timeout=timeout_seconds)  # Wait for process to complete, wait at most timeout_seconds seconds
    
    # If process is still running, terminate it and return False
    if p.is_alive():
        p.terminate()
        p.join()  # Ensure process is completely cleaned up
        return False
        
    # If process completed normally, get results
    try:
        return q.get_nowait()
    except:
        return False   

import re 
def preprocess_box_response_for_qwen_prompt(sequence, answer):
    model_output= re.sub(r'^.*?<\|im_start\|>assistant', '<|im_start|>assistant', sequence, flags=re.DOTALL,count = 1)
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extract_answer = qwen_extract_answer(model_output, data_name="math") #TODO: check the data_name, hard code here for now
    
    if qwen_math_equal_subprocess(prediction=extract_answer, reference=answer):
        return 1.0
    else:
        return 0.0
