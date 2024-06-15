"""
Executes the story inference in parallel to speed up the sequentilality computation
"""
import multiprocessing as mp
import os
from story_processor_main import StoryProcessor


def worker(arg):
    obj, m = arg
    return obj.infer(m)

if __name__ == '__main__':
    base_file_directory = "/Users/pranoysarath/PycharmProjects/pythonProject1/data/"
    type = "retold"
    index = [5, 7]
    files = [os.path.join(base_file_directory, f"{type}_files", f"{type}_part_{i}.csv") for i in range(index[0], index[1] + 1)]
    processors = [StoryProcessor(files[i]) for i in range(index[1] - index[0] + 1)]
    pool = mp.Pool(5)
    list_of_results = pool.map(worker, ((processors[i], None) for i in range(len(files))))
    pool.close()
    pool.join()
