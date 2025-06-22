import psutil
import os
import time

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # MB
    print(f"当前内存占用: {mem:.2f} MB")



_start_time = time.time()
def print_runtime():
    runtime = time.time() - _start_time
    print(f"当前程序运行时间: {runtime:.2f} 秒")