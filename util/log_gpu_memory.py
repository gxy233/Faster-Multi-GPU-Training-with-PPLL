import torch
import time


def log_gpu_memory(log_file, interval=10):
    with open(log_file, 'w') as f:
            f.write(f"------------------ start ---------------\n")
    while True:
        num_gpus = torch.cuda.device_count()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        with open(log_file, 'a') as f:
            f.write(f"Timestamp: {timestamp}\n")
            for i in range(num_gpus):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                
                max_memory = torch.cuda.get_device_properties(i).total_memory
                
                # Calculate percentage for memory usage
                alloc_percent = (memory_allocated / max_memory) * 100
                reserved_percent = (memory_reserved / max_memory) * 100
                
                f.write(f"GPU {i}:\n")
                f.write(f"  Memory Allocated: {memory_allocated / (1024 ** 2):.2f} MB ({alloc_percent:.2f}%)\n")
                # f.write(f"  Memory Reserved: {memory_reserved / (1024 ** 2):.2f} MB ({reserved_percent:.2f}%)\n")
            f.write("\n")
        
        time.sleep(interval)