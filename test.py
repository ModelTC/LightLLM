import torch
import multiprocessing
from multiprocessing.reduction import ForkingPickler


def worker(serialized):
    torch.cuda.set_device(0)
    tensor = ForkingPickler.loads(serialized)
    print("In worker process:", tensor, type(tensor))
    # import time
    # time.sleep(100)


def worker1(serialized):
    torch.cuda.set_device(1)
    tensor = ForkingPickler.loads(serialized)
    print("In worker process:", tensor, type(tensor))
    # import time
    # time.sleep(100)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    torch.cuda.set_device(0)
    # Create a tensor on the CUDA device
    a = torch.zeros((100,), device="cuda").cuda()

    same = ForkingPickler.dumps(a)
    serialized = same.tobytes()

    # Create a new process
    process = multiprocessing.Process(target=worker, args=(serialized,))

    # Start the process
    process.start()

    process1 = multiprocessing.Process(target=worker1, args=(serialized,))

    # Start the process
    process1.start()

    # Wait for the process to finish
    process.join()
    process1.join()
    print(a)
    import time

    time.sleep(10)

    print("Main process finished.")
