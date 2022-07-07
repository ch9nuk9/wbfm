import concurrent
import threading
import timeit

import tifffile

from wbfm.utils.general.video_and_data_conversion.import_video_as_array import get_single_volume

fname = r'D:\freely_immobilized\ZIM2051_trial_21_HEAD_mcherry_FULL_bigtiff.btf'
import_opt = {'num_slices': 33, 'alpha': 1.0, 'dtype': 'uint16'}
frame_range = list(range(1000, 1010))
print(frame_range)


def sequential_read():
    with tifffile.TiffFile(fname) as video_stream:
        for i in frame_range:
            _ = get_single_volume(video_stream, i, **import_opt)
            # print(f"Got volume {i}")


def simple_parallel():
    def parallel_func(i):
        get_single_volume(fname, i, **import_opt)
        # print(f"Got volume {i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(parallel_func, i) for i in frame_range}
        for future in concurrent.futures.as_completed(futures):
            future.result()


def filehandle_parallel():
    lock = threading.Lock()
    with tifffile.TiffFile(fname) as video_stream:
        def parallel_func(i):
            with lock:
                _ = get_single_volume(video_stream, i, **import_opt)
                # print(f"Got volume {i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(parallel_func, i) for i in frame_range}
            for future in concurrent.futures.as_completed(futures):
                future.result()


print("Sequential read")
print(timeit.timeit('sequential_read()', globals=globals(), number=10))

print("Parallel read")
print(timeit.timeit('simple_parallel()', globals=globals(), number=10))

print("Parallel fstream read")
print(timeit.timeit('filehandle_parallel()', globals=globals(), number=10))
