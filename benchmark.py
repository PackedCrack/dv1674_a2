import pathlib
import subprocess
import shutil
import os
import signal
import time


def erase_dir(name: str):
    result_dir = pathlib.Path.cwd() / name
    if result_dir.exists():
        if result_dir.is_dir() and not result_dir.is_symlink():
            shutil.rmtree(result_dir)
        else:
            result_dir.unlink()

def make_subdirs(name: str, subdirs):
    root = pathlib.Path.cwd() / name

    root.mkdir(parents=True, exist_ok=True)
    for name in subdirs:
        (root / name).mkdir(exist_ok=True)

def make_pearson_outdirs():
    name = "pearson_result"
    erase_dir(name)

    subdirs = ["128", "128_t1", "128_t3", "128_t6","128_t9", "128_t12",
               "256", "256_t1", "256_t3", "256_t6","256_t9", "256_t12",
               "512", "512_t1", "512_t3", "512_t6","512_t9", "512_t12",
               "1024", "1024_t1", "1024_t3", "1024_t6","1024_t9", "1024_t12"]
    make_subdirs(name, subdirs)

def make_blur_outdirs():
    name = "blur_result"
    erase_dir(name)

    subdirs = ["im1", "im2", "im3", "im4"]
    subdirs = ["im1", "im1_t1", "im1_t3", "im1_t6","im1_t9", "im1_t12",
               "im2", "im2_t1", "im2_t3", "im2_t6","im2_t9", "im2_t12",
               "im3", "im3_t1", "im3_t3", "im3_t6","im3_t9", "im3_t12",
               "im4", "im4_t1", "im4_t3", "im4_t6","im4_t9", "im4_t12"]
    make_subdirs(name, subdirs)

def profile_time(exe: str, inData : str, outData: str, outDir: str):
    print("Time Profiling {} {}".format(exe, inData))

    out = outDir + "/time.txt"
    cmd = [ 
        str(exe),
        str(inData),
        str(outData)
    ]
    
    start = time.perf_counter()
    subprocess.run(cmd, check = True, stdout=subprocess.DEVNULL)
    end = time.perf_counter()
    
    with open(out, "wb") as f:
        delta = end - start
        f.write("Wall Time: " + f"{delta:.9f}\n")
    
def profile_cpu(exe: str, inData : str, outData: str, outDir: str, radius = None, threads = None):
    print("\n\nCPU Profiling {} {}".format(exe, inData))

    out = outDir + "/hotspot"

    command = [ 
        "sudo",
        "perf", "record", 
        "-F", "400", 
        "-e", "cycles", 
        "-g", 
        "--call-graph=dwarf", 
        "-o", str(out), 
        "--", 
        str(exe)
    ]
    if radius != None:
        command.append(str(radius))
    command.append(str(inData))
    command.append(str(outData))
    if threads != None:
        command.append(str(threads))
    
    with open(os.devnull, "wb") as n:
        subprocess.run(command, check = True, stdout = n)

    out = outDir + "/cache.txt"
    command = [ 
        "sudo",
        "perf", "stat", 
        "--no-big-num",
        "-e", "cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches", 
        "-o", str(out), 
        "--", 
        str(exe)
    ]
    if radius != None:
        command.append(str(radius))
    command.append(str(inData))
    command.append(str(outData))
    if threads != None:
        command.append(str(threads))
        
    with open(os.devnull, "wb") as n:
        subprocess.run(command, check = True, stdout = n, stderr=subprocess.PIPE)

def profile_memory(exe: str, inData : str, outData: str, outDir: str, radius = None, threads = None):
    print("\n\nMemory Profiling {} {}".format(exe, inData))

    out = outDir + "/heaptrack-out"
    cmd = [ 
        "sudo",
        "heaptrack",
        "--output",
        str(out), 
        str(exe)
    ]
    if radius != None:
        cmd.append(str(radius))
    cmd.append(str(inData))
    cmd.append(str(outData))
    if threads != None:
        cmd.append(str(threads))

    with open(os.devnull, "wb") as n:
        subprocess.run(cmd, check = True, stdout = n, stderr = n)
    
    oldOut = out + ".zst"
    cmd = [ 
        "sudo",
        "heaptrack_print",
        str(oldOut)
    ]
    newOut = outDir + "/heaptrack.txt"
    with open(newOut,"w") as f:
        subprocess.run(cmd, check = True, stdout = f)

def profile_io(exe: str, inData : str, outData: str, outDir: str, radius = None, threads = None):
    print("\n\nIO Profiling {} {}".format(exe, inData))

    out = outDir + "/disk.txt"
    with open(out, "wb") as f:
        cmd = [
            str(exe)
        ]
        if radius != None:
            cmd.append(str(radius))
        cmd.append(str(inData))
        cmd.append(str(outData))
        if threads != None:
            cmd.append(str(threads))
        p = subprocess.Popen(cmd, stdout = subprocess.DEVNULL)

        cmd = [
            "pidstat", 
            "-d", 
            "1", 
            "-p", 
            str(p.pid)
        ]
        p2 = subprocess.Popen(cmd, stdout = f, start_new_session = True)
        
        p.wait()
        os.killpg(p2.pid, signal.SIGINT)
        try:
            p2.wait(timeout = 2)
        except subprocess.TimeoutExpired:
            os.killpg(p2.pid, signal.SIGKILL)
            p2.wait()

def pearson_baseline():
    exe = "./pearson/pearson"
    inData = [
        "./pearson/data/128.data",
        "./pearson/data/256.data",
        "./pearson/data/512.data",
        "./pearson/data/1024.data"
    ]
    outData = [
        "./pearson/data_o/128_seq.data",
        "./pearson/data_o/256_seq.data",
        "./pearson/data_o/512_seq.data",
        "./pearson/data_o/1024_seq.data"
    ]
    outDirs = [
        "./pearson_result/128",
        "./pearson_result/256",
        "./pearson_result/512",
        "./pearson_result/1024"
    ]
    

    for index in range(0, 4):
        i = inData[index]
        o = outData[index]
        outDir = outDirs[index]

        #profile_time(exe, i, o, outDir)
        profile_cpu(exe, i, o, outDir)
        profile_memory(exe, i, o, outDir)
        profile_io(exe, i, o, outDir)

def blur_baseline():
    exe = "./blur/blur"
    inData = [
        "./blur/data/im1.ppm",
        "./blur/data/im2.ppm",
        "./blur/data/im3.ppm",
        "./blur/data/im4.ppm"
    ]
    outData = [
        "./blur/data_o/im1_seq.ppm",
        "./blur/data_o/im2_seq.ppm",
        "./blur/data_o/im3_seq.ppm",
        "./blur/data_o/im4_seq.ppm"
    ]
    outDirs = [
        "./blur_result/im1",
        "./blur_result/im2",
        "./blur_result/im3",
        "./blur_result/im4"
    ]
    

    for index in range(0, 4):
        i = inData[index]
        o = outData[index]
        outDir = outDirs[index]

        #profile_time(exe, i, o, outDir)
        profile_cpu(exe, i, o, outDir, 15)
        profile_memory(exe, i, o, outDir, 15)
        profile_io(exe, i, o, outDir, 15)

def baseline():
    pearson_baseline()
    blur_baseline()

def pearson_benchmark():
    exe = "./pearson/pearson_par"
    inData = [
        "./pearson/data/128.data",
        "./pearson/data/256.data",
        "./pearson/data/512.data",
        "./pearson/data/1024.data"
    ]
    outData = [
        "./pearson/data_o/128_seq.data",
        "./pearson/data_o/256_seq.data",
        "./pearson/data_o/512_seq.data",
        "./pearson/data_o/1024_seq.data"
    ]
    outDirs = [
        "./pearson_result/128",
        "./pearson_result/256",
        "./pearson_result/512",
        "./pearson_result/1024"
    ]

    threads = [1, 3, 6, 9, 12]
    for index in range(0, 4):
        for threadCount in threads:
            i = inData[index]
            o = outData[index]
            outDir = outDirs[index]
            outDir = outDir + "_t" + str(threadCount)

            profile_cpu(exe, i, o, outDir, threads = threadCount)
            profile_memory(exe, i, o, outDir, threads = threadCount)
            #profile_io(exe, i, o, outDir, threads = threadCount)

def blur_benchmark():
    exe = "./blur/blur_par"
    inData = [
        "./blur/data/im1.ppm",
        "./blur/data/im2.ppm",
        "./blur/data/im3.ppm",
        "./blur/data/im4.ppm"
    ]
    outData = [
        "./blur/data_o/im1_par_bench.ppm",
        "./blur/data_o/im2_par_bench.ppm",
        "./blur/data_o/im3_par_bench.ppm",
        "./blur/data_o/im4_par_bench.ppm"
    ]
    outDirs = [
        "./blur_result/im1",
        "./blur_result/im2",
        "./blur_result/im3",
        "./blur_result/im4"
    ]

    threads = [1, 3, 6, 9, 12]
    for index in range(0, 4):
        for threadCount in threads:
            i = inData[index]
            o = outData[index]
            outDir = outDirs[index]
            outDir = outDir + "_t" + str(threadCount)

            profile_cpu(exe, i, o, outDir, radius = 15, threads = threadCount)
            profile_memory(exe, i, o, outDir, radius = 15, threads = threadCount)
            #profile_io(exe, i, o, outDir, threads = threadCount)

def benchmark():
    #pearson_benchmark()
    blur_benchmark()


make_pearson_outdirs()
make_blur_outdirs()

#baseline()
benchmark()