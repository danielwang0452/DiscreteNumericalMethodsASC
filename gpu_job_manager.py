import subprocess
import threading
import queue
import time
import os
from itertools import product
from pathlib import Path

class GPUJobManager:
    def __init__(self, num_gpus=8, jobs_per_gpu=3, log_dir='logs'):
        self.num_gpus = num_gpus
        self.jobs_per_gpu = jobs_per_gpu
        self.job_queue = queue.Queue()
        self.active_jobs = {gpu_id: [] for gpu_id in range(num_gpus)}
        self.lock = threading.Lock()
        self.completed = []
        self.failed = []
        self.log_dir = Path(log_dir)
        
        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)
        
    def add_job(self, seed, method, categorical_dim, latent_dim, optimizer_type, learning_rate, temperature):
        """Add a job to the queue"""
        job = {
            'seed': seed,
            'method': method,
            'categorical_dim': categorical_dim,
            'latent_dim': latent_dim,
            'optimizer_type': optimizer_type,
            'learning_rate': learning_rate,
            'temperature': temperature,
        }
        self.job_queue.put(job)
        
    def get_log_filename(self, gpu_id, job):
        """Generate a unique log filename for a job"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            f"gpu{gpu_id}_seed{job['seed']}_{job['method']}_"
            f"cat{job['categorical_dim']}_lat{job['latent_dim']}_"
            f"{job['optimizer_type']}_lr{job['learning_rate']}_"
            f"temp{job['temperature']}_{timestamp}.txt"
        )
        return self.log_dir / filename
        
    def run_job(self, gpu_id, job):
        """Run a single job on specified GPU"""
        cmd = [
            './env/bin/python3', 'run.py',
            '--seed', str(job['seed']),
            '--method', job['method'],
            '--categorical_dim', str(job['categorical_dim']),
            '--latent_dim', str(job['latent_dim']),
            '--optimizer_type', job['optimizer_type'],
            '--learning_rate', str(job['learning_rate']),
            '--temperature', str(job['temperature']),
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        job_str = f"GPU{gpu_id}: seed={job['seed']}, method={job['method']}, cat={job['categorical_dim']}, lat={job['latent_dim']}, opt={job['optimizer_type']}, lr={job['learning_rate']}, temp={job['temperature']}"
        log_file = self.get_log_filename(gpu_id, job)
        
        print(f"Starting {job_str}")
        print(f"  Log: {log_file}")
        
        try:
            with open(log_file, 'w', buffering=1) as f:  # Line buffering
                # Write header
                f.write(f"Job: {job_str}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"{'='*60}\n\n")
                f.flush()
                
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffering
                )
                
                return_code = process.wait(timeout=86400)
                
                if return_code == 0:
                    print(f"✓ Completed {job_str}")
                    with self.lock:
                        self.completed.append(job)
                else:
                    print(f"✗ Failed {job_str} (exit code: {return_code})")
                    with self.lock:
                        self.failed.append(job)
                    
        except subprocess.TimeoutExpired:
            print(f"⏱ Timeout {job_str}")
            with open(log_file, 'a') as f:
                f.write(f"\n\n{'='*60}\n")
                f.write("ERROR: Job timed out after 24 hours\n")
            with self.lock:
                self.failed.append(job)
        except Exception as e:
            print(f"✗ Exception {job_str}: {e}")
            with open(log_file, 'a') as f:
                f.write(f"\n\n{'='*60}\n")
                f.write(f"ERROR: {e}\n")
            with self.lock:
                self.failed.append(job)
    
    def gpu_worker(self, gpu_id):
        """Worker thread for a single GPU"""
        while True:
            # Check if we can run more jobs on this GPU
            with self.lock:
                active_count = len([t for t in self.active_jobs[gpu_id] if t.is_alive()])
            
            if active_count < self.jobs_per_gpu:
                try:
                    # Try to get a job (non-blocking)
                    job = self.job_queue.get(timeout=1)
                    
                    # Start job in a new thread
                    job_thread = threading.Thread(
                        target=self.run_job,
                        args=(gpu_id, job),
                        daemon=True
                    )
                    job_thread.start()
                    
                    with self.lock:
                        self.active_jobs[gpu_id].append(job_thread)
                    
                    self.job_queue.task_done()
                    
                except queue.Empty:
                    # No jobs available, check if we're done
                    if self.job_queue.empty() and all(
                        all(not t.is_alive() for t in threads) 
                        for threads in self.active_jobs.values()
                    ):
                        break
            
            # Clean up finished threads
            with self.lock:
                self.active_jobs[gpu_id] = [t for t in self.active_jobs[gpu_id] if t.is_alive()]
            
            time.sleep(1)
    
    def run_all(self):
        """Start all GPU workers and wait for completion"""
        print(f"Starting job manager with {self.num_gpus} GPUs, {self.jobs_per_gpu} jobs per GPU")
        print(f"Total jobs in queue: {self.job_queue.qsize()}")
        print(f"Logs will be saved to: {self.log_dir.absolute()}")
        
        # Start worker threads for each GPU
        gpu_threads = []
        for gpu_id in range(self.num_gpus):
            t = threading.Thread(target=self.gpu_worker, args=(gpu_id,), daemon=True)
            t.start()
            gpu_threads.append(t)
        
        # Wait for all jobs to complete
        try:
            for t in gpu_threads:
                t.join()
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        
        print(f"\n{'='*60}")
        print(f"Completed: {len(self.completed)} jobs")
        print(f"Failed: {len(self.failed)} jobs")
        print(f"{'='*60}")
        
        if self.failed:
            print("\nFailed jobs:")
            for job in self.failed:
                print(f"  {job}")


if __name__ == "__main__":
    manager = GPUJobManager(num_gpus=8, jobs_per_gpu=3)
    
    # Define parameter grid
    optimizer_options = ["Adam", "RAdam"]
    # learning_rate_options = [0.001, 0.0007, 0.0005, 0.0003]
    learning_rate_options = [0.0007, 0.0005, 0.0003]
    temperature_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    categorical_dim_options = [8, 4, 8, 16, 64, 10]
    latent_dim_options = [4, 24, 16, 12, 8, 30]
    # methods = ["reinmax_v2", "reinmax_v3"]
    # methods = ["reinmax_v3"]
    methods = ["reinmax_cv"]
    seeds = range(10)
    
    # Add all jobs to queue

    for seed in seeds:
        for cat_dim, lat_dim in reversed(list(zip(categorical_dim_options, latent_dim_options))):
            for optimizer in optimizer_options:
                for lr in learning_rate_options:
                    for temp in temperature_options:
                        for method in methods:
                            manager.add_job(
                                seed=seed,
                                method=method,
                                categorical_dim=cat_dim,
                                latent_dim=lat_dim,
                                optimizer_type=optimizer,
                                learning_rate=lr,
                                temperature=temp
                            )
    
    # Run all jobs
    manager.run_all()