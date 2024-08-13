import logging
import time

from wbfm.utils.general.hardcoded_paths import load_paper_datasets
from wbfm.utils.projects.finished_project_data import ProjectData
from submitit import AutoExecutor, LocalJob, DebugJob


def load_project_and_create_traces(project_path):
    p = ProjectData.load_final_project_data_from_config(project_path)
    p.logger.info(f"Building paper traces for {project_path}")
    output = p.calc_all_paper_traces()
    return output


def main():
    # Load all paths to datasets used in the paper
    all_paths_gcamp = load_paper_datasets(['gcamp', 'hannah_O2_fm'], only_load_paths=True)
    all_paths_gfp = load_paper_datasets(['gfp'], only_load_paths=True)
    all_paths_immob = load_paper_datasets(['immob'], only_load_paths=True)

    all_project_paths = all_paths_gcamp.copy()
    all_project_paths.update(all_paths_gfp)
    all_project_paths.update(all_paths_immob)

    # Set up the executor
    executor = AutoExecutor(folder="/tmp/submitit_runs", cluster='slurm')
    executor.update_parameters(slurm_time=f"0-04:00:00")
    executor.update_parameters(cpus_per_task=16)
    executor.update_parameters(slurm_mem="128G")
    executor.update_parameters(slurm_partition="basic,gpu")
    executor.update_parameters(slurm_job_name="create_paper_traces")

    jobs = []
    # Run until all the jobs have finished and our budget is used up.
    while jobs:
        for job in jobs[:]:
            # Poll if any jobs completed
            # Local and debug jobs don't run until .result() is called.
            if job.done() or type(job) in [LocalJob, DebugJob]:
                # The log file isn't being produced, so print the stdout instead
                result = job.result()
                jobs.remove(job)

        # Schedule new jobs
        for path in all_project_paths.items():
            # Make a new folder in the parent folder
            # Add the baseline parameters, and save in this folder
            job = executor.submit(load_project_and_create_traces, path)
            jobs.append(job)
            time.sleep(1)

        # Sleep for a bit before checking the jobs again to avoid overloading the cluster.
        # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
        time.sleep(5*60)


if __name__ == "__main__":
    main()
