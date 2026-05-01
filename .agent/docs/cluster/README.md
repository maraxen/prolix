# Cluster Job Logging

This directory logs SLURM job submissions and outcomes for tracking and debugging.

## Schema (JSONL format)

Each job is logged as a single JSON line in `engaging.jsonl` (or site-specific file):

```json
{
  "date": "2026-04-28T15:30:00Z",
  "site": "engaging",
  "project": "PROJECT_NAME",
  "job_id": "12345678",
  "script": "sbatch_script.sh",
  "partition": "mit_preemptable",
  "gpu_count": 0,
  "walltime": "12:00:00",
  "array_spec": "0-7%4 (if array job)",
  "status": "submitted|running|completed|failed",
  "outcome": "success|timeout|OOM|cancelled|other",
  "notes": "Brief description of job purpose"
}
```

**Usage:** Manually add entries after submitting jobs to track history, debug patterns, and analyze performance.
