import uuid
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Dict, Optional

logger = logging.getLogger(__name__)


class TaskRunner:
    """Bounded, thread-safe background task executor with UUID tracking."""

    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="NNTask")
        self.jobs: Dict[str, dict] = {}
        self.lock = threading.Lock()

    def submit_task(self, task_type: str, fn: Callable, *args, **kwargs) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job_record = {
            "job_id": job_id,
            "type": task_type,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
        }
        with self.lock:
            self.jobs[job_id] = job_record

        def _wrapper():
            with self.lock:
                self.jobs[job_id]["status"] = "running"
                self.jobs[job_id]["started_at"] = time.time()
            try:
                result = fn(*args, **kwargs)
                with self.lock:
                    self.jobs[job_id]["status"] = "completed"
                    self.jobs[job_id]["completed_at"] = time.time()
                    self.jobs[job_id]["result"] = result
            except Exception as exc:
                logger.error("Task %s (%s) failed: %s", job_id, task_type, exc, exc_info=True)
                with self.lock:
                    self.jobs[job_id]["status"] = "failed"
                    self.jobs[job_id]["completed_at"] = time.time()
                    self.jobs[job_id]["error"] = str(exc)

        self.executor.submit(_wrapper)
        return job_id

    def get_status(self, job_id: str) -> Optional[dict]:
        with self.lock:
            rec = self.jobs.get(job_id)
            return dict(rec) if rec else None

    def get_latest_by_type(self, task_type: str) -> Optional[dict]:
        with self.lock:
            matching = [j for j in self.jobs.values() if j["type"] == task_type]
            if not matching:
                return None
            return dict(sorted(matching, key=lambda x: x["created_at"])[-1])

    def shutdown(self, wait: bool = False):
        self.executor.shutdown(wait=wait)


global_task_runner = TaskRunner(max_workers=3)
