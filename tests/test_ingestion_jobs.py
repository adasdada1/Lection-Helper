import unittest

from core.ingestion import IngestJobManager


class IngestJobManagerTests(unittest.TestCase):
    def test_active_job_tracks_pending_and_processing_only(self):
        manager = IngestJobManager()

        self.assertFalse(manager.has_active_job())

        job_id = manager.create_job("lecture.mp3")
        self.assertTrue(manager.has_active_job())

        manager._update(job_id, status="processing")
        self.assertTrue(manager.has_active_job())

        manager._update(job_id, status="done")
        self.assertFalse(manager.has_active_job())

    def test_error_job_is_not_active(self):
        manager = IngestJobManager()
        job_id = manager.create_job("lecture.mp3")

        manager._update(job_id, status="error")

        self.assertFalse(manager.has_active_job())


if __name__ == "__main__":
    unittest.main()
