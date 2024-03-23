# Copyright © 2023 Apple Inc.

"""Tests bastion orchestrator."""
# pylint: disable=no-self-use,protected-access
import contextlib
import os
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Sequence
from unittest import mock

from absl.testing import parameterized

from axlearn.cloud.common import bastion
from axlearn.cloud.common.bastion import (
    _JOB_DIR,
    _LOG_DIR,
    Bastion,
    Job,
    JobState,
    _load_runtime_options,
    _PipedProcess,
    deserialize_jobspec,
    download_job_batch,
    new_jobspec,
    serialize_jobspec,
    set_runtime_options,
)
from axlearn.cloud.common.cleaner import Cleaner
from axlearn.cloud.common.scheduler import JobMetadata, JobScheduler
from axlearn.cloud.common.scheduler_test import mock_quota_config
from axlearn.cloud.common.types import JobSpec, ResourceMap
from axlearn.cloud.common.uploader import Uploader
from axlearn.common.config import config_for_function


class TestDownloadJobBatch(parameterized.TestCase):
    """Tests download utils."""

    def test_download_job_batch(self):
        spec_dir = "gs://test_spec_dir"
        state_dir = "gs://test_state_dir"
        user_state_dir = "gs://user_state_dir"

        user_states = {
            "job_test1": JobState.CANCELLING,
            "job_test2": JobState.ACTIVE,
            "job_test0": JobState.CANCELLING,
            "job_test3": JobState.CANCELLING,
        }
        states = {
            "job_test1": JobState.ACTIVE,
            "job_test0": JobState.CLEANING,
            "job_test3": JobState.COMPLETED,
            "job_test4": JobState.PENDING,
        }
        jobspecs = {
            "job_test2": mock.Mock(),
            "job_test1": mock.Mock(),
            "job_test0": mock.Mock(),
            "job_test3": mock.Mock(),
            "job_test4": mock.Mock(),
        }
        expected = {
            # User state is invalid and is ignored. Job state defaults to PENDING, since it's
            # missing a state.
            "job_test2": JobState.PENDING,
            # User state should take effect.
            "job_test1": JobState.CANCELLING,
            # User state should not affect CLEANING/COMPLETED.
            "job_test0": JobState.CLEANING,
            "job_test3": JobState.COMPLETED,
            # Has no user state.
            "job_test4": JobState.PENDING,
        }

        def mock_listdir(path):
            if path == spec_dir:
                return list(jobspecs.keys())
            if path == state_dir:
                return list(states.keys())
            if path == user_state_dir:
                return list(user_states.keys())
            assert False  # Should not be reached.

        def mock_download_jobspec(job_name, **kwargs):
            del kwargs
            return jobspecs[job_name]

        def mock_download_job_state(job_name, *, remote_dir, **kwargs):
            del kwargs
            if remote_dir == state_dir:
                # Job state may be initially missing, thus defaults to PENDING.
                return states.get(job_name, JobState.PENDING)
            if remote_dir == user_state_dir:
                # We should only query user states if one exists, so don't use get().
                return user_states[job_name]
            assert False  # Should not be reached.

        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _download_jobspec=mock.Mock(side_effect=mock_download_jobspec),
            _download_job_state=mock.Mock(side_effect=mock_download_job_state),
        )
        patch_tfio = mock.patch(f"{bastion.__name__}.tf_io.gfile.listdir", side_effect=mock_listdir)

        # Ensure that results are in the right order and pairing.
        with patch_fns, patch_tfio, tempfile.TemporaryDirectory() as tmpdir:
            jobs, jobs_with_user_states = download_job_batch(
                spec_dir=spec_dir,
                state_dir=state_dir,
                user_state_dir=user_state_dir,
                local_spec_dir=tmpdir,
            )
            self.assertSameElements(expected.keys(), jobs.keys())
            # "job_test1" is the only valid user state, but we still cleanup the others.
            self.assertSameElements(jobs_with_user_states, user_states.keys())
            for job_name, job in jobs.items():
                self.assertEqual(job.state, expected[job_name])
                self.assertEqual(job.spec, jobspecs[job_name])


class TestJobSpec(parameterized.TestCase):
    """Tests job specs."""

    @parameterized.parameters(
        [
            {"env_vars": None},
            {"env_vars": {"TEST_ENV": "TEST_VAL", "TEST_ENV2": 'VAL_WITH_SPECIAL_:,"-{}'}},
        ],
    )
    def test_serialization_job_spec(self, env_vars):
        test_spec = new_jobspec(
            name="test_job",
            command="test command",
            env_vars=env_vars,
            metadata=JobMetadata(
                user_id="test_id",
                project_id="test_project",
                creation_time=datetime.now(),
                resources={"test": 8},
                priority=1,
            ),
        )
        with tempfile.NamedTemporaryFile("w+b") as f:
            serialize_jobspec(test_spec, f.name)
            deserialized_jobspec = deserialize_jobspec(f=f.name)
            for key in test_spec.__dataclass_fields__:
                self.assertIn(key, deserialized_jobspec.__dict__)
                self.assertEqual(deserialized_jobspec.__dict__[key], test_spec.__dict__[key])


class TestRuntimeOptions(parameterized.TestCase):
    """Tests runtime options."""

    def test_load_and_set_runtime_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initially empty.
            self.assertEqual({}, _load_runtime_options(temp_dir))

            # Set some values.
            set_runtime_options(temp_dir, a="1", b={"c": "2"})
            self.assertEqual({"a": "1", "b": {"c": "2"}}, _load_runtime_options(temp_dir))

            # Update.
            set_runtime_options(temp_dir, a="2", b={"c": "3"})
            self.assertEqual({"a": "2", "b": {"c": "3"}}, _load_runtime_options(temp_dir))


# Returns a new mock Popen for each subprocess.Popen call.
def _mock_popen_fn(mock_spec: Dict[str, Dict]):
    """Returns a callable that outputs mocked Popens for predetermined commands.

    For example:
        Input:
            {'my_command': {'terminate.side_effect': ValueError}}
        Result:
            mock = subprocess.Popen('my_command')
            mock.terminate()  # Raises ValueError.
    """

    def popen(cmd, **kwargs):
        del kwargs
        if cmd not in mock_spec:
            raise ValueError(f"Don't know how to mock: {cmd}")
        m = mock.MagicMock()
        m.configure_mock(**mock_spec[cmd])
        return m

    return popen


# Returns a new mock _PipedProcess.
def _mock_piped_popen_fn(mock_spec: Dict[str, Dict]):
    """See `_mock_popen_fn`."""
    mock_popen_fn = _mock_popen_fn(mock_spec)

    def piped_popen(cmd, f, env_vars=None):
        del env_vars
        mock_fd = mock.MagicMock()
        mock_fd.name = f
        return _PipedProcess(popen=mock_popen_fn(cmd), fd=mock_fd)

    return piped_popen


class BastionTest(parameterized.TestCase):
    """Tests Bastion."""

    @contextlib.contextmanager
    def _patch_bastion(self, mock_popen_spec: Optional[Dict] = None):
        mocks = []
        module_name = bastion.__name__

        if mock_popen_spec:
            mock_popen = mock.patch.object(subprocess, "Popen", autospec=True)
            mock_popen.side_effect = _mock_popen_fn(mock_popen_spec)
            mocks.extend(
                [
                    mock_popen,
                    mock.patch(
                        f"{module_name}._piped_popen",
                        side_effect=_mock_piped_popen_fn(mock_popen_spec),
                    ),
                ]
            )

        class NoOpCleaner(Cleaner):
            def sweep(self, jobs):
                del jobs

        def noop_upload_fn(*args, **kwargs):
            del args, kwargs

        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory() as tmpdir:
            # Boilerplate to register multiple mocks at once.
            for m in mocks:
                stack.enter_context(m)

            cfg = Bastion.default_config().set(
                scheduler=JobScheduler.default_config().set(
                    quota=config_for_function(mock_quota_config)
                ),
                cleaner=NoOpCleaner.default_config(),
                uploader=Uploader.default_config().set(
                    upload_fn=config_for_function(lambda: noop_upload_fn)
                ),
                output_dir=tmpdir,
            )
            yield cfg.instantiate()

    @parameterized.product(
        [
            dict(
                # Command has not terminated -- expect kill() to be called.
                # We should not need to consult terminate() or poll().
                popen_spec={
                    "command": {
                        "wait.return_value": None,
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                    # cleanup should have no effect here, so we just raise if it's ever used.
                    "cleanup": {
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                },
            ),
            dict(
                # Command has already terminated. Expect state to transition to PENDING and
                # command_proc to be None.
                popen_spec={
                    "cleanup": {"poll.return_value": 0, "terminate.side_effect": ValueError},
                },
            ),
        ],
        user_state_exists=[False, True],
    )
    def test_pending(self, popen_spec, user_state_exists):
        """Test PENDING state transitions.

        1. If command_proc is still running, it should be terminated (killed).
        2. The state should remain PENDING, command_proc must be None, and log file should be
            uploaded.
        """
        mock_proc = _mock_piped_popen_fn(popen_spec)
        job = Job(
            spec=new_jobspec(
                name="test_job",
                command="command",
                cleanup_command="cleanup",
                metadata=JobMetadata(
                    user_id="test_user",
                    project_id="test_project",
                    creation_time=datetime.now(),
                    resources={"v4": 8},
                ),
            ),
            state=JobState.PENDING,
            command_proc=mock_proc("command", "test_command") if "command" in popen_spec else None,
            cleanup_proc=mock_proc("cleanup", "test_cleanup") if "cleanup" in popen_spec else None,
        )
        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
            send_signal=mock.DEFAULT,
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.Mock(return_value=user_state_exists),
            copy=mock.DEFAULT,
            remove=mock.DEFAULT,
        )
        with self._patch_bastion(
            popen_spec
        ) as mock_bastion, patch_fns as mock_fns, patch_tfio as mock_tfio:
            # Run a couple updates to test transition to PENDING and staying in PENDING.
            for _ in range(2):
                orig_command_proc = job.command_proc
                updated_job = mock_bastion._update_single_job(job)
                # Job should now be in pending.
                self.assertEqual(updated_job.state, JobState.PENDING)
                # Command should be None.
                self.assertIsNone(updated_job.command_proc)

                if orig_command_proc is not None:
                    # Kill should have been called, and fd should have been closed.
                    mock_fns["send_signal"].assert_called()
                    self.assertTrue(
                        orig_command_proc.fd.close.called  # pytype: disable=attribute-error
                    )

                    # Log should be uploaded if command was initially running.
                    upload_call = mock.call(
                        orig_command_proc.fd.name,
                        os.path.join(
                            mock_bastion._log_dir, os.path.basename(orig_command_proc.fd.name)
                        ),
                        overwrite=True,
                    )
                    mock_tfio["copy"].assert_has_calls([upload_call], any_order=False)

                # Cleanup command should not be involved.
                updated_job.cleanup_proc.popen.poll.assert_not_called()
                updated_job.cleanup_proc.popen.terminate.assert_not_called()

                updated_job = job

    @parameterized.product(
        [
            dict(
                popen_spec={
                    # Runs for one update step and then completes.
                    # terminate() raises, since we don't expect it to be called.
                    "command": {
                        "poll.side_effect": [None, 0],
                        "terminate.side_effect": ValueError,
                    },
                    # cleanup should have no effect here, so we just raise if it's ever used.
                    "cleanup": {
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                },
                expect_poll_calls=2,
            ),
            dict(
                popen_spec={
                    # Command terminates instantly.
                    "command": {
                        "poll.return_value": 1,
                        "terminate.side_effect": ValueError,
                    },
                    # cleanup should have no effect here, so we just raise if it's ever used.
                    "cleanup": {
                        "poll.side_effect": ValueError,
                        "terminate.side_effect": ValueError,
                    },
                },
                expect_poll_calls=1,
            ),
        ],
        logfile_exists=[False, True],
    )
    def test_active(self, popen_spec, expect_poll_calls, logfile_exists):
        """Test ACTIVE state transitions.

        1. If command_proc is not running, it should be started. If a log file exists remotely, it
            should be downloaded.
        2. If command_proc is already running, stay in ACTIVE.
        3. If command_proc is completed, move to CLEANING.
        """
        mock_proc = _mock_piped_popen_fn(popen_spec)
        job = Job(
            spec=new_jobspec(
                name="test_job",
                command="command",
                cleanup_command="cleanup",
                metadata=JobMetadata(
                    user_id="test_user",
                    project_id="test_job",
                    creation_time=datetime.now(),
                    resources={"v4": 8},
                ),
            ),
            state=JobState.ACTIVE,
            command_proc=None,  # Initially, command is None.
            cleanup_proc=mock_proc("cleanup", "test_cleanup"),
        )

        def mock_tfio_exists(f):
            if "logs" in f and os.path.basename(f) == "test_job":
                return logfile_exists
            return False

        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.MagicMock(side_effect=mock_tfio_exists),
            copy=mock.DEFAULT,
        )
        with patch_fns, self._patch_bastion(popen_spec) as mock_bastion, patch_tfio as mock_tfio:
            # Initially, job should have no command.
            self.assertIsNone(job.command_proc)

            # Run single update step to start the job.
            updated_job = mock_bastion._update_single_job(job)

            # Command should be started on the first update.
            self.assertIsNotNone(updated_job.command_proc)
            # Log should be downloaded if it exists.
            download_call = mock.call(
                os.path.join(mock_bastion._log_dir, job.spec.name),
                os.path.join(_LOG_DIR, job.spec.name),
                overwrite=True,
            )
            mock_tfio["copy"].assert_has_calls([download_call], any_order=False)

            # Run until expected job completion.
            for _ in range(expect_poll_calls - 1):
                self.assertEqual(updated_job.state, JobState.ACTIVE)
                updated_job = mock_bastion._update_single_job(updated_job)

            # Job state should be CLEANING.
            self.assertEqual(updated_job.state, JobState.CLEANING)

    # pylint: disable-next=too-many-branches
    def test_update_jobs(self):
        """Tests the global update step."""

        def popen_spec(command_poll=2, cleanup_poll=2):
            return {
                # Constructs a command_proc that "completes" after `command_poll` updates.
                "command": {
                    "wait.return_value": None,
                    "poll.side_effect": [None] * (command_poll - 1) + [0],
                    "terminate.side_effect": None,
                },
                # Constructs a cleanup_proc that completes after `cleanup_poll` updates.
                "cleanup": {
                    "poll.side_effect": [None] * (cleanup_poll - 1) + [0],
                    "terminate.side_effect": ValueError,
                },
            }

        def mock_proc(cmd, **kwargs):
            fn = _mock_piped_popen_fn(popen_spec(**kwargs))
            return fn(cmd, "test_file")

        yesterday = datetime.now() - timedelta(days=1)

        # Test state transitions w/ interactions between jobs (scheduling).
        # See also `mock_quota_config` for mock project quotas and limits.
        active_jobs = {
            # This job will stay PENDING, since user "b" has higher priority.
            "pending": Job(
                spec=new_jobspec(
                    name="pending",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="a",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=3),
                        resources={"v4": 12},  # Doesn't fit if "resume" job is scheduled.
                    ),
                ),
                state=JobState.PENDING,
                command_proc=None,  # No command proc for PENDING jobs.
                cleanup_proc=None,
            ),
            # This job will go from PENDING to ACTIVE.
            "resume": Job(
                spec=new_jobspec(
                    name="resume",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="b",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v4": 5},  # Fits within v4 budget in project2.
                    ),
                ),
                state=JobState.PENDING,
                command_proc=None,  # No command proc for PENDING jobs.
                cleanup_proc=None,
            ),
            # This job will stay in ACTIVE, since it takes 2 updates to complete.
            "active": Job(
                spec=new_jobspec(
                    name="active",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="c",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v3": 2},  # Fits within the v3 budget in project2.
                    ),
                ),
                state=JobState.PENDING,
                command_proc=mock_proc("command"),
                cleanup_proc=None,  # No cleanup_proc for ACTIVE jobs.
            ),
            # This job will go from ACTIVE to PENDING, since it's using part of project2's v4
            # quota, and "b" is requesting project2's v4 quota.
            # Even though poll()+terminate() typically takes a few steps, we instead go through
            # kill() to forcefully terminate within one step.
            "preempt": Job(
                spec=new_jobspec(
                    name="preempt",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="d",
                        project_id="project1",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v4": 12},  # Uses part of project2 budget.
                    ),
                ),
                state=JobState.ACTIVE,
                command_proc=mock_proc("command"),
                cleanup_proc=None,  # No cleanup_proc for ACTIVE.
            ),
            # This job will go from ACTIVE to CLEANING.
            "cleaning": Job(
                spec=new_jobspec(
                    name="cleaning",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="f",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v3": 2},  # Fits within the v3 budget in project2.
                    ),
                ),
                state=JobState.ACTIVE,
                command_proc=mock_proc("command", command_poll=1),
                cleanup_proc=None,
            ),
            # This job will go from CANCELLING to CLEANING.
            # Note that CANCELLING jobs will not be "pre-empted" by scheduler; even though this job
            # is out-of-budget, it will go to CLEANING instead of SUSPENDING.
            "cleaning_cancel": Job(
                spec=new_jobspec(
                    name="cleaning_cancel",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="g",
                        project_id="project2",
                        creation_time=yesterday + timedelta(seconds=4),
                        resources={"v4": 100},  # Does not fit into v4 budget.
                    ),
                ),
                state=JobState.CANCELLING,
                command_proc=mock_proc("command", command_poll=1),
                cleanup_proc=None,
            ),
            # This job will go from CLEANING to COMPLETED.
            "completed": Job(
                spec=new_jobspec(
                    name="completed",
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id="e",
                        project_id="project3",
                        creation_time=yesterday + timedelta(seconds=2),
                        resources={"v5": 2},
                    ),
                ),
                state=JobState.CLEANING,
                command_proc=None,
                cleanup_proc=mock_proc("cleanup", cleanup_poll=1),  # Should have cleanup_proc.
            ),
        }
        # Pretend that only 'cleaning_cancel' came from a user state.
        jobs_with_user_states = {"cleaning_cancel"}

        # Patch all network calls and utils.
        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
            send_signal=mock.DEFAULT,
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.DEFAULT,
            copy=mock.DEFAULT,
            remove=mock.DEFAULT,
        )
        with self._patch_bastion(
            popen_spec()
        ) as mock_bastion, patch_fns as mock_fns, patch_tfio as mock_tfio:
            mock_bastion._active_jobs = active_jobs
            mock_bastion._jobs_with_user_states = jobs_with_user_states
            mock_bastion._update_jobs()

            # Ensure _active_jobs membership stays same.
            self.assertEqual(mock_bastion._active_jobs.keys(), active_jobs.keys())

            expected_states = {
                "pending": JobState.PENDING,
                "resume": JobState.ACTIVE,
                "active": JobState.ACTIVE,
                "preempt": JobState.PENDING,
                "cleaning": JobState.CLEANING,
                "cleaning_cancel": JobState.CLEANING,
                "completed": JobState.COMPLETED,
            }
            for job_name in active_jobs:
                self.assertEqual(
                    mock_bastion._active_jobs[job_name].state, expected_states[job_name]
                )

            for job in mock_bastion._active_jobs.values():
                # For jobs that are ACTIVE, expect command_proc to be non-None.
                if job.state == JobState.ACTIVE:
                    self.assertIsNotNone(job.command_proc)
                    self.assertIsNone(job.cleanup_proc)
                # For jobs that are COMPLETED, expect both procs to be None.
                elif job.state == JobState.COMPLETED:
                    self.assertIsNone(job.command_proc)
                    self.assertIsNone(job.cleanup_proc)

                    # Remote jobspec should not be deleted until gc.
                    for delete_call in mock_tfio["remove"].mock_calls:
                        self.assertNotIn(
                            os.path.join(_JOB_DIR, job.spec.name),
                            delete_call.args,
                        )

                # User states should only be deleted if the job's state was read from
                # user_state_dir.
                self.assertEqual(
                    any(
                        os.path.join(mock_bastion._user_state_dir, job.spec.name)
                        in delete_call.args
                        for delete_call in mock_tfio["remove"].mock_calls
                    ),
                    job.spec.name in mock_bastion._jobs_with_user_states,
                )

                # For jobs that went from ACTIVE to PENDING, expect kill() to have been called.
                if active_jobs[job.spec.name] == JobState.ACTIVE and job.state == JobState.PENDING:
                    mock_fns["send_signal"].assert_called()
                    self.assertFalse(
                        active_jobs[
                            job.spec.name
                        ].command_proc.popen.terminate.called  # pytype: disable=attribute-error
                    )

            for job_name in active_jobs:
                history_file = os.path.join(mock_bastion._job_history_dir, job_name)
                if job_name in ("active", "pending"):
                    # The 'active'/'pending' jobs do not generate hisotry.
                    self.assertFalse(os.path.exists(history_file), msg=history_file)
                else:
                    self.assertTrue(os.path.exists(history_file), msg=history_file)
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = f.read()
                        expected_msg = {
                            "resume": "ACTIVE: start process command",
                            "preempt": "PENDING: pre-empting",
                            "cleaning": "CLEANING: process finished",
                            "cleaning_cancel": "CLEANING: process terminated",
                            "completed": "COMPLETED: cleanup finished",
                        }
                        self.assertIn(expected_msg[job_name], history)

            all_history_files = []
            for project_id in [f"project{i}" for i in range(1, 3)]:
                project_history_dir = os.path.join(mock_bastion._project_history_dir, project_id)
                project_history_files = list(os.scandir(project_history_dir))
                for history_file in project_history_files:
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = f.read()
                        print(f"[{project_id}] {history}")
                all_history_files.extend(project_history_files)
            # "project1" and "project2".
            self.assertLen(all_history_files, 2)

    def test_gc_jobs(self):
        """Tests GC mechanism.

        1. Only PENDING/COMPLETED jobs are cleaned.
        2. COMPLETED jobs that finish gc'ing should remove jobspecs.
        """
        # Note: command_proc and cleanup_proc shouldn't matter for GC. We only look at state +
        # resources.
        active_jobs = {}
        init_job_states = {
            "pending": JobState.PENDING,
            "active": JobState.ACTIVE,
            "cleaning": JobState.CLEANING,
            "completed": JobState.COMPLETED,
            "completed_gced": JobState.COMPLETED,
        }
        for job_name, job_state in init_job_states.items():
            active_jobs[job_name] = Job(
                spec=new_jobspec(
                    name=job_name,
                    command="command",
                    cleanup_command="cleanup",
                    metadata=JobMetadata(
                        user_id=f"{job_name}_user",
                        project_id="project1",
                        creation_time=datetime.now() - timedelta(days=1),
                        resources={"v4": 1},
                    ),
                ),
                state=job_state,
                command_proc=None,
                cleanup_proc=None,
            )
        # We pretend that only some jobs are "fully gc'ed".
        fully_gced = ["completed_gced"]

        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            remove=mock.DEFAULT,
        )
        with self._patch_bastion() as mock_bastion, patch_tfio as mock_tfio:

            def mock_clean(jobs: Dict[str, ResourceMap]) -> Sequence[str]:
                self.assertTrue(
                    all(
                        active_jobs[job_name].state in {JobState.PENDING, JobState.COMPLETED}
                        for job_name in jobs
                    )
                )
                for job_spec in jobs.values():
                    self.assertIsInstance(job_spec, JobSpec)
                return fully_gced

            with mock.patch.object(mock_bastion, "_cleaner") as mock_cleaner:
                mock_cleaner.configure_mock(**{"sweep.side_effect": mock_clean})
                mock_bastion._active_jobs = active_jobs
                mock_bastion._gc_jobs()

            # Ensure that each fully GC'ed COMPLETED job deletes jobspec and state.
            for job_name in fully_gced:
                deleted_state = any(
                    os.path.join(mock_bastion._state_dir, job_name) in delete_call.args
                    for delete_call in mock_tfio["remove"].mock_calls
                )
                deleted_jobspec = any(
                    os.path.join(mock_bastion._active_dir, job_name) in delete_call.args
                    for delete_call in mock_tfio["remove"].mock_calls
                )
                self.assertEqual(
                    active_jobs[job_name].state == JobState.COMPLETED,
                    deleted_state and deleted_jobspec,
                )

    @parameterized.parameters(
        dict(
            initial_jobs={
                "pending": JobState.PENDING,
                "active": JobState.ACTIVE,
                "cancelling": JobState.CANCELLING,
                "completed": JobState.COMPLETED,
            },
            runtime_options={},
            expect_schedulable=["pending", "active"],
        ),
        # Test runtime options.
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"dry_run": True, "verbosity": 1}},
            expect_schedulable=[],
            expect_dry_run=True,
            expect_verbosity=1,
        ),
        # Test invalid runtime options.
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"dry_run": "hello", "verbosity": None}},
            expect_schedulable=[],
        ),
        # Test invalid runtime options schema.
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"verbosity": None}},
            expect_schedulable=[],
        ),
        dict(
            initial_jobs={},
            runtime_options={"scheduler": {"unknown": 123}},
            expect_schedulable=[],
        ),
        dict(
            initial_jobs={},
            runtime_options={"scheduler": [], "unknown": 123},
            expect_schedulable=[],
        ),
        dict(
            initial_jobs={},
            runtime_options={"unknown": 123},
            expect_schedulable=[],
        ),
    )
    def test_update_scheduler(
        self,
        *,
        initial_jobs: Dict[str, JobState],
        runtime_options: Optional[Dict[str, Any]],
        expect_schedulable: Sequence[str],
        expect_dry_run: bool = False,
        expect_verbosity: int = 0,
    ):
        with self._patch_bastion() as mock_bastion:
            patch_update = mock.patch.object(mock_bastion, "_update_single_job")
            patch_history = mock.patch.object(mock_bastion, "_append_to_project_history")
            patch_scheduler = mock.patch.object(mock_bastion, "_scheduler")

            with patch_update, patch_history, patch_scheduler as mock_scheduler:
                mock_bastion._active_jobs = {
                    job_name: Job(
                        spec=mock.Mock(), state=state, command_proc=None, cleanup_proc=None
                    )
                    for job_name, state in initial_jobs.items()
                }
                mock_bastion._runtime_options = runtime_options
                mock_bastion._update_jobs()
                args, kwargs = mock_scheduler.schedule.call_args
                self.assertSameElements(expect_schedulable, args[0].keys())
                self.assertEqual({"dry_run": expect_dry_run, "verbosity": expect_verbosity}, kwargs)


class BastionDirectoryTest(parameterized.TestCase):
    """Tests BastionDirectory."""

    @parameterized.parameters(True, False)
    def test_submit_job(self, spec_exists):
        job_name = "test-job"
        job_spec_file = "spec"
        bastion_dir = (
            bastion.BastionDirectory.default_config().set(root_dir="test-dir").instantiate()
        )
        self.assertEqual("test-dir", str(bastion_dir))
        self.assertEqual("test-dir/logs", bastion_dir.logs_dir)
        self.assertEqual("test-dir/jobs/active", bastion_dir.active_job_dir)
        self.assertEqual("test-dir/jobs/complete", bastion_dir.complete_job_dir)
        self.assertEqual("test-dir/jobs/states", bastion_dir.job_states_dir)
        self.assertEqual("test-dir/jobs/user_states", bastion_dir.user_states_dir)
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.MagicMock(return_value=spec_exists),
            copy=mock.DEFAULT,
        )
        with patch_tfio as mock_tfio:
            bastion_dir.submit_job(job_name, job_spec_file=job_spec_file)
            if not spec_exists:
                mock_tfio["copy"].assert_called_with(
                    job_spec_file,
                    os.path.join(bastion_dir.active_job_dir, job_name),
                )
            else:
                mock_tfio["copy"].assert_not_called()

    @parameterized.parameters(True, False)
    def test_delete(self, spec_exists):
        job_name = "test-job"
        bastion_dir = (
            bastion.BastionDirectory.default_config().set(root_dir="test-dir").instantiate()
        )
        patch_tfio = mock.patch.multiple(
            f"{bastion.__name__}.tf_io.gfile",
            exists=mock.MagicMock(side_effect=[spec_exists, False]),
            copy=mock.DEFAULT,
        )
        patch_fns = mock.patch.multiple(
            bastion.__name__,
            _upload_job_state=mock.DEFAULT,
        )
        with patch_tfio, patch_fns as mock_fns:
            bastion_dir.cancel_job(job_name)
            if not spec_exists:
                mock_fns["_upload_job_state"].assert_not_called()
            else:
                mock_fns["_upload_job_state"].assert_called_with(
                    job_name,
                    JobState.CANCELLING,
                    remote_dir=bastion_dir.user_states_dir,
                )
