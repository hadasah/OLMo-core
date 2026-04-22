from glob import glob

import pytest
import requests

from olmo_core.io import (
    _http_get_bytes_range,
    copy_dir,
    copy_file,
    deserialize_from_tensor,
    file_exists,
    glob_directory,
    list_directory,
    remove_file,
    serialize_to_tensor,
    upload,
)


def test_serde_from_tensor():
    data = {"a": (1, 2)}
    assert deserialize_from_tensor(serialize_to_tensor(data)) == data


def test_local_functionality(tmp_path):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2").touch()
    (tmp_path / "dir1" / "file3.json").touch()

    # Should only list immediate children (files and dirs), but not files in subdirs.
    # The paths returned should be full paths.
    assert set(list_directory(tmp_path)) == {f"{tmp_path}/file1.json", f"{tmp_path}/dir1"}
    assert set(list_directory(tmp_path, recurse=True)) == {
        f"{tmp_path}/file1.json",
        f"{tmp_path}/dir1",
        f"{tmp_path}/dir1/file2",
        f"{tmp_path}/dir1/file3.json",
    }

    (tmp_path / "dir1" / "subdir1").mkdir()
    (tmp_path / "dir1" / "subdir1" / "file1").touch()
    (tmp_path / "dir1" / "subdir1" / "file4.json").touch()

    copy_dir(tmp_path / "dir1", tmp_path / "dir2")
    assert set(list_directory(tmp_path / "dir2", recurse=True)) == {
        f"{tmp_path}/dir2/file2",
        f"{tmp_path}/dir2/file3.json",
        f"{tmp_path}/dir2/subdir1",
        f"{tmp_path}/dir2/subdir1/file1",
        f"{tmp_path}/dir2/subdir1/file4.json",
    }

    # Test glob_directory with local files
    # Should list top-level json files
    assert set(glob_directory(f"{tmp_path}/*.json")) == {
        f"{tmp_path}/file1.json",
    }

    # Should list all json files
    assert set(glob_directory(f"{tmp_path}/**/*.json")) == {
        f"{tmp_path}/file1.json",
        f"{tmp_path}/dir1/file3.json",
        f"{tmp_path}/dir1/subdir1/file4.json",
        f"{tmp_path}/dir2/file3.json",
        f"{tmp_path}/dir2/subdir1/file4.json",
    }

    # Should list nested json files in dir1
    assert set(glob_directory(f"{tmp_path}/dir1/**/file*.json")) == {
        f"{tmp_path}/dir1/file3.json",
        f"{tmp_path}/dir1/subdir1/file4.json",
    }


def _run_remote_functionality(tmp_path, remote_dir):
    (tmp_path / "file1.json").touch()
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir1" / "file2.json").touch()

    assert not file_exists(f"{remote_dir}/dir1/file2.json")

    for path in tmp_path.glob("**/*"):
        if not path.is_file():
            continue
        rel_path = path.relative_to(tmp_path)
        upload(path, f"{remote_dir}/{rel_path}")
        assert file_exists(f"{remote_dir}/{rel_path}")

    # Should only list immediate children (files and dirs), but not files in subdirs.
    # The paths returned should be full paths.
    assert set(list_directory(remote_dir)) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1",
    }

    # Should list all children.
    assert set(list_directory(remote_dir, recurse=True)) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1",
        f"{remote_dir}/dir1/file2.json",
    }

    # Should list top-level json files.
    assert set(glob_directory(f"{remote_dir}/*.json")) == {
        f"{remote_dir}/file1.json",
    }

    # Should list all json files.
    assert set(glob_directory(f"{remote_dir}/**/*.json")) == {
        f"{remote_dir}/file1.json",
        f"{remote_dir}/dir1/file2.json",
    }

    # Should list nested json file
    assert set(glob_directory(f"{remote_dir}/dir1/file*.json")) == {
        f"{remote_dir}/dir1/file2.json",
    }

    # Try copying to a file that already exists.
    with pytest.raises(FileExistsError):
        copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir1/file2.json")
    copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir1/file2.json", save_overwrite=True)

    # Copy to a new file that doesn't exist.
    copy_file(f"{remote_dir}/dir1/file2.json", tmp_path / "dir2/file2.json")
    assert (tmp_path / "dir2/file2.json").is_file()

    # Copy dir.
    copy_dir(f"{remote_dir}", tmp_path / "dir3")
    assert (tmp_path / "dir3/dir1/file2.json").is_file()

    # Remove a file from the remote dir.
    remove_file(f"{remote_dir}/file1.json")
    assert set(list_directory(remote_dir, recurse=True)) == {
        f"{remote_dir}/dir1",
        f"{remote_dir}/dir1/file2.json",
    }


def test_s3_functionality(tmp_path, s3_checkpoint_dir):
    from botocore.exceptions import NoCredentialsError

    try:
        _run_remote_functionality(tmp_path, s3_checkpoint_dir)
    except NoCredentialsError:
        pytest.skip("Requires AWS credentials")


def test_gcs_functionality(tmp_path, gcs_checkpoint_dir):
    from google.auth.exceptions import DefaultCredentialsError

    try:
        _run_remote_functionality(tmp_path, gcs_checkpoint_dir)
    except DefaultCredentialsError:
        pytest.skip("Requires authentication with Google Cloud")


def test_glob_directory():
    assert set(glob("*.md")) == set(glob_directory("*.md"))
    assert set(glob("src/examples/**/*.py", recursive=True)) == set(
        glob_directory("src/examples/**/*.py")
    )


def test_http_get_bytes_range_retries_cloudflare_524(monkeypatch):
    class FakeResponse:
        def __init__(self, *, status_code: int, content: bytes = b"", text: str = ""):
            self.status_code = status_code
            self.content = content
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(
                    f"{self.status_code} error",
                    response=self,
                )

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, headers):
            del url, headers
            self.calls += 1
            if self.calls == 1:
                return FakeResponse(status_code=524, text="timeout")
            return FakeResponse(status_code=206, content=b"abcd")

    fake_session = FakeSession()

    monkeypatch.setattr("olmo_core.io._get_http_session", lambda: fake_session)
    monkeypatch.setattr("olmo_core.io._wait_before_retry", lambda attempt: None)

    assert _http_get_bytes_range("https://example.com/file.npy", 0, 4) == b"abcd"
    assert fake_session.calls == 2
