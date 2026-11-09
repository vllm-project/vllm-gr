# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from pathlib import Path

import pytest

from vllm_gr.sidecar import HDFSDownloader


@pytest.fixture  # type: ignore[misc]
def hdfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[HDFSDownloader, Path, Path]:
    binary = tmp_path / "hdfs"
    remote = tmp_path / "remote"
    remote.mkdir()
    log = tmp_path / "calls.jsonl"
    binary.write_text(
        f"#!{sys.executable}\n"
        + """
import json,os,pathlib,shutil,sys,time
args=sys.argv[1:]
with open(os.environ['HDFS_TEST_LOG'],'a') as f:f.write(json.dumps(args)+'\\n')
if os.environ.get('HDFS_TEST_SLEEP'):time.sleep(5)
if os.environ.get('HDFS_TEST_NOTICE'):print('WARNING: skip config refresh: must run as tiger user',file=sys.stderr)
if os.environ.get('HDFS_TEST_DENIED'):
 print('Permission denied',file=sys.stderr);sys.exit(1)
if args[0]=='dfs':args=args[1:]
command=args[0].lstrip('-');args=args[1:]
root=pathlib.Path(os.environ['HDFS_TEST_REMOTE'])
def remote(uri):return root/uri.removeprefix('hdfs://cluster/')
if command=='ls':
 for p in sorted(remote(args[0]).iterdir()):
  print(('drwxr-xr-x' if p.is_dir() else '-rw-r--r--')+' 1 user group 0 2026-11-03 00:00 '+args[0].rstrip('/')+'/'+p.name)
elif command=='test':sys.exit(0 if remote(args[-1]).exists() else 1)
elif command=='get':shutil.copytree(remote(args[-2]),args[-1])
"""
    )
    binary.chmod(0o755)
    monkeypatch.setenv("HDFS_TEST_REMOTE", str(remote))
    monkeypatch.setenv("HDFS_TEST_LOG", str(log))
    return HDFSDownloader(str(binary)), remote, log
