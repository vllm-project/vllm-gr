#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared helpers for benchmark dataset download and extraction."""

from __future__ import annotations

import abc
import logging
from pathlib import Path
import ssl
import tarfile
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm.auto import tqdm


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class DataProcessor(abc.ABC):
    """Base class for downloading and preprocessing benchmark data."""

    def __init__(
        self,
        download_url: str,
        output_root: Path,
        archive_name: str,
        prefix: str,
        insecure: bool = False,
    ) -> None:
        self.download_url = download_url
        self.output_root = output_root
        self.archive_path = output_root / "tmp" / archive_name
        self.extract_root = output_root / "tmp"
        self.dataset_root = self.extract_root / prefix
        self.prefix = prefix
        self.insecure = insecure
        self.log = logging.getLogger(self.__class__.__module__)
        _ensure_dir(self.archive_path.parent)

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        if not self.insecure:
            return None
        return ssl._create_unverified_context()

    def download(self) -> None:
        if self.archive_path.exists():
            return
        if self.insecure:
            self.log.warning(
                "Downloading %s with SSL verification disabled", self.download_url
            )
        else:
            self.log.info("Downloading %s", self.download_url)

        progress_bar = tqdm(
            total=None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {self.archive_path.name}",
        )
        try:
            with urlopen(
                self.download_url,
                context=self._build_ssl_context(),
            ) as response:
                total_size = response.headers.get("Content-Length")
                if total_size is not None:
                    progress_bar.total = int(total_size)

                with self.archive_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        progress_bar.update(len(chunk))
        finally:
            progress_bar.close()
        self.log.info("Downloaded to %s", self.archive_path)

    def extract(self) -> None:
        if self.dataset_root.exists():
            return
        if self.archive_path.exists():
            self.log.info("Extracting dataset from local archive %s", self.archive_path)
        else:
            self.log.info("Local archive %s not found; downloading dataset", self.archive_path)
            self.download()
        if self.archive_path.suffix == ".zip":
            with ZipFile(self.archive_path, "r") as archive:
                archive.extractall(path=self.extract_root)
        else:
            with tarfile.open(self.archive_path, "r:*") as archive:
                archive.extractall(path=self.extract_root)
        self.log.info("Extracted dataset into %s", self.extract_root)

    @abc.abstractmethod
    def preprocess(self) -> None:
        raise NotImplementedError
