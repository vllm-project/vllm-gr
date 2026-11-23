# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Resolve links to files outside the documentation on GitHub during site builds."""

import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from mkdocs.exceptions import ConfigurationError

log = logging.getLogger("mkdocs.plugins.repository_links")


class RepositoryLinkProcessor(Treeprocessor):
    def __init__(self, md, source, docs, repo_url, repo_ref):
        super().__init__(md)
        self.source = source
        self.docs = docs
        self.repo_url = repo_url
        self.repo_ref = repo_ref

    def run(self, root):
        for link in root.iter("a"):
            href = link.get("href", "")
            parts = urlsplit(href)
            if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
                continue
            path = (self.source.parent / unquote(parts.path)).resolve()
            if path.is_relative_to(self.docs) and path.is_file() and not path.name.startswith("."):
                continue
            if not path.is_relative_to(self.docs.parent) or not path.exists():
                log.warning("Missing repository target %r in %s", href, self.source)
                continue
            kind = "tree" if path.is_dir() else "blob"
            relative = quote(path.relative_to(self.docs.parent).as_posix())
            url = urlsplit(f"{self.repo_url}/{kind}/{self.repo_ref}/{relative}")
            link.set("href", urlunsplit(url._replace(query=parts.query, fragment=parts.fragment)))


class RepositoryLinks(Extension):
    def extendMarkdown(self, md):
        processor = RepositoryLinkProcessor(
            md, self.source, self.docs, self.repo_url, self.repo_ref
        )
        # Run after Markdown creates links and before MkDocs rewrites site URLs.
        md.treeprocessors.register(processor, "repository_links", 2)


links = RepositoryLinks()


def on_config(config):
    links.docs = Path(config.docs_dir).resolve()
    links.repo_url = config.repo_url.rstrip("/")
    repo_ref = os.environ.get("DOCS_REPO_REF", "").strip()
    if not repo_ref:
        try:
            repo_ref = subprocess.check_output(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=links.docs.parent,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ConfigurationError(
                "Set DOCS_REPO_REF when building outside a Git checkout."
            ) from error
    links.repo_ref = quote(repo_ref)
    config.markdown_extensions.append(links)


def on_page_markdown(markdown, page, **kwargs):
    links.source = Path(page.file.abs_src_path)
