# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Write legacy page and section redirects after MkDocs renders the site."""

import json
from html import escape
from html.parser import HTMLParser
from pathlib import Path

import yaml
from mkdocs.exceptions import ConfigurationError
from mkdocs.structure.files import File

SCRIPT_START = '<script id="legacy-section-redirects">'


class PageIds(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.feed(Path(path).read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        self.ids.update(value for name, value in attrs if name == "id" and value)


def on_files(files, config, **kwargs):
    global redirects, pages
    pages = {file.src_uri: file for file in files.documentation_pages()}
    redirects = yaml.safe_load(Path(__file__).with_suffix(".yml").read_text(encoding="utf-8"))
    outputs = {file.dest_uri.casefold() for file in files}
    for source, spec in redirects.items():
        if "page" not in spec:
            if source not in pages:
                raise ConfigurationError(f"Legacy section source {source!r} is missing.")
            continue
        redirect = File(source, config.docs_dir, config.site_dir, config.use_directory_urls)
        output = redirect.dest_uri.casefold()
        if output in outputs:
            raise ConfigurationError(f"Legacy redirect {source!r} collides with a site file.")
        outputs.add(output)


def on_post_build(config, **kwargs):
    ids = {}
    for source, spec in redirects.items():
        redirect = File(source, config.docs_dir, config.site_dir, config.use_directory_urls)
        default = spec.get("page", source)
        if default not in pages:
            raise ConfigurationError(
                f"Legacy redirect {source!r} points to missing page {default!r}."
            )
        default_page = pages[default]
        default_url = default_page.url_relative_to(redirect)
        if default not in ids:
            ids[default] = PageIds(default_page.abs_dest_path).ids
        moved = "page" in spec
        sections = {anchor: f"{default_url}#{anchor}" for anchor in ids[default]} if moved else {}
        mapped = set()
        for target, anchors in spec.get("sections", {}).items():
            page, _, anchor = target.partition("#")
            if page not in pages:
                raise ConfigurationError(
                    f"Legacy redirect {source!r} points to missing page {page!r}."
                )
            target_page = pages[page]
            if page not in ids:
                ids[page] = PageIds(target_page.abs_dest_path).ids
            if anchor and anchor not in ids[page]:
                raise ConfigurationError(
                    f"Legacy redirect {source!r} points to missing section {target!r}."
                )
            url = target_page.url_relative_to(redirect)
            for old in anchors:
                if old in mapped:
                    raise ConfigurationError(f"Legacy redirect {source!r} repeats section {old!r}.")
                if not moved and old in ids[source]:
                    raise ConfigurationError(
                        f"Legacy alias {source!r}#{old} shadows a current section."
                    )
                mapped.add(old)
                sections[old] = f"{url}#{anchor}" if anchor else url

        urls = json.dumps(sections, sort_keys=True).replace("<", "\\u003c")
        fallback = json.dumps(default_url).replace("<", "\\u003c")
        action = f"else location.replace({fallback});" if moved else ""
        script = f"""{SCRIPT_START}
  (() => {{
    const sections = {urls};
    const followLegacySection = () => {{
      let fragment = location.hash.slice(1);
      try {{ fragment = decodeURIComponent(fragment); }} catch {{}}
      if (Object.prototype.hasOwnProperty.call(sections, fragment)) location.replace(sections[fragment]);
      {action}
    }};
    followLegacySection();
    {"" if moved else 'window.addEventListener("hashchange", followLegacySection);'}
  }})();
</script>"""
        output = Path(redirect.abs_dest_path)
        if not moved:
            page_html = output.read_text(encoding="utf-8")
            before, marker, after = page_html.partition(SCRIPT_START)
            if marker:
                # A dirty build may reuse the HTML from the previous build.
                page_html = before + after.partition("</script>\n")[2]
            page_html = page_html.replace("</head>", f"{script}\n</head>", 1)
            output.write_text(page_html, encoding="utf-8")
            continue

        destination = escape(default_url, quote=True)
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Redirecting</title>
  <link rel="canonical" href="{destination}">
  <noscript><meta http-equiv="refresh" content="0; url={destination}"></noscript>
  {script}
</head>
<body><p>This page has moved. <a href="{destination}">Open the current page</a>.</p></body>
</html>
"""
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
