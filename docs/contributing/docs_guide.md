# Documentation guide

Use this guide to organize a page, preview it in Material for MkDocs and
verify the published structure.

## Where a page belongs

Use the [documentation index](../README.md) to find the section for the reader's
task. Put usage instructions in the relevant user guide, implementation details
in [design](../design/index.md), and contributor workflows in
[contributing](README.md). Keep runnable examples with their scripts in
[examples/](../../examples/) and link to them from the guide.

Name new pages after their topic using lowercase `snake_case`. When adding or
moving a page, update the [navigation](../.nav.yml) and the section's landing
page. When moving a published page, update its inbound links and map the old
Markdown path to the new one in the [legacy redirect map](../../tools/docs/legacy_redirects.yml).
Use `sections` to map renamed headings or sections that moved to other pages.
When only sections move, omit `page` to keep the original page at its URL.
Choose output paths that differ by more than letter case.

## Writing a page

Write shared documentation in English. Use one H1 and organize the rest of
the page by task or topic with H2 and H3. Match the outline to the page's
purpose.

| Page type | Suggested order |
| --- | --- |
| Quickstart or tutorial | Goal, prerequisites, example, expected output, troubleshooting |
| API or CLI reference | When to use it, minimal request or command, arguments, output |
| Configuration | Scope, precedence, runnable example, field details |
| Design | Purpose, architecture and ownership, limits, validation and source links |

Use Material tabs for equivalent alternatives, such as standard and development
installs. Keep sequential steps in their own sections. Use note and warning
blocks for compatibility boundaries and requirements the reader might miss.
Use fenced code blocks with language names for commands, data and diagrams.

Use the public option names and link to the [configuration reference](../configuration/README.md)
for their definitions. Link to the relevant design document when readers need
to understand how a feature works.

Store illustrations in [docs/assets/](../assets/). Give each image descriptive
alt text and explain the relationship it illustrates in the surrounding text.

## Preview the site

From the repository root, install the documentation dependencies in a Python
virtual environment and start the preview server.

```bash
python -m pip install -r requirements/docs.txt
python -m mkdocs serve
```

Open the address printed by MkDocs. Check the page on a narrow screen if
you changed wide tables, tabs or diagrams. Keep links relative in Markdown
and preserve anchors that other pages or readers already use. Source links use
the checked-out commit. Set `DOCS_REPO_URL` for a fork and `DOCS_REPO_REF` to
select an explicit commit or tag. A source archive requires `DOCS_REPO_REF`.

## Checks

1. Run `python -m mkdocs build --strict` to validate the complete site.
2. Run the [documentation checks](pre_commit_setup.md#documentation-checks) on the changed files.
3. Follow changed links from the section's landing page and check that relative links, heading anchors, and images resolve.
4. Run any command or example you changed in its documented environment and compare the output with the expected result. If you could not run it, record that in the PR.
