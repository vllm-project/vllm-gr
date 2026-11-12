# Documentation guide

Documentation is maintained as Markdown and can be read on GitHub or built as
a searchable site. The site uses Material for MkDocs with user and developer
navigation, code copying, light/dark themes and local assets.

## Build and preview

Use a separate Python environment; the documentation build does not import
vLLM, load model weights or require an accelerator.

```bash
python -m venv /tmp/vllm-gr-docs
/tmp/vllm-gr-docs/bin/python -m pip install -r requirements/docs.txt
/tmp/vllm-gr-docs/bin/python -m mkdocs build --strict --site-dir /tmp/vllm-gr-site
/tmp/vllm-gr-docs/bin/python -m mkdocs serve
```

Run these commands from the repository root. The preview server uses
`http://127.0.0.1:8000/`. For a normal local build without `--site-dir`, MkDocs
writes `site/`; generated site output must not be committed. Building a site
does not publish it or configure hosting.

## Write for the reader's task

- Start a user guide with its goal, prerequisites, smallest working example
  and expected result. Put advanced tuning and failure diagnosis after that path.
- Use canonical configuration keys from the current source. Link to the owning
  reference rather than duplicating a long option list.
- Distinguish model capabilities, runtime support and validation evidence.
  Keep historical measurements tied to their recorded environment and revision.
- Keep internal ownership and wire contracts in design or feature sections.
  The shared architecture belongs in [Architecture](../design/main-arch.md).
- Add pages to the explicit navigation in `mkdocs.yml`. Every page needs a
  descriptive title, usable relative links and a sensible next step.

## Diagrams and assets

The illustrations in `docs/assets/` explain architecture, SID expansion, graph
storage lifetime and resource updates. They are authored for this project;
their concepts must agree with the linked contracts and current source.

Use descriptive alt text and explain the important relationships in adjacent
text so the image is not the only source of information. Keep labels short and
legible at article width. A conceptual drawing is not a benchmark result.

The four documentation illustrations were produced with the built-in image
generation tool and reviewed against the GR contracts. The complete generation
prompts are in [illustration prompts](../assets/illustration-prompts.txt).
The existing project logo is reused. New illustrations do not copy Omni assets.

## Check a change

Run the [documentation hooks](PRE_COMMIT.md#documentation-checks), then build
with `mkdocs build --strict`. Review the rendered home page, navigation, images
and at least one long reference on desktop and narrow screens. Verify linked
commands and test nodes against the checkout; a clean Markdown lint run does
not establish that an inference command works on hardware.

The navigation and presentation take inspiration from
[vLLM-Omni](https://github.com/vllm-project/vllm-omni/tree/main/docs).
vLLM-GR documentation describes its own runtime and supported workloads.
