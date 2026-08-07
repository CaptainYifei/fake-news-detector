# Hand-drawn Research Workflow Design

## Goal

Add a hand-drawn research workflow figure to the README that explains the
project's three-node fact-checking method through one compact example.

## Placement And Asset

- Save the final bitmap as `docs/images/research-workflow-handdrawn.png`.
- Place it in the README project-preview section after the animated workflow
  GIF and before the core-features section.
- Use a wide 16:9 composition suitable for GitHub's README content column.

## Visual Direction

- White paper background with subtle notebook or graph-paper texture.
- Loose hand-drawn ink lines, slightly irregular arrows, and small research
  annotations without visual clutter.
- Three horizontal nodes use distinct accents: amber for extraction, blue for
  retrieval, and green for verification.
- Use simple hand-drawn icons: newspaper, magnifying glass with documents, and
  verification checklist.
- Keep high contrast, generous margins, and readable text at README scale.

## Information Architecture

Title: `AI虚假新闻三节点核查流程`

Input note: `输入新闻：“西班牙1:0阿根廷夺冠”`

Node 1:

- `01 新闻中心声明抽取`
- `LLM语义理解`
- `输出声明：西班牙击败阿根廷`

Node 2:

- `02 查询改写与证据检索`
- `DuckDuckGo + BGE-M3`
- `查询：2026世界杯决赛结果`
- `Top K证据`

Node 3:

- `03 多步分解声明验证`
- `原子声明 + 5W`
- `SUPPORTED`
- `结论：正确`

## Generation And Quality Gates

- Generate the artwork with the image-generation tool using the approved A1
  layout and the exact content above.
- Reject outputs with incorrect Chinese characters, missing nodes, reversed
  arrows, illegible labels, or cropped content.
- Preserve the three-node left-to-right sequence and keep the example within
  the corresponding node.
- Verify the final image visually at full size and at 900 pixels wide, matching
  a typical GitHub README content width.
- Run `git diff --check` and confirm the README image path resolves locally.
