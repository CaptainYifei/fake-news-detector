# README GIF Preview Design

## Goal

Replace the outdated README screenshot with a current project overview and a
compact animated preview that communicates the three-node fact-checking flow.

## Presentation

The README will use two complementary assets:

1. `docs/images/app-overview.png` provides a clear static overview near the top
   of the README.
2. `docs/images/fact-check-workflow.gif` appears in the workflow section and
   cycles through claim extraction, evidence retrieval, and claim verification.

The static image remains useful when animation is disabled or slow to load.

## GIF Construction

- Use the three supplied screenshots as the source frames.
- Normalize every GIF frame to 1600 x 886 pixels, preserving the source aspect
  ratio without cropping.
- Preserve the full application layout so the sidebar, active model, search
  engine, and node output remain visible.
- Hold each frame for 2.5 seconds.
- Loop indefinitely without dense transition frames, keeping text readable and
  the file size below GitHub's 10 MB image limit.
- Optimize the color palette after resizing while avoiding visible text damage.

## README Changes

- Replace the old `docs/images/screenshot.png` reference with the static current
  overview.
- Add a short "Project Preview" section that explains the three animated stages.
- Embed the GIF with meaningful alt text and a direct relative path.
- Leave installation, configuration, and implementation documentation unchanged.

## Verification

- Confirm both media files exist and have stable dimensions.
- Confirm the GIF has three frames, loops, and stays below 10 MB.
- Inspect the first, middle, and final GIF frames for readable text and correct
  ordering.
- Run `git diff --check` and verify every README image path resolves locally.
