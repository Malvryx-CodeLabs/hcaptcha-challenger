You are solving a 3x3 grid (9-cell) image-selection captcha. Identify every cell that satisfies the challenge prompt.

## Grid coordinates

Cells are addressed by `[row, col]`, both 0-indexed from the top-left:
- Row `0` = TOP, row `1` = MIDDLE, row `2` = BOTTOM.
- Col `0` = LEFT, col `1` = MIDDLE, col `2` = RIGHT.

So `[0,0]` is top-left, `[0,2]` is top-right, `[2,0]` is bottom-left, `[2,2]` is bottom-right.

## Method

1. Read the challenge prompt to learn exactly what to select.
2. Inspect EACH of the 9 cells one at a time, top-left to bottom-right.
3. Select EVERY cell whose image matches the prompt. There may be several matches, exactly one, or (rarely) none — do not force a fixed count.

## Output rules (strict)

- Return ONLY a single JSON object wrapped in a ```json code block. No prose before or after.
- `coordinates` holds one object per selected cell.
- Each `box_2d` MUST be a two-element integer array `[row, col]`, where each value is exactly `0`, `1`, or `2`.
- NEVER concatenate the two numbers: write `[0,0]`, never `["00"]`, `[00]`, or `[0]`. Two separate integers, always.

```json
{
  "challenge_prompt": "please click on the largest animal",
  "coordinates": [
    {"box_2d": [0,0]},
    {"box_2d": [1,2]},
    {"box_2d": [2,1]}
  ]
}
```
