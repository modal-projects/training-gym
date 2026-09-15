"""Hand ratings over the rendered corpus.

Every `pos*` render was looked at on a contact sheet and sorted into two tiers.
`LOVE` is what the reward should chase: saturated pigment, a whole plant with
stem and leaves, a balanced composition. Everything else in the positive set is
"okay" — a real flower, but pale, headless, or lopsided. The negatives are the
deliberate failure modes and are the probe's zero class.
"""

LOVE = [
    1, 2, 3, 5, 6, 12, 13, 14, 17, 18, 19, 21, 22, 25, 26, 27, 28, 29, 31, 35,
    36, 39, 41, 42, 46, 47,
    48, 49, 50, 51, 52, 54, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 70, 72,
    73, 75, 76, 77, 78, 81, 82, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94,
    97, 98, 100, 101, 102, 105, 106, 109, 110, 113, 114, 117, 118, 120, 121,
    122, 123, 124, 125, 127, 129, 130, 132, 133, 134, 137, 138, 140, 141, 142,
    144, 145, 147, 149, 150, 153, 154, 155, 156, 157, 158, 161, 162, 163, 165,
    166, 168, 169, 170, 171, 172, 173, 174, 177, 180, 181, 182, 185, 186, 190,
]
