#!/usr/bin/env sh
# Assemble the static site for GitHub Pages into $1 (default _site).
#
# app.html is a fragment: the artifact host supplies the doctype, charset and
# viewport around it. Served bare, a browser renders it in quirks mode with no
# viewport meta, which breaks the phone layout. So wrap it the way the artifact
# host does (the same skeleton tools/uitest reproduces) before publishing.
set -eu
out="${1:-_site}"
here="$(dirname "$0")"
mkdir -p "$out"
{
cat <<'HTML'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Predicts a steel's yield strength, tensile strength and elongation from its chemistry, and pulls a 3D bar to failure with them.">
<meta property="og:title" content="Live Demo: Interactive Tensile Test Bench">
<meta property="og:description" content="Predicts a steel's yield strength, tensile strength and elongation from its chemistry, and pulls a 3D bar to failure with them.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://daviddruker.github.io/stress-strain-predictor/">
<meta property="og:image" content="https://daviddruker.github.io/stress-strain-predictor/image.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://daviddruker.github.io/stress-strain-predictor/image.png">
<style>
:root { padding-top: env(safe-area-inset-top, 0px);
padding-bottom: env(safe-area-inset-bottom, 0px); }
body { margin: 0; }
img { max-width: 100%; }
[hidden] { display: none !important; }
</style>
</head>
<body>
HTML
cat "$here/app.html"
printf '\n</body>\n</html>\n'
} > "$out/index.html"
cp "$here/model.js" "$out/model.js"
cp "$here/image.png" "$out/image.png"
