#!/usr/bin/env bash
# Check that every relative link in a Markdown file points at a file that
# exists.
#
# The documentation in this repository carries the results, and every result
# names the command behind it. A link into docs/ that has gone stale takes a
# reader away from the evidence for a number, which is the one thing the
# documentation is for.
#
# The script tests relative links only. It opens no network connection, so it
# never fails because a remote site is slow or gone.

set -uo pipefail

fail=0
checked=0

while IFS= read -r file; do
  # Take the target out of every ](...) link. Drop the ones that point at a
  # URL, at a mail address, or at an anchor on the same page.
  while IFS= read -r target; do
    [ -z "$target" ] && continue
    case "$target" in
      \#* | http:* | https:* | mailto:* | //*) continue ;;
    esac

    # A link may carry an anchor, such as file.md#section. Only the part
    # before the hash names a file.
    path="${target%%#*}"
    [ -z "$path" ] && continue

    resolved="$(dirname "$file")/$path"
    checked=$((checked + 1))

    if [ ! -e "$resolved" ]; then
      echo "::error file=${file#./}::Broken link to '$target'"
      fail=1
    fi
  done < <(grep -oE '\]\([^)]+\)' "$file" | sed -E 's/^\]\(//; s/\)$//; s/ .*$//')
done < <(find . -name '*.md' -not -path './.git/*' | sort)

echo "Checked $checked relative links in Markdown files."
if [ "$fail" -eq 0 ]; then
  echo "Every one of them points at a file that exists."
fi

exit "$fail"
