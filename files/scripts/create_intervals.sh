
# Script to create WGS intervals list from a reference fasta
# Usage: ./create_wgs_intervals.sh /path/to/reference.fasta
#
# # Exit if any command fails

set -e

# Check input
if [ $# -ne 1 ]; then
  echo "Usage: $0 /path/to/reference.fasta"
    exit 1
    fi

    REFERENCE=$1

    # Derive filenames
    BASE=$(basename "$REFERENCE")
    PREFIX="${BASE%%.*}"

    DICT="${PREFIX}.dict"
    INTERVALS="${PREFIX}_wgs.interval_list"

    echo "==> Creating sequence dictionary..."
    picard CreateSequenceDictionary R="$REFERENCE" O="$DICT"

    echo "==> Generating intervals..."
    awk '/^@SQ/ {
      split($2, sn, ":")
        split($3, ln, ":")
         print sn[2] "\t1\t" ln[2] "\t+\t."
          }' "$DICT" > intervals_body.txt

          echo "==> Combining header and intervals..."
          cat "$DICT" intervals_body.txt > "$INTERVALS"

          echo "==> Cleaning up..."
          rm intervals_body.txt
          rm -f "$DICT"

          echo "✅ Done!"
          echo "Created interval list: $INTERVALS"



echo "==> Creating Y chromosome interval list..."
awk '/^@/ || $1=="chrY"' GCA_000001405_wgs.interval_list > GCA_000001405_chrY.interval_list
echo "✅ Created interval list for chrY: $Y_INTERVALS"