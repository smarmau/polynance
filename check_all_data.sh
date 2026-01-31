#!/bin/bash
echo "=== Window Count Per Asset ==="
for asset in btc eth sol xrp; do
    count=$(sqlite3 "data/${asset}.db" "SELECT COUNT(*) FROM windows" 2>/dev/null)
    echo "$asset: $count windows"
done

echo ""
echo "=== Recent Windows ==="
for asset in btc eth sol xrp; do
    echo "--- $asset ---"
    sqlite3 "data/${asset}.db" "SELECT window_id, outcome, spot_change_bps FROM windows ORDER BY window_start_utc DESC LIMIT 3" 2>/dev/null
done
