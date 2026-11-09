"""Génère results.jsonl (fixture_id + result H/D/A) pour les matchs terminés.

Usage :
  FOOTBALL_DATA_TOKEN=... python audit/fetch_audit_results.py \
      data/prospective_forecasts.jsonl audit/results.jsonl
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.football-data.org/v4/matches/{}"
WINNER = {"HOME_TEAM": "H", "DRAW": "D", "AWAY_TEAM": "A"}


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch(provider_id, token):
    req = urllib.request.Request(API.format(provider_id),
                                 headers={"X-Auth-Token": token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main(forecasts_path, results_path):
    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        sys.exit("Définissez FOOTBALL_DATA_TOKEN (clé football-data.org).")
    done = {r["fixture_id"] for r in read_jsonl(results_path)}
    todo = sorted({f["fixture_id"] for f in read_jsonl(forecasts_path)} - done)
    print(f"{len(todo)} match(s) à vérifier")
    with open(results_path, "a", encoding="utf-8") as out:
        for fid in todo:
            pid = fid.split(":", 1)[1]
            try:
                m = fetch(pid, token)
            except urllib.error.HTTPError as e:
                print(f"  {fid}: erreur HTTP {e.code}")
                time.sleep(7)
                continue
            winner = (m.get("score") or {}).get("winner")
            if m.get("status") == "FINISHED" and winner in WINNER:
                out.write(json.dumps({
                    "fixture_id": fid,
                    "result": WINNER[winner],
                    "home_goals": m["score"]["fullTime"]["home"],
                    "away_goals": m["score"]["fullTime"]["away"],
                    "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime()),
                }, ensure_ascii=False) + "\n")
                print(f"  {fid}: {WINNER[winner]}")
            else:
                print(f"  {fid}: pas terminé ({m.get('status')})")
            time.sleep(7)  # 10 requêtes/minute max


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
