import sqlite3
import json
from backend.database import connect
from backend.agents.coordinator_agent import run_pipeline
from backend.agents.perception_agent import urgent_requests, observe_all
import sys

def main():
    with connect() as conn:
        requests = observe_all(conn)
        urgent = urgent_requests(requests)
        urgent_skus = [r['sku_id'] for r in urgent][:10]
        non_urgent_skus = [k for k, v in requests.items() if not v['Urgent']][:2]
        
        skus_to_test = urgent_skus + non_urgent_skus
        
        results = {}
        for sku in skus_to_test:
            print(f"Running pipeline for {sku}...")
            res = run_pipeline(conn, sku)
            results[sku] = res
            
    with open("pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Saved pipeline results to pipeline_results.json")
    print(f"Tested SKUs: {skus_to_test}")

if __name__ == "__main__":
    main()
