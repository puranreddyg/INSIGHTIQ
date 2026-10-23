import sqlite3
from backend.database import connect
from backend.agents.coordinator_agent import run_pipeline

def main():
    with connect() as conn:
        print("Running SKU_44 first time...")
        res1 = run_pipeline(conn, "SKU_44")
        
        print("Running SKU_19...")
        res2 = run_pipeline(conn, "SKU_19")
        
        print("Running SKU_44 second time...")
        res3 = run_pipeline(conn, "SKU_44")
        
        print("Are results 1 and 3 equal (ignoring timestamps)?")
        
        # Strip timestamps
        res1.pop("started_at")
        res1.pop("completed_at")
        for log in res1["stage_log"]:
            log.pop("duration_ms")
            
        res3.pop("started_at")
        res3.pop("completed_at")
        for log in res3["stage_log"]:
            log.pop("duration_ms")
            
        if res1 == res3:
            print("YES - Complete Isolation Confirmed")
        else:
            print("NO - State leakage detected")
            
if __name__ == "__main__":
    main()
