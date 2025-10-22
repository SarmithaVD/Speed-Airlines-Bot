# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 18:58:49 2025

@author: keertanapriya
"""

# store_to_redis_windows.py
import os
import argparse
import pandas as pd
import redis
import sys

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Push flights and passengers CSVs to Redis")
parser.add_argument("--redis-host", default="localhost", help="Redis host")
parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
parser.add_argument("--db", type=int, default=0, help="Redis DB")
parser.add_argument("--folder", default=".", help="Folder containing flights.csv and passengers.csv")
args = parser.parse_args()

# --- Paths for CSV files ---
flights_csv = os.path.join(args.folder, "flights.csv")
passengers_csv = os.path.join(args.folder, "passengers.csv")

# Check files exist
if not os.path.exists(flights_csv):
    print(f"Flights CSV not found at {flights_csv}")
    sys.exit(1)
if not os.path.exists(passengers_csv):
    print(f"Passengers CSV not found at {passengers_csv}")
    sys.exit(1)

# --- Load CSVs ---
flights = pd.read_csv(flights_csv)
passengers = pd.read_csv(passengers_csv)

# --- Connect to Redis ---
try:
    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.db)
    if not r.ping():
        raise Exception("Ping failed")
except Exception as e:
    print("Could not connect to Redis:", e)
    sys.exit(1)

print("Connected to Redis at {}:{}".format(args.redis_host, args.redis_port))

# --- Push flights ---
for _, row in flights.iterrows():
    key = f"flight:{row['flight_no']}"
    for field, value in row.items():
        r.hset(key, field, str(value))
print(f"Pushed {len(flights)} flights to Redis")

# --- Push passengers ---
for _, row in passengers.iterrows():
    key = f"passenger:{row['passenger_id']}"
    for field, value in row.items():
        r.hset(key, field, str(value))
print(f"Pushed {len(passengers)} passengers to Redis")

print("All data pushed successfully!")
