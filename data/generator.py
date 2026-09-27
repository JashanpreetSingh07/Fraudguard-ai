import os
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class FinancialTransactionGenerator:
    """
    Generates realistic synthetic banking & e-commerce transaction data
    infused with subtle, enterprise-grade fraud typologies:
    1. Card-Not-Present (CNP) Velocity Burst & Card Testing
    2. Account Takeover (ATO) with Device/IP Hops & Abnormal Hours
    3. Impossible Travel (Geographic anomalies exceeding flight speed)
    4. AML Structuring / Smurfing (<$10,000 CTR evasion)
    5. Mule Network Rings & Collusion (Bipartite graph cycles)
    """

    CITIES = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060, "country": "US"},
        {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "country": "US"},
        {"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "country": "US"},
        {"name": "Houston", "lat": 29.7604, "lon": -95.3698, "country": "US"},
        {"name": "Miami", "lat": 25.7617, "lon": -80.1918, "country": "US"},
        {"name": "London", "lat": 51.5074, "lon": -0.1278, "country": "GB"},
        {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "country": "FR"},
        {"name": "Frankfurt", "lat": 50.1109, "lon": 8.6821, "country": "DE"},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "country": "JP"},
        {"name": "Singapore", "lat": 1.3521, "lon": 103.8198, "country": "SG"},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "country": "AU"},
        {"name": "Toronto", "lat": 43.6532, "lon": -79.3832, "country": "CA"},
    ]

    MERCHANT_CATEGORIES = [
        {"mcc": "5411", "category": "Grocery Stores", "risk_weight": 0.05, "avg_amount": 75.0, "std_amount": 35.0},
        {"mcc": "5812", "category": "Restaurants & Dining", "risk_weight": 0.08, "avg_amount": 45.0, "std_amount": 25.0},
        {"mcc": "5541", "category": "Gas & Fuel Stations", "risk_weight": 0.12, "avg_amount": 40.0, "std_amount": 15.0},
        {"mcc": "5311", "category": "Department Stores", "risk_weight": 0.15, "avg_amount": 120.0, "std_amount": 60.0},
        {"mcc": "5732", "category": "Electronics & Gadgets", "risk_weight": 0.45, "avg_amount": 450.0, "std_amount": 300.0},
        {"mcc": "4511", "category": "Airlines & Travel", "risk_weight": 0.35, "avg_amount": 650.0, "std_amount": 400.0},
        {"mcc": "5944", "category": "Jewelry & Luxury Goods", "risk_weight": 0.55, "avg_amount": 1200.0, "std_amount": 800.0},
        {"mcc": "6051", "category": "Crypto & Money Orders", "risk_weight": 0.75, "avg_amount": 2500.0, "std_amount": 1500.0},
        {"mcc": "7995", "category": "Gaming & Casinos", "risk_weight": 0.65, "avg_amount": 300.0, "std_amount": 250.0},
        {"mcc": "4814", "category": "Telecommunications", "risk_weight": 0.10, "avg_amount": 85.0, "std_amount": 30.0},
    ]

    CHANNELS = ["pos", "web", "mobile", "wire_transfer"]

    def __init__(
        self,
        num_transactions: int = config.DEFAULT_NUM_TRANSACTIONS,
        num_users: int = config.DEFAULT_NUM_USERS,
        num_merchants: int = config.DEFAULT_NUM_MERCHANTS,
        fraud_ratio: float = 0.045,  # 4.5% realistic fraud rate
        random_seed: int = config.RANDOM_SEED,
    ):
        self.num_transactions = num_transactions
        self.num_users = num_users
        self.num_merchants = num_merchants
        self.fraud_ratio = fraud_ratio
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        self.users = self._create_user_profiles()
        self.merchants = self._create_merchant_profiles()

    def _create_user_profiles(self) -> List[Dict]:
        users = []
        for i in range(self.num_users):
            home_city = random.choice(self.CITIES)
            primary_device = f"DEV_{uuid.uuid4().hex[:10].upper()}"
            primary_ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
            avg_spend = max(15.0, float(np.random.lognormal(mean=3.8, sigma=0.65)))
            users.append({
                "user_id": f"USR_{i:05d}",
                "name": f"User_{i:05d}",
                "home_city": home_city["name"],
                "home_lat": home_city["lat"] + np.random.normal(0, 0.05),
                "home_lon": home_city["lon"] + np.random.normal(0, 0.05),
                "home_country": home_city["country"],
                "primary_device": primary_device,
                "primary_ip": primary_ip,
                "avg_spend": avg_spend,
                "account_age_days": random.randint(30, 1800),
                "card_id": f"CARD_{i:05d}",
            })
        return users

    def _create_merchant_profiles(self) -> List[Dict]:
        merchants = []
        for i in range(self.num_merchants):
            cat = random.choice(self.MERCHANT_CATEGORIES)
            city = random.choice(self.CITIES)
            merchants.append({
                "merchant_id": f"MERCH_{i:04d}",
                "merchant_name": f"{cat['category']} Inc #{i:03d}",
                "mcc": cat["mcc"],
                "category": cat["category"],
                "risk_weight": cat["risk_weight"],
                "avg_amount": cat["avg_amount"],
                "std_amount": cat["std_amount"],
                "city": city["name"],
                "lat": city["lat"] + np.random.normal(0, 0.03),
                "lon": city["lon"] + np.random.normal(0, 0.03),
                "country": city["country"],
            })
        return merchants

    def generate(self) -> pd.DataFrame:
        """Generates full transaction stream with injected fraud typologies."""
        total_fraud_count = int(self.num_transactions * self.fraud_ratio)
        legit_count = self.num_transactions - total_fraud_count

        start_time = datetime(2026, 8, 1, 0, 0, 0)
        end_time = datetime(2026, 9, 25, 23, 59, 59)
        time_span_seconds = int((end_time - start_time).total_seconds())

        records: List[Dict] = []

        # 1. Generate Legitimate Transactions
        for _ in range(legit_count):
            user = random.choice(self.users)
            merchant = random.choice(self.merchants)

            # Legitimate diurnal pattern: more during 8am - 10pm
            hour_prob = np.array([
                0.01, 0.005, 0.005, 0.005, 0.005, 0.01,  # 0-5
                0.02, 0.04, 0.06, 0.07, 0.08, 0.08,      # 6-11
                0.09, 0.08, 0.07, 0.07, 0.08, 0.08,      # 12-17
                0.07, 0.06, 0.05, 0.04, 0.02, 0.01       # 18-23
            ])
            hour_prob = hour_prob / hour_prob.sum()
            chosen_hour = np.random.choice(24, p=hour_prob)

            random_days = random.randint(0, 54)
            chosen_time = start_time + timedelta(
                days=random_days,
                hours=int(chosen_hour),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )

            # Channel & Device consistency
            channel = random.choices(["mobile", "web", "pos"], weights=[0.45, 0.35, 0.20])[0]
            device_id = user["primary_device"] if channel != "pos" else "POS_TERMINAL"
            ip_address = user["primary_ip"] if channel != "pos" else "POS_NETWORK"

            # Location: mostly within home city radius (92% of time)
            if random.random() < 0.92:
                lat = user["home_lat"] + np.random.normal(0, 0.04)
                lon = user["home_lon"] + np.random.normal(0, 0.04)
                loc_country = user["home_country"]
            else:
                dest = random.choice(self.CITIES)
                lat = dest["lat"] + np.random.normal(0, 0.05)
                lon = dest["lon"] + np.random.normal(0, 0.05)
                loc_country = dest["country"]

            # Amount: centered around user and merchant typical habits
            base_amount = (user["avg_spend"] * 0.4) + (merchant["avg_amount"] * 0.6)
            amount = max(2.50, round(float(np.random.normal(base_amount, base_amount * 0.25)), 2))

            records.append({
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "timestamp": chosen_time,
                "user_id": user["user_id"],
                "card_id": user["card_id"],
                "merchant_id": merchant["merchant_id"],
                "merchant_category": merchant["category"],
                "mcc": merchant["mcc"],
                "amount": amount,
                "channel": channel,
                "device_id": device_id,
                "ip_address": ip_address,
                "location_lat": lat,
                "location_lon": lon,
                "country": loc_country,
                "is_fraud": 0,
                "fraud_type": "legitimate",
            })

        # 2. Inject Complex Fraud Typologies
        fraud_distribution = {
            "card_not_present_burst": int(total_fraud_count * 0.25),
            "account_takeover": int(total_fraud_count * 0.25),
            "impossible_travel": int(total_fraud_count * 0.20),
            "aml_structuring": int(total_fraud_count * 0.15),
            "mule_network_ring": total_fraud_count - int(total_fraud_count * 0.85),
        }

        # Typology 1: Card-Not-Present (CNP) Velocity Burst
        records.extend(self._inject_cnp_burst(fraud_distribution["card_not_present_burst"], start_time))

        # Typology 2: Account Takeover (ATO)
        records.extend(self._inject_account_takeover(fraud_distribution["account_takeover"], start_time))

        # Typology 3: Impossible Travel
        records.extend(self._inject_impossible_travel(fraud_distribution["impossible_travel"], start_time))

        # Typology 4: AML Structuring / Smurfing
        records.extend(self._inject_aml_structuring(fraud_distribution["aml_structuring"], start_time))

        # Typology 5: Mule Network Collusion Rings
        records.extend(self._inject_mule_network(fraud_distribution["mule_network_ring"], start_time))

        df = pd.DataFrame(records)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _inject_cnp_burst(self, count: int, base_time: datetime) -> List[Dict]:
        """Injects card testing micro-transactions followed by high-dollar bursts."""
        bursts: List[Dict] = []
        burst_size = 5
        num_bursts = max(1, count // burst_size)

        high_risk_merchants = [m for m in self.merchants if m["mcc"] in ["5732", "6051", "5944"]]
        if not high_risk_merchants:
            high_risk_merchants = self.merchants[:5]

        for _ in range(num_bursts):
            victim = random.choice(self.users)
            attacker_device = f"DEV_ROGUE_{uuid.uuid4().hex[:8].upper()}"
            attacker_ip = f"104.28.{random.randint(10, 240)}.{random.randint(1, 250)}"  # Cloudflare/VPN IP
            burst_start = base_time + timedelta(days=random.randint(5, 50), hours=random.randint(0, 23))

            # Step 1: Card probe / micro-auth
            probe_time = burst_start
            bursts.append({
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "timestamp": probe_time,
                "user_id": victim["user_id"],
                "card_id": victim["card_id"],
                "merchant_id": random.choice(self.merchants)["merchant_id"],
                "merchant_category": "Telecommunications",
                "mcc": "4814",
                "amount": round(random.uniform(1.05, 3.50), 2),
                "channel": "web",
                "device_id": attacker_device,
                "ip_address": attacker_ip,
                "location_lat": victim["home_lat"] + 5.0,
                "location_lon": victim["home_lon"] - 5.0,
                "country": victim["home_country"],
                "is_fraud": 1,
                "fraud_type": "card_not_present_burst",
            })

            # Step 2: High velocity expensive drain
            for step in range(1, burst_size):
                probe_time += timedelta(minutes=random.randint(1, 4), seconds=random.randint(10, 50))
                merch = random.choice(high_risk_merchants)
                bursts.append({
                    "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                    "timestamp": probe_time,
                    "user_id": victim["user_id"],
                    "card_id": victim["card_id"],
                    "merchant_id": merch["merchant_id"],
                    "merchant_category": merch["category"],
                    "mcc": merch["mcc"],
                    "amount": round(random.uniform(1200.0, 3800.0), 2),
                    "channel": "web",
                    "device_id": attacker_device,
                    "ip_address": attacker_ip,
                    "location_lat": victim["home_lat"] + 5.0,
                    "location_lon": victim["home_lon"] - 5.0,
                    "country": victim["home_country"],
                    "is_fraud": 1,
                    "fraud_type": "card_not_present_burst",
                })

        return bursts

    def _inject_account_takeover(self, count: int, base_time: datetime) -> List[Dict]:
        """Injects account takeover at abnormal hours with new device & foreign IP."""
        ato_records: List[Dict] = []
        for _ in range(count):
            victim = random.choice(self.users)
            foreign_city = random.choice([c for c in self.CITIES if c["country"] != victim["home_country"]])
            hacker_ip = f"185.{random.randint(100, 220)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            hacker_device = f"DEV_HIJACK_{uuid.uuid4().hex[:8].upper()}"

            # 2:00 AM - 4:30 AM
            ato_time = base_time + timedelta(
                days=random.randint(3, 52),
                hours=random.choice([2, 3, 4]),
                minutes=random.randint(5, 55),
            )
            merch = random.choice([m for m in self.merchants if m["risk_weight"] > 0.4] or self.merchants)

            # High amount compared to user normal
            ato_amount = round(victim["avg_spend"] * random.uniform(6.0, 15.0) + random.uniform(800.0, 2500.0), 2)

            ato_records.append({
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "timestamp": ato_time,
                "user_id": victim["user_id"],
                "card_id": victim["card_id"],
                "merchant_id": merch["merchant_id"],
                "merchant_category": merch["category"],
                "mcc": merch["mcc"],
                "amount": ato_amount,
                "channel": random.choice(["mobile", "web"]),
                "device_id": hacker_device,
                "ip_address": hacker_ip,
                "location_lat": foreign_city["lat"],
                "location_lon": foreign_city["lon"],
                "country": foreign_city["country"],
                "is_fraud": 1,
                "fraud_type": "account_takeover",
            })
        return ato_records

    def _inject_impossible_travel(self, count: int, base_time: datetime) -> List[Dict]:
        """Injects consecutive transactions spaced < 60 mins apart across distant cities."""
        travel_records: List[Dict] = []
        pairs_count = max(1, count // 2)

        for _ in range(pairs_count):
            victim = random.choice(self.users)
            city_a = random.choice(self.CITIES)
            city_b = random.choice([c for c in self.CITIES if c["name"] != city_a["name"]])

            t1 = base_time + timedelta(days=random.randint(5, 50), hours=random.randint(8, 20), minutes=random.randint(0, 30))
            t2 = t1 + timedelta(minutes=random.randint(12, 45))  # Physically impossible to fly across continents

            merch1 = random.choice(self.merchants)
            merch2 = random.choice(self.merchants)

            # Legit appearance for 1st
            travel_records.append({
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "timestamp": t1,
                "user_id": victim["user_id"],
                "card_id": victim["card_id"],
                "merchant_id": merch1["merchant_id"],
                "merchant_category": merch1["category"],
                "mcc": merch1["mcc"],
                "amount": round(random.uniform(25.0, 180.0), 2),
                "channel": "pos",
                "device_id": "POS_TERMINAL_LOC1",
                "ip_address": "POS_NET_1",
                "location_lat": city_a["lat"],
                "location_lon": city_a["lon"],
                "country": city_a["country"],
                "is_fraud": 0,  # original was legit
                "fraud_type": "legitimate",
            })

            # Impossible duplicate clone
            travel_records.append({
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "timestamp": t2,
                "user_id": victim["user_id"],
                "card_id": victim["card_id"],
                "merchant_id": merch2["merchant_id"],
                "merchant_category": merch2["category"],
                "mcc": merch2["mcc"],
                "amount": round(random.uniform(350.0, 1400.0), 2),
                "channel": "pos",
                "device_id": "POS_TERMINAL_LOC2",
                "ip_address": "POS_NET_2",
                "location_lat": city_b["lat"],
                "location_lon": city_b["lon"],
                "country": city_b["country"],
                "is_fraud": 1,
                "fraud_type": "impossible_travel",
            })

        return travel_records

    def _inject_aml_structuring(self, count: int, base_time: datetime) -> List[Dict]:
        """Injects smurfing/structuring transactions designed to evade the $10k CTR threshold."""
        struct_records: List[Dict] = []
        group_size = 3
        num_groups = max(1, count // group_size)

        for _ in range(num_groups):
            user = random.choice(self.users)
            group_start = base_time + timedelta(days=random.randint(5, 50), hours=random.randint(9, 17))
            device = f"DEV_SMURF_{uuid.uuid4().hex[:6].upper()}"

            for step in range(group_size):
                tx_time = group_start + timedelta(hours=step * random.randint(6, 18), minutes=random.randint(10, 50))
                # Amount clustered just below $10,000 CTR threshold (e.g. $9,200 to $9,950)
                amount = round(random.uniform(config.AML_STRUCTURING_MIN, config.AML_STRUCTURING_MAX), 2)
                merch = random.choice([m for m in self.merchants if m["mcc"] in ["6051", "5944"]] or self.merchants)

                struct_records.append({
                    "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                    "timestamp": tx_time,
                    "user_id": user["user_id"],
                    "card_id": user["card_id"],
                    "merchant_id": merch["merchant_id"],
                    "merchant_category": merch["category"],
                    "mcc": merch["mcc"],
                    "amount": amount,
                    "channel": "wire_transfer",
                    "device_id": device,
                    "ip_address": f"198.51.100.{random.randint(1, 200)}",
                    "location_lat": user["home_lat"] + 0.01,
                    "location_lon": user["home_lon"] + 0.01,
                    "country": user["home_country"],
                    "is_fraud": 1,
                    "fraud_type": "aml_structuring",
                })

        return struct_records

    def _inject_mule_network(self, count: int, base_time: datetime) -> List[Dict]:
        """Injects mule ring transactions where a shared device/IP links multiple mule accounts."""
        mule_records: List[Dict] = []
        syndicate_device = f"DEV_MULE_RING_MASTER"
        syndicate_ip = "203.0.113.88"
        mule_users = random.sample(self.users, min(6, len(self.users)))
        mule_start = base_time + timedelta(days=random.randint(10, 48), hours=random.randint(1, 22))

        for idx, mule_user in enumerate(mule_users):
            if len(mule_records) >= count:
                break
            tx_time = mule_start + timedelta(minutes=idx * random.randint(15, 45))
            merch = random.choice([m for m in self.merchants if m["mcc"] == "6051"] or self.merchants)
            amount = round(random.uniform(2800.0, 7500.0), 2)

            mule_records.append({
                "transaction_id": f"TX_{uuid.uuid4().hex[:12].upper()}",
                "timestamp": tx_time,
                "user_id": mule_user["user_id"],
                "card_id": mule_user["card_id"],
                "merchant_id": merch["merchant_id"],
                "merchant_category": merch["category"],
                "mcc": merch["mcc"],
                "amount": amount,
                "channel": "web",
                "device_id": syndicate_device,  # Shared infrastructure
                "ip_address": syndicate_ip,      # Shared infrastructure
                "location_lat": mule_user["home_lat"],
                "location_lon": mule_user["home_lon"],
                "country": mule_user["home_country"],
                "is_fraud": 1,
                "fraud_type": "mule_network_ring",
            })

        return mule_records


def generate_and_save_data(output_path: str = None) -> pd.DataFrame:
    """Convenience function to generate and persist transaction dataset."""
    if output_path is None:
        output_path = config.RAW_DATA_DIR / "transactions.csv"
    
    generator = FinancialTransactionGenerator()
    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"[Data Generator] Successfully generated {len(df)} transactions -> {output_path}")
    print(f"                 Total Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print("                 Fraud breakdown by typology:")
    for ftype, cnt in df[df['is_fraud'] == 1]['fraud_type'].value_counts().items():
        print(f"                   - {ftype}: {cnt}")
    return df


if __name__ == "__main__":
    generate_and_save_data()
