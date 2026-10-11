-- Minimal rebuild of the gold/silver objects read by forecasting/, from the bronze restore.
-- Usage: psql -h localhost -U postgres -d sadinhub -f rebuild_gold_from_bronze.sql
-- NOT the original recovered SQL (absent from the repo). See notes in the README of this session.
DROP SCHEMA IF EXISTS gold CASCADE;
DROP SCHEMA IF EXISTS silver CASCADE;
CREATE SCHEMA gold;
CREATE SCHEMA silver;

-- One canonical row per meter: energy_meter_key = bronze device_id.
-- The crossed ZN.20 mapping (device ZN.20-PB.1 also listed as meter ZN.20-PB.2) is excluded.
CREATE TABLE gold.meter AS
SELECT em.device_id AS energy_meter_key,
       em.meter_label,
       em.unit_id AS zone_key,
       em.facility_id AS facility_key,
       em.grid_billing_account_id
FROM bronze.energy_meters em
JOIN bronze.device d ON d.id = em.device_id
WHERE NOT (em.meter_label = 'ZN.20-PB.2_child__1_ch3pm'
           AND d.name = 'ZN.20-PB.1_child__1_ch3pm');

CREATE TABLE gold.zone AS
SELECT u.id AS zone_key, u.facility_id AS facility_key, u.name AS zone_name,
       u.latitude, u.longitude
FROM bronze.units u
WHERE u.id IN (SELECT zone_key FROM gold.meter);

-- One row per meter per timestamp: cumulative kWh (telemetry key 'kWh').
CREATE TABLE gold.energy_raw AS
SELECT t.device_id AS energy_meter_key,
       TIMESTAMP '1970-01-01' + t."timestamp" * INTERVAL '1 millisecond' AS reading_timestamp_utc,
       t.dbl AS active_energy_cumulative_kwh
FROM bronze.telemetries t
JOIN bronze.telemetries_dictionaries k ON k.id = t.key_id AND k.key = 'kWh'
WHERE t.device_id IN (SELECT energy_meter_key FROM gold.meter);
ALTER TABLE gold.energy_raw ADD PRIMARY KEY (energy_meter_key, reading_timestamp_utc);

-- STAND-IN: the original hand-maintained breaker ratings (80-250 A) were not recovered.
-- This uses the billing-account breaker_capacity from bronze. Only the bill forecast depends on it.
CREATE TABLE silver.dim_meter_cb_rating AS
SELECT m.energy_meter_key, g.breaker_capacity::double precision AS cb_max_rating_a, now() AS updated_at
FROM gold.meter m
JOIN bronze.grid_billing_accounts g ON g.id = m.grid_billing_account_id;

ANALYZE gold.energy_raw;
