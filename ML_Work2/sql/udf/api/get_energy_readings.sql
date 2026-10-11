-- Every cumulative-kWh reading of one meter, oldest first. Called by EnergyReadingRepository.
CREATE SCHEMA IF NOT EXISTS api;

CREATE OR REPLACE FUNCTION api.get_energy_readings(p_meter_key uuid)
RETURNS TABLE (
    reading_timestamp_utc         timestamp,
    active_energy_cumulative_kwh  double precision
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    WITH readings AS (
        SELECT r.reading_timestamp_utc::timestamp AS reading_timestamp_utc,
               r.active_energy_cumulative_kwh::double precision AS active_energy_cumulative_kwh
        FROM gold.energy_raw r
        WHERE r.energy_meter_key = p_meter_key
    )
    SELECT rd.reading_timestamp_utc, rd.active_energy_cumulative_kwh
    FROM readings rd
    ORDER BY rd.reading_timestamp_utc;
END;
$$;
