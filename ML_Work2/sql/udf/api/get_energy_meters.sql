-- One row per canonical meter with its zone. Read-only; called by MeterRepository.
CREATE SCHEMA IF NOT EXISTS api;

CREATE OR REPLACE FUNCTION api.get_energy_meters()
RETURNS TABLE (
    meter_key       text,
    meter_label     text,
    zone_name       text,
    zone_latitude   double precision,
    zone_longitude  double precision
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    WITH meters AS (
        SELECT m.energy_meter_key, m.meter_label::text AS meter_label, m.zone_key
        FROM gold.meter m
    ),
    zones AS (
        SELECT z.zone_key, z.zone_name::text AS zone_name,
               z.latitude::double precision AS latitude, z.longitude::double precision AS longitude
        FROM gold.zone z
    )
    SELECT m.energy_meter_key::text,
           m.meter_label,
           (SELECT z.zone_name FROM zones z WHERE z.zone_key = m.zone_key),
           (SELECT z.latitude  FROM zones z WHERE z.zone_key = m.zone_key),
           (SELECT z.longitude FROM zones z WHERE z.zone_key = m.zone_key)
    FROM meters m
    ORDER BY m.meter_label;
END;
$$;
