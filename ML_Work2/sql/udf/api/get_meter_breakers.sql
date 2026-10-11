-- Circuit-breaker rating per meter (NULL when none recorded). Called by MeterRepository.
CREATE SCHEMA IF NOT EXISTS api;

CREATE OR REPLACE FUNCTION api.get_meter_breakers()
RETURNS TABLE (
    meter_key        text,
    meter_label      text,
    cb_max_rating_a  double precision
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    WITH meters AS (
        SELECT m.energy_meter_key, m.meter_label::text AS meter_label
        FROM gold.meter m
    ),
    ratings AS (
        SELECT cb.energy_meter_key, cb.cb_max_rating_a::double precision AS cb_max_rating_a
        FROM silver.dim_meter_cb_rating cb
    )
    SELECT m.energy_meter_key::text,
           m.meter_label,
           (SELECT r.cb_max_rating_a FROM ratings r WHERE r.energy_meter_key = m.energy_meter_key)
    FROM meters m
    ORDER BY m.meter_label;
END;
$$;
