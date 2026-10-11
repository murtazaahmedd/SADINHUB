-- Daily validated kWh per meter as the gold layer books it. Called by GoldReferenceRepository.
CREATE SCHEMA IF NOT EXISTS api;

CREATE OR REPLACE FUNCTION api.get_gold_daily_consumption()
RETURNS TABLE (
    meter_label  text,
    reading_date date,
    gold_kwh     double precision
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    RETURN QUERY
    WITH booked AS (
        SELECT c.meter_label::text AS meter_label,
               c.reading_date_saudi::date AS reading_date,
               c.validated_kwh_consumption AS kwh
        FROM gold.energy_consumption_validated c
    )
    SELECT b.meter_label, b.reading_date, SUM(b.kwh)::double precision
    FROM booked b
    GROUP BY b.meter_label, b.reading_date;
END;
$$;
