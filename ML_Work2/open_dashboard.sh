#!/usr/bin/env bash
# Open the offline dashboard in the default browser (Linux/macOS). Windows: Open_Dashboard.cmd
cd "$(dirname "$0")" || exit 1
xdg-open dashboard/SADIN_Energy_Forecast_Dashboard.html 2>/dev/null || open dashboard/SADIN_Energy_Forecast_Dashboard.html
