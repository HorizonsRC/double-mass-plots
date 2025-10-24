from contextlib import suppress

from hilltoppy import Hilltop
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os
import sqlalchemy as db
import platform
import xmltodict
import json


load_dotenv()
base_url = os.getenv("BASE_URL")
hts = os.getenv("HTS")
ht = Hilltop(base_url, hts)
measurement = "Rainfall [Rainfall]"

# Sites that don't have primary scada, are too new to be processed, or causes some other problem.
excluded_sites = [
    "Akitio at Weber Intersection",
    "L Horowhenua Inflow at Lindsay Road",
    "Makino at Floodgates",
    "Makino at MDC Water Supply",
    "Makuri at Tuscan Hills",
    "Manawatu at Opiki Floodway",
    "Manawatu at Victoria Ave",
    "Mangahao at Ballance",
    "Ohau at Haines Ford",
    "Waihi at S.H.52",
    "Waikawa at North Manakau Road",
    "Wainui at Weber Intersection",
]

backup_measurements = [
    "Rainfall [SCADA Rainfall]",
    "Rainfall (backup) [SCADA Rainfall (backup)]",
    "Rainfall (backup) [Rainfall (backup sensors)]",
]


def monthly_sum(df):
    """Give monthly sums."""
    monthly = df.groupby([lambda x: x.year, lambda x: x.month]).sum()
    return monthly.iloc[1:-1]  # exclude partial months


def generate_site_data(site: str, filter_quality: int | None):
    """
    Generates data for a site read for parsing as double mass plots.

    Parameters
    ----------
    site
    filter_quality

    Returns
    -------

    """
    site_with_only_backup = False
    try:
        data = ht.get_data(site, measurement, from_date=None, to_date=None).set_index(
            ["Time"]
        )
    except ValueError:
        # This catches missing sites
        # a missing measurement returns an empty df, so we do that
        data = pd.DataFrame()

    # check if we need to try for a backup datasource
    if data.empty:
        for meas in backup_measurements:
            with suppress(ValueError):
                data = ht.get_data(site, meas, from_date=None, to_date=None).set_index(
                    ["Time"]
                )
            if not data.empty:
                site_with_only_backup = True

    if filter_quality:
        try:
            qc_data = ht.get_data(
                site,
                measurement,
                from_date=None,
                to_date=None,
                tstype="Quality",
            ).set_index(["Time"])
            if qc_data.empty:
                data = pd.Series()
            else:
                qc_series = qc_data.Value

                worst_qc_this_month = qc_series.copy()
                worst_qc_this_month.index = worst_qc_this_month.index.map(
                    lambda x: pd.Timestamp(year=x.year, month=x.month, day=1)
                )
                worst_qc_this_month = worst_qc_this_month.groupby(
                    worst_qc_this_month.index
                ).min()

                monthly_qc = pd.concat(
                    [
                        worst_qc_this_month.reindex(data.Value.index, fill_value=1000),
                        qc_series.reindex(data.Value.index, method="ffill"),
                    ],
                    axis=1,
                ).min(axis=1)

                data = data[monthly_qc > filter_quality]
        except ValueError:
            data = pd.Series()

    return data, site_with_only_backup


def title_sites_str(site1, site2, sites_using_backup):
    """"""
    backup1 = site1 in sites_using_backup
    if backup1:
        backup1 = "BACKUP " + site1
    else:
        backup1 = site1
    backup2 = site2 in sites_using_backup
    if backup2:
        backup2 = "BACKUP " + site2
    else:
        backup2 = site2
    return backup1 + " vs " + backup2


def plot_site_supplement_pairs(
    main_site: str, supplementary_sites: [str], filter_quality: int | None = 400
):
    """Give plots and correlation values for rainfall."""

    base_save_dir = f"./output/{main_site}/"
    os.makedirs(base_save_dir, exist_ok=True)

    monthly_data = []
    sites_using_backup = []
    for site in [main_site] + supplementary_sites:
        print(f"Obtaining {site} data for comparison to {main_site}")
        data, used_backup = generate_site_data(site=site, filter_quality=filter_quality)
        if used_backup:
            sites_using_backup.append(site)

        if not data.empty:
            monthly_data.append(monthly_sum(data.Value.rename(site)))
        elif site == main_site:
            print("")
            print(f"No valid data for main site {site}")
            print("")
            return {site: {}}, {site: {}}, {site: {}}
        else:
            supplementary_sites.remove(site)

    combined_df = pd.concat(monthly_data, axis=1).sort_index()

    correlation_matrix = combined_df.corr()

    for site in supplementary_sites:
        plt.figure()
        plt.scatter(combined_df[main_site], combined_df[site])
        plt.title(
            f"R2 {title_sites_str(main_site, site, sites_using_backup)}, PCC={correlation_matrix.loc[main_site, site]}"
        )
        plt.savefig(base_save_dir + f"{main_site}_correlation_{site}.png", format="png")
        plt.close()

    scale_dict = {}
    for site in supplementary_sites:
        cumulative_df = combined_df[
            ~combined_df[main_site].isnull() & ~combined_df[site].isnull()
        ].cumsum()
        if (not cumulative_df[main_site].empty) and (not cumulative_df[site].empty):
            regression = LinearRegression(fit_intercept=False).fit(
                cumulative_df[main_site].values.reshape(-1, 1),
                cumulative_df[site].values.reshape(-1, 1),
            )
            scale_dict[site] = float(regression.coef_[0][0])
            plt.figure()
            plt.scatter(cumulative_df[main_site], cumulative_df[site])
            max_value = max(
                cumulative_df[main_site][~cumulative_df[main_site].isnull()]
            )
            plt.plot(
                np.array([0, max_value]).reshape(-1, 1),
                regression.predict(np.array([0, max_value]).reshape(-1, 1)),
                color="k",
            )
            plt.title(
                f"DMP {title_sites_str(main_site, site, sites_using_backup)}, Scale={float(regression.coef_[0][0])}"
            )
            plt.savefig(base_save_dir + f"{main_site}_fit_{site}.png", format="png")
            plt.close()

    # plt.show()
    return (
        dict(correlation_matrix.loc[main_site].drop(main_site)),
        scale_dict,
        sites_using_backup,
    )


def sql_server_url():
    """Return URL for SQL server host computer."""
    if platform.system() == "Windows":
        hostname = os.getenv("WINDOWS_HOSTNAME")
    elif platform.system() == "Linux":
        # Nic's WSL support (with apologies). THIS IS NOT STABLE.
        hostname = os.getenv("LINUX_HOSTNAME")
    else:
        raise OSError("What is this, a mac? Get up on out of here, capitalist pig.")
    return hostname


def find_r_rain(site):
    """Given a site, look up the r_rain in the hilltop sites table."""
    query = db.text(
        """
        SELECT TOP (1) [SiteInfo]
        FROM [hilltop].[dbo].[Sites]
        WHERE SiteName = :site
        """
    )

    result = pd.read_sql(
        query,
        db.create_engine(
            db.engine.URL.create(
                os.getenv("DRIVER_NAME"),
                host=sql_server_url(),
                database="hilltop",
                query={"driver": os.getenv("QUERY_DRIVER")},
            )
        ),
        params={
            "site": site,
        },
    )

    xml_string = result.loc[0, "SiteInfo"]
    xml_dict = xmltodict.parse(xml_string)["SiteInfo"]
    return [x for x in [i.strip() for i in xml_dict["R_Rain"].split(",")] if len(x) > 0]


def get_site_list():
    """Returns all sites with R_Rain in the siteinfo."""
    query = db.text(
        """
        SELECT TOP (1000) [SiteName]
        FROM [hilltop].[dbo].[Sites]
        WHERE cast(SiteInfo as nvarchar(max)) LIKE '%R_Rain%'
        """
    )

    result = pd.read_sql(
        query,
        db.create_engine(
            db.engine.URL.create(
                os.getenv("DRIVER_NAME"),
                host=sql_server_url(),
                database="hilltop",
                query={"driver": os.getenv("QUERY_DRIVER")},
            )
        ),
    )

    site_list = list(result.SiteName)

    return sorted(site_list)


def find_empty_sites():
    backup_only_sites = []
    for site in sorted(get_site_list()):
        print(site)
        try:
            data = ht.get_data(site, measurement, from_date=None, to_date=None)
            if data.empty:
                backup_only_sites.append(site)
                print("MISSING")
                print(backup_only_sites)
        except ValueError:
            backup_only_sites.append(site)
            print("FAILED")
            print(backup_only_sites)
    print("FINAL LIST")
    print(backup_only_sites)
    return backup_only_sites


if __name__ == "__main__":
    main_save_dir = f"./output/aa_stats/"
    scales = {}
    correlations = {}
    backups = {}
    site = "Lower Retaruke"
    correlation, scale, backup = plot_site_supplement_pairs(
        site,
        [
            "Whanganui at Te Porere",
            "Whangamomona at Marco Road",
            "Ruatiti at Ruatiti Station",
            "Ohura at Ohura River Road",
            "Makahiwi at Repeater",
            "Air Quality at Taumarunui",
        ],
        filter_quality=400,
    )
    scales[site] = scale
    correlations[site] = correlation
    backups[site] = backup
    """
    for site in get_site_list():
        correlation, scale, backup = plot_site_supplement_pairs(
            site, find_r_rain(site), filter_quality=400
        )
        scales[site] = scale
        correlations[site] = correlation
        backups[site] = backup
    """
    os.makedirs(main_save_dir, exist_ok=True)
    with open(main_save_dir + "correlation_values.json", "w") as file:
        file.write(json.dumps(correlations))
    with open(main_save_dir + "scale_values.json", "w") as file:
        file.write(json.dumps(scales))
    with open(main_save_dir + "backup_sites.json", "w") as file:
        file.write(json.dumps(backups))
