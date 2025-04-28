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
]


def monthly_sum(df):
    """Give monthly sums."""
    monthly = df.groupby([lambda x: x.year, lambda x: x.month]).sum()
    return monthly.iloc[1:-1]  # exclude partial months


def plot_site_supplement_pairs(main_site: str):
    """Give plots and correlation values for rainfall."""

    base_save_dir = f"./output/{main_site}/"
    os.makedirs(base_save_dir, exist_ok=True)
    supplementary_sites = find_r_rain(main_site)
    all_sites = [main_site] + supplementary_sites

    monthly_data = []
    for site in all_sites:
        print(f"Obtaining {site} data for comparison to {main_site}")
        data = ht.get_data(site, measurement, from_date=None, to_date=None).set_index(
            ["Time"]
        )
        if not data.empty:
            monthly_data.append(monthly_sum(data.Value.rename(site)))
        else:
            raise ValueError(f"No data at Site: {site}, supplement of {main_site}")
    combined_df = pd.concat(monthly_data, axis=1).sort_index()

    correlations = combined_df.corr()

    for site in supplementary_sites:
        print(f"Calculating correlation for {main_site} & {site}")
        plt.figure()
        plt.scatter(combined_df[main_site], combined_df[site])
        plt.title(
            f"Monthly rain {main_site} vs {site}, PCC={correlations.loc[main_site, site]}"
        )
        plt.savefig(base_save_dir + f"{main_site}_correlation_{site}.png", format="png")
        plt.close()

    scale_dict = {}
    for site in supplementary_sites:
        print(f"Plotting double mass for {main_site} & {site}")
        cumulative_df = combined_df[
            ~combined_df[main_site].isnull() & ~combined_df[site].isnull()
        ].cumsum()
        regression = LinearRegression(fit_intercept=False).fit(
            cumulative_df[main_site].values.reshape(-1, 1),
            cumulative_df[site].values.reshape(-1, 1),
        )
        scale_dict[site] = float(regression.coef_[0][0])
        plt.figure()
        plt.scatter(cumulative_df[main_site], cumulative_df[site])
        max_value = max(cumulative_df[main_site][~cumulative_df[main_site].isnull()])
        plt.plot(
            np.array([0, max_value]).reshape(-1, 1),
            regression.predict(np.array([0, max_value]).reshape(-1, 1)),
            color="k",
        )
        plt.title(
            f"Double mass rain {main_site} vs {site}, Scale={float(regression.coef_[0][0])}"
        )
        plt.savefig(base_save_dir + f"{main_site}_fit_{site}.png", format="png")
        plt.close()

    # plt.show()
    return dict(correlations.loc[main_site].drop(main_site)), scale_dict


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
    return [
        s
        for s in [i.strip() for i in xml_dict["R_Rain"].split(",")]
        if ((len(s) > 0) and (s not in excluded_sites))
    ]


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

    return sorted([s for s in site_list if s not in excluded_sites])


def find_empty_sites():
    backup_only_sites = []
    for site in sorted(get_site_list())[9:]:
        print(site)
        try:
            data = ht.get_data(site, measurement, from_date=None, to_date=None)
            if data.empty:
                backup_only_sites.append(site)
                print("MISSING")
                print(backup_only_sites)
        except:
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
    for s in get_site_list():
        correlation, scale = plot_site_supplement_pairs(s)
        scales[s] = scale
        correlations[s] = correlation
    os.makedirs(main_save_dir, exist_ok=True)
    with open(main_save_dir + "correlation_values.txt", "w") as file:
        file.write(json.dumps(correlations))
    with open(main_save_dir + "scale_values.txt", "w") as file:
        file.write(json.dumps(scales))
